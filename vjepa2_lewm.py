"""Training-only V-JEPA 2.1 + LeWM model kept separate from vanilla LeWM.

The frozen V-JEPA backbone encodes a causal 64-frame history for each control
state. A trainable learned-query pooler consumes every spatiotemporal patch
token and produces the 192-dimensional state used by a newly initialized LeWM
action-conditioned predictor.
"""

from pathlib import Path
import sys

from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn

from module import ARPredictor, Embedder, MLP


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LearnedQueryPooler(nn.Module):
    """Efficient single-query attention over all V-JEPA video tokens."""

    def __init__(self, input_dim=768, output_dim=192):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.key = nn.Linear(input_dim, output_dim, bias=False)
        self.value = nn.Linear(input_dim, output_dim, bias=False)
        self.query = nn.Parameter(torch.empty(1, 1, output_dim))
        self.output = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, 4 * output_dim),
            nn.GELU(),
            nn.Linear(4 * output_dim, output_dim),
        )
        nn.init.normal_(self.query, std=0.02)

    def forward(self, tokens):
        tokens = self.norm(tokens)
        keys = self.key(tokens)
        values = self.value(tokens)
        query = self.query.expand(tokens.size(0), -1, -1)
        scores = torch.matmul(query, keys.transpose(-2, -1))
        scores = scores * (keys.size(-1) ** -0.5)
        weights = scores.softmax(dim=-1)
        pooled = torch.matmul(weights, values).squeeze(1)
        return pooled + self.output(pooled)


class VJEPAActionPredictor(ARPredictor):
    """Fresh action-conditioned predictor for V-JEPA state latents.

    It intentionally shares vanilla LeWM's proven architecture while owning a
    separate parameter set and checkpoint namespace.
    """


class VJEPA21LeWM(nn.Module):
    """Independent V-JEPA-state LeWM; does not alter vanilla JEPA/ARPredictor."""

    def __init__(
        self,
        checkpoint_path,
        vjepa2_repo="/home/coder/P-Rep/vjepa2",
        image_size=384,
        num_video_frames=64,
        embed_dim=192,
        history_size=3,
        action_dim=25,
        predictor_depth=6,
        predictor_heads=16,
        predictor_mlp_dim=2048,
        encoder_chunk_size=1,
        freeze_backbone=True,
    ):
        super().__init__()
        repo = str(Path(vjepa2_repo))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from app.vjepa_2_1.models.vision_transformer import vit_base

        self.image_size = image_size
        self.num_video_frames = num_video_frames
        self.encoder_chunk_size = encoder_chunk_size
        self.freeze_backbone = freeze_backbone
        self.backbone = vit_base(
            img_size=(image_size, image_size),
            patch_size=16,
            num_frames=num_video_frames,
            tubelet_size=2,
            use_sdpa=True,
            use_silu=False,
            wide_silu=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in checkpoint["ema_encoder"].items()
        }
        self.backbone.load_state_dict(state_dict, strict=True)
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        self.pooler = LearnedQueryPooler(self.backbone.embed_dim, embed_dim)
        self.projector = MLP(embed_dim, 2048, embed_dim, norm_fn=nn.BatchNorm1d)
        self.predictor = VJEPAActionPredictor(
            num_frames=history_size,
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            depth=predictor_depth,
            heads=predictor_heads,
            dim_head=64,
            mlp_dim=predictor_mlp_dim,
            dropout=0.1,
            emb_dropout=0.0,
        )
        self.action_encoder = Embedder(input_dim=action_dim, emb_dim=embed_dim)
        self.pred_proj = MLP(embed_dim, 2048, embed_dim, norm_fn=nn.BatchNorm1d)

        self.register_buffer(
            "pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1)
        )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _prepare_video(self, video):
        # Input is (B,F,C,H,W), uint8 or [0,255] float.
        video = video.to(device=next(self.backbone.parameters()).device)
        video = video.to(dtype=next(self.backbone.parameters()).dtype)
        if video.max() > 1:
            video = video / 255.0
        if video.shape[-2:] != (self.image_size, self.image_size):
            b, f, c, h, w = video.shape
            video = F.interpolate(
                video.reshape(b * f, c, h, w),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(b, f, c, self.image_size, self.image_size)
        video = video.permute(0, 2, 1, 3, 4)
        return (video - self.pixel_mean) / self.pixel_std

    def _encode_clips(self, clips):
        pooled = []
        for start in range(0, len(clips), self.encoder_chunk_size):
            video = self._prepare_video(clips[start : start + self.encoder_chunk_size])
            if self.freeze_backbone:
                with torch.no_grad():
                    tokens = self.backbone(video)
            else:
                tokens = self.backbone(video)
            pooled.append(self.pooler(tokens))
        return torch.cat(pooled, dim=0)

    def encode(self, info):
        pixels = info["pixels"]
        if pixels.ndim != 6:
            raise ValueError(
                "VJEPA21LeWM training expects pixels shaped (B,T,F,C,H,W); "
                f"received {tuple(pixels.shape)}"
            )
        b, t, f = pixels.shape[:3]
        if f != self.num_video_frames:
            raise ValueError(f"Expected {self.num_video_frames} context frames, got {f}")
        clips = rearrange(pixels, "b t f c h w -> (b t) f c h w")
        state = self._encode_clips(clips)
        state = self.projector(state)
        info["emb"] = rearrange(state, "(b t) d -> b t d", b=b, t=t)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(self, emb, act_emb):
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=emb.size(0), t=emb.size(1))


__all__ = ["LearnedQueryPooler", "VJEPAActionPredictor", "VJEPA21LeWM"]
