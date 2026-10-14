"""V-JEPA 2.1 encoder adapter for the frozen LeWM Cube predictor."""

from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import torch
from torch import nn
from einops import rearrange


VJEPA2_REPO = Path("/home/coder/P-Rep/vjepa2")


class VJEPA21LeWMAdapter(nn.Module):
    """Expose V-JEPA 2.1 ViT-B image features through LeWM's HF-ViT API.

    V-JEPA produces 768-dimensional patch tokens and has no CLS token. We mean
    pool the tokens, then use a fixed seeded orthogonal projection to the 192
    dimensions expected by the already-trained LeWM projector and predictor.
    This projection is deliberately untrained: the adapter is an inference-only
    compatibility experiment, not a trained representation alignment.
    """

    def __init__(self, checkpoint_path, output_dim=192, projection_seed=0):
        super().__init__()
        if str(VJEPA2_REPO) not in sys.path:
            sys.path.insert(0, str(VJEPA2_REPO))

        from app.vjepa_2_1.models.vision_transformer import vit_base

        self.backbone = vit_base(
            img_size=(384, 384),
            patch_size=16,
            num_frames=64,
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
        self.supports_video_context = True

        generator = torch.Generator(device="cpu").manual_seed(projection_seed)
        matrix = torch.randn(self.backbone.embed_dim, output_dim, generator=generator)
        projection, _ = torch.linalg.qr(matrix, mode="reduced")
        self.register_buffer("projection", projection)

    def forward(self, pixels, **_kwargs):
        if pixels.ndim == 5 and pixels.size(1) > 1:
            # LeWM supplies (B,T,C,H,W); V-JEPA expects (B,C,T,H,W).
            video = pixels.permute(0, 2, 1, 3, 4)
            patch_tokens = self.backbone(video)
            temporal_tokens = video.size(2) // self.backbone.tubelet_size
            spatial_tokens = patch_tokens.size(1) // temporal_tokens
            patch_tokens = patch_tokens.reshape(
                patch_tokens.size(0), temporal_tokens, spatial_tokens, -1
            )
            # Represent the current state with the final two-frame tubelet.
            pooled = patch_tokens[:, -1].mean(dim=1)
        else:
            # T=1 selects V-JEPA 2.1's image pathway (used for the goal).
            if pixels.ndim == 5:
                pixels = pixels[:, 0]
            patch_tokens = self.backbone(pixels.unsqueeze(2))
            pooled = patch_tokens.mean(dim=1)
        embedding = pooled @ self.projection.to(dtype=pooled.dtype)
        return SimpleNamespace(last_hidden_state=embedding.unsqueeze(1))


def replace_lewm_encoder(model, checkpoint_path):
    model.encoder = VJEPA21LeWMAdapter(checkpoint_path)
    model._lewm_original_rollout = model.rollout
    model.encode = MethodType(_video_aware_encode, model)
    model.rollout = MethodType(_video_context_rollout, model)
    return model


def _video_aware_encode(self, info):
    """Encode T>1 jointly, while preserving LeWM's regular T=1 behavior."""
    pixels = info["pixels"].to(next(self.encoder.parameters()).dtype)
    b = pixels.size(0)
    if pixels.ndim == 5 and pixels.size(1) > 1:
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        info["emb"] = self.projector(pixels_emb).unsqueeze(1)
    else:
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

    if "action" in info:
        info["act_emb"] = self.action_encoder(info["action"])
    return info


def _video_context_rollout(self, info, action_sequence, history_size=None):
    """Pre-encode video context without treating 64 frames as 64 actions."""
    if "pixels_video_context" in info and "emb" not in info:
        batch_size, num_samples = info["pixels"].shape[:2]
        initial = {
            key: value[:, 0]
            for key, value in info.items()
            if torch.is_tensor(value)
        }
        initial["pixels"] = initial.pop("pixels_video_context")
        initial.pop("action", None)
        initial = self.encode(initial)
        info["emb"] = (
            initial["emb"].detach().unsqueeze(1).expand(
                batch_size, num_samples, -1, -1
            )
        )
    return self._lewm_original_rollout(info, action_sequence, history_size)
