"""Frozen LeWM encoder/projector adapter used by downstream policy heads."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


def _load_local_world_model(source_checkpoint):
    """Load an arbitrary local stable-worldmodel export without cache-path assumptions."""
    from hydra.utils import instantiate

    source = Path(source_checkpoint).expanduser().resolve()
    if source.is_dir():
        candidates = sorted(source.glob("*.pt"))
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one .pt in {source}, found {len(candidates)}")
        weights = candidates[0]
        config_path = source / "config.json"
    elif source.suffix == ".pt":
        weights = source
        config_path = source.parent / "config.json"
    else:
        raise ValueError("source_checkpoint must be a local export directory or .pt file")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing LeWM config: {config_path}")
    model = instantiate(json.loads(config_path.read_text()))
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


class LeWMFrozenEncoder(nn.Module):
    """Expose the real LeWM encoder+projector as one global 192-D latent."""

    def __init__(self, source_checkpoint=None, world_model=None, latent_dim=192):
        super().__init__()
        if (source_checkpoint is None) == (world_model is None):
            raise ValueError("Provide exactly one source_checkpoint or world_model")
        if int(latent_dim) != 192:
            raise ValueError("This port supports the original LeWM 192-D latent only")
        if world_model is None:
            world_model = _load_local_world_model(source_checkpoint)
        self.encoder = world_model.encoder
        self.projector = world_model.projector
        self.config = SimpleNamespace(hidden_size=192)
        self.num_patches = 1
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        # Calling policy.train() must not reactivate encoder dropout or BN updates.
        super().train(False)
        return self

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1:] != (3, 224, 224):
            raise ValueError(f"Expected normalized BCHW 224 RGB, got {tuple(inputs.shape)}")
        with torch.no_grad():
            output = self.encoder(inputs.float(), interpolate_pos_encoding=True)
            pixels_emb = output.last_hidden_state[:, 0]
            latent = self.projector(pixels_emb)
        if latent.shape != (inputs.shape[0], 192) or not torch.isfinite(latent).all():
            raise ValueError(f"Expected finite global latent (batch,192), got {tuple(latent.shape)}")
        return SimpleNamespace(logits=latent)
