"""Rebuild LeWM + downstream policy heads from a policy-only checkpoint."""

from pathlib import Path

import torch

from .models.frozen_lewm import LeWMFrozenEncoder
from .models.gciql import FrozenLeWMGCIQL
from .models.policy import FrozenLeWMGCBC
from .models.state_fusion import FrozenStateVisualGCBC


def load_policy(checkpoint, lewm_checkpoint=None, map_location="cpu"):
    payload = torch.load(Path(checkpoint), map_location=map_location, weights_only=False)
    if payload.get("format") != "lewm-ogbench-policy-v1":
        raise ValueError("Unsupported downstream checkpoint format")
    source = lewm_checkpoint or payload["source_lewm_checkpoint"]
    encoder = LeWMFrozenEncoder(source_checkpoint=source)
    config = payload.get("config", {})
    if payload["algorithm"] == "gcbc":
        if payload["state_fusion"]:
            model = FrozenStateVisualGCBC(
                encoder,
                seed=int(config.get("seed", 0)),
                goal_state_mode=payload["goal_state_mode"],
            )
        else:
            model = FrozenLeWMGCBC(encoder, seed=int(config.get("seed", 0)))
    elif payload["algorithm"] == "gciql":
        model = FrozenLeWMGCIQL(
            encoder,
            state_fusion=payload["state_fusion"],
            goal_state_mode=payload["goal_state_mode"],
            seed=int(config.get("seed", 0)),
            discount=float(config.get("discount", 0.99)),
            tau=float(config.get("tau", 0.005)),
            expectile=float(config.get("expectile", 0.9)),
            actor_loss=config.get("actor_loss", "ddpgbc"),
            alpha=float(config.get("alpha", 0.3)),
        )
    else:
        raise ValueError(f"Unknown algorithm: {payload['algorithm']}")
    model.load_downstream_state_dict(payload["downstream_model"])
    return model.eval()
