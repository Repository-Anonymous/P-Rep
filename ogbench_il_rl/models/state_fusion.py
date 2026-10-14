"""Frozen 2D LeWM plus same-trajectory official 28-D Cube state."""

import math
from collections.abc import Mapping

import torch
from torch import nn

from . import policy as base

unit_gaussian_nll = base.unit_gaussian_nll
make_optimizer = base.make_optimizer


def mask_goal_state(goal_state, mode):
    if mode == "full":
        return goal_state
    if mode != "object_xyz":
        raise ValueError("goal_state_mode must be full or object_xyz")
    masked = goal_state.clone()
    masked[:, :19] = 0
    masked[:, 22:] = 0
    return masked


class FrozenStateVisualGCBC(base.FrozenLeWMGCBC):
    """The linked JEPA4R state_fusion.py interface, restricted to RGB LeWM."""

    def __init__(self, encoder, modality="pixels", seed=0, goal_state_mode="full", latent_dim=192):
        if modality != "pixels" or int(latent_dim) != 192:
            raise ValueError("This LeWM port intentionally supports only 2D RGB LeWM192")
        super().__init__(encoder=encoder, modality=modality, seed=seed)
        self.goal_state_mode = goal_state_mode
        self.state_dim = 28
        with torch.random.fork_rng(devices=[]):
            self.action_head[0] = nn.Linear(440, 512)
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            for index, layer in enumerate(self.action_head):
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(
                        layer.weight,
                        gain=math.sqrt(0.01 if index == 6 else 1.0),
                        generator=generator,
                    )
                    nn.init.zeros_(layer.bias)

    @staticmethod
    def _validate_states(state, batch_size, device):
        if state.shape != (batch_size, 28) or state.dtype != torch.float32 or state.device != device:
            raise ValueError("Same-device float32 Bx28 state is required")
        if not torch.isfinite(state).all():
            raise ValueError("State contains nonfinite values")

    def representations(self, current, goal, current_state, goal_state):
        self._validate_frames(current, goal)
        self._validate_states(current_state, len(current), current.device)
        self._validate_states(goal_state, len(current), current.device)
        goal_state = mask_goal_state(goal_state, self.goal_state_mode)
        current_latent = self.encoder(current).logits
        goal_latent = self.encoder(goal).logits
        return torch.cat((current_latent, current_state), -1), torch.cat((goal_latent, goal_state), -1)

    def forward(self, current, goal, current_state, goal_state):
        current_repr, goal_repr = self.representations(current, goal, current_state, goal_state)
        mean = self.action_head(torch.cat((current_repr, goal_repr), -1))
        if mean.shape != (len(current), 5) or not torch.isfinite(mean).all():
            raise ValueError("Nonfinite or malformed action means")
        return mean

    def get_action(self, info):
        keys = ("pixels", "goal", "current_state", "goal_state")
        if not isinstance(info, Mapping) or any(key not in info for key in keys):
            raise ValueError("Current/goal RGB and matching current/goal state28 are required")
        return self(*(info[key] for key in keys))


def create_policy(
    source_checkpoint=None,
    world_model=None,
    modality="pixels",
    latent_dim=192,
    history_size=1,
    action_dim=5,
    seed=0,
    goal_state_mode="full",
    model_folder=None,
):
    if modality != "pixels" or int(latent_dim) != 192 or model_folder is not None:
        raise ValueError("This LeWM port intentionally supports only 2D RGB LeWM192")
    if int(history_size) != 1 or int(action_dim) != 5:
        raise ValueError("Single-frame/action5 contract required")
    from .frozen_lewm import LeWMFrozenEncoder

    encoder = LeWMFrozenEncoder(source_checkpoint=source_checkpoint, world_model=world_model)
    return FrozenStateVisualGCBC(
        encoder,
        modality=modality,
        seed=seed,
        goal_state_mode=goal_state_mode,
        latent_dim=latent_dim,
    )
