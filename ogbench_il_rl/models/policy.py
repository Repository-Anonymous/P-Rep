"""JEPA4R-compatible frozen-LeWM goal-conditioned behavior cloning head."""

import math
from collections.abc import Mapping

import torch
from torch import nn


def make_actor(input_dim, seed=0):
    """Official OGBench GCActor topology and initialization in PyTorch."""
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(approximate="tanh"),
            nn.Linear(512, 512),
            nn.GELU(approximate="tanh"),
            nn.Linear(512, 512),
            nn.GELU(approximate="tanh"),
            nn.Linear(512, 5),
        )
        for index, layer in enumerate(head):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(
                    layer.weight,
                    gain=math.sqrt(0.01 if index == 6 else 1.0),
                    generator=generator,
                )
                nn.init.zeros_(layer.bias)
    return head


class FrozenLeWMGCBC(nn.Module):
    """One current RGB + one goal RGB -> frozen LeWM -> trainable mean action."""

    def __init__(self, encoder, modality="pixels", seed=0):
        super().__init__()
        if modality != "pixels":
            raise ValueError("This LeWM port intentionally supports only 2D RGB")
        self.encoder = encoder
        self.modality = modality
        self.history_size = 1
        self.action_dim = 5
        self.latent_dim = 192
        self.fixed_std = 1.0
        self.action_head = make_actor(384, seed)

    @staticmethod
    def _validate_frames(current, goal):
        if not torch.is_tensor(current) or not torch.is_tensor(goal):
            raise ValueError("Preprocessed tensors are required")
        if current.ndim != 4 or current.shape != goal.shape or current.shape[1:] != (3, 224, 224):
            raise ValueError("Exactly one matching BCHW 224 current and goal frame is required")
        if not current.is_floating_point() or not goal.is_floating_point():
            raise ValueError("Preprocessed RGB must be floating point")

    def representations(self, current, goal):
        self._validate_frames(current, goal)
        current_latent = self.encoder(current).logits
        goal_latent = self.encoder(goal).logits
        return current_latent, goal_latent

    def forward(self, current, goal):
        current_latent, goal_latent = self.representations(current, goal)
        mean = self.action_head(torch.cat((current_latent, goal_latent), dim=-1))
        if mean.shape != (len(current), 5) or not torch.isfinite(mean).all():
            raise ValueError("Nonfinite or malformed action means")
        return mean

    def get_action(self, info):
        if not isinstance(info, Mapping) or "pixels" not in info or "goal" not in info:
            raise ValueError("Current pixels and goal pixels are required")
        return self(info["pixels"], info["goal"])

    def downstream_state_dict(self):
        return {f"action_head.{key}": value for key, value in self.action_head.state_dict().items()}

    def load_downstream_state_dict(self, state):
        prefix = "action_head."
        selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if len(selected) != len(self.action_head.state_dict()) or len(selected) != len(state):
            raise ValueError("Checkpoint must contain exactly the GCBC action head")
        self.action_head.load_state_dict(selected, strict=True)


def create_policy(
    source_checkpoint=None,
    world_model=None,
    modality="pixels",
    latent_dim=192,
    history_size=1,
    action_dim=5,
    seed=0,
):
    if modality != "pixels" or int(latent_dim) != 192:
        raise ValueError("This LeWM port intentionally supports only 2D RGB LeWM192")
    if int(history_size) != 1 or int(action_dim) != 5:
        raise ValueError("Single-frame/action5 contract required")
    from .frozen_lewm import LeWMFrozenEncoder

    encoder = LeWMFrozenEncoder(source_checkpoint=source_checkpoint, world_model=world_model)
    return FrozenLeWMGCBC(encoder, modality=modality, seed=seed)


def unit_gaussian_nll(means, raw_actions):
    if means.ndim != 2 or means.shape != raw_actions.shape or means.shape[-1] != 5:
        raise ValueError("Means and raw actions must both be Bx5")
    if not torch.isfinite(means).all() or not torch.isfinite(raw_actions).all():
        raise ValueError("Finite means/actions required")
    if (raw_actions.abs() > 1.00001).any():
        raise ValueError("OGBench actions must remain in the published [-1,1] domain")
    return (0.5 * (means - raw_actions).square().sum(dim=-1) + 2.5 * math.log(2 * math.pi)).mean()


def make_optimizer(policy, lr=3e-4):
    expected = {id(parameter) for parameter in policy.action_head.parameters()}
    actual = {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
    if actual != expected:
        raise ValueError("Trainable parameters must be exactly the action head")
    return torch.optim.Adam(policy.action_head.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
