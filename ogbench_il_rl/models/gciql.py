"""Goal-conditioned IQL heads trained on top of a frozen LeWM representation."""

import contextlib
import copy
import math
from collections.abc import Mapping

import torch
from torch import nn

from .policy import make_actor
from .state_fusion import mask_goal_state


class ValueMLP(nn.Module):
    """OGBench GCIQL value topology: 512x3, GELU then LayerNorm, scalar output."""

    def __init__(self, input_dim, seed=0):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 512), nn.GELU(approximate="tanh"), nn.LayerNorm(512),
                nn.Linear(512, 512), nn.GELU(approximate="tanh"), nn.LayerNorm(512),
                nn.Linear(512, 512), nn.GELU(approximate="tanh"), nn.LayerNorm(512),
                nn.Linear(512, 1),
            )
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, generator=generator)
                    nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        return self.layers(inputs).squeeze(-1)


@contextlib.contextmanager
def _frozen_parameters(*modules):
    parameters = [parameter for module in modules for parameter in module.parameters()]
    prior = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, required in zip(parameters, prior):
            parameter.requires_grad_(required)


class FrozenLeWMGCIQL(nn.Module):
    """Official-style GCIQL with frozen LeWM and trainable actor/Q/V heads only."""

    def __init__(
        self,
        encoder,
        state_fusion=False,
        goal_state_mode="object_xyz",
        seed=0,
        discount=0.99,
        tau=0.005,
        expectile=0.9,
        actor_loss="ddpgbc",
        alpha=0.3,
    ):
        super().__init__()
        if actor_loss not in ("ddpgbc", "awr"):
            raise ValueError("actor_loss must be ddpgbc or awr")
        self.encoder = encoder
        self.state_fusion = bool(state_fusion)
        self.goal_state_mode = goal_state_mode
        self.repr_dim = 220 if self.state_fusion else 192
        pair_dim = 2 * self.repr_dim
        self.actor = make_actor(pair_dim, seed=seed)
        self.value = ValueMLP(pair_dim, seed=seed + 1)
        self.q1 = ValueMLP(pair_dim + 5, seed=seed + 2)
        self.q2 = ValueMLP(pair_dim + 5, seed=seed + 3)
        self.target_q1 = copy.deepcopy(self.q1).requires_grad_(False)
        self.target_q2 = copy.deepcopy(self.q2).requires_grad_(False)
        self.discount = float(discount)
        self.tau = float(tau)
        self.expectile = float(expectile)
        self.actor_loss_name = actor_loss
        self.alpha = float(alpha)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.train(False)
        self.target_q1.train(False)
        self.target_q2.train(False)
        return self

    def _validate_state(self, state, batch_size, device):
        if state is None or state.shape != (batch_size, 28):
            raise ValueError("state_fusion requires Bx28 state for every image")
        if state.dtype != torch.float32 or state.device != device or not torch.isfinite(state).all():
            raise ValueError("state must be same-device finite float32")

    def encode(self, frames, state=None, goal=False):
        if frames.ndim != 4 or frames.shape[1:] != (3, 224, 224):
            raise ValueError("Expected normalized BCHW 224 RGB")
        latent = self.encoder(frames).logits
        if not self.state_fusion:
            if state is not None:
                raise ValueError("State supplied while state_fusion is disabled")
            return latent
        self._validate_state(state, len(frames), frames.device)
        if goal:
            state = mask_goal_state(state, self.goal_state_mode)
        return torch.cat((latent, state), dim=-1)

    @staticmethod
    def _pair(observation, goal):
        return torch.cat((observation, goal), dim=-1)

    def actor_mean(self, observation, goal):
        return self.actor(self._pair(observation, goal))

    def get_action(self, info):
        if not isinstance(info, Mapping) or "pixels" not in info or "goal" not in info:
            raise ValueError("Current and goal pixels are required")
        current_state = info.get("current_state")
        goal_state = info.get("goal_state")
        observation = self.encode(info["pixels"], current_state, goal=False)
        goal = self.encode(info["goal"], goal_state, goal=True)
        return self.actor_mean(observation, goal)

    def compute_loss(self, batch):
        required = ("current", "next", "value_goal", "actor_goal", "action", "reward", "mask")
        if any(key not in batch for key in required):
            raise ValueError(f"GCIQL batch must contain {required}")

        states = batch.get("states")
        current_state = None if states is None else states["current"]
        next_state = None if states is None else states["next"]
        value_goal_state = None if states is None else states["value_goal"]
        actor_goal_state = None if states is None else states["actor_goal"]

        # The encoder is frozen. Gradients begin at the policy/value/critic heads.
        observation = self.encode(batch["current"], current_state, goal=False)
        next_observation = self.encode(batch["next"], next_state, goal=False)
        value_goal = self.encode(batch["value_goal"], value_goal_state, goal=True)
        actor_goal = self.encode(batch["actor_goal"], actor_goal_state, goal=True)
        action = batch["action"]

        value_pair = self._pair(observation, value_goal)
        q_input = torch.cat((value_pair, action), dim=-1)
        with torch.no_grad():
            target_q = torch.minimum(self.target_q1(q_input), self.target_q2(q_input))
        value = self.value(value_pair)
        advantage = target_q - value
        weight = torch.where(advantage >= 0, self.expectile, 1.0 - self.expectile)
        value_loss = (weight * advantage.square()).mean()

        with torch.no_grad():
            next_value = self.value(self._pair(next_observation, value_goal))
            q_target = batch["reward"] + self.discount * batch["mask"] * next_value
        q1 = self.q1(q_input)
        q2 = self.q2(q_input)
        critic_loss = ((q1 - q_target).square() + (q2 - q_target).square()).mean()

        actor_pair = self._pair(observation, actor_goal)
        action_mean = self.actor(actor_pair)
        if self.actor_loss_name == "awr":
            with torch.no_grad():
                actor_value = self.value(actor_pair)
                actor_q_input = torch.cat((actor_pair, action), dim=-1)
                actor_q = torch.minimum(self.q1(actor_q_input), self.q2(actor_q_input))
                actor_weight = torch.exp((actor_q - actor_value) * self.alpha).clamp(max=100.0)
            nll = 0.5 * (action_mean - action).square().sum(dim=-1) + 2.5 * math.log(2 * math.pi)
            actor_loss = (actor_weight * nll).mean()
            q_loss = action_mean.new_zeros(())
            bc_loss = actor_loss
        else:
            # Let Q provide dQ/da while excluding Q parameters from the actor term.
            with _frozen_parameters(self.q1, self.q2):
                proposed = action_mean.clamp(-1.0, 1.0)
                actor_q_input = torch.cat((actor_pair, proposed), dim=-1)
                actor_q = torch.minimum(self.q1(actor_q_input), self.q2(actor_q_input))
            q_loss = -actor_q.mean() / actor_q.abs().mean().detach().clamp_min(1e-6)
            bc_loss = 0.5 * (action_mean - action).square().sum(dim=-1).mean()
            actor_loss = q_loss + self.alpha * bc_loss

        total = value_loss + critic_loss + actor_loss
        metrics = {
            "loss": total.detach(),
            "value_loss": value_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "q_loss": q_loss.detach(),
            "bc_loss": bc_loss.detach(),
            "v_mean": value.detach().mean(),
            "q_mean": torch.minimum(q1.detach(), q2.detach()).mean(),
            "adv_mean": advantage.detach().mean(),
        }
        return total, metrics

    @torch.no_grad()
    def update_targets(self):
        for online, target in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
            for source_parameter, target_parameter in zip(online.parameters(), target.parameters()):
                target_parameter.lerp_(source_parameter, self.tau)

    def downstream_parameters(self):
        for module in (self.actor, self.value, self.q1, self.q2):
            yield from module.parameters()

    def downstream_state_dict(self):
        state = {}
        for name in ("actor", "value", "q1", "q2", "target_q1", "target_q2"):
            for key, value in getattr(self, name).state_dict().items():
                state[f"{name}.{key}"] = value
        return state

    def load_downstream_state_dict(self, state):
        consumed = set()
        for name in ("actor", "value", "q1", "q2", "target_q1", "target_q2"):
            prefix = name + "."
            selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            getattr(self, name).load_state_dict(selected, strict=True)
            consumed.update(prefix + key for key in selected)
        if consumed != set(state):
            raise ValueError("Unexpected downstream checkpoint keys")


def create_policy(source_checkpoint=None, world_model=None, state_fusion=False, seed=0, **kwargs):
    from .frozen_lewm import LeWMFrozenEncoder

    encoder = LeWMFrozenEncoder(source_checkpoint=source_checkpoint, world_model=world_model)
    return FrozenLeWMGCIQL(encoder, state_fusion=state_fusion, seed=seed, **kwargs)


def make_optimizer(policy, lr=3e-4):
    parameters = list(policy.downstream_parameters())
    expected = {id(parameter) for parameter in parameters}
    actual = {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
    if expected != actual:
        raise ValueError("Trainable parameters must be exactly actor, value, and online Q heads")
    return torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8)
