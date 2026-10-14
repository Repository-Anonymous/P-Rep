from types import SimpleNamespace

import torch
from torch import nn

from ogbench_il_rl.models import gciql, policy, state_fusion


class DummyFrozenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(192), requires_grad=False)
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, images):
        value = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return SimpleNamespace(logits=value * self.weight.unsqueeze(0))


def frames(batch=4):
    return torch.randn(batch, 3, 224, 224)


def test_gcbc_only_trains_action_head():
    model = policy.FrozenLeWMGCBC(DummyFrozenEncoder())
    optimizer = policy.make_optimizer(model)
    loss = policy.unit_gaussian_nll(model(frames(), frames()), torch.zeros(4, 5))
    loss.backward()
    assert model.encoder.weight.grad is None
    assert all(parameter.grad is not None for parameter in model.action_head.parameters())
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in model.action_head.parameters()
    }


def test_state_fusion_goal_uses_only_object_xyz():
    model = state_fusion.FrozenStateVisualGCBC(DummyFrozenEncoder(), goal_state_mode="object_xyz")
    current, goal = frames(2), frames(2)
    current_state = torch.randn(2, 28)
    goal_state = torch.randn(2, 28)
    changed = goal_state.clone()
    changed[:, :19] += 100
    changed[:, 22:] -= 100
    assert torch.equal(
        model(current, goal, current_state, goal_state),
        model(current, goal, current_state, changed),
    )


def test_gciql_gradients_and_policy_only_checkpoint():
    model = gciql.FrozenLeWMGCIQL(DummyFrozenEncoder(), state_fusion=True)
    optimizer = gciql.make_optimizer(model)
    batch_size = 3
    batch = {
        key: frames(batch_size) for key in ("current", "next", "value_goal", "actor_goal")
    }
    batch.update(
        action=torch.rand(batch_size, 5) * 2 - 1,
        reward=-torch.ones(batch_size),
        mask=torch.ones(batch_size),
        states={key: torch.randn(batch_size, 28) for key in ("current", "next", "value_goal", "actor_goal")},
    )
    target_before = next(model.target_q1.parameters()).detach().clone()
    loss, metrics = model.compute_loss(batch)
    loss.backward()
    optimizer.step()
    model.update_targets()
    assert set(metrics) >= {"value_loss", "critic_loss", "actor_loss"}
    assert model.encoder.weight.grad is None
    assert all(parameter.grad is None for parameter in model.target_q1.parameters())
    assert not torch.equal(target_before, next(model.target_q1.parameters()))
    state = model.downstream_state_dict()
    assert state and not any("encoder" in key for key in state)
