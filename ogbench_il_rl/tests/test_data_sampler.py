import numpy as np

from ogbench_il_rl.data import Native224GoalDataset


def test_official_uniform_future_goal_boundaries():
    dataset = object.__new__(Native224GoalDataset)
    dataset.valid_indices = np.r_[np.arange(4), np.arange(5, 9)]
    dataset.terminal_locs = np.array([3, 4, 8, 9])
    dataset.rng = np.random.RandomState(0)
    current = np.array([0, 3, 5, 8])
    goals = dataset.sample_goals(current, 0.0, 1.0, 0.0, geometric=False)
    assert np.all(goals >= current)
    assert np.all(goals <= np.array([3, 3, 8, 8]))
