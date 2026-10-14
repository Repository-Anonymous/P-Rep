"""Causal rolling-video training samples for Cube V-JEPA-LeWM."""

from bisect import bisect_right
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from torch.utils.data import Dataset


class CubeVideoContextDataset(Dataset):
    """Return T states, each represented by its preceding F-frame video clip.

    Actions preserve vanilla LeWM semantics: each state gets the next
    ``frameskip`` raw actions flattened into one action block.
    """

    def __init__(
        self,
        path,
        episode_indices,
        context_frames=64,
        num_steps=4,
        frameskip=5,
        sample_stride=1,
        action_mean=None,
        action_std=None,
    ):
        self.path = Path(path)
        self.episode_indices = np.asarray(episode_indices, dtype=np.int64)
        self.context_frames = int(context_frames)
        self.num_steps = int(num_steps)
        self.frameskip = int(frameskip)
        self.sample_stride = int(sample_stride)
        self._h5 = None

        with h5py.File(self.path, "r") as dataset:
            self.lengths = np.asarray(dataset["ep_len"][:], dtype=np.int64)
            self.offsets = np.asarray(dataset["ep_offset"][:], dtype=np.int64)
            if action_mean is None or action_std is None:
                actions = np.asarray(dataset["action"][:], dtype=np.float32)
                valid_actions = actions[~np.isnan(actions).any(axis=1)]
                action_mean = valid_actions.mean(axis=0)
                action_std = valid_actions.std(axis=0)
        self.action_mean = np.asarray(action_mean, dtype=np.float32)
        self.action_std = np.maximum(np.asarray(action_std, dtype=np.float32), 1e-6)

        self.first_state = self.context_frames - 1
        self.action_span = self.num_steps * self.frameskip
        self.starts_per_episode = []
        for episode in self.episode_indices:
            last_state = int(self.lengths[episode]) - self.action_span
            count = max(0, (last_state - self.first_state) // self.sample_stride + 1)
            self.starts_per_episode.append(count)
        self.cumulative = np.cumsum(self.starts_per_episode).tolist()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r", swmr=True)

    def __len__(self):
        return self.cumulative[-1] if self.cumulative else 0

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        episode_slot = bisect_right(self.cumulative, index)
        previous = 0 if episode_slot == 0 else self.cumulative[episode_slot - 1]
        local_sample = index - previous
        episode = int(self.episode_indices[episode_slot])
        state_start = self.first_state + local_sample * self.sample_stride
        global_start = int(self.offsets[episode])

        # One contiguous read supplies all four overlapping 64-frame clips.
        pixel_start = global_start + state_start - self.context_frames + 1
        pixel_end = global_start + state_start + (self.num_steps - 1) * self.frameskip + 1
        action_start = global_start + state_start
        action_end = action_start + self.action_span
        self._open()
        pixels = np.asarray(self._h5["pixels"][pixel_start:pixel_end])
        actions = np.asarray(self._h5["action"][action_start:action_end], dtype=np.float32)

        clips = np.stack(
            [
                pixels[step * self.frameskip : step * self.frameskip + self.context_frames]
                for step in range(self.num_steps)
            ]
        )
        clips = torch.from_numpy(clips).permute(0, 1, 4, 2, 3)
        actions = (actions - self.action_mean) / self.action_std
        actions = np.nan_to_num(actions, nan=0.0)
        actions = torch.from_numpy(actions.reshape(self.num_steps, -1))
        return {
            "pixels": clips,
            "action": actions,
            "episode_idx": torch.tensor(episode),
            "state_start": torch.tensor(state_start),
        }


def make_episode_split(num_episodes, train_fraction=0.9, seed=3072):
    generator = np.random.default_rng(seed)
    episodes = generator.permutation(num_episodes)
    split = int(num_episodes * train_fraction)
    return np.sort(episodes[:split]), np.sort(episodes[split:])


__all__ = ["CubeVideoContextDataset", "make_episode_split"]
