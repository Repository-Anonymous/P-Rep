"""Native-224 OGBench Visual Cube reader and official goal samplers."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class Native224GoalDataset:
    """Read the local lossless native-224 cache without unpacking the 25 GB NPZ."""

    def __init__(self, dataset, dataset_dir, state_path=None, seed=0):
        self.dataset = dataset
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.numeric_dir = self.dataset_dir / ".lewm-npy-cache" / dataset
        self.pixel_path = self.dataset_dir / ".lewm-native224" / dataset / "pixels.h5"
        if not self.pixel_path.is_file():
            raise FileNotFoundError(f"Missing native-224 pixels: {self.pixel_path}")
        self.actions = np.load(self.numeric_dir / "actions.npy", mmap_mode="r")
        self.terminals = np.load(self.numeric_dir / "terminals.npy", mmap_mode="r")
        if self.actions.dtype != np.float32 or self.actions.shape != (len(self.terminals), 5):
            raise ValueError("Expected OGBench float32 actions Nx5")
        if not bool(self.terminals[-1]):
            raise ValueError("The final raw row must be terminal")

        self.valid_indices = np.flatnonzero(~self.terminals.astype(bool))
        shifted = np.concatenate((self.terminals[1:], np.ones(1, dtype=bool)))
        official_terminals = np.logical_or(self.terminals, shifted)
        self.terminal_locs = np.flatnonzero(official_terminals)
        self.initial_locs = np.concatenate(([0], np.flatnonzero(self.terminals)[:-1] + 1))
        self.rng = np.random.RandomState(seed)
        self.states = None
        self.state_path = None
        if state_path is not None:
            self.state_path = Path(state_path).expanduser().resolve()
            self.states = np.load(self.state_path, mmap_mode="r", allow_pickle=False)
            if self.states.dtype != np.float32 or self.states.shape != (len(self.terminals), 28):
                raise ValueError("state fusion requires same-split float32 state28 for every raw row")
            manifest = self.state_path.with_suffix(".json")
            if not manifest.is_file():
                raise FileNotFoundError(f"state28 provenance manifest is required: {manifest}")
            metadata = json.loads(manifest.read_text())
            if (
                metadata.get("dataset") != dataset
                or metadata.get("rows") != len(self.terminals)
                or metadata.get("shape") != [len(self.terminals), 28]
                or metadata.get("dtype") != "float32"
            ):
                raise ValueError("state28 manifest does not match this Visual Cube split")
            bindings = {
                "qpos_sha256": self.numeric_dir / "qpos.npy",
                "qvel_sha256": self.numeric_dir / "qvel.npy",
                "state28_sha256": self.state_path,
            }
            for key, path in bindings.items():
                if metadata.get(key) != sha256(path):
                    raise ValueError(f"state28 provenance mismatch for {key}")
        self._pixels_file = None
        self._pixels = None
        self._open_pixels()

    def _open_pixels(self):
        if self._pixels is not None:
            return
        import hdf5plugin  # noqa: F401 -- registers Zstd before opening the VDS.
        import h5py

        self._pixels_file = h5py.File(self.pixel_path, "r")
        self._pixels = self._pixels_file["pixels"]
        if self._pixels.shape != (len(self.terminals), 224, 224, 3) or self._pixels.dtype != np.uint8:
            raise ValueError("Expected renderer-native uint8 NHWC 224 RGB")
        if not bool(self._pixels_file.attrs.get("native_renderer", False)):
            raise ValueError("Refusing an image cache not marked renderer-native")

    def close(self):
        if getattr(self, "_pixels_file", None) is not None:
            self._pixels_file.close()
        self._pixels_file = self._pixels = None

    def __del__(self):
        self.close()

    def sample_current(self, batch_size):
        return self.valid_indices[self.rng.randint(len(self.valid_indices), size=int(batch_size))]

    def sample_goals(self, current, p_current, p_future, p_random, geometric, discount=0.99):
        if not np.isclose(p_current + p_future + p_random, 1.0):
            raise ValueError("Goal probabilities must sum to one")
        current = np.asarray(current, dtype=np.int64)
        random_goal = self.valid_indices[self.rng.randint(len(self.valid_indices), size=len(current))]
        final = self.terminal_locs[np.searchsorted(self.terminal_locs, current)]
        if geometric:
            offsets = self.rng.geometric(p=1.0 - float(discount), size=len(current))
            future_goal = np.minimum(current + offsets, final)
        else:
            distances = self.rng.rand(len(current))
            future_goal = np.round(
                np.minimum(current + 1, final) * distances + final * (1.0 - distances)
            ).astype(np.int64)
        if p_current == 1.0:
            return current.copy()
        goal = np.where(self.rng.rand(len(current)) < p_future / (1.0 - p_current), future_goal, random_goal)
        return np.where(self.rng.rand(len(current)) < p_current, current, goal)

    def _read_frames(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        unique, inverse = np.unique(indices, return_inverse=True)
        frames = np.asarray(self._pixels[unique], dtype=np.uint8)[inverse]
        frames = frames.astype(np.float32) / 255.0
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(frames.transpose(0, 3, 1, 2)))

    def _read_states(self, indices):
        if self.states is None:
            return None
        return torch.from_numpy(np.array(self.states[np.asarray(indices, dtype=np.int64)], copy=True))

    def sample_gcbc(self, batch_size):
        current = self.sample_current(batch_size)
        goal = self.sample_goals(current, 0.0, 1.0, 0.0, geometric=False)
        return {
            "current": self._read_frames(current),
            "goal": self._read_frames(goal),
            "action": torch.from_numpy(np.array(self.actions[current], copy=True)),
            "states": None if self.states is None else {
                "current": self._read_states(current),
                "goal": self._read_states(goal),
            },
            "indices": {"current": current, "goal": goal},
        }

    def sample_gciql(self, batch_size, discount=0.99):
        current = self.sample_current(batch_size)
        next_indices = current + 1
        value_goal = self.sample_goals(current, 0.2, 0.5, 0.3, geometric=True, discount=discount)
        actor_goal = self.sample_goals(current, 0.0, 1.0, 0.0, geometric=False, discount=discount)
        success = current == value_goal
        index_map = {
            "current": current,
            "next": next_indices,
            "value_goal": value_goal,
            "actor_goal": actor_goal,
        }
        result = {
            key: self._read_frames(indices) for key, indices in index_map.items()
        }
        result.update(
            action=torch.from_numpy(np.array(self.actions[current], copy=True)),
            reward=torch.from_numpy(success.astype(np.float32) - 1.0),
            mask=torch.from_numpy((~success).astype(np.float32)),
            indices=index_map,
        )
        result["states"] = None if self.states is None else {
            key: self._read_states(indices) for key, indices in index_map.items()
        }
        return result


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        if key == "indices":
            continue
        if isinstance(value, dict):
            moved[key] = {name: tensor.to(device=device, non_blocking=True) for name, tensor in value.items()}
        elif torch.is_tensor(value):
            moved[key] = value.to(device=device, non_blocking=True)
        else:
            moved[key] = value
    return moved
