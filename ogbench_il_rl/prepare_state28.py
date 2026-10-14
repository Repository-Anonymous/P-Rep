"""Derive same-trajectory official Cube state28 from Visual Cube qpos/qvel."""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _worker(dataset_dir, dataset, destination, start, stop):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import gymnasium as gym
    import ogbench  # noqa: F401 -- environment registration

    cache = Path(dataset_dir) / ".lewm-npy-cache" / dataset
    qpos = np.load(cache / "qpos.npy", mmap_mode="r")
    qvel = np.load(cache / "qvel.npy", mmap_mode="r")
    output = np.lib.format.open_memmap(destination, mode="r+")
    env = gym.make("cube-single-v0", mode="data_collection")
    env.reset(seed=0)
    raw = env.unwrapped
    for index in range(start, stop):
        raw.set_state(qpos[index], qvel[index])
        value = raw.compute_observation()
        if value.shape != (28,) or not np.isfinite(value).all():
            raise ValueError(f"Bad state28 at raw row {index}")
        output[index] = value.astype(np.float32)
    output.flush()
    env.close()


def prepare(dataset, dataset_dir, output=None, workers=8, limit=None):
    if not dataset.startswith("visual-cube-single-"):
        raise ValueError("The JEPA4R state28 interface is defined only for visual-cube-single")
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    cache = dataset_dir / ".lewm-npy-cache" / dataset
    qpos_path, qvel_path = cache / "qpos.npy", cache / "qvel.npy"
    qpos = np.load(qpos_path, mmap_mode="r")
    qvel = np.load(qvel_path, mmap_mode="r")
    if qpos.shape[0] != qvel.shape[0]:
        raise ValueError("qpos/qvel row count differs")
    rows = len(qpos) if limit is None else min(int(limit), len(qpos))
    output = Path(output) if output else cache / "state28.npy"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(rows, 28))
    array.flush()
    del array
    boundaries = np.linspace(0, rows, min(int(workers), rows) + 1, dtype=int)
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_worker,
            args=(str(dataset_dir), dataset, str(temporary), int(start), int(stop)),
        )
        for start, stop in zip(boundaries[:-1], boundaries[1:])
        if start < stop
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failures = [process.exitcode for process in processes if process.exitcode != 0]
    if failures:
        raise RuntimeError(f"state28 workers failed: {failures}")
    result = np.load(temporary, mmap_mode="r")
    if result.shape != (rows, 28) or not np.isfinite(result).all():
        raise ValueError("Generated state28 failed validation")
    temporary.replace(output)
    metadata = {
        "dataset": dataset,
        "rows": rows,
        "shape": [rows, 28],
        "dtype": "float32",
        "source": "same Visual Cube qpos/qvel restored with OGBench CubeEnv.compute_observation",
        "qpos_sha256": _sha256(qpos_path),
        "qvel_sha256": _sha256(qvel_path),
        "state28_sha256": _sha256(output),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("/home/coder/P-Rep/ogbench"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    prepare(args.dataset, args.dataset_dir, args.output, args.workers, args.limit)


if __name__ == "__main__":
    main()
