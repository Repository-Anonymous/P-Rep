"""Export an observation-only clip from the local Cube HDF5 dataset."""

import argparse
from pathlib import Path

import hdf5plugin  # noqa: F401; register HDF5 compression filters
import h5py
import imageio
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    with h5py.File(args.dataset, "r") as dataset:
        episodes = np.asarray(dataset["ep_idx"][:])
        rows = np.flatnonzero(episodes == args.episode)
        if len(rows) == 0:
            raise ValueError(f"Episode {args.episode} does not exist")
        first_row = int(rows[0]) + args.start
        pixels = dataset["pixels"][first_row : first_row + args.frames]
        steps = dataset["step_idx"][first_row : first_row + args.frames]

    if len(pixels) != args.frames or not np.array_equal(
        steps, np.arange(args.start, args.start + args.frames)
    ):
        raise ValueError("Requested clip is outside the episode or non-contiguous")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), fps=args.fps, codec="libx264")
    for frame in pixels:
        writer.append_data(frame)
    writer.close()


if __name__ == "__main__":
    main()
