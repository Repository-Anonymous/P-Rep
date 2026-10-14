"""Replay official OGBench states into losslessly stored, renderer-native 224 RGB."""

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

from eval_ogbench_visual_cube import ROOT, write_json

DATASETS = [f'visual-cube-{variant}-{kind}-v0' for variant in ['single', 'double', 'triple', 'quadruple']
            for kind in ['play', 'noisy']]
NATIVE_ROOT = Path('/home/coder/P-Rep/ogbench/.lewm-native224')
CACHE_ROOT = Path('/home/coder/P-Rep/ogbench/.lewm-npy-cache')
SHARD_SIZE = 16384


def create_renderer(variant):
    import gymnasium as gym
    import ogbench
    env = gym.make(f'visual-cube-{variant}-v0', width=224, height=224, mode='data_collection')
    env.reset(seed=123)
    assert env.unwrapped._visualize_info is False
    assert env.unwrapped._pixel_transparent_arm is True
    return env


def render_worker(gpu, rank, jobs, native_root, cache_root, errors, log_root):
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID=str(gpu), CUDA_VISIBLE_DEVICES=str(gpu),
                      OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (Path(log_root) / f'render_gpu{gpu}_worker{rank}.log').open('a', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            env = None
            try:
                import h5py
                import hdf5plugin
                import imageio.v2 as imageio
                import mujoco
                import numpy as np
                from OpenGL import GL
                current_variant = None
                arrays = {}
                while True:
                    job = jobs.get()
                    if job is None:
                        break
                    dataset, start, stop = job['dataset'], job['start'], job['stop']
                    variant = dataset.split('-')[2]
                    if current_variant != variant:
                        if env is not None:
                            env.unwrapped._renderer.close()
                            env.unwrapped._renderer = None
                            env.close()
                        env = create_renderer(variant)
                        current_variant = variant
                        # MuJoCo initialization can leave an old GL error flag;
                        # querying hardware metadata must not attribute it to
                        # the subsequent glGetString call.
                        while GL.glGetError() != GL.GL_NO_ERROR:
                            pass
                        renderer_name = GL.glGetString(GL.GL_RENDERER).decode()
                        print(f'GPU {gpu} native renderer: {renderer_name}', flush=True)
                    if dataset not in arrays:
                        directory = Path(cache_root) / dataset
                        arrays[dataset] = {name: np.load(directory / f'{name}.npy', mmap_mode='r')
                                           for name in ['qpos', 'qvel', 'observations']}
                    data = arrays[dataset]
                    raw = env.unwrapped
                    assert data['qpos'].shape[1] == raw._model.nq
                    assert data['qvel'].shape[1] == raw._model.nv
                    # Verify stored-state replay with the original native-64
                    # camera before producing each shard; this catches a wrong
                    # scene, camera, alpha, frame alignment or simulator model.
                    renderer64 = mujoco.Renderer(raw._model, height=64, width=64)
                    replay_errors = []
                    for index in sorted(set([start, (start + stop) // 2, stop - 1])):
                        raw.set_state(data['qpos'][index], data['qvel'][index])
                        renderer64.update_scene(raw._data, camera='front_pixels', scene_option=raw._scene_option)
                        frame = renderer64.render()
                        error = float(np.abs(frame.astype(np.float32) - data['observations'][index].astype(np.float32)).mean())
                        assert error < 1.0, f'Native replay mismatch: {dataset}, row {index}, MAE {error}'
                        replay_errors.append(error)
                    renderer64.close()
                    destination = Path(native_root) / dataset / 'shards'
                    destination.mkdir(parents=True, exist_ok=True)
                    file = destination / f'{start:09d}.h5'
                    temporary = file.with_suffix('.h5.tmp')
                    begun = time.monotonic()
                    with h5py.File(temporary, 'w') as store:
                        pixels = store.create_dataset('pixels', shape=(stop - start, 224, 224, 3), dtype='u1',
                                                      chunks=(1, 224, 224, 3), fletcher32=True,
                                                      **hdf5plugin.Zstd(clevel=3))
                        for first in range(start, stop, 64):
                            last = min(first + 64, stop)
                            buffer = np.empty((last - first, 224, 224, 3), dtype=np.uint8)
                            for local, index in enumerate(range(first, last)):
                                raw.set_state(data['qpos'][index], data['qvel'][index])
                                buffer[local] = raw.render()
                            pixels[first - start:last - start] = buffer
                            if first == 0:
                                imageio.imwrite(Path(log_root) / f'{dataset}_sample224.png', buffer[0])
                        store.attrs.update(native_renderer=True, width=224, height=224,
                                           dataset=dataset, start=start, stop=stop,
                                           camera='front_pixels', visualize_info=False,
                                           source='official stored qpos/qvel, no simulator step, no resize')
                    temporary.replace(file)
                    elapsed = time.monotonic() - begun
                    write_json(file.with_suffix('.json'), {**job, 'gpu': gpu, 'worker': rank,
                               'passed': True, 'frames': stop - start, 'native_shape': [224, 224, 3],
                               'renderer': renderer_name,
                               'replay_mae64': replay_errors, 'elapsed_seconds': elapsed,
                               'frames_per_second': (stop - start) / elapsed, 'bytes': file.stat().st_size})
                    print(json.dumps({'dataset': dataset, 'start': start, 'stop': stop,
                                      'gpu': gpu, 'fps': (stop - start) / elapsed}), flush=True)
            except BaseException:
                errors.put(traceback.format_exc())
                traceback.print_exc()
                raise
            finally:
                if env is not None:
                    if env.unwrapped._renderer is not None:
                        env.unwrapped._renderer.close()
                        env.unwrapped._renderer = None
                    env.close()


def create_virtual_dataset(native_root, cache_root, dataset, shards, frame_count):
    import h5py
    directory = Path(native_root) / dataset
    file = directory / 'pixels.h5'
    layout = h5py.VirtualLayout(shape=(frame_count, 224, 224, 3), dtype='u1')
    for shard in shards:
        path = directory / 'shards' / f'{shard["start"]:09d}.h5'
        record = json.loads(path.with_suffix('.json').read_text())
        assert record['passed'] and record['frames'] == shard['stop'] - shard['start']
        with h5py.File(path, 'r') as source:
            assert source['pixels'].shape == (record['frames'], 224, 224, 3)
            assert bool(source.attrs['native_renderer'])
        virtual = h5py.VirtualSource(str(path.resolve()), 'pixels', shape=(record['frames'], 224, 224, 3))
        layout[shard['start']:shard['stop']] = virtual
    temporary = file.with_suffix('.h5.tmp')
    with h5py.File(temporary, 'w', libver='latest') as destination:
        destination.create_virtual_dataset('pixels', layout)
        destination.attrs.update(native_renderer=True, width=224, height=224,
                                 source_dataset=dataset, camera='front_pixels')
    temporary.replace(file)
    identity = json.loads((Path(cache_root) / dataset / 'manifest.json').read_text())
    write_json(directory / 'manifest.json', {'complete': True, 'dataset': dataset,
               'frames': frame_count, 'shape': [frame_count, 224, 224, 3],
               'native_renderer': True, 'source_identity': identity,
               'shards': len(shards), 'compression': 'lossless Zstd level 3 and per-frame Fletcher32',
               'camera': 'front_pixels', 'pixel_transparent_arm': True,
               'visualize_info': False, 'numerical_state_model_input': False})


def prepare(output, gpus, workers_per_gpu=4, native_root=NATIVE_ROOT, smoke=False):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from finetune_ogbench_visual_cube import prepare_cache
    output = Path(output)
    log_root = output / 'native224-preparation'
    log_root.mkdir(parents=True, exist_ok=True)
    selected = [dataset + suffix for dataset in DATASETS for suffix in ['', '-val']]
    ready = [Path(native_root) / dataset / 'manifest.json' for dataset in selected]
    if not smoke and all(path.exists() and json.loads(path.read_text()).get('complete') for path in ready):
        from replace_ogbench_native224 import file_identity
        for dataset in selected:
            directory = Path(native_root) / dataset
            manifest = json.loads((directory / 'manifest.json').read_text())
            assert (directory / 'pixels.h5').exists() and manifest['native_renderer']
            current = file_identity(Path('/home/coder/P-Rep/ogbench') / (dataset + '.npz'))
            expected = [manifest['source_identity']]
            record = directory / 'npz-replacement.json'
            if record.exists():
                expected.append(json.loads(record.read_text())['new_identity'])
            assert current in expected, 'Original dataset changed since native replay'
        print('All sixteen native224 image caches already complete; skip re-rendering', flush=True)
        return
    source_root = Path('/home/coder/P-Rep/ogbench')
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda dataset: prepare_cache(source_root / (dataset + '.npz'), CACHE_ROOT,
                                                   include_observations=True), selected))
    pending, all_shards, counts = [], {}, {}
    for dataset in selected:
        count = len(np.load(CACHE_ROOT / dataset / 'terminals.npy', mmap_mode='r'))
        if smoke:
            count = min(count, 1001)
        counts[dataset] = count
        shard_size = 256 if smoke else SHARD_SIZE
        shards = [{'dataset': dataset, 'start': start, 'stop': min(start + shard_size, count)}
                  for start in range(0, count, shard_size)]
        all_shards[dataset] = shards
        directory = Path(native_root) / dataset
        if (directory / 'manifest.json').exists():
            prior = json.loads((directory / 'manifest.json').read_text())
            identity = json.loads((CACHE_ROOT / dataset / 'manifest.json').read_text())
            assert prior['source_identity'] == identity and prior['frames'] == count, 'Native cache source changed'
        for shard in shards:
            file = directory / 'shards' / f'{shard["start"]:09d}.h5'
            if file.exists() and file.with_suffix('.json').exists():
                record = json.loads(file.with_suffix('.json').read_text())
                assert record['passed'] and record['frames'] == shard['stop'] - shard['start']
                continue
            pending.append(shard)
    total_frames = sum(counts.values())
    context = mp.get_context('spawn')
    jobs, errors = context.Queue(), context.Queue()
    # Rotate datasets to avoid one GPU being stuck with a five-million-frame
    # dataset. Every worker can take any shard of any official split.
    pending.sort(key=lambda item: (item['start'], item['dataset']))
    for job in pending:
        jobs.put(job)
    processes = []
    for gpu in gpus:
        for rank in range(workers_per_gpu):
            jobs.put(None)
            process = context.Process(target=render_worker,
                      args=(gpu, rank, jobs, str(native_root), str(CACHE_ROOT), errors, str(log_root)))
            processes.append(process)
    print(f'NATIVE224 PREPARATION: {total_frames:,} frames, {len(pending)} pending shards, {len(processes)} workers', flush=True)
    write_json(output / 'status.json', {'phase': 'preparing_native224', 'total_frames': total_frames,
               'pending_shards': len(pending), 'gpus': gpus, 'workers_per_gpu': workers_per_gpu})
    for process in processes:
        process.start()
    try:
        while any(process.is_alive() for process in processes):
            failures = [process.exitcode for process in processes if process.exitcode not in (None, 0)]
            if failures:
                raise RuntimeError(f'Native renderer workers failed: {failures}; inspect render logs')
            time.sleep(5)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
    assert all(process.exitcode == 0 for process in processes)
    for dataset in selected:
        create_virtual_dataset(native_root, CACHE_ROOT, dataset, all_shards[dataset], counts[dataset])
    write_json(log_root / 'complete.json', {'passed': True, 'total_frames': total_frames,
               'native_root': str(native_root), 'datasets': selected, 'shape': [224, 224, 3]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', type=int, default=list(range(8)))
    parser.add_argument('--workers-per-gpu', type=int, default=4)
    parser.add_argument('--native-root', type=Path, default=NATIVE_ROOT)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    prepare(args.output, args.gpus, args.workers_per_gpu, args.native_root, args.smoke)
