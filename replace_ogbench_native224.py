"""Atomically replace visual Cube NPZ pixels with completed native224 replays."""

import argparse
import fcntl
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

from eval_ogbench_visual_cube import write_json

SOURCE_ROOT = Path('/home/coder/P-Rep/ogbench')
NATIVE_ROOT = SOURCE_ROOT / '.lewm-native224'
CACHE_ROOT = SOURCE_ROOT / '.lewm-npy-cache'


def file_identity(path):
    return {'source': str(path), 'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}


def header(stream):
    import numpy as np
    version = np.lib.format.read_magic(stream)
    shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
    return shape, fortran, dtype


def replace_dataset(dataset, native_root=NATIVE_ROOT, source_root=SOURCE_ROOT, cache_root=CACHE_ROOT):
    import h5py
    import hdf5plugin
    import numpy as np
    source = Path(source_root) / f'{dataset}.npz'
    native = Path(native_root) / dataset
    cache = Path(cache_root) / dataset
    native.mkdir(parents=True, exist_ok=True)
    with (native / 'replacement.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _replace_locked(dataset, source, native, cache)


def _replace_locked(dataset, source, native, cache):
    import h5py
    import numpy as np
    record_path = native / 'npz-replacement.json'
    if record_path.exists():
        record = json.loads(record_path.read_text())
        assert record['completed'] and record['new_identity'] == file_identity(source)
        with zipfile.ZipFile(source) as archive, archive.open('observations.npy') as stream:
            shape, _, dtype = header(stream)
            assert shape == tuple(record['shape']) and dtype == np.uint8
        # A stopped job can have completed the atomic swap before cleanup.
        synchronize_metadata(record, native, cache)
        (cache / 'observations.npy').unlink(missing_ok=True)
        record_all_replacements(source.parent, native.parent)
        return record
    manifest_path = native / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['complete'] and manifest['native_renderer']
    old_identity = file_identity(source)
    temporary = source.with_suffix('.npz.native224.tmp')
    status_path = native / 'replacement-status.json'
    write_json(status_path, {'phase': 'replacing_npz', 'dataset': dataset,
                            'frames': manifest['frames'], 'source': str(source)})
    before_hashes, entry_sizes = {}, {}
    started = time.monotonic()
    with h5py.File(native / 'pixels.h5', 'r') as rendered, zipfile.ZipFile(source) as original:
        pixels = rendered['pixels']
        shape = tuple(pixels.shape)
        assert bool(rendered.attrs['native_renderer']) and shape[1:] == (224, 224, 3)
        assert shape[0] == manifest['frames'] and pixels.dtype == np.uint8
        with original.open('observations.npy') as stream:
            old_shape, old_fortran, old_dtype = header(stream)
        assert old_shape[0] == shape[0] and old_dtype == np.uint8 and not old_fortran
        assert old_shape[1:] in [(64, 64, 3), (224, 224, 3)]
        if old_shape[1:] == (64, 64, 3):
            assert old_identity == manifest['source_identity'], 'Dataset changed since native rendering'
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1, allowZip64=True) as replacement:
            # Preserve every non-image NPY member byte for byte, including
            # actions, qpos/qvel and terminal rows. Keep the standard NPZ API.
            for name in original.namelist():
                if name == 'observations.npy':
                    continue
                digest = hashlib.sha256()
                with original.open(name) as src, replacement.open(name, 'w', force_zip64=True) as dst:
                    while data := src.read(8 * 1024 * 1024):
                        digest.update(data)
                        dst.write(data)
                before_hashes[name] = digest.hexdigest()
                entry_sizes[name] = original.getinfo(name).file_size
            with replacement.open('observations.npy', 'w', force_zip64=True) as dst:
                np.lib.format.write_array_header_2_0(dst, {'descr': '|u1', 'fortran_order': False, 'shape': shape})
                for first in range(0, shape[0], 128):
                    stop = min(first + 128, shape[0])
                    data = np.ascontiguousarray(pixels[first:stop]).tobytes()
                    dst.write(data)
                    if first % 16384 == 0 or stop == shape[0]:
                        write_json(status_path, {'phase': 'replacing_npz', 'dataset': dataset,
                                   'completed_frames': stop, 'frames': shape[0],
                                   'elapsed_seconds': time.monotonic() - started})
        # Validate container structure, a real RGB block and all preserved
        # arrays from the newly written file before removing the old NPZ.
        with zipfile.ZipFile(temporary) as check:
            assert set(check.namelist()) == set(original.namelist())
            for name, expected in before_hashes.items():
                digest = hashlib.sha256()
                with check.open(name) as stream:
                    while data := stream.read(8 * 1024 * 1024):
                        digest.update(data)
                assert digest.hexdigest() == expected and check.getinfo(name).file_size == entry_sizes[name]
            with check.open('observations.npy') as stream:
                new_shape, fortran, dtype = header(stream)
                assert new_shape == shape and not fortran and dtype == np.uint8
                header_bytes = stream.tell()
                count = min(128, shape[0])
                block = np.frombuffer(stream.read(count * 224 * 224 * 3), dtype=np.uint8).reshape(count, 224, 224, 3)
                assert np.array_equal(block, pixels[:count])
                assert check.getinfo('observations.npy').file_size == header_bytes + int(np.prod(shape))
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    assert file_identity(source) == old_identity, 'Source changed while packaging'
    temporary.replace(source)
    record = {'completed': True, 'dataset': dataset, 'shape': list(shape),
              'native_renderer': True, 'old_identity': old_identity, 'new_identity': file_identity(source),
              'preserved_member_sha256': before_hashes,
              'compression': 'standard NPZ ZIP_DEFLATED level 1; lossless',
              'old_npz_replaced_atomically': True, 'old_64_npy_cache_removed': True,
              'elapsed_seconds': time.monotonic() - started}
    write_json(record_path, record)
    # Update source identities so training/resuming never recreates a huge
    # uncompressed image NPY. The existing scalar/state caches are unchanged.
    synchronize_metadata(record, native, cache)
    (cache / 'observations.npy').unlink(missing_ok=True)
    write_json(status_path, {'phase': 'complete', 'dataset': dataset, 'frames': shape[0]})
    record_all_replacements(source.parent, native.parent)
    print(f'REPLACED {dataset}: observations={shape}, non-image arrays verified unchanged; old64 removed', flush=True)
    return record


def synchronize_metadata(record, native, cache):
    cache.mkdir(parents=True, exist_ok=True)
    write_json(cache / 'manifest.json', record['new_identity'])
    path = native / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['render_source_identity'] = manifest.get('render_source_identity', manifest['source_identity'])
    manifest['source_identity'] = record['new_identity']
    manifest['npz_replaced_with_native224'] = True
    write_json(path, manifest)


def record_all_replacements(source_root, native_root):
    from prepare_ogbench_native224 import DATASETS
    names = [dataset + suffix for dataset in DATASETS for suffix in ['', '-val']]
    with (Path(native_root) / 'replacement-summary.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        paths = [Path(native_root) / name / 'npz-replacement.json' for name in names]
        if all(path.exists() for path in paths):
            records = [json.loads(path.read_text()) for path in paths]
            assert all(record['completed'] for record in records)
            write_json(Path(source_root) / 'native224-replacement-manifest.json',
                       {'completed': True, 'datasets': records,
                        'scope': 'four visual Cube variants × play/noisy × train/val'})


def replace_all(native_root=NATIVE_ROOT, source_root=SOURCE_ROOT, cache_root=CACHE_ROOT):
    from concurrent.futures import ProcessPoolExecutor
    from prepare_ogbench_native224 import DATASETS
    datasets = [dataset + suffix for dataset in DATASETS for suffix in ['', '-val']]
    with ProcessPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(replace_dataset, dataset, native_root, source_root, cache_root) for dataset in datasets]
        records = [future.result() for future in futures]
    write_json(Path(source_root) / 'native224-replacement-manifest.json',
               {'completed': True, 'datasets': records, 'scope': 'four visual Cube variants × play/noisy × train/val'})
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', help='One split name; otherwise replace all sixteen splits')
    args = parser.parse_args()
    if args.dataset:
        replace_dataset(args.dataset)
    else:
        replace_all()
