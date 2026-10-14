"""Warm-start LeWM on one official OGBench visual Cube train/validation pair."""

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from eval_ogbench_visual_cube import ROOT, sha256, write_json
from module import SIGReg


def prepare_cache(source, cache_root, include_observations=False):
    """Extract official arrays; qpos/qvel are only used for renderer replay."""
    destination = Path(cache_root) / source.stem
    destination.mkdir(parents=True, exist_ok=True)
    identity = {'source': str(source), 'bytes': source.stat().st_size,
                'mtime_ns': source.stat().st_mtime_ns}
    manifest = destination / 'manifest.json'
    names = ['actions.npy', 'terminals.npy', 'qpos.npy', 'qvel.npy']
    if include_observations:
        names.append('observations.npy')
    same_source = manifest.exists() and json.loads(manifest.read_text()) == identity
    if same_source:
        if all((destination / name).exists() for name in names):
            return destination
    with zipfile.ZipFile(source) as archive:
        for name in names:
            if same_source and (destination / name).exists():
                continue
            temporary = destination / (name + '.tmp')
            with archive.open(name) as src, temporary.open('wb') as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            temporary.replace(destination / name)
    write_json(manifest, identity)
    return destination


class OfficialSequences(Dataset):
    """Original LeWM layout: 20 actions and RGB frames at offsets 0,5,10,15."""

    def __init__(self, directory, normalizer, native_pixels, smoke_prefix=0):
        self.directory = str(directory)
        self.native_pixels = str(native_pixels)
        self.smoke_prefix = smoke_prefix
        self.mean = np.asarray(normalizer['mean'], dtype=np.float32)
        self.scale = np.asarray(normalizer['scale'], dtype=np.float32)
        self._open()
        terminals = np.asarray(self.terminals)
        ends = np.flatnonzero(terminals)
        assert len(ends) and ends[-1] == len(terminals) - 1
        starts = np.r_[0, ends[:-1] + 1]
        # Terminal rows contain a dummy action. Require all 20 real actions
        # inside a single trajectory, matching the original frameskip=5 reader.
        clips = [np.arange(start, end - 19, dtype=np.int64)
                 for start, end in zip(starts, ends) if end - start >= 20]
        self.indices = np.concatenate(clips)
        assert len(self.indices) and self.pixels.dtype == np.uint8
        assert self.pixels.shape == (len(terminals), 224, 224, 3)
        assert self.actions.shape == (len(terminals), 5)
        self.episodes = len(ends)

    def _open(self):
        import h5py
        import hdf5plugin  # Register the lossless Zstd filter.
        root = Path(self.directory)
        self._pixels_file = h5py.File(self.native_pixels, 'r')
        assert bool(self._pixels_file.attrs['native_renderer'])
        self.pixels = self._pixels_file['pixels']
        self.actions = np.load(root / 'actions.npy', mmap_mode='r')
        self.terminals = np.load(root / 'terminals.npy', mmap_mode='r')
        if self.smoke_prefix:
            self.actions = self.actions[:self.smoke_prefix]
            self.terminals = self.terminals[:self.smoke_prefix]

    def __getstate__(self):
        state = dict(self.__dict__)
        for name in ['pixels', '_pixels_file', 'actions', 'terminals']:
            state.pop(name)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        start = int(self.indices[index])
        pixels = np.array(self.pixels[start:start + 20:5], copy=True)
        actions = np.array(self.actions[start:start + 20], copy=True)
        actions = ((actions - self.mean) / self.scale).reshape(4, 25)
        return {'pixels': torch.from_numpy(pixels).permute(0, 3, 1, 2),
                'action': torch.from_numpy(actions)}


def prepare_pixels(pixels):
    batch, steps = pixels.shape[:2]
    assert pixels.shape[-3:] == (3, 224, 224), 'Training requires renderer-native 224 RGB'
    pixels = pixels.to(device='cuda', dtype=torch.float32, non_blocking=True)
    pixels = pixels.reshape(batch * steps, 3, 224, 224) / 255.0
    mean = pixels.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = pixels.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    pixels = (pixels - mean) / std
    return pixels.reshape(batch, steps, 3, 224, 224)


def losses(model, regularizer, batch):
    prepared = {'pixels': prepare_pixels(batch['pixels']),
                'action': batch['action'].cuda(non_blocking=True)}
    assert prepared['pixels'].shape[-3:] == (3, 224, 224)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        encoded = model.encode(prepared)
        embeddings = encoded['emb']
        prediction = model.predict(embeddings[:, :3], encoded['act_emb'][:, :3])
    prediction_loss = F.mse_loss(prediction.float(), embeddings[:, 1:].float())
    regularization_loss = regularizer(embeddings.float().transpose(0, 1))
    return prediction_loss + 0.09 * regularization_loss, prediction_loss, regularization_loss


def atomic_torch_save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def export_model(model, config, destination):
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / 'config.json', config)
    atomic_torch_save(destination / 'weights.pt',
                      {name: value.detach().cpu() for name, value in model.state_dict().items()})


def train(args):
    import stable_worldmodel as swm
    import wandb
    from sklearn.preprocessing import StandardScaler

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    write_json(output / 'status.json', {'phase': 'preparing_data', 'dataset': args.dataset,
                                       'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')})
    source = args.dataset_dir / (args.dataset + '.npz')
    validation_source = args.dataset_dir / (args.dataset + '-val.npz')
    if not args.smoke_prefix:
        request = output.parent.parent / 'native224-replacement-request.json'
        if request.exists():
            from replace_ogbench_native224 import replace_dataset
            from concurrent.futures import ThreadPoolExecutor
            write_json(output / 'status.json', {'phase': 'replacing_npz_with_native224', 'dataset': args.dataset})
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(replace_dataset, name, args.native_pixels_dir, args.dataset_dir, args.cache_dir)
                           for name in [args.dataset, args.dataset + '-val']]
                for future in futures:
                    future.result()
    train_cache = prepare_cache(source, args.cache_dir)
    val_cache = prepare_cache(validation_source, args.cache_dir)
    actions = np.load(train_cache / 'actions.npy', mmap_mode='r')
    terminals = np.load(train_cache / 'terminals.npy', mmap_mode='r')
    assert np.isfinite(actions).all()
    scaler = StandardScaler().fit(actions[~terminals])
    normalizer = {'source': str(source), 'split': 'train only; terminal dummy actions excluded',
                  'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                  'variance': scaler.var_.tolist(), 'samples': int(scaler.n_samples_seen_)}
    write_json(output / 'action-normalizer.json', normalizer)
    train_data = OfficialSequences(train_cache, normalizer, args.native_pixels_dir / source.stem / 'pixels.h5', args.smoke_prefix)
    val_data = OfficialSequences(val_cache, normalizer, args.native_pixels_dir / validation_source.stem / 'pixels.h5', args.smoke_prefix)
    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': args.workers,
                     'pin_memory': True}
    if args.workers:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=3,
                             multiprocessing_context='spawn')
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True,
                              generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, drop_last=False, **loader_kwargs)
    steps_per_epoch = min(len(train_loader), args.max_batches or len(train_loader))
    total_steps = args.steps or steps_per_epoch * args.epochs
    epochs = math.ceil(total_steps / steps_per_epoch)
    model_config = json.loads((args.checkpoint / 'config.json').read_text())
    model = swm.wm.utils.load_pretrained(str(args.checkpoint)).cuda().train()
    model.requires_grad_(True)
    original_state = torch.load(args.checkpoint / 'weights.pt', weights_only=True, map_location='cpu')
    assert all(torch.equal(value.detach().cpu(), original_state[name])
               for name, value in model.state_dict().items())
    del original_state
    regularizer = SIGReg(knots=17, num_proj=1024).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3, fused=True)
    warmup = max(1, int(total_steps * 0.05))

    def lr_factor(step):
        if step < warmup:
            return 0.1 + 0.9 * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(train_sequences=len(train_data), val_sequences=len(val_data),
                  train_episodes=train_data.episodes, val_episodes=val_data.episodes,
                  steps_per_epoch=steps_per_epoch, total_optimizer_steps=total_steps, effective_epochs=epochs,
                  initial_checkpoint_sha256=sha256(args.checkpoint / 'weights.pt'),
                  warm_start_strictly_verified=True, trainable_parameters=sum(p.numel() for p in model.parameters()),
                  optimizer='AdamW', weight_decay=1e-3, gradient_clip=1.0,
                  warmup_steps=warmup, warmup_start_factor=0.1, scheduler='linear warmup then cosine',
                  precision='bf16 mixed; FP32 prediction and SIGReg losses',
                  input_resolution=224, training_image_resolution=224, source_render_resolution=224,
                  training_images='native 224 MuJoCo replay of official qpos/qvel; no upscaling',
                  history=3, prediction_steps=1, frameskip=5, sigreg_weight=0.09,
                  validation='official separate -val NPZ; never used for optimizer/scaler',
                  action_normalizer=normalizer,
                  numerical_state_input=False,
                  validation_rng_seed=args.seed + 1000000,
                  checkpoint_selection='minimum held-out prediction + 0.09*SIGReg loss')
    write_json(output / 'training-config.json', config)
    (output / 'checkpoints').mkdir(exist_ok=True)
    resume_path = output / 'checkpoints' / 'training-state.pt'
    initial_epoch, global_step, best = 0, 0, float('inf')
    if resume_path.exists():
        state = torch.load(resume_path, map_location='cpu', weights_only=False)
        assert state.get('native_training_resolution') == 224, 'Refusing to resume optimizer/model state trained on upscaled images'
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        initial_epoch, global_step, best = state['next_epoch'], state['global_step'], state['best']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        generator.set_state(state['loader_rng'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        del state
    wandb_id_path = output / 'wandb-run-id.txt'
    run_id = wandb_id_path.read_text().strip() if wandb_id_path.exists() else uuid.uuid4().hex[:12]
    wandb_id_path.write_text(run_id + '\n')
    run = wandb.init(entity=os.environ.get('WANDB_ENTITY', 'visual-intelligence-laboratory'),
                     project=os.environ.get('WANDB_PROJECT', 'P-Rep'),
                     name=f'{args.group}_{args.dataset}', group=args.group,
                     job_type='lewm-ogbench-finetune', id=run_id, resume='allow',
                     mode=args.wandb_mode, dir=str(output), config=config)
    run.define_metric('optimizer_step')
    run.define_metric('train/*', step_metric='optimizer_step')
    run.define_metric('val/*', step_metric='optimizer_step')
    write_json(output / 'wandb-run.json', {'id': run.id, 'url': run.url, 'mode': run.settings.mode})
    print(f'TRAIN START {args.dataset}: {len(train_data)} train / {len(val_data)} val sequences, '
          f'{total_steps} steps; W&B {run.url}', flush=True)
    model.encoder.register_forward_pre_hook(
        lambda module, positional, kwargs: check_encoder_input(output, positional, kwargs), with_kwargs=True)
    first_parameter = next(model.encoder.parameters())
    before = first_parameter.detach().clone()
    running = np.zeros(3)
    updates_since_log = 0
    with (output / 'losses.jsonl').open('a', buffering=1) as history:
        for epoch in range(initial_epoch, epochs):
            model.train()
            for batch_index, batch in enumerate(train_loader):
                if batch_index >= steps_per_epoch or global_step >= total_steps:
                    break
                optimizer.zero_grad(set_to_none=True)
                values = losses(model, regularizer, batch)
                values[0].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                global_step += 1
                numeric = np.array([value.detach().item() for value in values])
                assert np.isfinite(numeric).all(), numeric
                running += numeric
                updates_since_log += 1
                if global_step == 1:
                    delta = float((first_parameter.detach() - before).abs().max())
                    assert delta > 0, 'Encoder did not update during finetuning'
                    write_json(output / 'optimizer-check.json', {'passed': True, 'losses_finite': True,
                               'encoder_parameter_max_change': delta, 'encoder_input': [3, 224, 224],
                               'all_model_parameters_trainable': all(p.requires_grad for p in model.parameters())})
                    del before
                if global_step % args.log_every == 0 or global_step <= 3 or batch_index + 1 == steps_per_epoch or global_step == total_steps:
                    mean_losses = running / updates_since_log
                    payload = {'optimizer_step': global_step, 'train/epoch': epoch + (batch_index + 1) / steps_per_epoch,
                               'train/loss': float(mean_losses[0]), 'train/pred_loss': float(mean_losses[1]),
                               'train/sigreg_loss': float(mean_losses[2]), 'train/grad_norm': float(grad_norm),
                               'train/lr': optimizer.param_groups[0]['lr'],
                               'train/elapsed_seconds': time.monotonic() - started,
                               'train/gpu_peak_memory_gb': torch.cuda.max_memory_allocated() / 1e9}
                    run.log(payload)
                    history.write(json.dumps(payload) + '\n')
                    write_json(output / 'status.json', {'phase': 'training', 'dataset': args.dataset,
                               'epoch': epoch + 1, 'global_step': global_step, 'total_steps': total_steps,
                               'loss': payload['train/loss'], 'wandb_url': run.url})
                    print(json.dumps(payload), flush=True)
                    running[:] = 0
                    updates_since_log = 0
            model.eval()
            totals, seen = np.zeros(3), 0
            # Keep regularizer projections identical across validation epochs,
            # and do not let validation consume the training RNG stream.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(args.seed + 1000000)
                with torch.inference_mode():
                    for index, batch in enumerate(val_loader):
                        if args.val_batches and index >= args.val_batches:
                            break
                        count = len(batch['pixels'])
                        totals += np.array([v.item() for v in losses(model, regularizer, batch)]) * count
                        seen += count
            validation = totals / seen
            assert np.isfinite(validation).all()
            run.log({'optimizer_step': global_step, 'val/epoch': epoch + 1,
                     'val/loss': float(validation[0]), 'val/pred_loss': float(validation[1]),
                     'val/sigreg_loss': float(validation[2]), 'val/sequences': seen})
            export_model(model, model_config, output / 'checkpoints' / 'last')
            export_model(model, model_config, output / 'checkpoints' / f'epoch_{epoch + 1:02d}')
            if validation[0] < best:
                best = float(validation[0])
                export_model(model, model_config, output / 'checkpoints' / 'best')
                write_json(output / 'best-checkpoint.json', {'epoch': epoch + 1, 'global_step': global_step,
                           'val_loss': best, 'validation_sequences': seen,
                           'checkpoint': str(output / 'checkpoints' / 'best'),
                           'sha256': sha256(output / 'checkpoints' / 'best' / 'weights.pt')})
            atomic_torch_save(resume_path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                              'native_training_resolution': 224,
                              'scheduler': scheduler.state_dict(), 'next_epoch': epoch + 1,
                              'global_step': global_step, 'best': best, 'torch_rng': torch.get_rng_state(),
                              'cuda_rng': torch.cuda.get_rng_state_all(), 'loader_rng': generator.get_state(),
                              'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()})
            print(f'EPOCH {epoch + 1} COMPLETE: validation={validation.tolist()}, best={best}', flush=True)
    assert global_step == total_steps
    write_json(output / 'train-complete.json', {'dataset': args.dataset, 'epochs': epochs,
               'optimizer_steps': global_step, 'best_val_loss': best,
               'best_checkpoint': str(output / 'checkpoints' / 'best'),
               'elapsed_seconds': time.monotonic() - started, 'wandb_run_id': run.id, 'wandb_url': run.url})
    run.summary['best_val_loss'] = best
    run.summary['training_complete'] = True
    run.finish()
    write_json(output / 'status.json', {'phase': 'complete', 'dataset': args.dataset,
               'global_step': global_step, 'best_val_loss': best})


def check_encoder_input(output, positional, kwargs):
    pixels = kwargs.get('pixel_values', positional[0] if positional else None)
    assert pixels.shape[-3:] == (3, 224, 224), pixels.shape
    destination = output / 'encoder-input-check.json'
    if not destination.exists():
        write_json(destination, {'passed': True, 'shape': list(pixels.shape), 'dtype': str(pixels.dtype)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path, default=Path('/home/coder/P-Rep/ogbench'))
    parser.add_argument('--cache-dir', type=Path, default=Path('/home/coder/P-Rep/ogbench/.lewm-npy-cache'))
    parser.add_argument('--native-pixels-dir', type=Path, default=Path('/home/coder/P-Rep/ogbench/.lewm-native224'))
    parser.add_argument('--checkpoint', type=Path, default=ROOT / '.stable-wm/checkpoints/cube-lewm')
    parser.add_argument('--group', default='lewm-visual-cube-finetune')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--steps', type=int, default=0,
                        help='Fixed update budget, overriding epochs; used to balance all eight GPUs')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=3072)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--max-batches', type=int, default=0, help='Only for smoke checks; 0 uses full training data')
    parser.add_argument('--val-batches', type=int, default=0, help='0 validates on the complete official held-out split')
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
    parser.add_argument('--smoke-prefix', type=int, default=0,
                        help='Only for native-render smoke checks, using the first complete episode')
    args = parser.parse_args()
    assert args.epochs > 0 and args.batch_size >= 2 and args.lr > 0
    if args.smoke_prefix:
        assert args.max_batches and args.val_batches and args.wandb_mode == 'disabled'
    train(args)


if __name__ == '__main__':
    main()
