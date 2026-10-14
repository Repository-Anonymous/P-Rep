"""Evaluate a fixed LeWM checkpoint on official OGBench visual Cube tasks.

Uses official reset(task_id), goal images, termination, and registered step limits.
No numerical state is passed to the policy and no training is performed.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
LIMITS = {'single': 200, 'double': 500, 'triple': 1000, 'quadruple': 1000}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def batch_safe_criterion(model, info):
    """Preserve LeWM's cost while giving goals an explicit candidate dimension."""
    import torch.nn.functional as functional
    prediction = info['predicted_emb']
    goal = info['goal_emb']
    if goal.ndim == prediction.ndim - 1:
        goal = goal.unsqueeze(1)
    goal = goal[..., -1:, :].expand_as(prediction)
    return functional.mse_loss(prediction[..., -1:, :], goal[..., -1:, :].detach(),
                               reduction='none').sum(dim=tuple(range(2, prediction.ndim)))


def make_official_env(env_name, video_size=224, observation_size=64):
    """Run each simulator in its own process; retain terminal frames until reset."""
    import gymnasium as gym
    import numpy as np
    import ogbench

    class FreezeFinished(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.finished = None
            self.record_video = False
            self.video_renderer = None

        def reset(self, **kwargs):
            self.finished = None
            seed = kwargs.get('seed')
            np.random.seed(seed)
            self.env.action_space.seed(seed)
            observation, info = self.env.reset(**kwargs)
            if self.record_video:
                info = {**info, 'video_rgb': self.get_video_frame()}
            return observation, info

        def get_video_frame(self):
            import mujoco
            raw = self.unwrapped
            if raw._render_height == raw._render_width == video_size:
                return raw.render()
            if self.video_renderer is None:
                # Increase framebuffer capacity, keeping the observation renderer 64px.
                raw._model.vis.global_.offwidth = max(raw._model.vis.global_.offwidth, video_size)
                raw._model.vis.global_.offheight = max(raw._model.vis.global_.offheight, video_size)
                self.video_renderer = mujoco.Renderer(raw._model, height=video_size, width=video_size)
            self.video_renderer.update_scene(raw._data, camera='front_pixels',
                                             scene_option=raw._scene_option)
            return self.video_renderer.render()

        def step(self, action):
            if self.finished is not None:
                return self.finished
            result = self.env.step(action)
            if self.record_video:
                result = (*result[:4], {**result[4], 'video_rgb': self.get_video_frame()})
            if result[2] or result[3]:
                self.finished = result
            return result

        def close(self):
            if self.video_renderer is not None:
                self.video_renderer.close()
                self.video_renderer = None
            if self.unwrapped._renderer is not None:
                self.unwrapped._renderer.close()
                self.unwrapped._renderer = None
            self.env.close()

        def benchmark_metadata(self):
            return {
                'limit': self.env.spec.max_episode_steps,
                'pixels': self.unwrapped._ob_type == 'pixels',
                'transparent_arm': bool(self.unwrapped._pixel_transparent_arm),
                'tasks': self.unwrapped.task_infos,
                'current_task': self.unwrapped.cur_task_id,
            }

    return FreezeFinished(gym.make(env_name, width=observation_size, height=observation_size))


def buffered_action(policy, terminated):
    """Reuse already planned actions without repeating stateless image transforms."""
    import numpy as np
    import torch
    assert policy.cfg.history_len == 1
    if any(not dead and not policy._action_buffer[index]
           for index, dead in enumerate(terminated)):
        return None
    action_dim = policy.env.single_action_space.shape[-1]
    action = torch.full((policy.env.num_envs, action_dim), float('nan'))
    for index, dead in enumerate(terminated):
        if not dead:
            action[index] = policy._action_buffer[index].popleft()
    action = action.reshape(*policy.env.action_space.shape).float().numpy()
    if 'action' in policy.process:
        action = policy.process['action'].inverse_transform(action)
    return action


def verify_batch_cost(model, original_criterion, policy, observations, goals, seed):
    """Check parallel costs against the original one-environment computation."""
    import torch
    import numpy as np
    from types import MethodType
    count = min(2, len(observations))
    prepared = policy._prepare_info({
        'pixels': observations[:count, None], 'goal': goals[:count, None],
        'action': np.zeros((count, 1, 5), dtype=np.float32),
    })
    expanded = {key: value.cuda().unsqueeze(1).expand(count, 3, *value.shape[1:])
                for key, value in prepared.items()}
    candidates = torch.randn(count, 3, 5, 25, device='cuda',
                             generator=torch.Generator(device='cuda').manual_seed(seed))
    with torch.inference_mode():
        parallel_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision('highest')
        parallel_predict = model.predict
        model.predict = getattr(model, '_reference_predict', parallel_predict)
        model.criterion = original_criterion
        reference = torch.cat([model.get_cost(
            {key: value[index:index + 1] for key, value in expanded.items()},
            candidates[index:index + 1]) for index in range(count)], dim=0)
        model.criterion = MethodType(batch_safe_criterion, model)
        model.predict = parallel_predict
        torch.set_float32_matmul_precision(parallel_precision)
        parallel = model.get_cost(dict(expanded), candidates)
    assert torch.allclose(reference, parallel, rtol=1e-3, atol=1e-3)
    assert torch.equal(reference.argmin(1), parallel.argmin(1))
    return {'passed': True, 'environments': count, 'candidates': 3,
            'max_absolute_difference': float((reference - parallel).abs().max()),
            'max_relative_difference': float(((reference - parallel).abs() / reference.abs().clamp_min(1e-8)).max()),
            'best_candidate_matches': True, 'reference_matmul_precision': 'highest',
            'parallel_matmul_precision': parallel_precision,
            'reference': 'original LeWM criterion, each environment separately'}


def worker(gpu, queue, settings):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(gpu)
    os.environ['STABLEWM_HOME'] = str(ROOT / '.stable-wm')
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    output = Path(settings['output'])
    with (output / f'gpu_{gpu}.log').open('a', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                from select_ogbench_checkpoint import evaluation_checkpoint
                settings = evaluation_checkpoint(settings)
                run_worker(gpu, queue, settings)
            except BaseException:
                traceback.print_exc()
                raise


def run_worker(gpu, queue, settings):
    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    import ogbench  # Registers the official environments.
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
    from functools import partial
    from sklearn.preprocessing import StandardScaler
    from torchvision.transforms import v2
    from types import SimpleNamespace

    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision(settings['matmul_precision'])
    output = Path(settings['output'])
    norm = json.loads((output / 'action-normalizer.json').read_text())
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(norm['mean'], dtype=np.float64)
    scaler.scale_ = np.asarray(norm['scale'], dtype=np.float64)
    scaler.var_ = np.asarray(norm['variance'], dtype=np.float64)
    scaler.n_features_in_ = 5
    scaler.n_samples_seen_ = norm['samples']
    model = swm.wm.utils.load_pretrained(settings['checkpoint']).cuda().eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    encoder_input_recorded = False

    def validate_encoder_input(module, args, kwargs):
        nonlocal encoder_input_recorded
        pixels = kwargs.get('pixel_values', args[0] if args else None)
        assert tuple(pixels.shape[-3:]) == (3, 224, 224), pixels.shape
        if not encoder_input_recorded:
            write_json(output / f'gpu_{gpu}_encoder_input_check.json', {
                'passed': True, 'actual_encoder_input_shape': list(pixels.shape),
                'dtype': str(pixels.dtype), 'renderer_observation_size': [settings['observation_size'], settings['observation_size'], 3],
                'preprocessing': 'ImageNet normalization and Resize224',
            })
            encoder_input_recorded = True

    model.encoder.register_forward_pre_hook(validate_encoder_input, with_kwargs=True)
    from types import MethodType
    original_criterion = model.criterion
    model._reference_predict = model.predict
    if settings['compile_predictor']:
        model.predict = torch.compile(model.predict, mode='default', fullgraph=True)
    model.criterion = MethodType(batch_safe_criterion, model)
    batch_cost_verified = False
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(**spt.data.dataset_stats.ImageNet),
        v2.Resize(size=224),
    ])
    environments = {}
    print(f'GPU {gpu}: {torch.cuda.get_device_name(0)} ready', flush=True)
    try:
        while True:
            job = queue.get()
            if job is None:
                break
            variant, task_id, eval_seed = job['variant'], job['task_id'], job['eval_seed']
            env_name = f'visual-cube-{variant}-v0'
            count = min(settings['parallel_envs'], settings['episodes_per_task'])
            job_seed = eval_seed * 100_000 + list(LIMITS).index(variant) * 10_000 + task_id * 1_000
            random.seed(job_seed)
            np.random.seed(job_seed)
            torch.manual_seed(job_seed)
            solver = swm.solver.CEMSolver(
                model=model, batch_size=count, num_samples=300,
                var_scale=1.0, n_steps=30, topk=30, device='cuda', seed=job_seed,
            )
            records = []
            started = time.perf_counter()
            for first in range(0, settings['episodes_per_task'], count):
                size = min(count, settings['episodes_per_task'] - first)
                key = (variant, size)
                if key not in environments:
                    for old_env in environments.values():
                        old_env.close()
                    environments.clear()
                    environments[key] = gym.vector.AsyncVectorEnv(
                        [partial(make_official_env, env_name, settings['video_size'], settings['observation_size']) for _ in range(size)],
                        context='spawn', autoreset_mode=gym.vector.AutoresetMode.DISABLED,
                    )
                active_envs = environments[key]
                lengths = np.zeros(size, dtype=np.int64)
                alive = np.ones(size, dtype=bool)
                success = np.zeros(size, dtype=bool)
                terminated_flags = np.zeros(size, dtype=bool)
                truncated_flags = np.zeros(size, dtype=bool)
                record_video = settings['videos'] and eval_seed == settings['seeds'][0] and first == 0
                active_envs.set_attr('record_video', [record_video and local == 0 for local in range(size)])
                seeds = [job_seed + first + local for local in range(size)]
                observations, reset_info = active_envs.reset(seed=seeds, options={'task_id': task_id})
                goals = reset_info['goal'].copy()
                assert observations.shape == goals.shape == (size, settings['observation_size'], settings['observation_size'], 3)
                assert observations.dtype == goals.dtype == np.uint8
                metadata = active_envs.call('benchmark_metadata')
                assert all(item['limit'] == LIMITS[variant] and item['pixels']
                           and item['transparent_arm'] and len(item['tasks']) == 5
                           and item['current_task'] == task_id for item in metadata)
                action_info = np.zeros((size, 5), dtype=np.float32)
                policy = swm.policy.WorldModelPolicy(
                    solver=solver,
                    config=swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5),
                    process={'action': scaler},
                    transform={'pixels': transform, 'goal': transform},
                )
                policy.set_env(SimpleNamespace(
                    num_envs=size, single_action_space=active_envs.single_action_space,
                    action_space=active_envs.action_space,
                ))
                if not batch_cost_verified:
                    comparison = verify_batch_cost(model, original_criterion, policy,
                                                   observations, goals, job_seed)
                    write_json(output / f'gpu_{gpu}_batch_cost_check.json', comparison)
                    batch_cost_verified = True
                video = None
                if record_video:
                    video_path = output / 'videos' / f'{variant}_task{task_id}.mp4'
                    video_path.parent.mkdir(exist_ok=True)
                    video = imageio.get_writer(str(video_path), fps=20, codec='libx264',
                                               quality=8, macro_block_size=None)
                    frame = reset_info['video_rgb'][0]
                    video.append_data(frame)
                    imageio.imwrite(output / 'videos' / f'{variant}_task{task_id}_start.png', frame)
                    imageio.imwrite(output / 'videos' / f'{variant}_task{task_id}_goal.png', goals[0])
                try:
                    with torch.inference_mode():
                        while alive.any():
                            # Deliberately pass only images, action, and episode status.
                            info_for_policy = {
                                'pixels': observations[:, None],
                                'goal': goals[:, None],
                                'action': action_info[:, None],
                                'terminated': ~alive,
                            }
                            actions = buffered_action(policy, ~alive)
                            if actions is None:
                                actions = policy.get_action(info_for_policy)
                            assert np.isfinite(actions[alive]).all()
                            actions = np.clip(actions, -1.0, 1.0)
                            actions[~alive] = 0
                            observations, rewards, terminated, truncated, step_info = active_envs.step(actions)
                            for local in np.flatnonzero(alive):
                                action_info[local] = actions[local]
                                lengths[local] += 1
                                success[local] = bool(step_info['success'][local])
                                if local == 0 and video is not None:
                                    video.append_data(step_info['video_rgb'][0])
                                if terminated[local] or truncated[local]:
                                    assert lengths[local] <= LIMITS[variant]
                                    if terminated[local]:
                                        assert step_info['success'][local], 'Unexpected non-success termination'
                                    terminated_flags[local] = terminated[local]
                                    truncated_flags[local] = truncated[local]
                                    alive[local] = False
                finally:
                    if video is not None:
                        video.close()
                for local in range(size):
                    records.append({
                        'episode': first + local, 'reset_seed': seeds[local],
                        'success': bool(success[local]), 'steps': int(lengths[local]),
                        'terminated': bool(terminated_flags[local]),
                        'truncated': bool(truncated_flags[local]),
                    })
                progress = {
                    **job, 'gpu': gpu, 'completed_episodes': len(records),
                    'successes': sum(row['success'] for row in records),
                    'elapsed_seconds': time.perf_counter() - started,
                }
                write_json(output / 'progress' / f'{job["id"]}.json', progress)
                print('PROGRESS ' + json.dumps(progress), flush=True)
            result = {
                **job, 'gpu': gpu, 'env_name': env_name,
                'task_name': metadata[0]['tasks'][task_id - 1]['task_name'],
                'max_episode_steps': LIMITS[variant], 'episodes': len(records),
                'successes': sum(row['success'] for row in records),
                'success_rate_percent': 100 * np.mean([row['success'] for row in records]),
                'elapsed_seconds': time.perf_counter() - started, 'records': records,
            }
            write_json(output / 'jobs' / f'{job["id"]}.json', result)
            print('RESULT ' + json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)
    finally:
        for env in environments.values():
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint', default=str(ROOT / '.stable-wm/checkpoints/cube-lewm'))
    parser.add_argument('--action-normalizer', type=Path,
                        help='Use the finetuned checkpoint training-split action scaler')
    parser.add_argument('--training-description',
                        default='none; fixed checkpoint trained on custom single-cube 224px expert data')
    parser.add_argument('--resume', action='store_true', help='Skip completed jobs with matching configuration')
    parser.add_argument('--variants', nargs='+', choices=list(LIMITS), default=list(LIMITS))
    parser.add_argument('--tasks', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--gpus', nargs='+', type=int, default=list(range(8)))
    parser.add_argument('--episodes-per-task', type=int, default=50)
    parser.add_argument('--parallel-envs', type=int, default=10)
    parser.add_argument('--no-videos', action='store_true')
    parser.add_argument('--video-size', type=int, default=224)
    parser.add_argument('--observation-size', type=int, choices=[64, 224], default=224)
    parser.add_argument('--compile-predictor', action='store_true')
    parser.add_argument('--matmul-precision', choices=['highest', 'high'], default='highest')
    args = parser.parse_args()
    assert set(args.tasks) <= {1, 2, 3, 4, 5}
    assert args.episodes_per_task > 0 and args.parallel_envs > 0
    assert len(args.gpus) == len(set(args.gpus))
    stamp = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d_%H%M%S')
    output = args.output or ROOT / 'outputs' / f'ogbench_visual_cube_lewm_{stamp}'
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, output / Path(__file__).name)
    settings = {**vars(args), 'output': str(output), 'videos': not args.no_videos}
    settings['checkpoint'] = str(Path(args.checkpoint).resolve())
    jobs = [
        {'id': f'{variant}_seed{seed}_task{task}', 'variant': variant,
         'eval_seed': seed, 'task_id': task}
        for seed in args.seeds for variant in sorted(args.variants, key=lambda x: -LIMITS[x])
        for task in args.tasks
    ]
    packages = {name: importlib.metadata.version(name) for name in
                ['torch', 'torchvision', 'stable-worldmodel', 'ogbench', 'mujoco']}
    config = {
        **settings, 'started_at': datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
        'checkpoint_sha256': sha256(Path(args.checkpoint) / 'weights.pt'),
        'packages': packages, 'jobs': jobs,
        'total_episodes': len(jobs) * args.episodes_per_task,
        'observation': f'visual environment native {args.observation_size}x{args.observation_size} RGB -> existing ImageNet/Resize224 preprocessing',
        'resolution_note': 'official Table 5 uses 64px; 224px renderer requested by user, with task definitions/actions/limits unchanged',
        'policy_inputs': ['pixels', 'goal_image', 'action', 'episode_terminated'],
        'numerical_state_input': False,
        'training': args.training_description,
        'plan': {'horizon': 5, 'receding_horizon': 5, 'action_block': 5, 'observed_history': 1},
        'cem': {'samples': 300, 'iterations': 30, 'topk': 30, 'variance_scale': 1.0},
        'batch_adapter': 'explicit candidate axis in latent goal cost; compared with original per-env costs',
        'simulators': 'AsyncVectorEnv spawn, automatic reset disabled, terminal environments frozen',
        'reset_seeding': 'seed simulator, numpy, and action_space; official reset samples two actions for goal settling',
        'video': {'native_render_size': args.video_size, 'layout': 'raw RGB, no border or overlay',
                  'policy_observation_size': args.observation_size},
        'buffer_optimization': 'skip stateless image preprocessing when executing an existing plan; history_len=1',
        'success': 'final info[success], terminated or truncated, equal average over 5 task IDs',
    }
    import numpy as np
    if args.action_normalizer:
        norm = json.loads(args.action_normalizer.read_text())
        assert len(norm['mean']) == len(norm['scale']) == 5
        assert np.isfinite(norm['mean']).all() and np.all(np.asarray(norm['scale']) > 0)
        settings['action_normalizer'] = str(args.action_normalizer.resolve())
        config['action_normalizer'] = str(args.action_normalizer.resolve())
    else:
        import h5py
        import hdf5plugin  # Register compression filters.
        from sklearn.preprocessing import StandardScaler
        with h5py.File(ROOT / '.stable-wm/cube_single_expert.h5', 'r') as dataset:
            actions = dataset['action'][:]
        actions = actions[np.isfinite(actions).all(axis=1)]
        scaler = StandardScaler().fit(actions)
        norm = {'source': str(ROOT / '.stable-wm/cube_single_expert.h5'),
                'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                'variance': scaler.var_.tolist(), 'samples': int(scaler.n_samples_seen_),
                'procedure': 'same finite-action StandardScaler.fit used in existing eval.py'}
        del actions
    if args.resume and (output / 'config.json').exists():
        previous = json.loads((output / 'config.json').read_text())
        for key in ['checkpoint_sha256', 'variants', 'tasks', 'seeds', 'episodes_per_task',
                    'parallel_envs', 'observation_size', 'video_size', 'matmul_precision']:
            assert previous[key] == config[key], f'Resume configuration changed: {key}'
        assert json.loads((output / 'action-normalizer.json').read_text()) == norm
    write_json(output / 'config.json', config)
    write_json(output / 'action-normalizer.json', norm)
    print(f'Output: {output}\nJobs: {len(jobs)}; episodes: {config["total_episodes"]}', flush=True)
    context = mp.get_context('spawn')
    queue = context.Queue()
    for job in jobs:
        if args.resume and (output / 'jobs' / f'{job["id"]}.json').exists():
            continue
        queue.put(job)
    for _ in args.gpus:
        queue.put(None)
    processes = [context.Process(target=worker, args=(gpu, queue, settings)) for gpu in args.gpus]
    for process in processes:
        process.start()
    while any(process.is_alive() for process in processes):
        time.sleep(5)
        failures = [process for process in processes if process.exitcode not in (None, 0)]
        if failures:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            raise RuntimeError(f'Worker failed: {[p.exitcode for p in failures]}; see GPU logs')
    for process in processes:
        process.join()
    results = [json.loads((output / 'jobs' / f'{job["id"]}.json').read_text()) for job in jobs]
    assert sum(row['episodes'] for row in results) == config['total_episodes']
    summary = {'config': config, 'variants': {}, 'jobs': results}
    for variant in args.variants:
        seed_rates = []
        per_task = {}
        for seed in args.seeds:
            selected = [row for row in results if row['variant'] == variant and row['eval_seed'] == seed]
            seed_rates.append(float(np.mean([row['success_rate_percent'] for row in selected])))
        for task in args.tasks:
            selected = [row for row in results if row['variant'] == variant and row['task_id'] == task]
            per_task[str(task)] = {
                'name': selected[0]['task_name'], 'episodes': sum(row['episodes'] for row in selected),
                'successes': sum(row['successes'] for row in selected),
                'success_rate_percent': float(np.mean([row['success_rate_percent'] for row in selected])),
            }
        summary['variants'][variant] = {
            'env_name': f'visual-cube-{variant}-v0', 'evaluation_seed_rates': seed_rates,
            'mean_success_rate_percent': float(np.mean(seed_rates)),
            'std_over_evaluation_seeds_percent': float(np.std(seed_rates)),
            'episodes': sum(value['episodes'] for value in per_task.values()),
            'successes': sum(value['successes'] for value in per_task.values()),
            'tasks': per_task,
        }
    summary['finished_at'] = datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()
    write_json(output / 'metrics.json', summary)
    print(json.dumps(summary['variants'], indent=2), flush=True)


if __name__ == '__main__':
    main()
