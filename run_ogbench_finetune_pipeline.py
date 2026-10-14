"""Eight independent warm starts, followed by a shared eight-GPU evaluation queue."""

import argparse
import gc
import json
import multiprocessing as mp
import os
import queue
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from eval_ogbench_visual_cube import LIMITS, ROOT, sha256, worker, write_json


DATASETS = [f'visual-cube-{variant}-{kind}-v0'
            for variant in LIMITS for kind in ['play', 'noisy']]


def evaluate_queue(gpu, shared_queue, settings_by_dataset, errors):
    os.setsid()
    try:
        while True:
            job = shared_queue.get()
            if job is None:
                return
            local_queue = queue.Queue()
            local_queue.put(job)
            local_queue.put(None)
            worker(gpu, local_queue, settings_by_dataset[job['dataset']])
            gc.collect()
    except BaseException:
        errors.put({'gpu': gpu, 'traceback': traceback.format_exc()})
        raise


def wait_training(processes, handles, output):
    try:
        while any(process.poll() is None for process in processes):
            failures = [index for index, process in enumerate(processes) if process.poll() not in (None, 0)]
            if failures:
                names = [processes[index].args[processes[index].args.index('--dataset') + 1] for index in failures]
                raise RuntimeError(f'Training jobs failed: {names}; see train.log')
            time.sleep(5)
        assert all(process.returncode == 0 for process in processes)
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for handle in handles:
            handle.close()


def run_training(args, output, manifest):
    write_json(output / 'status.json', {'phase': 'training', 'datasets': DATASETS, 'gpus': args.gpus})
    processes, handles = [], []
    for gpu, dataset in zip(args.gpus, DATASETS):
        destination = output / 'training' / dataset
        destination.mkdir(parents=True, exist_ok=True)
        if (destination / 'train-complete.json').exists():
            result = json.loads((destination / 'train-complete.json').read_text())
            assert result['optimizer_steps'] == manifest['optimizer_steps_per_model']
            continue
        command = [sys.executable, '-u', str(ROOT / 'finetune_ogbench_visual_cube.py'),
                   '--dataset', dataset, '--output', str(destination), '--checkpoint', str(args.checkpoint),
                   '--group', args.group, '--steps', str(manifest['optimizer_steps_per_model']),
                   '--batch-size', str(args.batch_size), '--lr', str(args.lr),
                   '--workers', str(args.workers), '--wandb-mode', args.wandb_mode]
        if args.smoke:
            command += ['--max-batches', '2', '--val-batches', '2', '--smoke-prefix', '1001',
                        '--native-pixels-dir', str(output / 'native-smoke-cache')]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4', STABLEWM_HOME=str(ROOT / '.stable-wm'),
                   WANDB_INIT_TIMEOUT='120', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
        handle = (destination / 'train.log').open('a', buffering=1)
        processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True))
        handles.append(handle)
        write_json(destination / 'launch.json', {'gpu': gpu, 'pid': processes[-1].pid, 'command': command})
        print(f'Training {dataset} on GPU {gpu}, PID {processes[-1].pid}', flush=True)
    wait_training(processes, handles, output)
    for dataset in DATASETS:
        destination = output / 'training' / dataset
        assert (destination / 'train-complete.json').exists()
        assert json.loads((destination / 'optimizer-check.json').read_text())['passed']
        assert json.loads((destination / 'encoder-input-check.json').read_text())['passed']
        best = json.loads((destination / 'best-checkpoint.json').read_text())
        assert sha256(Path(best['checkpoint']) / 'weights.pt') == best['sha256']


def evaluation_settings(args, output, dataset):
    training = output / 'training' / dataset
    settings = {'output': str(output / 'evaluation' / dataset),
            'checkpoint': str(training / 'checkpoints' / 'best'),
            'observation_size': 224, 'video_size': 224, 'videos': True,
            'matmul_precision': 'highest', 'compile_predictor': False,
            'parallel_envs': min(10, args.episodes_per_task),
            'episodes_per_task': args.episodes_per_task, 'seeds': args.seeds,
            'gpus': args.gpus,
            'training': f'finetuned on {dataset} official train NPZ, checkpoint selected using separate -val NPZ'}
    from select_ogbench_checkpoint import evaluation_checkpoint
    return evaluation_checkpoint(settings)


def run_evaluation(args, output, manifest):
    import numpy as np
    write_json(output / 'status.json', {'phase': 'evaluation', 'total_episodes': manifest['total_evaluation_episodes']})
    settings = {dataset: evaluation_settings(args, output, dataset) for dataset in DATASETS}
    jobs = []
    for dataset in DATASETS:
        variant = dataset.split('-')[2]
        destination = Path(settings[dataset]['output'])
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / 'training' / dataset / 'action-normalizer.json', destination / 'action-normalizer.json')
        selected = [{'id': f'{variant}_seed{seed}_task{task}', 'dataset': dataset, 'variant': variant,
                     'eval_seed': seed, 'task_id': task}
                    for seed in args.seeds for task in args.tasks]
        config = {**settings[dataset], 'jobs': selected,
                  'checkpoint_sha256': sha256(Path(settings[dataset]['checkpoint']) / 'weights.pt'),
                  'total_episodes': len(selected) * args.episodes_per_task,
                  'dataset': dataset, 'numerical_state_input': False,
                  'plan': {'horizon': 5, 'receding_horizon': 5, 'action_block': 5, 'observed_history': 1},
                  'cem': {'samples': 300, 'iterations': 30, 'topk': 30, 'variance_scale': 1.0},
                  'resolution_note': 'train/validation/evaluation RGB all renderer-native 224, official states re-rendered without upscaling',
                  'reset_seeding': 'same simulator/numpy/action_space reset seeding as preceding fixed-checkpoint benchmark'}
        if (destination / 'config.json').exists():
            old = json.loads((destination / 'config.json').read_text())
            for key in ['checkpoint_sha256', 'jobs', 'episodes_per_task', 'observation_size', 'video_size']:
                assert old[key] == config[key], f'Cannot mix different evaluations: {dataset}/{key}'
        write_json(destination / 'config.json', config)
        jobs.extend(selected)
    # Keep all cards busy: any card can evaluate any checkpoint/task. Longest
    # episode limits first, then let the common queue balance remaining work.
    jobs.sort(key=lambda job: (-LIMITS[job['variant']], job['eval_seed'], job['task_id'], job['dataset']))
    context = mp.get_context('spawn')
    shared_queue, errors = context.Queue(), context.Queue()
    for job in jobs:
        destination = Path(settings[job['dataset']]['output'])
        if not (destination / 'jobs' / f'{job["id"]}.json').exists():
            shared_queue.put(job)
    for _ in args.gpus:
        shared_queue.put(None)
    processes = [context.Process(target=evaluate_queue, args=(gpu, shared_queue, settings, errors))
                 for gpu in args.gpus]
    for process in processes:
        process.start()
    try:
        while any(process.is_alive() for process in processes):
            failed = [process.exitcode for process in processes if process.exitcode not in (None, 0)]
            if failed:
                messages = []
                while not errors.empty():
                    messages.append(errors.get())
                write_json(output / 'evaluation-errors.json', messages)
                raise RuntimeError(f'Evaluation worker failure: {failed}; inspect evaluation GPU logs')
            time.sleep(5)
    finally:
        for process in processes:
            if process.is_alive():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in processes:
            process.join(timeout=20)
            if process.is_alive():
                os.killpg(process.pid, signal.SIGKILL)
                process.join()
    assert all(process.exitcode == 0 for process in processes)
    combined = {'manifest': manifest, 'datasets': {}}
    for dataset in DATASETS:
        destination = Path(settings[dataset]['output'])
        config = json.loads((destination / 'config.json').read_text())
        rows = [json.loads((destination / 'jobs' / f'{job["id"]}.json').read_text()) for job in config['jobs']]
        variant = dataset.split('-')[2]
        rates = [float(np.mean([row['success_rate_percent'] for row in rows if row['eval_seed'] == seed]))
                 for seed in args.seeds]
        tasks = {}
        for task in args.tasks:
            selected = [row for row in rows if row['task_id'] == task]
            tasks[str(task)] = {'name': selected[0]['task_name'],
                               'episodes': sum(row['episodes'] for row in selected),
                               'successes': sum(row['successes'] for row in selected),
                               'success_rate_percent': float(np.mean([row['success_rate_percent'] for row in selected]))}
        summary = {'env_name': f'visual-cube-{variant}-v0', 'evaluation_seed_rates': rates,
                   'mean_success_rate_percent': float(np.mean(rates)),
                   'std_over_evaluation_seeds_percent': float(np.std(rates)),
                   'episodes': sum(row['episodes'] for row in rows),
                   'successes': sum(row['successes'] for row in rows), 'tasks': tasks}
        assert summary['episodes'] == config['total_episodes']
        write_json(destination / 'metrics.json', {'config': config, 'jobs': rows, 'variants': {variant: summary}})
        combined['datasets'][dataset] = summary
    assert sum(row['episodes'] for row in combined['datasets'].values()) == manifest['total_evaluation_episodes']
    write_json(output / 'metrics.json', combined)
    verify_videos(args, output)
    if args.wandb_mode == 'online':
        log_evaluation_to_wandb(args, output, combined)


def verify_videos(args, output):
    import imageio.v2 as imageio
    import imageio_ffmpeg
    videos = []
    for dataset in DATASETS:
        variant = dataset.split('-')[2]
        destination = output / 'evaluation' / dataset
        assert list(destination.glob('gpu_*_encoder_input_check.json'))
        for check in destination.glob('gpu_*_encoder_input_check.json'):
            data = json.loads(check.read_text())
            assert data['passed'] and data['renderer_observation_size'] == [224, 224, 3]
        for check in destination.glob('gpu_*_batch_cost_check.json'):
            assert json.loads(check.read_text())['passed']
        for task in args.tasks:
            video = destination / 'videos' / f'{variant}_task{task}.mp4'
            row = json.loads((destination / 'jobs' / f'{variant}_seed{args.seeds[0]}_task{task}.json').read_text())
            reader = imageio.get_reader(video)
            size = list(reader.get_meta_data()['size'])
            reader.close()
            assert size == [224, 224]
            frames, _ = imageio_ffmpeg.count_frames_and_secs(str(video))
            assert frames == row['records'][0]['steps'] + 1
            videos.append({'dataset': dataset, 'task': task, 'path': str(video), 'size': size, 'frames': frames})
    write_json(output / 'video-manifest.json', {'passed': True, 'videos': videos,
               'layout': 'native 224 RGB, no white borders, no overlays'})


def log_evaluation_to_wandb(args, output, combined):
    import wandb
    for dataset in DATASETS:
        training = output / 'training' / dataset
        run_info = json.loads((training / 'wandb-run.json').read_text())
        with wandb.init(entity=os.environ['WANDB_ENTITY'], project=os.environ['WANDB_PROJECT'],
                        id=run_info['id'], resume='must', dir=str(training),
                        job_type='lewm-ogbench-finetune', mode='online') as run:
            summary = combined['datasets'][dataset]
            payload = {'eval/success_rate_percent': summary['mean_success_rate_percent'],
                       'eval/std_over_evaluation_seeds_percent': summary['std_over_evaluation_seeds_percent'],
                       'eval/episodes': summary['episodes']}
            for task, values in summary['tasks'].items():
                payload[f'eval/task_{task}/success_rate_percent'] = values['success_rate_percent']
                variant = dataset.split('-')[2]
                video = output / 'evaluation' / dataset / 'videos' / f'{variant}_task{task}.mp4'
                payload[f'eval/task_{task}/video'] = wandb.Video(str(video), format='mp4')
            run.log(payload)
            run.summary['evaluation_complete'] = True


def update_document(output, complete=False):
    """Only replace this run's marked block; preserve all preceding answers."""
    manifest = json.loads((output / 'manifest.json').read_text())
    if manifest['smoke']:
        return
    document = ROOT / 'zeyu-understand.md'
    start = f'<!-- lewm-finetune:{manifest["group"]}:start -->'
    end = f'<!-- lewm-finetune:{manifest["group"]}:end -->'
    lines = [start, f'### OGBench visual Cube 分数据集微调与自动评测（{manifest["group"]}）', '',
             '官方主 NPZ 用于训练，独立 `-val.npz` 仅用于验证及选择 checkpoint。在线成功率在模拟器的 5 个官方目标任务上测量；`-val` 不是在线测试集。', '',
             f'八个模型分别从原始 `cube-lewm` checkpoint 严格加载全部权重，然后在四种 Cube × play/noisy 的各自 train 数据上独立微调。原始权重 SHA256：`{manifest["initial_checkpoint_sha256"]}`。', '',
             f'每模型 {manifest["optimizer_steps_per_model"]:,} 次更新，batch size={manifest["batch_size"]}，AdamW lr={manifest["lr"]}，weight decay=0.001，5% warmup（起始 lr=目标值的 10%）后 cosine，bf16 mixed，梯度裁剪 1.0。history=3，prediction=1，frameskip=5，预测 MSE + 0.09×SIGReg。所有模型参数参与更新。', '',
             '八张卡各训练一个模型，统一更新次数以平衡不同数据量；每遍 train 数据后和最后一步均在完整官方 val 数据上验证。训练目标仍为预测MSE+0.09×SIGReg；自动评测前按完整独立val集上最低预测MSE选checkpoint。连续轨迹组批会抬高验证SIGReg，原组合指标仅留作诊断。动作标准化只拟合各自 train 数据的非 terminal 动作，并用于该模型的训练、验证和在线规划。qpos/qvel 等数值状态不输入模型。', '',
             '训练及验证图像由官方 NPZ 保存的 qpos/qvel 还原同一场景，直接用 MuJoCo 渲染器生成 224×224 RGB；保留官方透明机械臂、front_pixels 相机及静态物体配色。图片用无损 Zstd 存储，训练仅做 ImageNet normalize，禁止 64 放大。qpos/qvel 只用于离线重渲染，不作为 LeWM 输入。重渲染时不执行动作或额外模拟 step，原始动作、状态次序和 train/val 划分保持一致。每个数据 shard 在 64 分辨率上与原官方图片核对平均像素误差 <1/255 后生成原生224缓存。', '',
             '全部重渲染完成并校验后，原位替换 ogbench 下对应 16 份 visual-cube train/val NPZ 的 observations 为原生224，删除其旧64图片 NPY缓存。所有非图片 NPZ成员逐字节 SHA256核对后保留，原文件名、动作、状态、terminal、帧序和 train/val 划分保持一致。替换记录见 ogbench/native224-replacement-manifest.json。', '',
             '在线评测输入、goal 图像和无白边原始视频也均由渲染器直接产生 224×224。本实验训练/验证/测试分辨率统一为原生224；PDF baseline 使用标准64像素协议，本表应作为参照比较。', '',
             f'训练完成后自动启动八卡共用任务队列：每模型 5 个 task × 4 个 evaluation seed（0,1,2,3）× 50 episodes = 1,000 episodes，总计 8,000。CEM 与之前固定 checkpoint 的测试完全一致：samples=300，iterations=30，topk=30，horizon=5，receding=5，action block=5，observed history=1。', '',
             '下表 baseline 严格来自本地 ogbench.pdf 第 21 页 Table 5。baseline 的标准差来自 4 个独立训练种子；本次每份数据只训练 1 个模型，LeWM 微调结果的标准差来自 4 个评测种子，统计含义不同。先前 LeWM 固定权重在四种 Cube 上均为 0/1,000；该旧结果对 play/noisy 共用，微调后每行是各自数据训练的独立模型。', '',
             '| Dataset | GCBC | GCIVL | GCIQL | QRL | CRL | HIQL | LeWM 原 checkpoint | LeWM 微调 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    baseline = json.loads((ROOT / 'ogbench_table5_visual_cube_baselines.json').read_text())
    metrics = json.loads((output / 'metrics.json').read_text()) if complete else None
    for row in baseline['rows']:
        cells = [row['dataset']]
        cells += [f'{row["methods"][name]["mean"]} ± {row["methods"][name]["std"]}'
                  for name in ['GCBC', 'GCIVL', 'GCIQL', 'QRL', 'CRL', 'HIQL']]
        cells.append('0 ± 0')
        if complete:
            values = metrics['datasets'][row['dataset']]
            cells.append(f'{values["mean_success_rate_percent"]:.2f} ± {values["std_over_evaluation_seeds_percent"]:.2f}')
        else:
            cells.append('待训练及自动评测完成')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', f'运行目录：`{output}`。tmux session：`{manifest["tmux_session"]}`。', '',
              '脚本：`launch_ogbench_finetune_8gpu.sh` → `run_ogbench_finetune_pipeline.py` → `prepare_ogbench_native224.py`（八卡共享 shard 队列重渲染）→ `replace_ogbench_native224.py`（原位替换 NPZ并删除旧图片）→ `finetune_ogbench_visual_cube.py`（八模型各一张卡微调）→ `eval_ogbench_visual_cube.py`（八卡共用 task 队列自动评测）。', '',
              '[W&B 项目](https://forge.coreweave.com/wandb/visual-intelligence-laboratory/P-Rep)，各模型的 loss 链接保存在对应 training 子目录的 `wandb-run.json`。密钥存储在仓库外的私有环境文件中。']
    if complete:
        lines += ['', '微调和全部在线评测已完成，checkpoint、逐 episode 记录、原生视频及输入检查均保存在运行目录。', '',
                  '| Dataset | 成功数 / episodes | best epoch | best val loss |', '|---|---:|---:|---:|']
        for dataset in DATASETS:
            values = metrics['datasets'][dataset]
            best = json.loads((output / 'training' / dataset / 'best-checkpoint.json').read_text())
            lines.append(f'| {dataset} | {values["successes"]} / {values["episodes"]} | {best["epoch"]} | {best["val_loss"]:.6f} |')
        lines += ['', '| Dataset | Task | Success % |', '|---|---|---:|']
        for dataset in DATASETS:
            for task, value in metrics['datasets'][dataset]['tasks'].items():
                lines.append(f'| {dataset} | {task}: {value["name"]} | {value["success_rate_percent"]:.2f} |')
    lines += [end, '']
    block = '\n'.join(lines)
    original = document.read_text()
    if start in original:
        before, remainder = original.split(start, 1)
        _, after = remainder.split(end, 1)
        updated = before + block.rstrip('\n') + after
    else:
        updated = original.rstrip('\n') + '\n\n' + block
    temporary = document.with_suffix('.md.tmp')
    temporary.write_text(updated)
    temporary.replace(document)
    (output / 'finetune-report.md').write_text(block)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render-workers-per-gpu', type=int, default=4)
    parser.add_argument('--group', required=True)
    parser.add_argument('--tmux-session', default='lewm-ogbench-ft8')
    parser.add_argument('--gpus', type=int, nargs='+', default=list(range(8)))
    parser.add_argument('--checkpoint', type=Path, default=ROOT / '.stable-wm/checkpoints/cube-lewm')
    parser.add_argument('--steps', type=int, default=80000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--episodes-per-task', type=int, default=50)
    parser.add_argument('--tasks', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--wandb-mode', choices=['online', 'disabled'], default='online')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert len(args.gpus) == 8 and len(set(args.gpus)) == 8
    if args.smoke:
        args.steps, args.batch_size, args.workers = 2, 8, 2
        args.episodes_per_task, args.tasks, args.seeds = 1, [1], [0, 1]
        args.wandb_mode = 'disabled'
    else:
        assert args.tasks == [1, 2, 3, 4, 5] and args.seeds == [0, 1, 2, 3] and args.episodes_per_task == 50
        assert args.wandb_mode == 'online', 'Production must log losses online to W&B'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'group': args.group, 'smoke': args.smoke, 'tmux_session': args.tmux_session,
                'started_at': datetime.now().astimezone().isoformat(), 'datasets': DATASETS,
                'gpus': args.gpus, 'optimizer_steps_per_model': args.steps,
                'batch_size': args.batch_size, 'lr': args.lr,
                'initial_checkpoint': str(args.checkpoint), 'native_training_resolution': 224,
                'render_workers_per_gpu': args.render_workers_per_gpu,
                'initial_checkpoint_sha256': sha256(args.checkpoint / 'weights.pt'),
                'total_evaluation_episodes': len(DATASETS) * len(args.tasks) * len(args.seeds) * args.episodes_per_task}
    if (output / 'manifest.json').exists():
        previous = json.loads((output / 'manifest.json').read_text())
        assert previous.get('native_training_resolution') == 224, 'Cannot resume a 64->224 training run'
        for key in ['group', 'smoke', 'datasets', 'optimizer_steps_per_model', 'batch_size', 'lr', 'initial_checkpoint_sha256']:
            assert previous[key] == manifest[key], f'Unsafe resume parameter change: {key}'
    write_json(output / 'manifest.json', manifest)
    if not args.smoke:
        write_json(output / 'validation-selection-request.json', {
            'enabled': True, 'metric': 'minimum full held-out prediction MSE',
            'reason': 'Validation SIGReg depends on temporal correlation within consecutive trajectory batches',
            'training_objective_unchanged': 'prediction MSE + 0.09*SIGReg',
        })
    for name in ['finetune_ogbench_visual_cube.py', 'run_ogbench_finetune_pipeline.py',
                 'eval_ogbench_visual_cube.py', 'launch_ogbench_finetune_8gpu.sh',
                 'prepare_ogbench_native224.py', 'replace_ogbench_native224.py',
                 'select_ogbench_checkpoint.py', 'ogbench_table5_visual_cube_baselines.json']:
        shutil.copy2(ROOT / name, output / name)
    try:
        update_document(output)
        from prepare_ogbench_native224 import prepare
        if args.smoke:
            prepare(output, args.gpus, 1, native_root=output / 'native-smoke-cache', smoke=True)
        else:
            prepare(output, args.gpus, args.render_workers_per_gpu)
            write_json(output / 'status.json', {'phase': 'replacing_ogbench_npz_with_native224'})
            from replace_ogbench_native224 import replace_all
            replace_all()
        run_training(args, output, manifest)
        run_evaluation(args, output, manifest)
        assert sha256(args.checkpoint / 'weights.pt') == manifest['initial_checkpoint_sha256']
        update_document(output, complete=True)
        write_json(output / 'status.json', {'phase': 'complete', 'group': args.group,
                   'total_evaluation_episodes': manifest['total_evaluation_episodes']})
        print('PIPELINE COMPLETE', flush=True)
    except BaseException:
        write_json(output / 'status.json', {'phase': 'failed', 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
