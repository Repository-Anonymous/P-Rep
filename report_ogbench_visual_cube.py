"""Append a completed official visual Cube evaluation to the local notes."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANTS = ['single', 'double', 'triple', 'quadruple']
METHODS = ['GCBC', 'GCIVL', 'GCIQL', 'QRL', 'CRL', 'HIQL']


def generate(metrics_path, document, baselines_path):
    data = json.loads(metrics_path.read_text())
    config = data['config']
    output = metrics_path.parent.resolve()
    baseline = json.loads(baselines_path.read_text())
    assert config['total_episodes'] == 4000
    assert config['seeds'] == [0, 1, 2, 3]
    assert config['gpus'] == list(range(8))
    assert config['observation_size'] == config['video_size'] == 224
    assert len(data['jobs']) == 80
    assert sum(job['episodes'] for job in data['jobs']) == 4000
    assert sum(job['successes'] for job in data['jobs']) == sum(
        row['successes'] for row in data['variants'].values())
    for variant in VARIANTS:
        assert data['variants'][variant]['episodes'] == 1000
        assert len(data['variants'][variant]['tasks']) == 5
        assert all(task['episodes'] == 200 for task in data['variants'][variant]['tasks'].values())
    checks = [json.loads((output / f'gpu_{gpu}_batch_cost_check.json').read_text())
              for gpu in range(8)]
    assert all(check['passed'] and check['best_candidate_matches'] for check in checks)
    assert len(list((output / 'videos').glob('*.mp4'))) == 20
    video_manifest = json.loads((output / 'video-manifest.json').read_text())
    assert len(video_manifest['videos']) == 20
    assert all(row['size'] == [224, 224] for row in video_manifest['videos'])
    checkpoint = Path(config['checkpoint'])
    heading = f"### OGBench visual Cube 原生 224×224 八卡评估（{config['started_at'][:10]}）"
    marker = f'<!-- ogbench-evaluation: {output.name} -->'
    old = document.read_text()
    assert '## ogbench' in old
    if marker in old:
        raise ValueError('This evaluation has already been recorded.')
    rows = [
        heading, '', marker, '',
        f"评估完成时间：`{data['finished_at']}`（America/Los_Angeles）。使用同一个固定 "
        f"[cube-lewm checkpoint]({checkpoint})，未重新训练或微调。模型的观测和目标均由 MuJoCo "
        "直接渲染为 224×224，没有经过 64×64 下采样。单方块结果包含从自采集图像到 visual 外观的迁移；"
        "double、triple、quadruple 均为**同一 "
        "single-cube checkpoint 的跨环境迁移**。", '',
        '#### 与 PDF Table 5 对照', '',
        '下表六个 baseline 列逐项取自 [OGBench PDF 第 21 页 Table 5]'
        '(/home/coder/P-Rep/ogbench/ogbench.pdf)。全部数值为成功率百分比。**论文使用原生 64×64 观测；'
        'LeWM 按用户要求使用原生 224×224，因此观测分辨率不同。**任务、动作、目标和步数预算沿用官方设置。', '',
        '| Table 5 数据集（baseline 64×64） | GCBC | GCIVL | GCIQL | QRL | CRL | HIQL | LeWM 原生224固定 checkpoint（均值 ± 评估种子标准差） | LeWM 评估性质 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in baseline['rows']:
        result = data['variants'][row['variant']]
        cells = [row['dataset']] + [f"{row['methods'][method]['mean']} ± {row['methods'][method]['std']}"
                                   for method in METHODS]
        cells += [f"{result['mean_success_rate_percent']:.2f} ± {result['std_over_evaluation_seeds_percent']:.2f}",
                  '单方块 visual 外观迁移' if row['variant'] == 'single' else '同一 checkpoint 跨环境迁移']
        rows.append('| ' + ' | '.join(cells) + ' |')
    rows += ['',
        '**对照口径：**`play/noisy` 指训练数据分布，二者共享同一个评估环境。LeWM 此次只使用原有 '
        '自采集单方块 expert 数据训练的 checkpoint，因此同一变体在 play/noisy 两行重复显示同一组 '
        '评估结果；没有在这两套官方数据上分别训练 LeWM。论文 baseline 的 ± 是 4 个独立训练种子之间的 '
        '标准差，每个训练种子的分数还平均最后 3 次评估 checkpoint；LeWM 的 ± 是一个固定 checkpoint '
        '在 4 个评估种子上的标准差。这是迁移性能参考，训练条件和误差条含义不同。', '',
        '#### 五个官方目标的实测结果', '',
        '每个环境共 1,000 回合；每个 task 共 200 回合（4 个评估种子 × 每种子 50 回合）。', '',
        '| 环境 | task 1 | task 2 | task 3 | task 4 | task 5 | 总成功数 / 回合数 | 平均成功率 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for variant in VARIANTS:
        result = data['variants'][variant]
        cells = [f'visual-cube-{variant}-v0']
        cells += [f"{result['tasks'][str(task)]['success_rate_percent']:.2f}% "
                  f"({result['tasks'][str(task)]['successes']}/200)" for task in range(1, 6)]
        cells += [f"{result['successes']}/1000", f"{result['mean_success_rate_percent']:.2f}%"]
        rows.append('| ' + ' | '.join(cells) + ' |')
    rows += ['', '| 环境 | seed 0 | seed 1 | seed 2 | seed 3 |',
             '|---|---:|---:|---:|---:|']
    for variant in VARIANTS:
        cells = [variant] + [f'{rate:.2f}%' for rate in data['variants'][variant]['evaluation_seed_rates']]
        rows.append('| ' + ' | '.join(cells) + ' |')
    rows += ['', '#### 评估设置与验证', '',
        '| 项目 | 设置 |', '|---|---|',
        '| 环境与目标 | 官方 `visual-cube-{single,double,triple,quadruple}-v0`；`reset(options={task_id: 1..5})` 返回的官方目标图像 |',
        '| 单回合预算 | single 200；double 500；triple 1,000；quadruple 1,000 个环境 step，与注册环境一致 |',
        '| 成功判定 | 官方最终 `info[success]`；成功终止或达到 TimeLimit 后停止；对 5 个 task 等权平均 |',
        '| 图像输入 | visual 相机和半透明机械臂，渲染器 width=height=224；观测与目标均为原生 224×224 RGB；ImageNet normalization + Resize224（尺寸不变） |',
        '| 数值输入 | 仅动作用于动作编码；没有把关节、夹爪、物体位置等 state 向量送入模型 |',
        '| 规划 | CEM：300 candidates、30 iterations、30 elites、variance_scale=1；horizon=5、action_block=5、receding_horizon=5（每 25 个环境 step 重规划） |',
        '| 历史 | 延续当前推理入口的 1 帧观测历史；预测器在想象 rollout 中保留至多 3 帧上下文 |',
        '| 动作处理 | 原训练 HDF5 有限动作行拟合 StandardScaler（2,000,000 行）；执行动作 clip 到 [-1,1] |',
        f"| 并行 | 8 张 H200；每 GPU 1 个规划进程，每批 {config['parallel_envs']} 个独立模拟器；80 个任务批次动态分配 |",
        f"| 数值精度 | float32；PyTorch matmul precision `{config['matmul_precision']}`；predictor compile={config['compile_predictor']} |",
        '| 终止处理 | AsyncVectorEnv 使用 spawn；关闭自动 reset；完成的环境保持终止状态，不计入新回合 |', '',
        '模型 encoder 的 forward 前置检查在 8 张 GPU 上直接读取实际输入张量，均为 '
        '`[batch, 3, 224, 224]`、float32。已规划动作执行期间跳过重复的无状态图像预处理；'
        '在 1、2、10 个环境以及混合终止状态下，25 步缓存动作与原实现逐值相同。', '',
        '固定了模拟器、NumPy 与 CEM 的随机种子。官方 Cube 在生成目标图像时还会调用 '
        'action_space.sample() 做两步稳定动作，而 action_space 属性每次创建新的 Box；'
        '此次保留官方采样行为，因此固定 reset seed 不保证目标图像逐像素重复。', '',
        '修复了原 LeWM 目标 embedding 在多环境批量规划时缺少 candidate 轴的问题，仅在评估适配器中 '
        '增加广播维度，未修改权重或代价函数。8 个 GPU 均使用真实起始图像，验证了批量代价与原来逐环境、'
        'matmul precision=highest 的代价一致（rtol=atol=1e-3），且候选动作最优索引一致。', '',
        f"Checkpoint weights SHA-256：`{config['checkpoint_sha256']}`。", '',
        f"运行版本：{'; '.join(name + '=' + value for name, value in config['packages'].items())}。", '',
        '官方环境的正常推理和统计流程通过验证；成功率表示这个固定模型的任务性能。'
        '训练数据来自原版单方块 224×224 渲染，visual 测试改变了相机、颜色和透明机械臂外观，'
        '多方块同时改变物体数量。本次已消除 64→224 放大，但仍存在训练与测试的外观和任务差异；'
        '这些结果不能代表重新训练后的 LeWM 上限。', '',
        '#### 可复现记录', '',
        f"- [完整指标与逐回合结果]({output / 'metrics.json'})；[运行配置]({output / 'config.json'})；"
        f"[动作 normalizer]({output / 'action-normalizer.json'})。",
        f"- [Table 5 baseline 原值]({output / baselines_path.name})；"
        f"[本次执行脚本快照]({output / 'eval_ogbench_visual_cube.py'})。",
        f"- [80 个 task/seed 原始记录]({output / 'jobs'})；[20 个目标的推理视频与起始图]({output / 'videos'})。"
        '视频为 MuJoCo 原生渲染的 224×224 当前观测，无白边、拼接或叠字。'
        '模型收到的也是这一原生 224×224 图像；20 个视频回合已包含在 4,000 回合统计中。',
        f"- [视频尺寸及模型输入检查]({output / 'video-manifest.json'})。", '',
        '复现命令：', '', '```bash', 'cd /home/coder/P-Rep/le-wm',
        '.venv/bin/python eval_ogbench_visual_cube.py \\',
        '  --gpus 0 1 2 3 4 5 6 7 --seeds 0 1 2 3 \\',
        '  --variants single double triple quadruple --tasks 1 2 3 4 5 \\',
        f"  --episodes-per-task 50 --parallel-envs 10 --matmul-precision {config['matmul_precision']} \\",
        '  --observation-size 224 --video-size 224',
        '```', '',
    ]
    section = '\n'.join(rows)
    (output / 'evaluation-report.md').write_text(section)
    shutil.copy2(baselines_path, output / baselines_path.name)
    before = document.read_bytes()
    shutil.copy2(document, output / 'zeyu-understand.before-evaluation.md')
    with document.open('a') as stream:
        stream.write('\n\n' + section)
    assert document.read_bytes().startswith(before)
    (output / 'documentation-update.json').write_text(json.dumps({
        'document': str(document), 'before_sha256': hashlib.sha256(before).hexdigest(),
        'before_bytes': len(before), 'prefix_preserved': True,
        'section': heading, 'metrics': str(metrics_path),
    }, indent=2) + '\n')
    print(f'Appended completed results to {document}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics', type=Path)
    parser.add_argument('--document', type=Path, default=ROOT / 'zeyu-understand.md')
    parser.add_argument('--baselines', type=Path, default=ROOT / 'ogbench_table5_visual_cube_baselines.json')
    args = parser.parse_args()
    generate(args.metrics.resolve(), args.document.resolve(), args.baselines.resolve())


if __name__ == '__main__':
    main()
