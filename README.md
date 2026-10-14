
# LeWorldModel
### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Checkpoints &amp; Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Using the code
This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. Together they reduce this repository to its core contribution: the model architecture and training objective.

**Installation:**
```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch training:
```bash
python train.py data=pusht
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## OGBench IL / offline RL policy training (frozen LeWM)

`ogbench_il_rl/` is a downstream training module ported into this repository from the [`state_fusion.py` interface on JEPA4R's `OGBench-IL-RL` branch](https://github.com/AIGeeksGroup/JEPA4R/blob/93ae307319a9d61b0e60ebc5e77b3ab6b6d12742/lewm_pcd/ogbench_il_rl/models/state_fusion.py). The inference architecture is **LeWM encoder + projector + policy head**: LeWM encodes the current and goal RGB224 images into one 192-dimensional latent each, and the following GCBC or GCIQL network outputs the action. LeWM remains fully frozen during training and is excluded from the optimizer; only the policy/value/critic networks are trained. Policy checkpoints likewise store only these downstream network parameters and record the path to the original LeWM checkpoint.

This directory covers only imitation learning and offline RL with 2D RGB LeWM:

- `gcbc`: the same 512×3 MLP, fixed unit-variance Gaussian NLL, and Adam at 3e-4 as JEPA4R and official OGBench.
- `gciql`: ports the GCIQL objective from official OGBench `572788d` onto the same frozen LeWM representation. It trains the actor, V, Q1, and Q2, with Polyak target updates for the two Q networks. Defaults are expectile=0.9, discount=0.99, tau=0.005, and DDPG+BC alpha=0.3.
- Optional `--state-fusion` preserves the input semantics of the linked code: the current input uses the complete state28, while the goal retains only the object XYZ at `state[19:22]`; the visual goal remains the complete RGB224 image. This state28 interface applies only to single-cube.
- Visual Cube and non-visual Cube contain independent trajectories, so observation28 from `cube-single-*.npz` cannot be directly paired with Visual Cube. `prepare_state28.py` restores the official state28 from the qpos/qvel on the same Visual Cube row. The trainer verifies the split, row count, dtype, dimensions, and qpos/qvel/state28 SHA-256 hashes.

That JEPA4R commit publishes the frozen LeWM, GCBC, state-fusion, and data interfaces, but it does not publish a GCIQL training loop. The GCIQL algorithm here therefore comes from the official OGBench implementation pinned by its documentation, with its input adapted to the same frozen LeWM/state-fusion interface. This directory does not include the JEPA4R/4R-JEPA model, 4D temporal aggregation, point clouds, 3D encoders, DINO, or JEPA4R world-model training code.

First generate same-trajectory state28 files for the single-cube train and validation splits. Skip this step for visual-only training:

```bash
source .venv/bin/activate
python -m ogbench_il_rl.prepare_state28 \
  --dataset visual-cube-single-play-v0 --workers 8
python -m ogbench_il_rl.prepare_state28 \
  --dataset visual-cube-single-play-v0-val --workers 8
```

Train GCBC (IL) or GCIQL (offline RL):

```bash
# visual-only IL
python -m ogbench_il_rl.train \
  --algorithm gcbc \
  --dataset visual-cube-single-play-v0 \
  --output outputs/ogbench_policy/gcbc_visual

# LeWM + state28 IL
python -m ogbench_il_rl.train \
  --algorithm gcbc --state-fusion \
  --dataset visual-cube-single-play-v0 \
  --output outputs/ogbench_policy/gcbc_state

# LeWM + state28 offline RL / GCIQL
python -m ogbench_il_rl.train \
  --algorithm gciql --state-fusion \
  --dataset visual-cube-single-play-v0 \
  --output outputs/ogbench_policy/gciql_state
```

By default, the trainer reads `/home/coder/P-Rep/ogbench/.lewm-native224/<dataset>/pixels.h5`. These images are rendered directly at 224×224 and are not upscaled from 64 to 224. The default LeWM checkpoint is `.stable-wm/checkpoints/cube-lewm`; use `--lewm-checkpoint` to select another local export. Training defaults to 100,000 steps, batch size 256, and FP32. Use `--wandb-mode online` to log to W&B and `--resume` to resume exactly from `training-state.pt` in the output directory.

Loading an inference model reloads the frozen LeWM and then applies the policy-only weights:

```python
from ogbench_il_rl.checkpoint import load_policy

policy = load_policy("outputs/ogbench_policy/gciql_state/policy_step_100000.pt").cuda()
action = policy.get_action(info).clamp(-1, 1)
```

See `ogbench_il_rl/configs/` for configuration details. The pinned commits above define the implementation provenance, scope, and algorithmic differences.

## Planning

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

### Default Cube inference workflow

For the locally installed Cube checkpoint and dataset, run:

```bash
./run_cube_inference.sh
```

Each invocation creates an isolated timestamped directory under `outputs/`.
That directory contains `cube_lewm_inference.mp4`, a PNG preview,
`cube_lewm_results.txt`, the evaluation log, Hydra config, and a copy of the exact
launcher as `run.sh`. The default evaluates one episode;
override it when needed:

```bash
NUM_EVAL=50 ./run_cube_inference.sh
```

Multi-episode runs save one numbered MP4 and PNG preview per episode.

<!-- ogbench-finetune-results:lewm_visual_cube_native224_ft8_20261005_211637:start -->
## OGBench Visual Cube native-224 fine-tuning and evaluation (local experiment)

8 independent models were fine-tuned across the four Cube variants × play/noisy, with 80,000 updates per model. Automated online evaluation covered 8,000 episodes in total, with 1,000 episodes per model.

The models succeeded in **16/8000 episodes (0.20%)**. Triple-noisy was highest at **1.40%**; double-noisy and triple-play each reached 0.10%, while the others reached 0%. The training prediction loss decreased, but task success remained low with this fine-tuning and CEM configuration.

### Comparison with OGBench Table 5

Values are success rates in percent. The six baseline columns are transcribed from [page 21, Table 5 of ogbench.pdf](../ogbench/ogbench.pdf#page=21). The paper uses native-64 observations, whereas this experiment uses native-224 observations. The paper's standard deviations come from 4 independent training seeds; here, 1 model was fine-tuned per dataset and the standard deviations come from 4 evaluation seeds. This table is therefore a reference comparison across different image resolutions and seed settings.

| Dataset | GCBC | GCIVL | GCIQL | QRL | CRL | HIQL | Original LeWM checkpoint | Fine-tuned LeWM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| visual-cube-single-play-v0 | 5 ± 1 | 60 ± 5 | 30 ± 5 | 41 ± 15 | 31 ± 15 | 89 ± 0 | 0 ± 0 | 0.00 ± 0.00 |
| visual-cube-double-play-v0 | 1 ± 1 | 10 ± 2 | 1 ± 1 | 5 ± 0 | 2 ± 1 | 39 ± 2 | 0 ± 0 | 0.00 ± 0.00 |
| visual-cube-triple-play-v0 | 15 ± 2 | 14 ± 2 | 15 ± 1 | 16 ± 1 | 17 ± 2 | 21 ± 0 | 0 ± 0 | 0.10 ± 0.17 |
| visual-cube-quadruple-play-v0 | 8 ± 1 | 0 ± 0 | 7 ± 1 | 5 ± 1 | 4 ± 1 | 14 ± 1 | 0 ± 0 | 0.00 ± 0.00 |
| visual-cube-single-noisy-v0 | 14 ± 3 | 75 ± 3 | 48 ± 3 | 10 ± 5 | 39 ± 30 | 99 ± 0 | 0 ± 0 | 0.00 ± 0.00 |
| visual-cube-double-noisy-v0 | 5 ± 1 | 17 ± 4 | 22 ± 2 | 6 ± 2 | 6 ± 3 | 59 ± 3 | 0 ± 0 | 0.10 ± 0.17 |
| visual-cube-triple-noisy-v0 | 16 ± 1 | 18 ± 1 | 12 ± 1 | 9 ± 4 | 16 ± 1 | 23 ± 2 | 0 ± 0 | 1.40 ± 0.66 |
| visual-cube-quadruple-noisy-v0 | 9 ± 0 | 0 ± 0 | 2 ± 2 | 0 ± 0 | 8 ± 2 | 12 ± 8 | 0 ± 0 | 0.00 ± 0.00 |

The original-checkpoint column comes from an earlier evaluation of fixed single-cube weights for 1,000 episodes on each of the four Cube variants; play and noisy share those results. The multi-cube original-checkpoint entries are cross-environment transfer results, while each fine-tuned entry uses an independent model trained on its corresponding dataset.

### Training and evaluation conditions

- All eight models were initialized from the local original `cube-lewm` checkpoint, and all parameters were updated. Training used batch=128, AdamW with lr=1e-5, and cosine decay after a 5% warmup. The objective was prediction MSE + 0.09×SIGReg.
- The official train NPZ was used for optimization and action normalization; the independent `-val.npz` was used only for validation. Images were restored from the qpos/qvel of the same frame and rendered directly at 224×224. Training images, validation images, online observations, goals, and raw videos were all native-224. Model inputs were RGB and actions; qpos/qvel were used only for re-rendering.
- Contiguous trajectories in the original sequential validation batches inflated the SIGReg statistic. Evaluation weights were ultimately selected from all saved epochs by the **lowest prediction MSE on the complete independent validation set**. The original combined metrics and training weights were retained; see each model's `prediction-checkpoint-selection.json`. W&B `val/pred_loss` is the prediction error, while the original `val/loss` includes SIGReg and cannot be compared directly with the total loss from randomly sampled training batches.
- Each model used 5 official goals × 4 evaluation seeds (0,1,2,3) × 50 episodes. CEM used 300 samples, 30 iterations, and 30 elites, with horizon=5, receding=5, action block=5, and observed history=1. 5 raw native-224 videos were saved per model, for 40 videos total, with no white borders.

### Success counts, selected weights, and training records

| Dataset | Successes / episodes | Validation prediction MSE | Selected checkpoint | W&B |
|---|---:|---:|---|---|
| visual-cube-single-play-v0 | 0 / 1000 | 0.003454 | [epoch 11](outputs/lewm_visual_cube_native224_ft8_20261005_211637/training/visual-cube-single-play-v0/checkpoints/epoch_11) | [loss and evaluation](https://wandb.ai/visual-intelligence-laboratory/P-Rep/runs/02d2eeeb956d) |
| visual-cube-single-noisy-v0 | 0 / 1000 | 0.012328 | [epoch 9](outputs/lewm_visual_cube_native224_ft8_20261005_211637/training/visual-cube-single-noisy-v0/checkpoints/epoch_09) | [loss and evaluation](https://wandb.ai/visual-intelligence-laboratory/P-Rep/runs/49644b5ddd5f) |
| visual-cube-double-play-v0 | 0 / 1000 | 0.005578 | [epoch 9](outputs/lewm_visual_cube_native224_ft8_20261005_211637/training/visual-cube-double-play-v0/checkpoints/epoch_09) | [loss and evaluation](https://wandb.ai/visual-intelligence-laboratory/P-Rep/runs/27125ca414fb) |
| visual-cube-double-noisy-v0 | 1 / 1000 | 0.013212 | [epoch 10](outputs/lewm_visual_cube_native224_ft8_20261005_211637/training/visual-cube-double-noisy-v0/checkpoints/epoch_10) | [loss and evaluation](https://wandb.ai/visual-intelligence-laboratory/P-Rep/runs/9e583c5d81b0) |
| visual-cube-triple-play-v0 | 1 / 1000 | 0.005754 | [epoch 3](outputs/lewm_visual_cube_native224_ft8_20261005_211637/training/visual-cube-triple-play-v0/checkpoints/epoch_03) | [loss and evaluation](https://wandb.ai/visual-intelligence-laboratory/P-Rep/runs/2f90dc4920b7) |
| visual-cube-triple-noisy-v0 | 14 / 1000 | 0.011127 | [epoch 3](outputs/lewm_visual_cube_native224_ft8_20261005_211637/training/visual-cube-triple-noisy-v0/checkpoints/epoch_03) | [loss and evaluation](https://wandb.ai/visual-intelligence-laboratory/P-Rep/runs/6c4bff782f40) |
| visual-cube-quadruple-play-v0 | 0 / 1000 | 0.006280 | [epoch 2](outputs/lewm_visual_cube_native224_ft8_20261005_211637/training/visual-cube-quadruple-play-v0/checkpoints/epoch_02) | [loss and evaluation](https://wandb.ai/visual-intelligence-laboratory/P-Rep/runs/05a5b67ba674) |
| visual-cube-quadruple-noisy-v0 | 0 / 1000 | 0.010974 | [epoch 3](outputs/lewm_visual_cube_native224_ft8_20261005_211637/training/visual-cube-quadruple-noisy-v0/checkpoints/epoch_03) | [loss and evaluation](https://wandb.ai/visual-intelligence-laboratory/P-Rep/runs/2d8210bffdf2) |

- [Complete metrics](outputs/lewm_visual_cube_native224_ft8_20261005_211637/metrics.json), [full result verification](outputs/lewm_visual_cube_native224_ft8_20261005_211637/completion-verification.json), and [run configuration](outputs/lewm_visual_cube_native224_ft8_20261005_211637/manifest.json).
- [Per-task success rates and experiment notes](zeyu-understand.md#ogbench), [evaluation records and 40 videos](outputs/lewm_visual_cube_native224_ft8_20261005_211637/evaluation), and [training logs and weights for all eight models](outputs/lewm_visual_cube_native224_ft8_20261005_211637/training).
- [W&B project](https://forge.coreweave.com/wandb/visual-intelligence-laboratory/P-Rep).
- Entry point: `launch_ogbench_finetune_8gpu.sh` → `run_ogbench_finetune_pipeline.py`; the output directory for this run is `outputs/lewm_visual_cube_native224_ft8_20261005_211637`.
<!-- ogbench-finetune-results:lewm_visual_cube_native224_ft8_20261005_211637:end -->

## Pretrained Checkpoints

Pretrained LeWM checkpoints for each environment are mirrored on the Hugging Face
Hub (model repos), alongside the datasets (dataset repos) in the same collection:

- [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht)
- [`quentinll/lewm-cube`](https://huggingface.co/quentinll/lewm-cube)
- [`quentinll/lewm-tworooms`](https://huggingface.co/quentinll/lewm-tworooms)
- [`quentinll/lewm-reacher`](https://huggingface.co/quentinll/lewm-reacher)

The full baseline checkpoint suite (PLDM, LeJEPA, IVL, IQL, GCBC, DINO-WM, DINO-WM-noprop)
is available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e):

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

### From the Drive archive

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

Place the extracted files under `$STABLEWM_HOME/` and load via:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

`AutoCostModel` accepts:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

### From the Hugging Face mirror

The HF model repos ship the LeWM checkpoint as a `weights.pt` (state dict) plus a
`config.json` describing the model. Convert once to produce the `_object.ckpt`
that `eval.py` expects:

```bash
# download weights.pt + config.json
hf download quentinll/lewm-pusht --local-dir $STABLEWM_HOME/hf_pusht

# convert to object checkpoint under $STABLEWM_HOME/pusht/lewm_object.ckpt
python - <<'PY'
import json, torch, stable_pretraining as spt
from pathlib import Path
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
import stable_worldmodel as swm

src = Path(swm.data.utils.get_cache_dir(), "hf_pusht")
out = Path(swm.data.utils.get_cache_dir(), "pusht", "lewm_object.ckpt")

cfg = json.loads((src / "config.json").read_text())
encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"],
    patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"],
    pretrained=False, use_mask_token=False,
)
mlp = lambda k: MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
                    hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)
model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**cfg["predictor"]),
    action_encoder=Embedder(**cfg["action_encoder"]),
    projector=mlp("projector"),
    pred_proj=mlp("pred_proj"),
)
sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
PY
```

After conversion, load via `swm.policy.AutoCostModel('pusht/lewm')` as usual.

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
