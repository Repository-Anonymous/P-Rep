#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/coder/P-Rep/le-wm"
cd "$REPO_DIR"

source .venv/bin/activate
export STABLEWM_HOME="$PWD/.stable-wm"
export LOCAL_DATASET_DIR="$STABLEWM_HOME"
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export WANDB_ENTITY="visual-intelligence-laboratory"
export WANDB_PROJECT="P-Rep"

RUN_ID="${RUN_ID:-cube_lewm_4gpu_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$PWD/outputs/$RUN_ID"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
mkdir -p "$RUN_DIR"
cp "$REPO_DIR/run_train_cube_wandb.sh" "$RUN_DIR/run.sh"

python train.py --config-name=lewm \
  data=ogb \
  output_model_name="$RUN_ID" \
  subdir="$RUN_ID" \
  trainer.max_epochs=10 \
  trainer.devices=4 \
  +trainer.strategy=ddp \
  loader.batch_size="$PER_DEVICE_BATCH_SIZE" \
  optimizer.lr="$LEARNING_RATE" \
  wandb.enabled=true \
  wandb.config.entity="$WANDB_ENTITY" \
  wandb.config.project="$WANDB_PROJECT" \
  wandb.config.name="$RUN_ID" \
  wandb.config.id="$RUN_ID" \
  wandb.config.resume=allow \
  periodic_eval.enabled=true \
  periodic_eval.every_n_epochs=2 \
  periodic_eval.output_dir="$RUN_DIR" \
  hydra.run.dir="$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/train.log"
