#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/coder/P-Rep/le-wm"
cd "$REPO_DIR"
source .venv/bin/activate
export STABLEWM_HOME="$PWD/.stable-wm"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_ID="${RUN_ID:-cube_vjepa2_lewm_train_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$PWD/outputs/$RUN_ID"
mkdir -p "$RUN_DIR"
cp "$REPO_DIR/run_train_vjepa2_cube.sh" "$RUN_DIR/run.sh"

python train_vjepa2_cube.py \
  trainer.max_epochs="${MAX_EPOCHS:-100}" \
  data.sample_stride="${SAMPLE_STRIDE:-1}" \
  2>&1 | tee "$RUN_DIR/train.log"
