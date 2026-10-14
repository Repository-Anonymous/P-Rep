#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/coder/P-Rep/le-wm"
cd "$REPO_DIR"

PARENT_ID="${RUN_ID:-cube_lewm_3examples_$(date +%Y%m%d_%H%M%S)}"
PARENT_DIR="$PWD/outputs/$PARENT_ID"
mkdir -p "$PARENT_DIR"
cp "$REPO_DIR/run_three_cube_examples.sh" "$PARENT_DIR/run_all.sh"

CUDA_VISIBLE_DEVICES=0 SEED=43 \
  RUN_ID="$PARENT_ID/example_1_gpu0_seed43" \
  ./run_cube_inference.sh &
PID_1=$!

CUDA_VISIBLE_DEVICES=1 SEED=44 \
  RUN_ID="$PARENT_ID/example_2_gpu1_seed44" \
  ./run_cube_inference.sh &
PID_2=$!

CUDA_VISIBLE_DEVICES=2 SEED=45 \
  RUN_ID="$PARENT_ID/example_3_gpu2_seed45" \
  ./run_cube_inference.sh &
PID_3=$!

STATUS=0
wait "$PID_1" || STATUS=1
wait "$PID_2" || STATUS=1
wait "$PID_3" || STATUS=1

printf 'Outputs: %s\n' "$PARENT_DIR"
exit "$STATUS"
