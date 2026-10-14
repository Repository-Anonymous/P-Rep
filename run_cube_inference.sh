#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/coder/P-Rep/le-wm"
cd "$REPO_DIR"

source .venv/bin/activate
export STABLEWM_HOME="$PWD/.stable-wm"
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_ID="${RUN_ID:-cube_lewm_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$PWD/outputs/$RUN_ID"
mkdir -p "$RUN_DIR"

# Preserve the exact launcher alongside this run's results.
cp "$REPO_DIR/run_cube_inference.sh" "$RUN_DIR/run.sh"

python eval.py --config-name=cube.yaml \
  policy=cube-lewm \
  seed="${SEED:-42}" \
  eval.num_eval="${NUM_EVAL:-1}" \
  output_dir="$RUN_DIR" \
  hydra.run.dir="$RUN_DIR"

mv "$RUN_DIR/ogb_cube_results.txt" "$RUN_DIR/cube_lewm_results.txt"

FFMPEG_BIN="$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
shopt -s nullglob
VIDEOS=("$RUN_DIR"/env_*.mp4)

if (( ${#VIDEOS[@]} == 1 )); then
  mv "${VIDEOS[0]}" "$RUN_DIR/cube_lewm_inference.mp4"
  "$FFMPEG_BIN" -loglevel error -y \
    -ss 0.8 -i "$RUN_DIR/cube_lewm_inference.mp4" \
    -frames:v 1 -update 1 "$RUN_DIR/cube_lewm_preview.png"
else
  for VIDEO in "${VIDEOS[@]}"; do
    INDEX="${VIDEO##*/env_}"
    INDEX="${INDEX%.mp4}"
    TARGET="$RUN_DIR/cube_lewm_inference_${INDEX}.mp4"
    mv "$VIDEO" "$TARGET"
    "$FFMPEG_BIN" -loglevel error -y \
      -ss 0.8 -i "$TARGET" \
      -frames:v 1 -update 1 "$RUN_DIR/cube_lewm_preview_${INDEX}.png"
  done
fi
