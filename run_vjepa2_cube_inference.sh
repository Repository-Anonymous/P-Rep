#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/coder/P-Rep/le-wm"
cd "$REPO_DIR"

source .venv/bin/activate
export STABLEWM_HOME="$PWD/.stable-wm"
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

VJEPA2_CHECKPOINT="$STABLEWM_HOME/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
RUN_ID="${RUN_ID:-cube_vjepa2_1_vitb_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$PWD/outputs/$RUN_ID"
mkdir -p "$RUN_DIR"
cp "$REPO_DIR/run_vjepa2_cube_inference.sh" "$RUN_DIR/run.sh"

python eval.py --config-name=cube.yaml \
  policy=cube-lewm \
  seed="${SEED:-42}" \
  eval.num_eval="${NUM_EVAL:-1}" \
  eval.img_size=384 \
  vjepa2_checkpoint="$VJEPA2_CHECKPOINT" \
  output_dir="$RUN_DIR" \
  hydra.run.dir="$RUN_DIR"

mv "$RUN_DIR/env_0.mp4" "$RUN_DIR/cube_vjepa2_inference.mp4"
mv "$RUN_DIR/ogb_cube_results.txt" "$RUN_DIR/cube_vjepa2_results.txt"

FFMPEG_BIN="$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
"$FFMPEG_BIN" -loglevel error -y \
  -ss 0.8 -i "$RUN_DIR/cube_vjepa2_inference.mp4" \
  -frames:v 1 -update 1 "$RUN_DIR/cube_vjepa2_preview.png"
