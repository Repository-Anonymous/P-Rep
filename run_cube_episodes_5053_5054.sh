#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/coder/P-Rep/le-wm"
cd "$REPO_DIR"

source .venv/bin/activate
export STABLEWM_HOME="$PWD/.stable-wm"
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_ID="${RUN_ID:-cube_lewm_episodes5053_5054_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$PWD/outputs/$RUN_ID"
mkdir -p "$RUN_DIR"
cp "$REPO_DIR/run_cube_episodes_5053_5054.sh" "$RUN_DIR/run.sh"

python eval.py --config-name=cube.yaml \
  policy=cube-lewm \
  eval.num_eval=2 \
  'eval.episodes_idx=[5053,5054]' \
  'eval.start_steps=[0,0]' \
  eval.goal_offset_steps=200 \
  eval.eval_budget=200 \
  world.terminate_at_goal=false \
  output_dir="$RUN_DIR" \
  hydra.run.dir="$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/eval.log"

mv "$RUN_DIR/ogb_cube_results.txt" "$RUN_DIR/cube_lewm_results.txt"
mv "$RUN_DIR/env_0.mp4" "$RUN_DIR/episode_5053.mp4"
mv "$RUN_DIR/env_1.mp4" "$RUN_DIR/episode_5054.mp4"

FFMPEG_BIN="$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
"$FFMPEG_BIN" -loglevel error -y \
  -i "$RUN_DIR/episode_5053.mp4" \
  -i "$RUN_DIR/episode_5054.mp4" \
  -filter_complex \
  '[0:v]setpts=PTS-STARTPTS[v0];[1:v]setpts=PTS-STARTPTS[v1];[v0][v1]concat=n=2:v=1:a=0[v]' \
  -map '[v]' -r 15 -c:v libx264 -pix_fmt yuv420p \
  "$RUN_DIR/cube_lewm_episodes5053_5054_continuous.mp4"

"$FFMPEG_BIN" -loglevel error -y \
  -ss 6.7 -i "$RUN_DIR/cube_lewm_episodes5053_5054_continuous.mp4" \
  -frames:v 1 -update 1 "$RUN_DIR/preview_episode5053.png"
"$FFMPEG_BIN" -loglevel error -y \
  -ss 20.1 -i "$RUN_DIR/cube_lewm_episodes5053_5054_continuous.mp4" \
  -frames:v 1 -update 1 "$RUN_DIR/preview_episode5054.png"

printf 'Output: %s\n' "$RUN_DIR"
