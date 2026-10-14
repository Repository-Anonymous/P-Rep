#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/coder/P-Rep/le-wm"
cd "$REPO_DIR"

source .venv/bin/activate
export STABLEWM_HOME="$PWD/.stable-wm"
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CHECKPOINT="$PWD/.stable-wm/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
RUN_ID="${RUN_ID:-cube_vjepa2_episode5053_64f_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$PWD/outputs/$RUN_ID"
mkdir -p "$RUN_DIR"
cp "$REPO_DIR/run_vjepa2_episode5053_64f.sh" "$RUN_DIR/run.sh"
cp "$REPO_DIR/vjepa2_encoder.py" "$RUN_DIR/vjepa2_encoder_snapshot.py"
cp "$REPO_DIR/video_context_policy.py" "$RUN_DIR/video_context_policy_snapshot.py"
cp "$REPO_DIR/export_cube_observation_clip.py" "$RUN_DIR/export_cube_observation_clip_snapshot.py"

python export_cube_observation_clip.py \
  --dataset "$PWD/.stable-wm/datasets/ogbench/cube_single_expert.h5" \
  --episode 5053 --start 0 --frames 64 \
  --output "$RUN_DIR/episode5053_observation_step000_063.mp4"

python eval.py --config-name=cube.yaml \
  policy=cube-lewm \
  vjepa2_checkpoint="$CHECKPOINT" \
  eval.img_size=384 \
  eval.num_eval=1 \
  'eval.episodes_idx=[5053]' \
  'eval.start_steps=[63]' \
  eval.goal_offset_steps=137 \
  eval.eval_budget=137 \
  world.terminate_at_goal=false \
  'vjepa2_video_context={episodes_idx:[5053],start_steps:[0],num_frames:64}' \
  output_dir="$RUN_DIR" \
  hydra.run.dir="$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/eval.log"

mv "$RUN_DIR/ogb_cube_results.txt" "$RUN_DIR/cube_vjepa2_results.txt"
mv "$RUN_DIR/env_0.mp4" "$RUN_DIR/cube_vjepa2_episode5053_64f.mp4"

FFMPEG_BIN="$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
"$FFMPEG_BIN" -loglevel error -y \
  -ss 4.0 -i "$RUN_DIR/cube_vjepa2_episode5053_64f.mp4" \
  -frames:v 1 -update 1 "$RUN_DIR/cube_vjepa2_episode5053_64f_preview.png"

printf 'Output: %s\n' "$RUN_DIR"
