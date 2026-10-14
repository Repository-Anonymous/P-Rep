#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/coder/P-Rep/le-wm"
cd "$REPO_DIR"
TASK_GROUP="${TASK_GROUP:-lewm_visual_cube_native224_ft8_$(date +%Y%m%d_%H%M%S)}"
TASK_OUTPUT="${TASK_OUTPUT:-$REPO_DIR/outputs/$TASK_GROUP}"
TASK_SESSION="${TASK_SESSION:-lewm-ogbench-ft8}"
TASK_SECRETS="${TASK_SECRETS:-/home/coder/.config/lewm/wandb-finetune.env}"
TASK_STEPS="${TASK_STEPS:-80000}"
TASK_BATCH_SIZE="${TASK_BATCH_SIZE:-128}"
TASK_LR="${TASK_LR:-1e-5}"

if [[ ! -r "$TASK_SECRETS" ]]; then
    printf 'Missing private W&B environment file: %s\n' "$TASK_SECRETS" >&2
    exit 1
fi
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then
    printf 'tmux session already exists: %s\n' "$TASK_SESSION" >&2
    exit 1
fi
mkdir -p "$TASK_OUTPUT"
printf -v TASK_COMMAND 'cd %q && set -a && source %q && set +a && exec %q -u %q --output %q --group %q --tmux-session %q --steps %q --batch-size %q --lr %q > %q 2>&1' \
    "$REPO_DIR" "$TASK_SECRETS" "$REPO_DIR/.venv/bin/python" \
    "$REPO_DIR/run_ogbench_finetune_pipeline.py" "$TASK_OUTPUT" "$TASK_GROUP" "$TASK_SESSION" \
    "$TASK_STEPS" "$TASK_BATCH_SIZE" "$TASK_LR" "$TASK_OUTPUT/pipeline.log"
if [[ "${TASK_SMOKE:-0}" == 1 ]]; then
    TASK_COMMAND="${TASK_COMMAND% >*} --smoke > $(printf '%q' "$TASK_OUTPUT/pipeline.log") 2>&1"
fi
tmux new-session -d -s "$TASK_SESSION" "bash -c $(printf '%q' "$TASK_COMMAND")"
printf 'Started tmux: %s\nOutput: %s\nAttach: tmux attach -t %s\n' "$TASK_SESSION" "$TASK_OUTPUT" "$TASK_SESSION"
