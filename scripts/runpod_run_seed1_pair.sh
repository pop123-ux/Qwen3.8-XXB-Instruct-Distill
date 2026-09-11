#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/Qwen3.8-XXB-Instruct-Distill}"
VENV="${VENV:-/workspace/.venvs/qwen-rq1-3h}"
SESSION_DIR="${SESSION_DIR:-/workspace/runpod_sessions/2026-09-11-rq1-3h}"
mkdir -p "$SESSION_DIR"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

run_guarded() {
  local name="$1"
  shift
  echo
  echo "======================================================================"
  echo "START $name  $(date -Is)"
  echo "======================================================================"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader
  python scripts/guard_vram.py \
    --max-vram-gib 45.0 \
    --interval 5 \
    --log "$SESSION_DIR/${name}_vram.log" \
    -- "$@"
  local rc=$?
  echo "END $name rc=$rc $(date -Is)"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader
  return "$rc"
}

# Order is deliberate: reproduce the winning treatment first, then immediately obtain its
# same-seed pointwise control in the identical software environment.
run_guarded run008_raw_delta_seed1 python scripts/run008_raw_delta_seed1.py
run_guarded run009_pointwise_seed1 python scripts/run009_pointwise_seed1.py

echo
printf 'Seed-1 matched pair complete at %s\n' "$(date -Is)"
echo 'Do not spend the remaining paid window on lengthy post-hoc analysis. Archive the pair,'
echo 'update research/STATE.yaml, classify the result, and launch the next admissible complete'
echo 'high-information experiment that fits the remaining time.'
