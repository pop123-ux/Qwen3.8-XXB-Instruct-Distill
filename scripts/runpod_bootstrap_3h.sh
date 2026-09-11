#!/usr/bin/env bash
set -Eeuo pipefail

# Bootstrap a fresh RunPod L40S shell for the 2026-09-11 three-hour RQ1 session.
# Python packages are intentionally UNPINNED at the user's request. Exact installed
# versions are captured so every run produced in this session can be attributed to the
# environment actually used.

REPO_ROOT="${REPO_ROOT:-/workspace/Qwen3.8-XXB-Instruct-Distill}"
VENV="${VENV:-/workspace/.venvs/qwen-rq1-3h}"
SESSION_DIR="${SESSION_DIR:-/workspace/runpod_sessions/2026-09-11-rq1-3h}"
TEACHER="/workspace/models/qwen3.8-27b-dbdc473"
TEACHER_REV="dbdc473dea0d6a9763042881cc33d6058d1742d2"
STUDENT="/workspace/runs/pilot001/transferred"
CORPUS_DIR="/workspace/corpora/gutenberg"
CORPUS="$CORPUS_DIR/train.txt"
EXPECTED_CORPUS_SHA="bc5972d9a52580ff14ab1b3b1753f9cd68c726c63cc625a7ed3913ec3c5dc5c5"
EXPECTED_CORPUS_BYTES="5554406"

mkdir -p "$SESSION_DIR" "$(dirname "$VENV")" /workspace/models /workspace/runs /workspace/corpora
cd "$REPO_ROOT"

printf '\n=== Git / GPU baseline ===\n'
git status --short --branch
git rev-parse HEAD
nvidia-smi

printf '\n=== Python environment (unpinned current releases) ===\n'
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade \
  torch \
  transformers \
  accelerate \
  peft \
  bitsandbytes \
  datasets \
  huggingface_hub \
  safetensors \
  sentencepiece \
  protobuf \
  pyyaml \
  numpy \
  scipy \
  psutil \
  tqdm \
  matplotlib \
  pytest \
  einops \
  packaging
python -m pip install -e . --no-deps

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

printf '\n=== Import smoke ===\n'
python - <<'PY'
import torch
import transformers
import accelerate
import peft
import bitsandbytes
import datasets
import safetensors
import yaml
import qwen_distill

print('python imports: PASS')
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('accelerate:', accelerate.__version__)
print('peft:', peft.__version__)
print('bitsandbytes:', bitsandbytes.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda runtime:', torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit('CUDA is not available')
print('gpu:', torch.cuda.get_device_name(0))
PY

printf '\n=== Teacher asset ===\n'
if [[ ! -f "$TEACHER/config.json" ]]; then
  echo "Teacher not found; downloading pinned Qwen3.8-27B revision to $TEACHER"
  hf download Qwen/Qwen3.8-27B --revision "$TEACHER_REV" --local-dir "$TEACHER"
else
  echo "Reusing existing teacher: $TEACHER"
fi

printf '\n=== Exact Gutenberg / Level-2R corpus ===\n'
if [[ ! -f "$CORPUS" ]]; then
  echo "Corpus not found; reconstructing with the repository's deterministic preparation script."
  python scripts/prepare_level2r_dataset.py --output "$CORPUS_DIR"
fi

ACTUAL_SHA="$(sha256sum "$CORPUS" | awk '{print $1}')"
ACTUAL_BYTES="$(wc -c < "$CORPUS" | tr -d ' ')"
echo "train.txt sha256: $ACTUAL_SHA"
echo "train.txt bytes : $ACTUAL_BYTES"
if [[ "$ACTUAL_SHA" != "$EXPECTED_CORPUS_SHA" || "$ACTUAL_BYTES" != "$EXPECTED_CORPUS_BYTES" ]]; then
  echo "FATAL: corpus identity mismatch. Do not train on substituted data." >&2
  echo "Expected sha=$EXPECTED_CORPUS_SHA bytes=$EXPECTED_CORPUS_BYTES" >&2
  exit 20
fi
python scripts/verify_corpus.py "$CORPUS_DIR" --level2r --json "$SESSION_DIR/corpus_verification.json"

printf '\n=== Canonical transferred student ===\n'
if [[ ! -f "$STUDENT/config.json" ]]; then
  echo "Canonical transferred student not found; materialising deterministically."
  python scripts/distill_pilot.py \
    --teacher-path "$TEACHER" \
    --output "$STUDENT" \
    --layer-strategy group \
    --kv-merge mean \
    --ffn-method contiguous_partition \
    --seed 0
else
  echo "Reusing existing canonical student: $STUDENT"
fi

printf '\n=== Capture exact session provenance ===\n'
{
  echo "captured_at=$(date -Is)"
  echo "repo=$REPO_ROOT"
  echo "git_sha=$(git rev-parse HEAD)"
  echo "git_branch=$(git branch --show-current)"
  echo "teacher_revision=$TEACHER_REV"
  echo "corpus_sha256=$ACTUAL_SHA"
  echo "corpus_bytes=$ACTUAL_BYTES"
  echo
  python --version
  echo
  nvidia-smi
  echo
  python - <<'PY'
import platform, torch, transformers, accelerate, peft, bitsandbytes
print('platform:', platform.platform())
print('torch:', torch.__version__)
print('torch_cuda:', torch.version.cuda)
print('transformers:', transformers.__version__)
print('accelerate:', accelerate.__version__)
print('peft:', peft.__version__)
print('bitsandbytes:', bitsandbytes.__version__)
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
PY
} | tee "$SESSION_DIR/environment.txt"
python -m pip freeze > "$SESSION_DIR/pip-freeze.txt"
git diff --quiet || {
  echo "FATAL: tracked repository files changed during bootstrap." >&2
  git status --short
  exit 21
}

printf '\nBootstrap complete.\n'
printf 'Activate later with: source %s/bin/activate\n' "$VENV"
printf 'Session provenance: %s\n' "$SESSION_DIR"
printf 'Next: start Claude Code from %s and resume the latest project session.\n' "$REPO_ROOT"
