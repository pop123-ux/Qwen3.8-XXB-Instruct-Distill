#!/usr/bin/env bash
set -euo pipefail

# Paid-GPU launcher for RQ1_V2_SCALE1024.
# Intentionally contains no environment installation, documentation generation,
# plotting, or exploratory work. Prepare those before starting the pod.

ROOT="${ROOT:-/workspace/Qwen3.8-XXB-Instruct-Distill}"
TEACHER="${TEACHER:-/workspace/models/qwen3.8-27b-dbdc473}"
STUDENT="${STUDENT:-/workspace/runs/pilot001/transferred}"
CORPUS="${CORPUS:-/workspace/corpora/gutenberg/train.txt}"
RUN_DIR="${RUN_DIR:-/workspace/runs/run004_behavioral_scale_1024}"
REVISION="dbdc473dea0d6a9763042881cc33d6058d1742d2"
EXPECTED_CORPUS_SHA="e11ca38bb099fc89c2f74e96f5d2f1209def6a16f6a8432d4e9972acd50c100d"
EXPECTED_GPU="NVIDIA L40S"

cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "$RUN_DIR"

fail() { echo "FATAL: $*" >&2; exit 2; }

[[ -d "$TEACHER" ]] || fail "teacher missing: $TEACHER"
[[ -d "$STUDENT" ]] || fail "materialised student missing: $STUDENT"
[[ -f "$CORPUS" ]] || fail "corpus missing: $CORPUS"
[[ -f scripts/run004_behavioral_kd.py ]] || fail "Run 004 behavioral launcher missing"
[[ -f scripts/guard_vram.py ]] || fail "VRAM guard missing"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)"
[[ "$GPU_NAME" == "$EXPECTED_GPU" ]] || fail "expected $EXPECTED_GPU, got $GPU_NAME"

CORPUS_SHA="$(sha256sum "$CORPUS" | awk '{print $1}')"
[[ "$CORPUS_SHA" == "$EXPECTED_CORPUS_SHA" ]] || fail "corpus SHA mismatch: $CORPUS_SHA"

python - <<'PY'
import sys, torch, transformers
expected = {"python": "3.12.3", "torch": "2.8.0+cu128", "transformers": "5.15.1"}
actual = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "transformers": transformers.__version__,
}
for key, value in expected.items():
    if actual[key] != value:
        raise SystemExit(f"FATAL: {key} expected {value}, got {actual[key]}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("FATAL: exactly one CUDA GPU is required")
print("environment gate: PASS", actual)
PY

# Capture immutable launch provenance before consuming paid GPU time.
{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_status=$(git status --porcelain | wc -l | xargs) dirty_paths"
  echo "gpu=$GPU_NAME"
  echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | xargs)"
  echo "corpus_sha256=$CORPUS_SHA"
  echo "teacher_revision=$REVISION"
  echo "protocol=RQ1_V2_SCALE1024"
} | tee "$RUN_DIR/preflight.txt"

# The only expensive action: guarded 1024-step behavioral KD.
python scripts/guard_vram.py \
  --max-vram-gib 45 \
  --interval 1.0 \
  --log "$RUN_DIR/vram_guard.log" \
  -- python -u scripts/run004_behavioral_kd.py \
    --teacher "$TEACHER" \
    --revision "$REVISION" \
    --quantization 4bit \
    --pretrained "$STUDENT" \
    --text-path "$CORPUS" \
    --max-tokens 700000 \
    --sequence-length 1536 \
    --steps 1024 \
    --batch-size 1 \
    --gradient-accumulation-steps 1 \
    --learning-rate 2e-4 \
    --kd-temperature 2.0 \
    --kd-top-k 64 \
    --chunk-pairs 4 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --optimizer adamw \
    --precision bf16 \
    --seed 0 \
    --log-every 1 \
    --eval-every 64 \
    --save-every 256 \
    --experiment-id run004_behavioral_scale_1024 \
    --output "$RUN_DIR"

# CPU/lightweight integrity work only after training exits successfully.
python scripts/run_record.py verify "$RUN_DIR" || true

# Small evidence bundle for immediate exfiltration before pod termination.
# Checkpoints/weights/optimizer state are deliberately excluded.
tar -czf "$RUN_DIR/evidence_bundle.tgz" \
  --exclude='checkpoints' \
  --exclude='*.safetensors' \
  --exclude='*.pt' \
  -C "$RUN_DIR" .

printf '\nRUN COMPLETE\nEvidence: %s\n' "$RUN_DIR/evidence_bundle.tgz"
