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
EXPECTED_BRANCH="prep/l40s-rq1-scale1024"
# Raw source-file identity from experiments/run002_logit_kd/dataset_provenance.json.
EXPECTED_SOURCE_CORPUS_SHA="bc5972d9a52580ff14ab1b3b1753f9cd68c726c63cc625a7ed3913ec3c5dc5c5"
# Deterministic 700k-token packed stream identity recorded by matched Run 003.
EXPECTED_PACKED_CORPUS_SHA="e11ca38bb099fc89c2f74e96f5d2f1209def6a16f6a8432d4e9972acd50c100d"
EXPECTED_GPU="NVIDIA L40S"
# Run 003 on the same GPU peaked at ~38.95 GiB allocated / 40.77 GiB reserved.
# The archived L40S exposed ~44.39 GiB total, so the historical 45-GiB external
# nvidia-smi threshold was above physical GiB capacity and could not act as a guard.
OPERATIONAL_VRAM_GUARD_GIB="44.0"

cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

fail() { echo "FATAL: $*" >&2; exit 2; }

CURRENT_BRANCH="$(git branch --show-current)"
[[ "$CURRENT_BRANCH" == "$EXPECTED_BRANCH" ]] || \
  fail "expected branch $EXPECTED_BRANCH, got ${CURRENT_BRANCH:-DETACHED}"
[[ -z "$(git status --porcelain)" ]] || fail "repository is dirty; refuse paid run"

[[ -d "$TEACHER" ]] || fail "teacher missing: $TEACHER"
[[ -d "$STUDENT" ]] || fail "materialised student missing: $STUDENT"
[[ -f "$CORPUS" ]] || fail "corpus missing: $CORPUS"
[[ -f scripts/run004_behavioral_kd.py ]] || fail "Run 004 behavioral launcher missing"
[[ -f scripts/guard_vram.py ]] || fail "VRAM guard missing"
[[ -f "$TEACHER/config.json" ]] || fail "teacher config missing: $TEACHER/config.json"
[[ -f "$STUDENT/config.json" ]] || fail "student config missing: $STUDENT/config.json"

# Never mix a paid run with stale or partial artifacts. To resume/retry deliberately,
# choose a new RUN_DIR after inspecting the failed record.
if [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -mindepth 1 -print -quit | grep -q .; then
  fail "run directory is not empty: $RUN_DIR"
fi
mkdir -p "$RUN_DIR"

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | xargs)"
[[ "$GPU_COUNT" == "1" ]] || fail "expected exactly one NVIDIA GPU, got $GPU_COUNT"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)"
[[ "$GPU_NAME" == "$EXPECTED_GPU" ]] || fail "expected $EXPECTED_GPU, got $GPU_NAME"

SOURCE_CORPUS_SHA="$(sha256sum "$CORPUS" | awk '{print $1}')"
[[ "$SOURCE_CORPUS_SHA" == "$EXPECTED_SOURCE_CORPUS_SHA" ]] || \
  fail "raw train.txt SHA mismatch: $SOURCE_CORPUS_SHA"

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
  echo "git_branch=$CURRENT_BRANCH"
  echo "git_status=clean"
  echo "gpu=$GPU_NAME"
  echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | xargs)"
  echo "source_corpus_sha256=$SOURCE_CORPUS_SHA"
  echo "expected_packed_corpus_sha256=$EXPECTED_PACKED_CORPUS_SHA"
  echo "teacher_revision=$REVISION"
  echo "operational_vram_guard_gib=$OPERATIONAL_VRAM_GUARD_GIB"
  echo "protocol=RQ1_V2_SCALE1024"
} | tee "$RUN_DIR/preflight.txt"

# The only expensive action: guarded 1024-step behavioral KD.
# Instrumentation cadence matches the historically preregistered 1024-step command.
python scripts/guard_vram.py \
  --max-vram-gib "$OPERATIONAL_VRAM_GUARD_GIB" \
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
    --log-every 16 \
    --eval-every 128 \
    --save-every 512 \
    --experiment-id run004_behavioral_scale_1024 \
    --output "$RUN_DIR"

# Cheap post-run integrity gate. The raw file hash above proves source identity; this
# proves that tokenizer + EOS + 700k cap reproduced the exact matched packed stream.
python - "$RUN_DIR" "$EXPECTED_PACKED_CORPUS_SHA" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_packed_sha = sys.argv[2]
summary_path = root / "summary.json"
manifest_path = root / "run004_behavioral_manifest.json"
if not summary_path.is_file():
    raise SystemExit(f"FATAL: missing {summary_path}")
if not manifest_path.is_file():
    raise SystemExit(f"FATAL: missing {manifest_path}")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
checks = {
    "outcome": summary.get("outcome") == "completed",
    "steps": summary.get("steps") == 1024,
    "packed_corpus_sha": summary.get("corpus", {}).get("sha256") == expected_packed_sha,
    "packed_tokens": summary.get("corpus", {}).get("n_tokens") == 700000,
    "sequence_length": summary.get("corpus", {}).get("sequence_length") == 1536,
    "behavioral_objective": manifest.get("objective") == "behavioral_kd",
    "behavioral_mode": manifest.get("behavioral_mode") == "delta",
    "teacher_revision": manifest.get("teacher_revision") == "dbdc473dea0d6a9763042881cc33d6058d1742d2",
    "max_tokens": manifest.get("max_tokens") == 700000,
}
failed = [name for name, ok in checks.items() if not ok]
for name, ok in checks.items():
    print(f"post-run {name}: {'PASS' if ok else 'FAIL'}")
if failed:
    raise SystemExit("FATAL: post-run integrity checks failed: " + ", ".join(failed))
PY

# Small evidence bundle for immediate exfiltration before pod termination.
# Checkpoints/weights/optimizer state are deliberately excluded. Exclude the archive
# itself because it is created inside RUN_DIR.
tar -czf "$RUN_DIR/evidence_bundle.tgz" \
  --exclude='checkpoints' \
  --exclude='*.safetensors' \
  --exclude='*.pt' \
  --exclude='evidence_bundle.tgz' \
  -C "$RUN_DIR" .

printf '\nRUN COMPLETE\nEvidence: %s\n' "$RUN_DIR/evidence_bundle.tgz"
