#!/usr/bin/env bash
# Launch a KD run so that neither the SSH session, the terminal, nor Claude Code owns it.
#
# Three things kill a rented-GPU run that has nothing wrong with it: the SSH connection
# drops, the terminal closes, or the agent that started it exits and takes its process
# group. `setsid` + `nohup` detaches from all three. `tee` puts stdout and stderr into
# the persistent record as they are produced, because a log that only exists in a
# scrollback is not a log.
#
# On exit — success, exception, OOM or SIGTERM — the termination reason is recorded and
# the checksums are refreshed. A failed run keeps its partial metrics, its log and its
# last complete checkpoint; nothing here deletes or restarts.
#
# Usage:
#   scripts/launch_kd_run.sh <training command...>
#
# Example:
#   scripts/launch_kd_run.sh python scripts/train_student.py \
#       --config configs/experiments/kd_run_001.yaml
#
# Then follow it with:  tail -f /workspace/runs/kd_run_001/training.log

set -uo pipefail

RUN_ROOT="${RUN_ROOT:-/workspace/runs/kd_run_001}"
# Names this run in its own log banner. Defaults to the directory it writes to, so a
# second run through this launcher cannot inherit the first one's label.
RUN_LABEL="${RUN_LABEL:-$(basename "${RUN_ROOT}")}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${RUN_ROOT}/training.log"
PIDFILE="${RUN_ROOT}/run.pid"

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <training command...>" >&2
    exit 2
fi

if [ ! -f "${RUN_ROOT}/manifest.json" ]; then
    echo "ERROR: ${RUN_ROOT}/manifest.json does not exist." >&2
    echo "       Run 'python scripts/run_record.py init ...' first: the manifest must" >&2
    echo "       exist BEFORE the run, or a run that dies early has no record of what" >&2
    echo "       it was." >&2
    exit 2
fi

if [ -f "${PIDFILE}" ] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
    echo "ERROR: a run is already active (pid $(cat "${PIDFILE}")). Refusing to start a" >&2
    echo "       second one against the same record." >&2
    exit 2
fi

mkdir -p "${RUN_ROOT}"
printf '%s\n' "$*" > "${RUN_ROOT}/command.txt"
{
    echo ""
    echo "=============================================================================="
    echo "${RUN_LABEL} launch  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "  command : $*"
    echo "  cwd     : ${REPO}"
    echo "  commit  : $(git -C "${REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "=============================================================================="
} >> "${LOG}"

# `setsid` gives the run its own session, so a hangup on this terminal does not reach it.
setsid nohup bash -c '
    set -o pipefail
    cd "$1"; shift
    RUN_ROOT="$1"; shift
    LOG="${RUN_ROOT}/training.log"
    "$@" 2>&1 | tee -a "${LOG}"
    code=${PIPESTATUS[0]}

    case ${code} in
        0)   status=completed;   reason="the training command exited 0" ;;
        130) status=interrupted; reason="SIGINT (Ctrl-C)" ;;
        137) status=interrupted; reason="SIGKILL (137) — OOM-killer or a hard stop" ;;
        143) status=interrupted; reason="SIGTERM (143)" ;;
        *)   status=failed;      reason="the training command exited ${code}" ;;
    esac
    if grep -qi "CUDA out of memory\|torch.OutOfMemoryError" "${LOG}" 2>/dev/null; then
        status=oom
        reason="CUDA out of memory (exit ${code}); see training.log"
    fi

    last_step=$(python - "${RUN_ROOT}" <<'"'"'PY'"'"'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "progress" / "latest.json"
try:
    print(json.loads(p.read_text()).get("step", ""))
except Exception:
    print("")
PY
)
    args=(--root "${RUN_ROOT}" terminate --status "${status}" --reason "${reason}" --exit-code "${code}")
    [ -n "${last_step}" ] && args+=(--last-step "${last_step}")
    python scripts/run_record.py "${args[@]}" >> "${LOG}" 2>&1

    # A fast checksum pass: the weights are hashed by the deliberate full pass before
    # archival, not on every exit, so a crash loop cannot spend the run re-hashing GB.
    python scripts/run_record.py --root "${RUN_ROOT}" checksums --max-bytes 268435456 \
        >> "${LOG}" 2>&1
    rm -f "${RUN_ROOT}/run.pid"
' _ "${REPO}" "${RUN_ROOT}" "$@" </dev/null >/dev/null 2>&1 &

echo $! > "${PIDFILE}"
echo "${RUN_LABEL} launched detached: pid $(cat "${PIDFILE}")"
echo "  log     : ${LOG}"
echo "  follow  : tail -f ${LOG}"
echo "  status  : python scripts/run_record.py verify"
