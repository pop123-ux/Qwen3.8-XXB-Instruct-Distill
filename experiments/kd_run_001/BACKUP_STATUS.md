# Run 001 — external backup status

Checked 2026-09-02 on the RunPod Pod that will host Run 001.

## Destinations inspected

| Destination | Available | Evidence |
| --- | --- | --- |
| GitHub (this repository) | **commit yes, push NO** | `git ls-remote` succeeds anonymously; `git push` fails with `could not read Username for 'https://github.com'`. No `GITHUB_TOKEN`/`GH_TOKEN`, no `credential.helper`, no `~/.git-credentials`, no `~/.netrc`, no `SSH_AUTH_SOCK`, no `gh` CLI. |
| Google Drive via rclone | **no** | The remote `gdrive-primary:` is configured but `rclone lsd` returns `empty token found - please run "rclone config reconnect gdrive-primary:"`. Reconnecting needs an interactive OAuth flow. |
| Hugging Face Hub | **no** | No `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`, no `~/.cache/huggingface/token`. `hf` CLI is installed but unauthenticated. |

## Consequence

The scientific record is committed to this branch but **is not yet on GitHub**. Until
someone with push credentials runs

    git push origin claude/qwen38-16gb-distill-t6d0gy

the only copy of Run 001's record is on the Pod, and destroying the Pod destroys it.

This is a blocking item for pre-termination verification. It is recorded here rather
than only reported, because a blocker that lives in a chat transcript is a blocker that
gets forgotten at the moment the Pod bill is being watched.

## What is deliberately never uploaded anywhere

- The ~55 GB teacher checkpoint. It is re-obtainable from its pinned revision
  `dbdc473dea0d6a9763042881cc33d6058d1742d2`; that revision is the archived artefact.
- Checkpoint weights and optimizer state. Referenced in `ARCHIVE_INDEX.json` by size,
  completeness and `metadata.json`, never committed.
- Environment variable *values*. `environment.txt` records names only.
