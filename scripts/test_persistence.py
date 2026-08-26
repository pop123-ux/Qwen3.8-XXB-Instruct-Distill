#!/usr/bin/env python3
"""Prove the persistence guarantee on this machine, end to end, in a few seconds.

Not a unit test — a self-test you can run on the box that will do the training, against
the filesystem that will hold the checkpoints. That distinction matters: the guarantee
this exercises is about what a *destination* does with your bytes, and a Drive mount, an
SMB share and a local SSD do different things with them.

The lifecycle, exactly as the hardening requires it:

    create two checkpoints
      -> persist both, verifying each at the destination
      -> confirm the newest is resumable
      -> DELETE model.safetensors from the newest, as if by hand or by a sync
      -> validate again  ..... MUST report INVALID
      -> resolve `latest` ..... MUST fall back to the older valid checkpoint
      -> restore ............. MUST refuse the damaged one and restore the good one
      -> truncate a file mid-copy ... MUST NOT advance the pointer

Every stage that must fail is asserted to fail. A self-test that only exercises the happy
path would have passed on the code that lost a run.

Tiny synthetic tensors, CPU only, everything inside a temporary directory that is removed
on exit. Nothing is written to your run directories and no GPU is touched.

Examples::

    python scripts/test_persistence.py
    python scripts/test_persistence.py --destination /content/drive/MyDrive/persist-selftest
    python scripts/test_persistence.py --level load --keep
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_distill.training.checkpoint_validation import (
    LEVELS,
    MANIFEST,
    STRUCTURE,
    resolve_latest,
    validate_checkpoint_dir,
)
from qwen_distill.training.persist import (
    persist_checkpoint,
    persistent_status,
    restore_run,
)

RULE = "=" * 78


class Outcome:
    """Stage results, accumulated so one failure does not hide the rest."""

    def __init__(self) -> None:
        self.stages: list[tuple[str, bool, str]] = []

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        self.stages.append((name, passed, detail))
        marker = "PASS" if passed else "FAIL"
        print(f"  [{marker}] {name}" + (f"\n         {detail}" if detail else ""))
        return passed

    @property
    def failed(self) -> list[tuple[str, bool, str]]:
        return [s for s in self.stages if not s[1]]


def build_source_checkpoint(root: Path, step: int) -> Path:
    """A real checkpoint, written by the real code path, with a tiny real model."""
    import torch

    from qwen_distill.training.checkpoints import capture_rng_state, save_checkpoint

    model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.Linear(128, 64))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(2, 64)).sum().backward()
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    return save_checkpoint(
        root, step, model=model, optimizer=optimizer, scheduler=scheduler,
        training_state={"epoch": 0, "tokens_seen": step * 16_384},
        config={"name": "persistence-selftest"}, rng_state=capture_rng_state(),
    )


def run_selftest(destination: Path, workspace: Path, *, level: str) -> Outcome:
    outcome = Outcome()
    local = workspace / "local"
    checkpoints = local / "checkpoints"

    print(f"{RULE}\nPERSISTENCE SELF-TEST\n{RULE}")
    print(f"  source      : {local}")
    print(f"  destination : {destination}")
    print(f"  level       : {level}\n")

    # --- 1. create ---------------------------------------------------------------
    print("1. create two checkpoints locally")
    first = build_source_checkpoint(checkpoints, 200)
    second = build_source_checkpoint(checkpoints, 400)
    outcome.check(
        "both checkpoints validate locally",
        validate_checkpoint_dir(first, level=level).valid
        and validate_checkpoint_dir(second, level=level).valid,
    )

    # --- 2. persist --------------------------------------------------------------
    print("\n2. persist both, verifying each AT THE DESTINATION")
    results = [
        persist_checkpoint(first, destination, verify_level=level),
        persist_checkpoint(second, destination, verify_level=level),
    ]
    outcome.check(
        "both copies verified at the destination",
        all(r.verified and r.pointer_updated for r in results),
        "; ".join(str(r.failure) for r in results if not r.verified),
    )
    outcome.check(
        "'persisted ->' is printed only after verification",
        all("persisted ->" in r.render() for r in results),
    )

    status = persistent_status(destination, level=level)
    outcome.check(
        "the destination reports 2 verified resumable checkpoints",
        len(status["checkpoints"]) == 2 and status["resumable_step"] == 400,
        f"checkpoints={status['checkpoints']} resumable_step={status['resumable_step']}",
    )

    # --- 3. delete the weights, as if by hand ------------------------------------
    print("\n3. DELETE model.safetensors from the newest persisted checkpoint")
    remote_newest = Path(destination) / "checkpoints" / "step_000400"
    (remote_newest / "model.safetensors").unlink()
    damaged = validate_checkpoint_dir(remote_newest, level=level)
    outcome.check(
        "the damaged checkpoint is reported INVALID",
        not damaged.valid and "model.safetensors" in (damaged.invalid_reason or ""),
        damaged.invalid_reason or "it was still reported valid",
    )

    # --- 4. latest falls back ------------------------------------------------------
    print("\n4. resolve `latest` — must fall back, and must say so")
    resolution = resolve_latest(Path(destination) / "checkpoints", level=level)
    outcome.check(
        "latest.json still names the damaged checkpoint",
        resolution.pointer_path == "step_000400" and not resolution.pointer_valid,
    )
    outcome.check(
        "it falls back to the older checkpoint that verifies",
        resolution.fell_back and resolution.resolved_step == 200,
        f"resolved to step {resolution.resolved_step}",
    )
    outcome.check(
        "the fallback is reported, not silent",
        "is invalid" in resolution.render() and "falling back" in resolution.render(),
    )

    status = persistent_status(destination, level=level)
    outcome.check(
        "status counts 1 verified resumable, not 2 directories",
        len(status["checkpoints"]) == 1 and len(status["invalid_checkpoints"]) == 1,
        f"valid={status['checkpoints']} invalid={[e['name'] for e in status['invalid_checkpoints']]}",
    )

    # --- 5. restore refuses the damaged one ----------------------------------------
    print("\n5. restore into a fresh local directory")
    fresh = workspace / "session-b"
    restored = restore_run(destination, fresh)
    outcome.check(
        "the damaged checkpoint is skipped, by name",
        [e["name"] for e in restored["skipped"]] == ["step_000400"],
        f"skipped={restored['skipped']}",
    )
    outcome.check(
        "the good checkpoint is restored and the pointer names it",
        restored["restored"] == ["step_000200"]
        and restored["pointer"] is not None
        and restored["pointer"]["step"] == 200,
    )
    outcome.check(
        "the restored copy validates locally",
        validate_checkpoint_dir(
            fresh / "checkpoints" / "step_000200", level=level
        ).valid,
    )

    # --- 6. a truncated copy must not advance the pointer --------------------------
    print("\n6. a copy that arrives truncated must not advance the pointer")
    third = build_source_checkpoint(checkpoints, 600)
    real_copyfileobj = shutil.copyfileobj

    def truncate_weights(reader, writer, length=0):
        if writer.name.endswith("model.safetensors"):
            writer.write(reader.read(64))
            return None
        return real_copyfileobj(reader, writer, length)

    shutil.copyfileobj = truncate_weights
    try:
        bad = persist_checkpoint(third, destination, verify_level=level)
    finally:
        shutil.copyfileobj = real_copyfileobj

    outcome.check(
        "the truncated copy is refused",
        not bad.verified and not bad.pointer_updated,
        str(bad.failure),
    )
    outcome.check(
        "no 'persisted ->' is printed for it",
        "persisted ->" not in bad.render() and "NOT updated" in bad.render(),
    )
    outcome.check(
        "step_000600 is not promoted at the destination",
        not (Path(destination) / "checkpoints" / "step_000600").exists(),
    )
    final = persistent_status(destination, level=STRUCTURE)
    outcome.check(
        "the pointer still names the last checkpoint that verified",
        final["pointer"]["path"] == "step_000400",
        f"pointer={final['pointer']['path']}",
    )
    return outcome


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--destination", type=Path,
                        help="persistent destination to exercise (default: a temporary "
                             "directory). Point this at Drive to test the real thing.")
    parser.add_argument("--level", choices=LEVELS, default=MANIFEST,
                        help="verification depth to exercise (default: manifest)")
    parser.add_argument("--keep", action="store_true",
                        help="leave the workspace behind for inspection")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("this self-test needs torch (pip install -r requirements/training.txt)",
              file=sys.stderr)
        return 2

    workspace = Path(tempfile.mkdtemp(prefix="persistence-selftest-"))
    destination = args.destination or (workspace / "persistent")
    try:
        outcome = run_selftest(destination, workspace, level=args.level)
    finally:
        if args.keep:
            print(f"\n  workspace kept at {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)
            if args.destination is None:
                shutil.rmtree(destination, ignore_errors=True)

    print(f"\n{RULE}")
    if outcome.failed:
        print(f"PERSISTENCE SELF-TEST FAILED — {len(outcome.failed)} of "
              f"{len(outcome.stages)} stages")
        for name, _passed, detail in outcome.failed:
            print(f"  ! {name}" + (f": {detail}" if detail else ""))
        print(RULE)
        return 1
    print(f"PERSISTENCE SELF-TEST PASSED — {len(outcome.stages)} stages")
    print("  A checkpoint is only reported persisted after the destination copy has been")
    print("  independently verified, and a checkpoint that loses its weights afterwards")
    print("  is detected at the next status, restore or resume.")
    print(RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
