#!/usr/bin/env python3
"""Back up experiment results from a Colab session to Google Drive.

Colab sessions are ephemeral: when the runtime recycles, anything not on Drive or in git
is gone. Checkpoints and experiment artifacts are usually too large for git, so Drive is
where they belong.

Run this **after** an experiment, not during one. Synchronising on every training step
would slow the run and thrash Drive for no benefit.

Safety is the priority here, because the destination is the user's personal Drive:

* **Nothing is ever deleted by default.** ``--delete-extraneous`` exists, is off, and
  refuses to run without ``--yes``.
* Symlinks are never followed, so a stray link cannot pull in an unrelated tree.
* Credentials and secrets are excluded by pattern and never copied.
* Files are only overwritten when the source is genuinely newer or a different size.
* ``--dry-run`` shows exactly what would happen and changes nothing.

Examples::

    python scripts/backup_colab_to_drive.py --dry-run
    python scripts/backup_colab_to_drive.py
    python scripts/backup_colab_to_drive.py --source /content/repo --destination /content/drive/MyDrive/repo
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SOURCE = Path("/content/Qwen3.8-XXB-Instruct-Distill")
DEFAULT_DESTINATION = Path("/content/drive/MyDrive/Qwen3.8-XXB-Instruct-Distill")
DRIVE_MOUNT = Path("/content/drive/MyDrive")

#: Never copied. Version control and caches are reproducible; the rest is either
#: regenerable or must not leave the machine.
EXCLUDE_PATTERNS: tuple[str, ...] = (
    ".git", ".git/*",
    "__pycache__", "*/__pycache__/*", "*.pyc", "*.pyo",
    ".pytest_cache", ".pytest_cache/*",
    ".ruff_cache", ".ruff_cache/*",
    ".mypy_cache", ".mypy_cache/*",
    ".venv", ".venv/*", "venv", "venv/*",
    "*.egg-info", "*.egg-info/*",
    "node_modules", "node_modules/*",
    ".ipynb_checkpoints", "*/.ipynb_checkpoints/*",
)

#: Credential-shaped paths. Excluded regardless of location, so a token dropped into the
#: repository root is never uploaded to Drive.
SECRET_PATTERNS: tuple[str, ...] = (
    ".env", ".env.*", "*.pem", "*.key", "id_rsa*", "id_ed25519*",
    ".netrc", ".git-credentials", "*token*.json", "*credentials*.json",
    "*.secret", "secrets.*", ".aws/*", ".ssh/*", ".config/gh/*",
    "huggingface_token*", "hf_token*",
)


@dataclass
class BackupPlan:
    """What a backup would do, before it does it."""

    source: Path
    destination: Path
    to_copy: list[tuple[Path, Path]] = field(default_factory=list)
    skipped_unchanged: int = 0
    excluded: int = 0
    secrets_excluded: list[str] = field(default_factory=list)
    symlinks_skipped: list[str] = field(default_factory=list)
    extraneous: list[Path] = field(default_factory=list)
    total_bytes: int = 0

    def summary(self) -> str:
        lines = [
            f"  source      : {self.source}",
            f"  destination : {self.destination}",
            f"  to copy     : {len(self.to_copy)} file(s), {self.total_bytes / 1024 / 1024:.1f} MiB",
            f"  unchanged   : {self.skipped_unchanged} file(s)",
            f"  excluded    : {self.excluded} path(s) by pattern",
        ]
        if self.secrets_excluded:
            lines.append(f"  secrets     : {len(self.secrets_excluded)} excluded (never copied)")
        if self.symlinks_skipped:
            lines.append(f"  symlinks    : {len(self.symlinks_skipped)} skipped (never followed)")
        if self.extraneous:
            lines.append(f"  extraneous  : {len(self.extraneous)} file(s) exist only at the destination")
        return "\n".join(lines)


def matches(relative: Path, patterns: tuple[str, ...]) -> bool:
    """Whether a repo-relative path matches any glob pattern, at any depth."""
    text = relative.as_posix()
    for pattern in patterns:
        if fnmatch.fnmatch(text, pattern) or fnmatch.fnmatch(relative.name, pattern):
            return True
        if any(fnmatch.fnmatch(part, pattern) for part in relative.parts):
            return True
    return False


def drive_mounted(mount: Path = DRIVE_MOUNT) -> bool:
    return mount.is_dir()


def build_plan(
    source: Path, destination: Path, *, include_extraneous: bool = False
) -> BackupPlan:
    """Walk the source and decide what needs copying. Touches nothing."""
    plan = BackupPlan(source=source, destination=destination)

    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)

        if path.is_symlink():
            # Never followed: a symlink could point anywhere on the filesystem.
            plan.symlinks_skipped.append(relative.as_posix())
            continue
        if matches(relative, SECRET_PATTERNS):
            plan.secrets_excluded.append(relative.as_posix())
            continue
        if matches(relative, EXCLUDE_PATTERNS):
            plan.excluded += 1
            continue
        if not path.is_file():
            continue

        target = destination / relative
        if target.exists():
            source_stat = path.stat()
            target_stat = target.stat()
            # Copy only when genuinely different, so re-running is cheap and idempotent.
            if (
                source_stat.st_size == target_stat.st_size
                and source_stat.st_mtime <= target_stat.st_mtime
            ):
                plan.skipped_unchanged += 1
                continue
        plan.to_copy.append((path, target))
        plan.total_bytes += path.stat().st_size

    if include_extraneous and destination.is_dir():
        for path in sorted(destination.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(destination)
            if matches(relative, EXCLUDE_PATTERNS) or matches(relative, SECRET_PATTERNS):
                continue
            if not (source / relative).exists():
                plan.extraneous.append(path)

    return plan


def execute(plan: BackupPlan, *, delete_extraneous: bool = False) -> int:
    """Perform the copies. Returns the number of files copied."""
    copied = 0
    for source_file, target in plan.to_copy:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)
        copied += 1
    if delete_extraneous:
        for path in plan.extraneous:
            path.unlink()
    return copied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--dry-run", action="store_true", help="show what would happen; change nothing")
    parser.add_argument(
        "--delete-extraneous", action="store_true",
        help="DANGEROUS: remove destination files absent from the source. Requires --yes.",
    )
    parser.add_argument("--yes", action="store_true", help="confirm a destructive operation")
    parser.add_argument(
        "--skip-mount-check", action="store_true",
        help="do not require /content/drive/MyDrive (for testing or a non-Colab target)",
    )
    parser.add_argument("--quiet", action="store_true", help="print only the summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.source.is_dir():
        print(f"Source directory not found: {args.source}", file=sys.stderr)
        return 2

    if not args.skip_mount_check and not drive_mounted():
        print(
            f"Google Drive is not mounted at {DRIVE_MOUNT}.\n\n"
            "In a Colab notebook, mount it first:\n\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')\n\n"
            "Then re-run this command. To back up somewhere other than Drive, pass\n"
            "--destination and --skip-mount-check.",
            file=sys.stderr,
        )
        return 2

    if args.delete_extraneous and not args.yes:
        print(
            "--delete-extraneous removes files from the destination and requires --yes.\n"
            "Run with --dry-run first to see exactly what would be deleted.",
            file=sys.stderr,
        )
        return 2

    plan = build_plan(args.source, args.destination, include_extraneous=args.delete_extraneous)

    print("=" * 60)
    print("COLAB -> DRIVE BACKUP" + (" (DRY RUN)" if args.dry_run else ""))
    print("=" * 60)
    print(plan.summary())

    if plan.secrets_excluded and not args.quiet:
        print("\n  excluded as credentials (never copied):")
        for name in plan.secrets_excluded[:10]:
            print(f"    - {name}")

    if not args.quiet and plan.to_copy:
        print("\n  files to copy:")
        for source_file, _ in plan.to_copy[:25]:
            relative = source_file.relative_to(args.source)
            print(f"    {relative.as_posix()}  ({source_file.stat().st_size / 1024:.0f} KiB)")
        if len(plan.to_copy) > 25:
            print(f"    ... and {len(plan.to_copy) - 25} more")

    if plan.extraneous:
        print(f"\n  {'WOULD DELETE' if args.dry_run else 'DELETING'} "
              f"{len(plan.extraneous)} extraneous destination file(s):")
        for path in plan.extraneous[:10]:
            print(f"    {path}")

    if args.dry_run:
        print("\n  DRY RUN: nothing was copied or deleted.")
        return 0

    try:
        copied = execute(plan, delete_extraneous=args.delete_extraneous)
    except OSError as exc:
        print(f"\n  Backup FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"\n  copied {copied} file(s) to {args.destination}")
    if not args.destination.is_dir():
        print("  WARNING: destination directory does not exist after copying", file=sys.stderr)
        return 1
    print("  backup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
