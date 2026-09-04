"""The experiment record for one KD run, written before the run starts.

A conversation is not an experiment record. Neither is a terminal scrollback, nor a
RunPod Pod: `/workspace` survives a process dying, but not the Pod being destroyed, and
the Pod is destroyed the moment the run is judged finished.

So the record is written in three places, each answering a different failure:

    /workspace/runs/kd_run_001/   survives the process, the SSH session, the OOM
    experiments/kd_run_001/       survives the Pod  (small text artefacts only)
    an external object store      survives GitHub's file-size limits (large artefacts)

This module writes the first and prepares the second. It records *what would let someone
else reproduce the run*: the exact commit, the exact teacher revision, the exact
tokenizer, the exact corpus bytes, the exact hardware, the exact command. Every field is
captured from the machine rather than restated from a document, because a document can
drift from the thing it describes and a captured field cannot.

The manifest is written **before** the run, so a run that dies in its first minute still
has a complete statement of what it was. Metrics, checksums and the termination reason
are appended as they become known.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "RUN_ID", "run_id_for", "DEFAULT_RUN_ROOT", "ARCHIVE_ROOT", "SUBDIRECTORIES", "ARCHIVED_FILES",
    "capture_git", "capture_environment", "capture_hardware", "capture_teacher",
    "capture_tokenizer", "capture_dataset", "build_manifest", "initialise_run",
    "write_checksums", "record_termination", "archive_to_repository", "verify_record",
    "sha256_file",
]

#: The default run, used only to derive :data:`DEFAULT_RUN_ROOT` and
#: :data:`ARCHIVE_ROOT`. A run's actual identity comes from :func:`run_id_for`, i.e. the
#: directory it writes to. A second run gets a second directory, never a reuse of this
#: one: overwriting a run destroys the only copy of what it did.
RUN_ID = "kd_run_001"

DEFAULT_RUN_ROOT = Path("/workspace/runs") / RUN_ID

#: Where the small text-based record lives inside the repository, so GitHub carries the
#: scientific record off the Pod.
ARCHIVE_ROOT = Path("experiments") / RUN_ID

SUBDIRECTORIES = ("checkpoints", "artifacts", "final", "progress")

#: What is small enough, and textual enough, to belong in ordinary Git history. Weights
#: and optimizer state are deliberately absent: they are referenced by checksum instead.
def run_id_for(root: Path | str) -> str:
    """The run's identity: the name of the directory it writes to.

    ``RUN_ID`` was a module constant while the project had exactly one run, and every
    record it wrote said ``kd_run_001`` regardless of where it was written. A second run
    through the same code would have produced a record, a README, a checksum header and a
    termination entry all claiming to be the first run -- the precise failure the record
    exists to prevent. Deriving it from the root means a record cannot misname itself.
    """
    return Path(root).name or RUN_ID


ARCHIVED_FILES = (
    "README.md", "manifest.json", "config.json", "command.txt", "environment.txt",
    "hardware.txt", "git.txt", "teacher_provenance.json", "tokenizer_provenance.json",
    "dataset_provenance.json", "metrics.jsonl", "training.log", "CHECKSUMS.txt",
    "summary.json", "termination.json",
)

#: Files above this size are recorded by size and location rather than copied into Git.
ARCHIVE_SIZE_LIMIT = 8 * 1024 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(command: list[str], *, cwd: Path | None = None, timeout: int = 30) -> str | None:
    """Best-effort capture of a subprocess's stdout. A missing tool is not an error."""
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False,
            cwd=None if cwd is None else str(cwd),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    _write(path, json.dumps(payload, indent=2, sort_keys=False))


# --------------------------------------------------------------------------- git


def capture_git(repository: Path) -> dict[str, Any]:
    """Exactly which code this is, including whether it matches anything published.

    ``dirty`` is recorded with the list of modified paths rather than as a bare flag: a
    dirty tree means the commit SHA alone does not identify what ran, and the only
    honest record of that is which files differ.
    """
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repository)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repository)
    status = _run(["git", "status", "--porcelain"], cwd=repository)
    remote = _run(["git", "remote", "get-url", "origin"], cwd=repository)
    modified = [line[3:] for line in (status or "").splitlines() if line.strip()]
    return {
        "repository": remote,
        "branch": branch,
        "commit": commit,
        "commit_subject": _run(["git", "log", "-1", "--pretty=%s"], cwd=repository),
        "commit_date": _run(["git", "log", "-1", "--pretty=%cI"], cwd=repository),
        "dirty": bool(modified),
        "modified_paths": modified,
        "captured_at": _utc_now(),
    }


# --------------------------------------------------------------------- environment


def capture_environment() -> dict[str, Any]:
    """The software stack. Versions are read from the interpreter that will run.

    Imports are attempted rather than assumed: a record claiming a torch version that
    is not importable is worse than one saying the import failed.
    """
    record: dict[str, Any] = {
        "captured_at": _utc_now(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    for module, keys in (("torch", ("__version__",)), ("transformers", ("__version__",)),
                         ("safetensors", ("__version__",)), ("huggingface_hub", ("__version__",))):
        try:
            imported = __import__(module)
            record[f"{module}_version"] = getattr(imported, keys[0], None)
        except Exception as exc:  # noqa: BLE001 - absence is a fact worth recording
            record[f"{module}_version"] = f"unavailable: {type(exc).__name__}"
    try:
        import torch

        record["cuda_version"] = torch.version.cuda
        record["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        record["cuda_available"] = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        record["cuda_available"] = None
    # A frozen dependency list is what actually reproduces an environment; the named
    # versions above are what a reader scans.
    record["pip_freeze"] = (_run([sys.executable, "-m", "pip", "freeze"], timeout=120) or "").splitlines()
    # Environment variables are recorded by *name* only. Their values are where tokens
    # live, and this record is pushed to a public repository.
    record["environment_variable_names"] = sorted(os.environ)
    return record


# ------------------------------------------------------------------------ hardware


def capture_hardware() -> dict[str, Any]:
    """What the run is renting. Includes the raw `nvidia-smi` text, deliberately.

    A parsed summary is easier to read and a raw dump is harder to be wrong about, so
    both are kept.
    """
    record: dict[str, Any] = {"captured_at": _utc_now()}
    record["nvidia_smi"] = _run(["nvidia-smi"]) or "unavailable"
    query = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,compute_cap,serial",
        "--format=csv,noheader",
    ])
    record["gpus"] = []
    for line in (query or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            record["gpus"].append({
                "name": parts[0], "memory_total": parts[1],
                "driver_version": parts[2], "compute_capability": parts[3],
            })
    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            record["torch_gpu"] = {
                "name": properties.name,
                "total_memory_gib": round(properties.total_memory / 1024**3, 3),
                "multi_processor_count": properties.multi_processor_count,
            }
    except Exception:  # noqa: BLE001
        pass
    record["cpu_count"] = os.cpu_count()
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        record["total_ram_gib"] = round(page_size * os.sysconf("SC_PHYS_PAGES") / 1024**3, 3)
    except (ValueError, OSError):
        record["total_ram_gib"] = None
    record["filesystems"] = []
    for mount in ("/workspace", "/", "/root"):
        if Path(mount).exists():
            usage = shutil.disk_usage(mount)
            record["filesystems"].append({
                "mount": mount,
                "total_gib": round(usage.total / 1024**3, 2),
                "used_gib": round(usage.used / 1024**3, 2),
                "free_gib": round(usage.free / 1024**3, 2),
            })
    return record


# ---------------------------------------------------------------------- provenance


def capture_teacher(teacher_directory: Path | None) -> dict[str, Any]:
    """The teacher, identified by its upstream revision rather than its path.

    A directory name is not an identity — two directories on this Pod hold two different
    revisions of the same repository — so the download manifest's recorded revision is
    the authority, and its absence is reported as a reproducibility gap rather than
    filled in with a guess.
    """
    record: dict[str, Any] = {
        "captured_at": _utc_now(),
        "local_path": str(teacher_directory) if teacher_directory else None,
        "revision": None,
        "model": None,
    }
    if teacher_directory is None or not Path(teacher_directory).is_dir():
        record["status"] = "absent: no teacher directory was supplied or it does not exist"
        return record
    teacher_directory = Path(teacher_directory)
    manifest = teacher_directory / "teacher_download_manifest.json"
    if manifest.is_file():
        try:
            downloaded = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            downloaded = {}
        record.update({
            "model": downloaded.get("model"),
            "revision": downloaded.get("revision"),
            "downloaded_at": downloaded.get("downloaded_at"),
            "n_files": downloaded.get("n_files"),
            "total_bytes": downloaded.get("total_bytes"),
            "huggingface_hub_version": downloaded.get("huggingface_hub_version"),
            "download_manifest_sha256": sha256_file(manifest),
        })
    else:
        record["status"] = (
            "no teacher_download_manifest.json: the upstream revision is unrecorded and "
            "this run is not reproducible from the Hub"
        )
    # Small metadata files are cheap to hash and are what a differing revision changes
    # first, so they are the practical fingerprint of the checkpoint on disk.
    fingerprint = {}
    for name in ("config.json", "generation_config.json", "tokenizer_config.json",
                 "model.safetensors.index.json", "chat_template.jinja"):
        path = teacher_directory / name
        if path.is_file():
            fingerprint[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    record["metadata_sha256"] = fingerprint
    record["weights_not_archived"] = (
        "The teacher checkpoint is ~55 GB and is never committed or uploaded. It is "
        "re-obtainable from its pinned revision; that revision is the archived artefact."
    )
    if record.get("revision") is None:
        record.setdefault("status", "revision unpinned")
    return record


def capture_tokenizer(tokenizer_path: Path | None) -> dict[str, Any]:
    """Which tokenizer produced the ids. A vocabulary mismatch is silent and fatal.

    The student's embedding has one row per teacher token; a different tokenizer means
    the ids index different rows, and nothing in the training loop notices.
    """
    record: dict[str, Any] = {
        "captured_at": _utc_now(),
        "path": str(tokenizer_path) if tokenizer_path else None,
    }
    if tokenizer_path is None or not Path(tokenizer_path).exists():
        record["status"] = "absent: no tokenizer path supplied or it does not exist"
        return record
    tokenizer_path = Path(tokenizer_path)
    files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "chat_template.jinja", "special_tokens_map.json"):
        path = tokenizer_path / name
        if path.is_file():
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    record["files_sha256"] = files
    config = tokenizer_path / "tokenizer_config.json"
    if config.is_file():
        try:
            loaded = json.loads(config.read_text(encoding="utf-8"))
            record["tokenizer_class"] = loaded.get("tokenizer_class")
            record["model_max_length"] = loaded.get("model_max_length")
        except json.JSONDecodeError:
            pass
    tokenizer_json = tokenizer_path / "tokenizer.json"
    if tokenizer_json.is_file():
        try:
            loaded = json.loads(tokenizer_json.read_text(encoding="utf-8"))
            record["vocab_size"] = len(loaded.get("model", {}).get("vocab", {})) or None
            record["n_added_tokens"] = len(loaded.get("added_tokens", []))
        except (json.JSONDecodeError, AttributeError):
            pass
    # The tokenizer travels with the teacher checkpoint, so it inherits that revision.
    manifest = tokenizer_path / "teacher_download_manifest.json"
    if manifest.is_file():
        with contextlib.suppress(json.JSONDecodeError):
            record["revision"] = json.loads(manifest.read_text(encoding="utf-8")).get("revision")
    return record


def capture_dataset(corpus_manifest: Path | None) -> dict[str, Any]:
    """The corpus, by content hash rather than by filename.

    `train.txt` is a name; `bc5972d9…` is the data. Only the second survives someone
    regenerating the corpus with a different seed and keeping the name.
    """
    record: dict[str, Any] = {
        "captured_at": _utc_now(),
        "manifest_path": str(corpus_manifest) if corpus_manifest else None,
    }
    if corpus_manifest is None or not Path(corpus_manifest).is_file():
        record["status"] = "absent: no corpus manifest supplied or it does not exist"
        return record
    corpus_manifest = Path(corpus_manifest)
    try:
        loaded = json.loads(corpus_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        record["status"] = "unreadable: the corpus manifest is not valid JSON"
        return record
    # The document list can run to thousands of entries; the counts and hashes are what
    # identify the corpus, and the full list stays where it was generated.
    record.update({
        key: loaded.get(key)
        for key in ("name", "preparation_version", "created_at", "n_documents",
                    "n_train_documents", "n_validation_documents", "total_bytes",
                    "train_bytes", "validation_bytes", "train_sha256",
                    "validation_sha256", "split_rule", "overlap_check", "warnings")
    })
    record["manifest_sha256"] = sha256_file(corpus_manifest)
    record["directory"] = str(corpus_manifest.parent)
    return record


# ------------------------------------------------------------------------ manifest


def build_manifest(
    *,
    repository: Path,
    config: dict[str, Any] | None,
    config_path: Path | None,
    teacher_directory: Path | None,
    tokenizer_path: Path | None,
    corpus_manifest: Path | None,
    command: str | None,
    student_id: str | None = None,
    student_parameters: dict[str, Any] | None = None,
    notes: list[str] | None = None,
    run_id: str = RUN_ID,
) -> dict[str, Any]:
    """Everything needed to say what one run was, assembled from captured facts."""
    training = (config or {}).get("training", {})
    runtime = (config or {}).get("runtime", {})
    data = (config or {}).get("data", {})
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "schema_version": 1,
        "created_at": _utc_now(),
        "started_at": None,
        "finished_at": None,
        "status": "initialised",
        "git": capture_git(repository),
        "environment": capture_environment(),
        "hardware": capture_hardware(),
        "student": {
            "id": student_id,
            "parameters": student_parameters or {},
        },
        "teacher": capture_teacher(teacher_directory),
        "tokenizer": capture_tokenizer(tokenizer_path),
        "dataset": capture_dataset(corpus_manifest),
        "config_path": str(config_path) if config_path else None,
        "config": config,
        "training": {
            "objective": training.get("objective"),
            "strategy": training.get("strategy"),
            "optimizer": training.get("optimizer"),
            "precision": training.get("precision"),
            "learning_rate": training.get("learning_rate"),
            "weight_decay": training.get("weight_decay"),
            "scheduler": training.get("scheduler"),
            "warmup_steps": training.get("warmup_steps"),
            "max_steps": training.get("max_steps"),
            "batch_size": training.get("batch_size"),
            "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
            "gradient_checkpointing": training.get("gradient_checkpointing"),
            "sequence_length": data.get("max_sequence_length"),
            "seed": training.get("seed"),
        },
        "kd": {
            "objective": training.get("objective"),
            "kd_weight": training.get("kd_weight"),
            "kd_temperature": training.get("kd_temperature"),
            "kd_tail": training.get("kd_tail"),
            "kd_top_k": training.get("kd_top_k"),
        },
        "checkpointing": {
            "save_every": training.get("save_every"),
            "log_every": training.get("log_every"),
            "progress_every": training.get("progress_every") or training.get("log_every"),
            "eval_every": training.get("eval_every"),
            "output_dir": runtime.get("output_dir"),
            "persistent_backup": training.get("persistent_backup"),
            "resume_from": runtime.get("resume_from"),
        },
        "stopping_criteria": {
            "max_steps": training.get("max_steps"),
            "note": (
                "The run stops at max_steps, on an unrecoverable exception, or when "
                "stopped by hand. A CUDA OOM is caught and recorded as a result: the "
                "summary is written and the last complete checkpoint stays valid."
            ),
        },
        "command": command,
        "notes": notes or [],
    }
    return manifest


# ------------------------------------------------------------------ initialisation


README_TEMPLATE = """# Run `{run_id}` — knowledge distillation

Run ID: `{run_id}`
Created: {created}
Status: see `manifest.json` -> `status`

This directory, not any terminal scrollback or chat transcript, is the record of this run.

## What is here

| Path | What it is |
| --- | --- |
| `manifest.json` | The complete statement of what this run is: commit, teacher revision, tokenizer, corpus, hardware, every hyperparameter. Written **before** the run started. |
| `config.json` | The resolved experiment config, exactly as the trainer read it. |
| `command.txt` | The exact command that launched the run. |
| `environment.txt` | Python, PyTorch, CUDA, Transformers, full `pip freeze`. |
| `hardware.txt` | `nvidia-smi`, GPU model and VRAM, driver, RAM, filesystems. |
| `git.txt` | Repository, branch, commit SHA, clean/dirty state, modified paths. |
| `teacher_provenance.json` | Teacher model id, **exact upstream revision SHA**, metadata checksums. |
| `tokenizer_provenance.json` | Tokenizer identity, vocabulary size, file checksums. |
| `dataset_provenance.json` | Corpus identity, byte counts, `train`/`validation` SHA-256. |
| `metrics.jsonl` | Append-only, fsynced per record: loss, KD components, LR, step, throughput, elapsed, validation. |
| `training.log` | Full stdout+stderr of the run, tee'd as it is produced. |
| `progress/latest.json` | Atomically updated "where did this run get to". |
| `checkpoints/` | Full resumable checkpoints (weights, optimizer, scheduler, scaler, RNG, data position). |
| `artifacts/` | Plots, evaluation outputs, anything derived. |
| `final/` | The end-state student, once the run completes. |
| `CHECKSUMS.txt` | SHA-256, size and source path for every important artefact. |
| `termination.json` | How the run ended, written whether it succeeded or failed. |

## Archival

{archival}

## If this run failed

Do not delete it, do not overwrite it, and do not restart it from step 0. A failed run
is a measurement. Preserve the partial `metrics.jsonl`, the `training.log`, the last
complete checkpoint and `termination.json`, and resume with
`runtime.resume_from` pointing at `checkpoints/latest.json`.

## Reproducing

    git clone {repository}
    git checkout {commit}
    # teacher: {teacher_model} @ {teacher_revision}
    {command}
"""


def initialise_run(
    root: Path,
    manifest: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    command: str | None = None,
    archival_note: str | None = None,
) -> list[Path]:
    """Create the run directory and write every record that is knowable up front.

    Idempotent by design: re-running refreshes the captured records but never removes a
    metrics file, a log or a checkpoint. Losing those is the failure this exists to
    prevent, so nothing here deletes.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in SUBDIRECTORIES:
        (root / name).mkdir(exist_ok=True)

    written: list[Path] = []

    def emit(name: str, writer) -> None:
        path = root / name
        writer(path)
        written.append(path)

    emit("manifest.json", lambda p: _write_json(p, manifest))
    if config is not None:
        emit("config.json", lambda p: _write_json(p, config))
    emit("teacher_provenance.json", lambda p: _write_json(p, manifest["teacher"]))
    emit("tokenizer_provenance.json", lambda p: _write_json(p, manifest["tokenizer"]))
    emit("dataset_provenance.json", lambda p: _write_json(p, manifest["dataset"]))

    git = manifest["git"]
    git_text = "\n".join([
        f"repository   : {git.get('repository')}",
        f"branch       : {git.get('branch')}",
        f"commit       : {git.get('commit')}",
        f"subject      : {git.get('commit_subject')}",
        f"commit date  : {git.get('commit_date')}",
        f"state        : {'DIRTY' if git.get('dirty') else 'clean'}",
        "modified     : " + (", ".join(git.get("modified_paths") or []) or "(none)"),
        f"captured at  : {git.get('captured_at')}",
    ])
    emit("git.txt", lambda p: _write(p, git_text))

    environment = manifest["environment"]
    environment_text = "\n".join([
        f"captured at        : {environment.get('captured_at')}",
        f"hostname           : {environment.get('hostname')}",
        f"platform           : {environment.get('platform')}",
        f"python             : {environment.get('python_version')}  ({environment.get('python_executable')})",
        f"torch              : {environment.get('torch_version')}",
        f"cuda (torch)       : {environment.get('cuda_version')}",
        f"cudnn              : {environment.get('cudnn_version')}",
        f"transformers       : {environment.get('transformers_version')}",
        f"safetensors        : {environment.get('safetensors_version')}",
        f"huggingface_hub    : {environment.get('huggingface_hub_version')}",
        f"cuda available     : {environment.get('cuda_available')}",
        "",
        "# pip freeze",
        *environment.get("pip_freeze", []),
        "",
        "# environment variable NAMES only (values are never recorded: they hold tokens)",
        *environment.get("environment_variable_names", []),
    ])
    emit("environment.txt", lambda p: _write(p, environment_text))

    hardware = manifest["hardware"]
    hardware_text = "\n".join([
        f"captured at   : {hardware.get('captured_at')}",
        f"cpu count     : {hardware.get('cpu_count')}",
        f"total RAM GiB : {hardware.get('total_ram_gib')}",
        "",
        "# gpus",
        *[json.dumps(gpu) for gpu in hardware.get("gpus", [])],
        f"torch gpu     : {json.dumps(hardware.get('torch_gpu'))}",
        "",
        "# filesystems",
        *[json.dumps(fs) for fs in hardware.get("filesystems", [])],
        "",
        "# nvidia-smi",
        hardware.get("nvidia_smi", "unavailable"),
    ])
    emit("hardware.txt", lambda p: _write(p, hardware_text))

    if command is not None:
        emit("command.txt", lambda p: _write(p, command))

    # Never truncate a metrics file or a log that already exists.
    for name in ("metrics.jsonl", "training.log"):
        path = root / name
        if not path.exists():
            path.touch()
            written.append(path)

    teacher = manifest["teacher"]
    identity = manifest.get("run_id") or run_id_for(root)
    if archival_note is None:
        archival_note = f"See the repository's `experiments/{identity}/` directory."
    emit("README.md", lambda p: _write(p, README_TEMPLATE.format(
        run_id=identity,
        created=manifest["created_at"],
        archival=archival_note,
        repository=git.get("repository") or "<repository>",
        commit=git.get("commit") or "<commit>",
        teacher_model=teacher.get("model") or "<teacher>",
        teacher_revision=teacher.get("revision") or "<UNPINNED>",
        command=command or "<command>",
    )))
    return written


# ----------------------------------------------------------------------- checksums


def write_checksums(
    root: Path,
    *,
    external_locations: dict[str, str] | None = None,
    max_bytes: int | None = None,
) -> Path:
    """SHA-256 every artefact under the run directory, with its size and location.

    Large files are hashed too — a checksum that skips the weights cannot verify the
    weights — but `max_bytes` allows a fast pass when a run is still writing.
    """
    root = Path(root)
    external_locations = external_locations or {}
    lines = [
        f"# Run {run_id_for(root)} artefact checksums",
        f"# generated: {_utc_now()}",
        f"# root: {root}",
        "#",
        "# sha256  bytes  relative-path  external-location",
        "",
    ]
    skipped: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "CHECKSUMS.txt":
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        if max_bytes is not None and size > max_bytes:
            skipped.append(f"{relative}  ({size} bytes)")
            continue
        external = external_locations.get(relative, "-")
        lines.append(f"{sha256_file(path)}  {size}  {relative}  {external}")
    if skipped:
        lines += ["", "# not hashed on this pass (over the size limit):", *[f"#   {s}" for s in skipped]]
    target = root / "CHECKSUMS.txt"
    _write(target, "\n".join(lines))
    return target


def record_termination(
    root: Path, *, status: str, reason: str, exit_code: int | None = None,
    last_step: int | None = None, extra: dict[str, Any] | None = None,
) -> Path:
    """How the run ended — written for a failure exactly as for a success.

    A run that OOMs at step 3,000 measured something; a record that only exists when a
    run succeeds throws that measurement away.
    """
    root = Path(root)
    payload = {
        "run_id": run_id_for(root), "status": status, "reason": reason,
        "exit_code": exit_code, "last_step": last_step,
        "recorded_at": _utc_now(), **(extra or {}),
    }
    target = root / "termination.json"
    _write_json(target, payload)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = status
            manifest["finished_at"] = payload["recorded_at"]
            _write_json(manifest_path, manifest)
        except json.JSONDecodeError:
            pass
    return target


# ------------------------------------------------------------------------ archival


def archive_to_repository(
    root: Path, archive: Path, *, size_limit: int = ARCHIVE_SIZE_LIMIT
) -> dict[str, Any]:
    """Copy the small text record into the repository, and *reference* the rest.

    Weights and optimizer state are never copied. What goes into Git is the record that
    lets someone reconstruct the run; what stays out is the thing the record points at.
    """
    root, archive = Path(root), Path(archive)
    archive.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    referenced: list[dict[str, Any]] = []
    for name in ARCHIVED_FILES:
        source = root / name
        if not source.is_file():
            continue
        size = source.stat().st_size
        entry = {"name": name, "bytes": size, "sha256": sha256_file(source),
                 "source": str(source)}
        if size > size_limit:
            referenced.append({**entry, "reason": "over the in-Git size limit"})
            continue
        shutil.copy2(source, archive / name)
        copied.append(entry)
    # Checkpoints are referenced, never copied: they are gigabytes of binary.
    checkpoints = root / "checkpoints"
    if checkpoints.is_dir():
        for directory in sorted(p for p in checkpoints.iterdir() if p.is_dir()):
            metadata = directory / "metadata.json"
            total = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
            referenced.append({
                "name": f"checkpoints/{directory.name}", "bytes": total,
                "source": str(directory),
                "complete": (directory / "COMPLETE").exists(),
                "metadata": json.loads(metadata.read_text(encoding="utf-8"))
                if metadata.is_file() else None,
                "reason": "binary checkpoint; not committed to Git",
            })
    index = {
        "run_id": run_id_for(root), "generated_at": _utc_now(), "run_root": str(root),
        "copied": copied, "referenced_not_copied": referenced,
    }
    _write_json(archive / "ARCHIVE_INDEX.json", index)
    return index


# -------------------------------------------------------------------- verification


#: The pre-termination checklist, as machine-checkable predicates. A checklist a human
#: ticks by eye is a checklist that gets ticked when the Pod bill is running.
CHECKLIST: tuple[tuple[str, str], ...] = (
    ("run directory exists", "root"),
    ("manifest exists", "manifest.json"),
    ("git SHA recorded", "manifest:git.commit"),
    ("branch recorded", "manifest:git.branch"),
    ("git dirty state recorded", "manifest:git.dirty"),
    ("teacher revision recorded", "manifest:teacher.revision"),
    ("tokenizer provenance recorded", "tokenizer_provenance.json"),
    ("dataset provenance recorded", "dataset_provenance.json"),
    ("environment recorded", "environment.txt"),
    ("hardware recorded", "hardware.txt"),
    ("exact command recorded", "command.txt"),
    ("logs persisted", "nonempty:training.log"),
    ("metrics persisted", "nonempty:metrics.jsonl"),
    ("checkpoint persisted", "checkpoint"),
    ("checkpoint metadata preserved", "checkpoint-metadata"),
    ("SHA-256 checksums generated", "nonempty:CHECKSUMS.txt"),
    ("termination reason recorded", "termination.json"),
)


def _manifest_field(manifest: dict[str, Any], dotted: str) -> Any:
    node: Any = manifest
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def verify_record(root: Path) -> dict[str, Any]:
    """Check every pre-termination item and return pass/fail per item.

    Returns a structure, not a print: the caller decides whether a missing item blocks
    terminating the Pod.
    """
    root = Path(root)
    manifest: dict[str, Any] = {}
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}

    results: list[dict[str, Any]] = []
    for label, predicate in CHECKLIST:
        detail = ""
        if predicate == "root":
            ok = root.is_dir()
            detail = str(root)
        elif predicate == "checkpoint":
            directories = [
                p for p in (root / "checkpoints").glob("step_*")
                if p.is_dir() and (p / "COMPLETE").exists()
            ] if (root / "checkpoints").is_dir() else []
            ok = bool(directories)
            detail = f"{len(directories)} complete checkpoint(s)"
        elif predicate == "checkpoint-metadata":
            directories = [
                p for p in (root / "checkpoints").glob("step_*")
                if p.is_dir() and (p / "COMPLETE").exists()
            ] if (root / "checkpoints").is_dir() else []
            ok = bool(directories) and all((p / "metadata.json").is_file() for p in directories)
            detail = "metadata.json present in every complete checkpoint" if ok else "missing"
        elif predicate.startswith("manifest:"):
            value = _manifest_field(manifest, predicate.split(":", 1)[1])
            ok = value is not None
            detail = str(value)
        elif predicate.startswith("nonempty:"):
            path = root / predicate.split(":", 1)[1]
            ok = path.is_file() and path.stat().st_size > 0
            detail = f"{path.stat().st_size} bytes" if path.is_file() else "missing"
        else:
            path = root / predicate
            ok = path.is_file()
            detail = f"{path.stat().st_size} bytes" if ok else "missing"
        results.append({"item": label, "ok": ok, "detail": detail})

    failed = [r["item"] for r in results if not r["ok"]]
    return {
        "run_id": run_id_for(root), "root": str(root), "checked_at": _utc_now(),
        "items": results, "failed": failed, "verified": not failed,
        "status": "PERSISTENCE / BACKUP STATUS: VERIFIED" if not failed
        else "PERSISTENCE / BACKUP STATUS: FAILED — DO NOT TERMINATE POD",
    }
