"""The checkpoint contract: one format, agreed on by trainer, validator and resume.

These reproduce a real Tesla T4 session in which three tools disagreed. A 20-step run
reported `wrote experiments/runs/t4_level2_100m_ckpt/final`; validating
`checkpoints/step_000020` said "Checkpoint directory not found"; and `--resume latest`
died with `FileNotFoundError: latest/training_state.pt`.

Two distinct causes, both now fixed, both pinned here:

1. **Two checkpoint systems coexisted.** The trainer wrote `final/` in a bespoke layout
   while the validator and resume logic expected the canonical `checkpoints/step_NNNNNN/`.
2. **`--resume` was not an option.** Only `--resume-from` existed, and argparse silently
   accepted `--resume` as an unambiguous *prefix abbreviation* of it — so `latest` became
   the literal relative path `latest/`, and the loader looked for
   `latest/training_state.pt`.

The second is the nastier failure: argparse's prefix matching turns a missing option into
a wrong value rather than an error, so the command looked valid and failed deep inside
the loader.

All CPU. No GPU, no Drive, no network.
"""

from __future__ import annotations

import json

import pytest
from conftest import HAS_STACK
from scripts_shim import load as load_script

from qwen_distill.training.checkpoints import (
    is_complete,
    list_checkpoints,
    read_latest_pointer,
    resolve_checkpoint,
    step_dirname,
)
from qwen_distill.training.resume_compat import (
    check_resume_compatibility,
)

train_student = load_script("train_student")


# --- cause 2: the CLI contract --------------------------------------------
def test_resume_is_a_real_option_not_a_prefix_of_resume_from():
    """`--resume latest` must bind to --resume. When only --resume-from existed,
    argparse's prefix matching made `latest` a filesystem path, and the loader then
    looked for `latest/training_state.pt` — the exact reported error."""
    args = train_student.build_parser().parse_args(
        ["--config", "x.yaml", "--resume", "latest", "--max-steps", "40"]
    )
    assert args.resume == "latest"
    assert args.resume_from is None


def test_resume_takes_a_string_so_latest_is_never_a_path():
    """Typed as Path, `latest` becomes Path('latest') and resolves against the working
    directory. It must stay a token the checkpoint resolver interprets."""
    args = train_student.build_parser().parse_args(["--config", "x.yaml", "--resume", "latest"])
    assert isinstance(args.resume, str)


def test_an_explicit_checkpoint_path_is_also_accepted():
    args = train_student.build_parser().parse_args(
        ["--config", "x.yaml", "--resume",
         "experiments/runs/t4_level2_100m_ckpt/checkpoints/step_000020"]
    )
    assert args.resume.endswith("step_000020")


def test_the_deprecated_alias_still_works():
    """Existing commands must not break."""
    args = train_student.build_parser().parse_args(
        ["--config", "x.yaml", "--resume-from", "some/checkpoint"]
    )
    assert str(args.resume_from) == "some/checkpoint"


# --- cause 1: one canonical layout ----------------------------------------
def test_the_trainer_writes_no_bespoke_final_directory():
    """`final/` was the second, incompatible checkpoint system. It is gone, not aliased:
    two layouts is what caused the disagreement in the first place."""
    source = (train_student.Path(__file__).resolve().parent.parent
              / "src" / "qwen_distill" / "training" / "trainer.py").read_text(encoding="utf-8")
    assert 'output / "final"' not in source
    assert 'step_dirname' in source, "the canonical layout is what the trainer writes"


# --- resume compatibility --------------------------------------------------
def base_config(**training):
    config = {
        "model": {"architecture": {"hidden_size": 64}},
        "data": {"max_sequence_length": 64},
        "training": {"max_steps": 20, "batch_size": 2, "strategy": "full",
                     "learning_rate": 6e-4},
    }
    config["training"].update(training)
    return config


def test_raising_max_steps_is_allowed_because_it_is_the_new_total():
    """Train 20, look at the curve, continue to 40. A normal workflow."""
    result = check_resume_compatibility(base_config(), base_config(max_steps=40))
    assert result.ok
    assert result.extends_schedule
    assert result.saved_max_steps == 20
    assert result.requested_max_steps == 40
    assert "20 -> 40" in result.render()


def test_the_schedule_rebuild_is_reported_not_silent():
    """OneCycleLR's LR depends on total_steps, so extending really does change the curve
    for the remaining steps. Silently is the wrong way to do that."""
    rendered = check_resume_compatibility(base_config(), base_config(max_steps=40)).render()
    assert "rebuilt" in rendered
    assert "new curve" in rendered


def test_an_unchanged_config_needs_no_schedule_rebuild():
    result = check_resume_compatibility(base_config(), base_config())
    assert result.ok
    assert not result.extends_schedule
    assert result.render() == ""


@pytest.mark.parametrize(
    ("path", "changed"),
    [
        (("model", "architecture"), {"hidden_size": 128}),
        (("data", "max_sequence_length"), 32),
        (("training", "batch_size"), 8),
        (("training", "strategy"), "lora"),
    ],
)
def test_genuinely_incompatible_changes_are_still_rejected(path, changed):
    """Section 6 of the mandate: allow the workflow, keep the safeguards."""
    current = base_config()
    node = current
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = changed

    result = check_resume_compatibility(base_config(), current)

    assert not result.ok
    assert any(".".join(path) in item for item in result.fatal)
    assert "cannot be resumed" in result.render()


def test_a_fatal_mismatch_reports_why_not_merely_that():
    result = check_resume_compatibility(base_config(), base_config(batch_size=8))
    assert "batch index" in result.render(), "the reason must be actionable"


def test_a_fatal_mismatch_suppresses_the_schedule_note():
    """Noise on top of the reason it stopped."""
    current = base_config(max_steps=40, batch_size=8)
    rendered = check_resume_compatibility(base_config(), current).render()
    assert "cannot be resumed" in rendered
    assert "rebuilt" not in rendered


def test_notable_changes_are_allowed_but_surfaced():
    result = check_resume_compatibility(base_config(), base_config(learning_rate=1e-3))
    assert result.ok
    assert any("learning_rate" in item for item in result.notable)
    assert "learning_rate" in result.render()


def test_a_checkpoint_without_a_saved_config_is_not_blocked():
    """Older checkpoints predate the saved config; refusing them helps nobody."""
    result = check_resume_compatibility(None, base_config(max_steps=40))
    assert result.ok
    assert not result.extends_schedule


# --- the exact reported lifecycle, end to end ------------------------------
pytestmark_stack = pytest.mark.skipif(not HAS_STACK, reason="requires torch and transformers")

TINY = {
    "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
    "vocab_size": 256, "num_attention_heads": 2, "num_key_value_heads": 1,
    "head_dim": 32, "linear_num_key_heads": 1, "linear_num_value_heads": 2,
    "linear_key_head_dim": 32, "linear_value_head_dim": 32,
    "full_attention_interval": 4, "tie_word_embeddings": True,
    "max_position_embeddings": 512,
}


def make_config(output, *, max_steps, resume=None, save_every=200):
    """Mirrors t4_level2_100m_ckpt.yaml: save_every deliberately exceeds max_steps, which
    is what made the 20-step smoke test produce no canonical checkpoint."""
    from qwen_distill.training.config import ExperimentConfig, ModelConfig

    config = ExperimentConfig(name="contract")
    config.model = ModelConfig(architecture=dict(TINY))
    config.data.text_corpus = True
    config.data.max_sequence_length = 64
    config.data.procedural_bytes = 20_000
    config.training.max_steps = max_steps
    config.training.batch_size = 2
    config.training.gradient_accumulation_steps = 1
    config.training.save_every = save_every
    config.training.log_every = 5
    config.training.eval_every = max_steps
    config.training.precision = "fp32"
    config.training.strategy = "full"
    config.training.objective = "sft"
    config.training.gradient_checkpointing = True
    config.runtime.output_dir = str(output)
    config.runtime.device = "cpu"
    config.runtime.resume_from = resume
    return config


def run(config):
    from qwen_distill.training.trainer import train

    return train(config, config.model.resolve_spec())


@pytestmark_stack
def test_the_exact_reported_lifecycle_train20_validate_resume_latest_to_40(tmp_path):
    """The regression test for the real T4 session, step for step.

    Fails on the old implementation at the first assertion: it wrote `final/`, not
    `checkpoints/step_000020/`.
    """
    from qwen_distill.training.validate_checkpoint import (
        validate_checkpoint,
        validate_resume,
    )

    output = tmp_path / "t4_level2_100m_ckpt"

    # 1. train 20 steps, with save_every (200) larger than max_steps
    assert run(make_config(output, max_steps=20)) == 0

    # 2. the canonical checkpoint exists — and no bespoke `final/` beside it
    checkpoint = output / "checkpoints" / step_dirname(20)
    assert checkpoint.is_dir(), "a successful run must leave a canonical checkpoint"
    assert is_complete(checkpoint)
    assert not (output / "final").exists(), "the second checkpoint system is gone"

    # 3. latest.json points at step 20
    pointer = read_latest_pointer(output / "checkpoints")
    assert pointer["step"] == 20
    assert pointer["path"] == step_dirname(20)
    assert pointer["complete"] is True

    # 4. validation accepts it
    report = validate_checkpoint(checkpoint, prompts=("The ",), max_new_tokens=4)
    assert report.passed, report.render()
    assert validate_resume(checkpoint).passed

    # 5. `latest` resolves to step 20 — not to a relative path called "latest"
    resolved = resolve_checkpoint(output / "checkpoints", "latest")
    assert resolved == checkpoint

    # 6. resume latest, extending the target to 40
    assert run(make_config(output, max_steps=40, resume="latest")) == 0

    # 7. it continued rather than restarting, and reached 40
    assert read_latest_pointer(output / "checkpoints")["step"] == 40
    assert [p.name for p in list_checkpoints(output / "checkpoints")] == [
        step_dirname(20), step_dirname(40),
    ]
    state = json.loads(
        (output / "checkpoints" / step_dirname(40) / "training_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["step"] == 40
    assert state["tokens_seen"] > 0

    # 8. the final checkpoint validates too
    assert validate_checkpoint(
        output / "checkpoints" / step_dirname(40), prompts=("The ",), max_new_tokens=4
    ).passed


@pytestmark_stack
def test_resume_by_explicit_checkpoint_path(tmp_path):
    """Section 8: the same lifecycle, naming the checkpoint directly."""
    output = tmp_path / "run"
    assert run(make_config(output, max_steps=20)) == 0

    explicit = str(output / "checkpoints" / step_dirname(20))
    assert run(make_config(output, max_steps=40, resume=explicit)) == 0

    assert read_latest_pointer(output / "checkpoints")["step"] == 40


@pytestmark_stack
def test_resume_by_step_number(tmp_path):
    output = tmp_path / "run"
    assert run(make_config(output, max_steps=20)) == 0
    assert run(make_config(output, max_steps=40, resume="20")) == 0
    assert read_latest_pointer(output / "checkpoints")["step"] == 40


@pytestmark_stack
def test_resumed_training_does_not_repeat_completed_steps(tmp_path):
    """Restarting at 0 would look like success and silently waste the first run."""
    from qwen_distill.training.progress import ProgressWriter

    def training_steps(directory):
        # Only the per-step training records; a validation record shares a step number
        # with one of them and is a different kind of event, not a repeated step.
        return [
            r["step"] for r in ProgressWriter(directory).read_history()
            if r.get("status") == "completed_step"
        ]

    output = tmp_path / "run"
    run(make_config(output, max_steps=20))
    before = training_steps(output)

    run(make_config(output, max_steps=40, resume="latest"))
    after = training_steps(output)

    assert max(before) == 20
    assert max(after) == 40
    assert after == sorted(after), "steps must advance monotonically across the resume"
    assert len(after) == len(set(after)), f"a step was trained twice: {after}"
    assert [s for s in after if s > 20] == [25, 30, 35, 40]


@pytestmark_stack
def test_validation_accepts_the_run_directory_too(tmp_path):
    """Pointing the validator at a run directory used to fail with a confusing error."""
    from qwen_distill.training.validate_checkpoint import validate_checkpoint

    output = tmp_path / "run"
    run(make_config(output, max_steps=20))

    report = validate_checkpoint(output, prompts=("The ",), max_new_tokens=4)

    assert report.passed, report.render()
    assert report.checkpoint.endswith(step_dirname(20)), "it resolved to the checkpoint"


@pytestmark_stack
def test_resuming_onto_an_incompatible_config_is_refused(tmp_path):
    """The safeguard survives the new flexibility."""
    output = tmp_path / "run"
    run(make_config(output, max_steps=20))

    config = make_config(output, max_steps=40, resume="latest")
    config.data.max_sequence_length = 32   # re-chunks the corpus

    assert run(config) == 2


@pytestmark_stack
def test_a_short_run_checkpoints_even_when_save_every_is_larger(tmp_path):
    """Section 10: max_steps=20 against save_every=200 previously left nothing usable."""
    output = tmp_path / "run"
    assert run(make_config(output, max_steps=20, save_every=200)) == 0
    assert [p.name for p in list_checkpoints(output / "checkpoints")] == [step_dirname(20)]


@pytestmark_stack
def test_an_unaligned_final_step_still_checkpoints(tmp_path):
    """save_every=5 with max_steps=13: the periodic writes land at 5 and 10, and 13 must
    still be captured."""
    output = tmp_path / "run"
    assert run(make_config(output, max_steps=13, save_every=5)) == 0
    names = [p.name for p in list_checkpoints(output / "checkpoints")]
    assert names == [step_dirname(5), step_dirname(10), step_dirname(13)]
    assert read_latest_pointer(output / "checkpoints")["step"] == 13
