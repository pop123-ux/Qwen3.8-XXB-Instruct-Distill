"""Throughput accounting must survive a resume.

The Level-2 run completed 2000 steps in 15,684.6 s for 32,768,000 tokens — a true
run-wide rate of ~2,090 tok/s. After resuming at step 1600, the logs reported 139,256
tok/s, then 70,945, 48,117, 36,623, decaying to 10,605 at the end.

That decay is the diagnosis. A rate that falls as 1/n is a numerator that barely moves
over a denominator growing linearly: **cumulative tokens** (restored from the checkpoint)
divided by **this session's elapsed time**. At the first log after resume that is 26.6M
tokens over ~196 s.

The fix separates three genuinely different quantities, because collapsing any two of
them is what produced the bug:

* **interval** — tokens since the last log / time since the last log. What you watch to
  see whether the run just got slower.
* **session** — tokens this process generated / time this process has run. What the
  current GPU is doing.
* **run-wide** — all tokens / all time including previous sessions. What the experiment
  cost.

These tests exercise the arithmetic directly rather than through a training loop, so
they pin the definitions rather than a particular loop's shape.
"""

from __future__ import annotations

import pytest

from qwen_distill.training.throughput import ThroughputTracker

#: The real Level-2 run, used to check the corrected arithmetic against known truth.
LEVEL2_TOKENS = 32_768_000
LEVEL2_STEPS = 2000
LEVEL2_SECONDS = 15_684.6
LEVEL2_RATE = LEVEL2_TOKENS / LEVEL2_SECONDS          # ~2,090 tok/s
TOKENS_PER_STEP = LEVEL2_TOKENS // LEVEL2_STEPS       # 16,384


def test_a_fresh_run_reports_all_three_rates_identically():
    """With no prior session, interval == session == run-wide over the first window."""
    tracker = ThroughputTracker()
    tracker.add_tokens(1000)
    rates = tracker.rates(now=10.0)

    assert rates["interval_tokens_per_second"] == pytest.approx(100.0)
    assert rates["session_tokens_per_second"] == pytest.approx(100.0)
    assert rates["tokens_per_second"] == pytest.approx(100.0)


def test_resumed_tokens_are_not_divided_by_session_time():
    """The exact bug: 26.6M cumulative tokens over 196 s of session time gave 135k tok/s.

    The run-wide rate must account for the time the earlier sessions took, and the
    session rate must count only what this process generated.
    """
    tracker = ThroughputTracker(
        resumed_tokens=26_214_400,          # 1600 steps x 16,384
        resumed_seconds=26_214_400 / LEVEL2_RATE,
    )
    tracker.add_tokens(25 * TOKENS_PER_STEP)          # 25 steps into the new session
    session_seconds = 25 * TOKENS_PER_STEP / LEVEL2_RATE

    rates = tracker.rates(now=session_seconds)

    assert rates["tokens_per_second"] == pytest.approx(LEVEL2_RATE, rel=0.01)
    assert rates["session_tokens_per_second"] == pytest.approx(LEVEL2_RATE, rel=0.01)
    assert rates["interval_tokens_per_second"] == pytest.approx(LEVEL2_RATE, rel=0.01)
    # The bug produced this. Nothing may report it.
    assert rates["tokens_per_second"] < 10_000


def test_the_first_log_after_resume_is_not_absurd():
    """Reproduces the reported 139,256 tok/s and asserts it cannot happen."""
    resumed_tokens = 1600 * TOKENS_PER_STEP
    tracker = ThroughputTracker(
        resumed_tokens=resumed_tokens, resumed_seconds=resumed_tokens / LEVEL2_RATE
    )
    tracker.add_tokens(25 * TOKENS_PER_STEP)
    session_seconds = 25 * TOKENS_PER_STEP / LEVEL2_RATE

    buggy = tracker.total_tokens / session_seconds     # what the old code computed
    correct = tracker.rates(now=session_seconds)["tokens_per_second"]

    assert buggy > 100_000, "the old arithmetic really did produce six-figure rates"
    assert correct == pytest.approx(LEVEL2_RATE, rel=0.01)
    assert correct < buggy / 50


def test_run_wide_rate_matches_the_real_level2_run():
    """End of the real run: 2000 steps, resumed once, must report ~2,090 tok/s."""
    resumed_tokens = 1600 * TOKENS_PER_STEP
    resumed_seconds = resumed_tokens / LEVEL2_RATE
    tracker = ThroughputTracker(resumed_tokens=resumed_tokens, resumed_seconds=resumed_seconds)
    tracker.add_tokens(400 * TOKENS_PER_STEP)
    session_seconds = 400 * TOKENS_PER_STEP / LEVEL2_RATE

    rates = tracker.rates(now=session_seconds)

    assert rates["tokens_per_second"] == pytest.approx(LEVEL2_RATE, rel=0.005)
    assert rates["total_tokens"] == LEVEL2_TOKENS
    assert rates["elapsed_s"] == pytest.approx(LEVEL2_SECONDS, rel=0.005)


def test_interval_rate_reflects_the_last_window_only():
    """A slowdown must show up immediately, not be averaged away by earlier speed."""
    tracker = ThroughputTracker()
    tracker.add_tokens(1000)
    first = tracker.rates(now=1.0)                    # 1000 tok/s
    tracker.add_tokens(100)
    second = tracker.rates(now=2.0)                   # 100 tok in the last second

    assert first["interval_tokens_per_second"] == pytest.approx(1000.0)
    assert second["interval_tokens_per_second"] == pytest.approx(100.0)
    # The session average is dragged only partway down, which is why both are reported.
    assert second["session_tokens_per_second"] == pytest.approx(550.0)


def test_the_interval_resets_at_every_report():
    tracker = ThroughputTracker()
    tracker.add_tokens(500)
    tracker.rates(now=1.0)
    tracker.add_tokens(500)
    assert tracker.rates(now=2.0)["interval_tokens_per_second"] == pytest.approx(500.0)


def test_three_sessions_accumulate_correctly():
    """A run resumed twice must still report what the experiment actually cost."""
    first = ThroughputTracker()
    first.add_tokens(1000)
    first.rates(now=10.0)

    second = ThroughputTracker(resumed_tokens=first.total_tokens,
                               resumed_seconds=first.total_seconds(10.0))
    second.add_tokens(1000)
    second.rates(now=10.0)

    third = ThroughputTracker(resumed_tokens=second.total_tokens,
                              resumed_seconds=second.total_seconds(10.0))
    third.add_tokens(1000)
    rates = third.rates(now=10.0)

    assert rates["total_tokens"] == 3000
    assert rates["elapsed_s"] == pytest.approx(30.0)
    assert rates["tokens_per_second"] == pytest.approx(100.0)
    assert rates["session_tokens_per_second"] == pytest.approx(100.0)


def test_a_resumed_session_that_does_nothing_reports_the_prior_rate():
    """Zero new tokens must not make the run-wide rate zero or undefined."""
    tracker = ThroughputTracker(resumed_tokens=10_000, resumed_seconds=100.0)
    rates = tracker.rates(now=5.0)

    assert rates["total_tokens"] == 10_000
    # Rates are rounded to 1 dp so logs stay readable; compare accordingly.
    assert rates["tokens_per_second"] == pytest.approx(10_000 / 105.0, rel=1e-3)
    assert rates["session_tokens_per_second"] == 0.0
    assert rates["interval_tokens_per_second"] == 0.0


def test_zero_elapsed_time_does_not_divide_by_zero():
    tracker = ThroughputTracker()
    tracker.add_tokens(100)
    rates = tracker.rates(now=0.0)
    assert rates["tokens_per_second"] == 0.0
    assert rates["interval_tokens_per_second"] == 0.0


def test_negative_or_backwards_time_is_clamped():
    """A monotonic clock should not go backwards, but a wrong reading must not produce
    a negative rate that looks like data."""
    tracker = ThroughputTracker()
    tracker.add_tokens(100)
    tracker.rates(now=10.0)
    rates = tracker.rates(now=5.0)
    assert rates["interval_tokens_per_second"] >= 0.0
    assert rates["tokens_per_second"] >= 0.0


def test_rates_are_reported_separately_and_never_conflated():
    """Collapsing any two of these is what produced the original bug."""
    tracker = ThroughputTracker(resumed_tokens=1_000_000, resumed_seconds=1000.0)
    tracker.add_tokens(2000)
    rates = tracker.rates(now=1.0)

    assert rates["interval_tokens_per_second"] == pytest.approx(2000.0)
    assert rates["session_tokens_per_second"] == pytest.approx(2000.0)
    assert rates["tokens_per_second"] == pytest.approx(1_002_000 / 1001.0)
    assert len({round(rates[k]) for k in
                ("interval_tokens_per_second", "tokens_per_second")}) == 2


def test_the_tracker_can_be_restored_from_checkpoint_state():
    """It reads the two fields TrainingState already carries; no new state is stored."""
    tracker = ThroughputTracker.from_state(tokens_seen=26_214_400, elapsed_seconds=12_547.0)
    assert tracker.resumed_tokens == 26_214_400
    assert tracker.resumed_seconds == 12_547.0
    assert tracker.total_tokens == 26_214_400
