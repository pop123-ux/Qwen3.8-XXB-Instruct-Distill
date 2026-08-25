"""Evaluation harness: tasks, generation measurement, metrics and reasoning sweeps."""

from .metrics import RunSummary, StratumSummary, compare, format_summary, summarise
from .reasoning import (
    DEFAULT_SETTINGS,
    ReasoningSweep,
    TemplateComparison,
    compare_rendered_prompts,
    sweep_reasoning_settings,
)
from .runner import (
    Backend,
    BackendProbe,
    GenerationResult,
    TransformersBackend,
    run_tasks,
    split_thinking,
)
from .tasks import (
    DIFFICULTY_ORDER,
    Task,
    long_context_suite,
    needle_in_haystack,
    reasoning_dev_set,
    tasks_by_difficulty,
)

__all__ = [
    "Task",
    "DIFFICULTY_ORDER",
    "reasoning_dev_set",
    "long_context_suite",
    "needle_in_haystack",
    "tasks_by_difficulty",
    "GenerationResult",
    "BackendProbe",
    "Backend",
    "TransformersBackend",
    "run_tasks",
    "split_thinking",
    "StratumSummary",
    "RunSummary",
    "summarise",
    "compare",
    "format_summary",
    "DEFAULT_SETTINGS",
    "TemplateComparison",
    "ReasoningSweep",
    "compare_rendered_prompts",
    "sweep_reasoning_settings",
]

from .benchmark import (  # noqa: E402
    CATEGORIES,
    BenchmarkItem,
    BenchmarkRun,
    BenchmarkSuite,
    comparable,
    new_run,
)
from .paired import (  # noqa: E402
    METRIC_DEFINITIONS,
    PairedRecord,
    PairedSummary,
    accuracy_at_token_budget,
    format_sweep_table,
    reasoning_sweep_table,
    result_to_record,
    summarise_paired,
    summarise_side,
)

__all__ = [*globals().get("__all__", []), "CATEGORIES", "METRIC_DEFINITIONS",
           "BenchmarkItem", "BenchmarkRun", "BenchmarkSuite", "PairedRecord",
           "PairedSummary", "accuracy_at_token_budget", "comparable",
           "format_sweep_table", "new_run", "reasoning_sweep_table",
           "result_to_record", "summarise_paired", "summarise_side"]
