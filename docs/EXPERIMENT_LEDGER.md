# Experiment ledger

Code: `src/qwen_distill/research/ledger.py`. Tests: `tests/test_research_ledger.py`.
Default location: `experiments/ledger.jsonl`.

A JSONL file, one entry per line, newest last. Deliberately not a database. A project this
size needs exactly three things from its record-keeping — that a result cannot be silently
edited after the fact, that every entry says where it came from, and that the file survives
being copied to a machine that does not have the codebase. A JSONL file does all three; a
database server does none of them without extra work.

## Rule 1 — append-only

`record()` only appends. There is no `update` and no `delete`, and tests assert the methods
do not exist. A result that turns out to be wrong is **superseded**:

```python
entry = ledger.measured("evaluation", "A3 pilot", {"score": 0.71})
ledger.retract(entry.id, "measured against the wrong checkpoint")
```

The original line stays on disk. `entries()` hides it by default; `include_superseded=True`
shows it. The retraction is itself part of the record — that is the difference between a
ledger and a scratchpad.

## Rule 2 — provenance is required

```
measured_here              produced by running something in this repository
reported_by_third_party    published elsewhere; must carry `source`
estimated                  produced by a model; must carry `method`
```

A closed set. There is no fourth option and an unknown value raises. Mixing these three is
the single most common way a research artifact becomes untrustworthy, so each has its own
constructor and its own validation:

- an **estimate without a method** is refused — otherwise it is indistinguishable from a
  measurement in the record;
- a **third-party number without a source** is refused — this is the rule that stops
  unsourced competitor benchmarks entering the record;
- a **measured entry that cites an external source** is refused — that combination reads as
  a measurement and is a citation.

## What every entry carries

`kind`, `title`, `provenance`, `payload`, plus `arm` for ablation results, `tags`, a UTC
timestamp, a content-addressed 16-character `id`, and an `env` block: Python version,
platform, git commit, branch, whether the tree was dirty, torch and transformers versions,
and GPU name if one is present. Cheap enough to attach to everything, and it is what makes
a six-month-old number interpretable.

Entry kinds: `architecture_audit`, `initialisation`, `training_run`, `evaluation`,
`memory_accounting`, `ablation_result`, `comparison`, `note`, `retraction`.

## Using it

```bash
python scripts/student_report.py --ledger        # append audit + memory accounting
```

```python
from qwen_distill.research.ledger import Ledger

ledger = Ledger()
ledger.measured("architecture_audit", "frozen student", audit())
ledger.reported("comparison", "Qwen3.5-9B MMLU-Pro", {"mmlu_pro": 82.5},
                source="Qwen3.5-9B model card")
ledger.estimated("memory_accounting", "Q4 at 32K", {"total_gib": 15.14},
                 method="analytical: audited component counts x quantisation bytes")

ledger.entries(kind="evaluation", arm="A3")
print(ledger.render())
```

`summary()` reports live and superseded counts, a breakdown by kind and by provenance, the
**measured fraction** — how much of the record is our own measurement rather than citation
or estimate — and which ablation arms have results.

## Reading it without the codebase

```bash
cat experiments/ledger.jsonl | jq -r 'select(.provenance=="measured_here") | .title'
```

That is the point of the format. A corrupt line raises with the file and line number rather
than being skipped.
