# Third-Party Components and Licensing

**Nothing has been redistributed from this repository yet. This file records what we
depend on and what must be checked before anything is released.**

## Upstream model

| Item | Status |
|---|---|
| `Qwen/Qwen3.8-27B` | **License NOT verified from the upstream repository.** Secondary sources consistently report Apache-2.0. |
| Naming / trademark requirements | **NOT verified.** |
| Attribution requirements | **NOT verified.** |
| Redistribution terms for derived weights | **NOT verified.** |

The authoring environment could not reach `huggingface.co` (blocked by egress policy),
so the upstream `LICENSE` was never read. See [`docs/VERIFICATION.md`](docs/VERIFICATION.md).

### Required before any release

1. Read the upstream `LICENSE` and model card directly.
2. Confirm derived weights may be redistributed, and under what terms.
3. Confirm attribution requirements and satisfy them in the model card.
4. Confirm naming requirements before choosing a public model name. The working name
   `Qwen3.8-XXB-Instruct-Distill` is a **placeholder** and must not be treated as
   approved.
5. Check the license of any base model used for initialization.
6. Check the license of every dataset used for training or evaluation.
7. Determine whether teacher-generated synthetic data carries constraints.

**Do not release weights until all seven are resolved and documented here.**

## Code dependencies

| Dependency | License | Use |
|---|---|---|
| Python ≥ 3.10 | PSF | runtime |
| PyYAML | MIT | configuration files |
| huggingface_hub (optional) | Apache-2.0 | metadata download for teacher inspection |
| pytest (dev) | MIT | tests |
| ruff (dev) | MIT | linting |

## Reference material

Architectural formulas in `src/qwen_distill/architecture/` were transcribed by reading
`transformers==5.15.1`, specifically `transformers/models/qwen3_5/`. That code is
Apache-2.0 (Copyright 2025 The Qwen Team and The HuggingFace Inc. team).

**No upstream code was copied into this repository.** The modules here are independent
implementations of parameter, memory and FLOP *accounting* — they compute sizes, they
do not reimplement the model. Should model code be vendored later, it must be recorded
here with its license and copyright notice preserved.

## Terminology

Per the project's reporting standards, describe precisely what is released:

- **open-weight** — weights published under a stated license
- **open-source code** — this repository's code, MIT
- **reproducible training recipe** — configs and scripts sufficient to reproduce a run

Do not describe the project as "fully open source" unless every component named in a
release actually justifies it. Note also that Qwen3.8 is itself reported to be
Apache-2.0, so it must not be described as closed or proprietary.
