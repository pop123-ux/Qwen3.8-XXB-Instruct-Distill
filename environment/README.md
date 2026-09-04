# Research execution environment

This directory separates **observed historical facts** from the reproducible environment used for new controlled research.

## Historical baseline

`research-baseline.json` records facts observed during completed Run 004-M. It is provenance, not a rebuild recipe. The historical run did not capture every transitive package version, so the repository does not invent them after the fact.

## New RQ1 environment

`Dockerfile.research` defines the new RQ1 image. It is built off the paid GPU by `.github/workflows/research-image.yml` and published to GHCR. The image contains:

- Python 3.12.3;
- PyTorch 2.8.0 from the CUDA 12.8 wheel index;
- exact direct RQ1 package pins from `requirements/research-rq1-direct.txt`;
- this repository installed editable with dependency resolution disabled after the pinned install;
- `/opt/research-pip-freeze.txt`, a snapshot of the complete resolved package set.

The registry digest is the immutable binary environment identity. Record it through `RESEARCH_CONTAINER_DIGEST` in every controlled run. The runtime capture also records the package set so the environment can be audited without trusting a tag such as `rq1-v2`.

The host NVIDIA driver is outside the container and is captured separately. A driver mismatch disables direct throughput comparability but does not automatically invalidate a quality ablation when every scientific lock still matches.

## Allocator variable

New environments use the current canonical PyTorch variable:

`PYTORCH_ALLOC_CONF=expandable_segments:True`

The older `PYTORCH_CUDA_ALLOC_CONF` name is accepted as a backward-compatible alias by the guard/capture code. If both names are set with different values, the run is refused.

## Runtime capture

Inside a GPU session, before the first controlled model load:

`python scripts/capture_research_environment.py --output <run-or-session-path>/environment.json`

This is a short provenance action, not an environment-debugging exercise. All repository, protocol, CI and container-build work should already have happened before the GPU is rented.

## What changes require a new protocol?

Scientific quality comparisons require the registered teacher/student/data/training recipe and critical software/GPU class to remain fixed. Changes to those require a new protocol version.

Container digest and driver are additionally required for direct systems/throughput claims. When only these differ, the guard records `throughput_comparable=false` instead of disguising the difference or wasting compute trying to recreate a host that is not scientifically relevant to the quality objective.
