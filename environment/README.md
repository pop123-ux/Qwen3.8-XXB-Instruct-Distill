# Research execution environment

This directory separates **observed historical environment facts** from the reproducible environment used for future controlled research.

`research-baseline.json` is factual provenance from the completed Run 004-M environment. It pins the critical versions that were actually observed; it does not invent versions for packages whose exact historical versions were not recorded.

`Dockerfile.research` provides the container baseline for future work. The exact built image digest must be recorded with each controlled run. The host NVIDIA driver remains outside the container and is therefore separately fingerprinted.

`requirements/research-critical.txt` pins the two critical package versions directly supported by archived run evidence:

- PyTorch 2.8.0, CUDA 12.8 wheel
- Transformers 5.15.1

Do not treat this short file as a complete historical dependency lock. Before the final long-running experiment, run `scripts/capture_research_environment.py` inside the controlled container and retain the full package set. That artifact is the authoritative full-environment fingerprint for that protocol family.

A new software version, image digest, CUDA stack, driver or GPU changes the execution environment. It must either be reverted to the protocol or recorded as a new protocol family; it must never silently enter a matched ablation.
