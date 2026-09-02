# Run 002 — trained LoRA adapters

The output of the pure logit-KD control: 128 optimizer steps at sequence length 1536 on the
frozen canonical student, teacher `Qwen/Qwen3.8-27B @ dbdc473…`
in 4-bit. Kept through Git LFS because it is the one artifact of a completed run that the
repository cannot otherwise reconstruct — everything else here is a number, and this is the
model those numbers describe.

| | |
| --- | --- |
| step | 128 of 128 (and the step-64 midpoint) |
| trainable parameters | 23,003,136 of 13,031,508,864 |
| strategy | qlora r=16 alpha=32 |
| precision | bf16 |
| optimizer | adamw |
| sequence length | 1536 |
| tokens seen | 196,608 |
| launch commit | `f5dd3f75bf4e069c75442396bcf43f2f52d0471d` |

Trajectory, from `../summary.json`:

| | first | final |
| --- | --- | --- |
| KD divergence | 7.1902 | 1.4101 |
| cross-entropy | 10.9596 | 4.879 |
| top-1 teacher agreement | 0.0065 | 0.3375 |

## What is here, and what is not

`adapter_model.safetensors` only — 92 MB per step, the trained LoRA weights. The
`optimizer.pt`, `scheduler.pt`, `scaler.pt` and `rng.pt` that sat beside them in the run
directory are **not** here. They are 184 MB per step and exist to *resume* training, which
a completed run does not need, and the project's rule against committing optimizer state
stands unchanged.

The base weights are not here either. These adapters are meaningless without the frozen
canonical student they were trained against; see `../../pilot001_materialisation/` for how
that checkpoint was produced and `../command.txt` for the exact invocation.

## What this is not

Not a capability result. 128 steps and 196,608 tokens cannot move a 13B
model, and Run 002's own record says so. These adapters exist so the arm can be *evaluated*
later, and so the Run 002 / Run 003 comparison can be made on models rather than only on
training curves.
