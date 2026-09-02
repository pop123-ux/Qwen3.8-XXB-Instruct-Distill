# pilot001 — canonical student materialisation

The provenance of the student **every** Run 001/002/003 experiment loads. `kd_run.py` is
always invoked with `--pretrained /workspace/runs/pilot001/transferred`, and this is the
record of how those weights were produced and that they were produced completely.

Kept here because the weights themselves are 25 GiB and live only on the RunPod instance.
They are regenerable — `scripts/distill_pilot.py` against the pinned teacher
`Qwen/Qwen3.8-27B @ dbdc473dea0d6a9763042881cc33d6058d1742d2` — but the record of the
materialisation that the existing runs actually used is not, and every number those runs
produced is attributable to it.

## Transfer coverage — `materialisation.json`

| | |
| --- | ---: |
| student parameters | 13,008,505,728 |
| transferred parameters | 13,006,293,888 |
| coverage | 0.999830 |
| copied | 471 |
| merged | 24 |
| decomposed | 240 |
| initialised fresh | 96 |
| missing | 0 |
| complete | True |

`student_parameters` is 13,008,505,728, the frozen count the skill pins for
`qwen38_19b_h5120_l48_moe`. `missing` is empty and `complete` is true: no student tensor
was left without a source or an explicit fresh initialisation. The 96
initialised tensors are the 0.017003% of parameters with no teacher
counterpart — the MoE router and expert structure the dense teacher has nothing to donate
to. It also carries the 48 -> 64 `layer_mapping`, which is the same correspondence
`layer_kd` supervises.

## First forward — `preflight.json`

| | |
| --- | ---: |
| model class | `Qwen3_5MoeForCausalLM` |
| parameters | 13,008,505,728 |
| dtype | torch.bfloat16 |
| weights | 24.23 GiB |
| load time | 175.9 s |
| forward | 1.18 s at (1, 1024) |
| loss | 9.2700 (ppl 10614.7) |
| finite | loss True, logits True |
| peak VRAM | 26.75 GiB of 44.43 |

A cross-entropy of 9.27 at initialisation is ln(vocab) territory and is what a
correctly assembled but untrained student should give; it is a smoke test, not a result.
The 26.75 GiB peak is bf16 inference on the A40, and is **not** a
deployment measurement — the 16 GB target is a separate, quantised measurement that has
not been made.

## Checksums

```
2e6944b05047b932c5c489c13e17786976284d9d16631d5d00229e7ec270da4c  materialisation.json
3ef6633dc85b8794dd13eeaea856628cb1a37a7417849ec86be30f8696ae8abc  preflight.json
```
