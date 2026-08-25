# Distillation configuration

Infrastructure examples. **No hyperparameter here is a recommendation** — the student
size, learning rate, sequence length and dataset size are all research results this
project has not produced yet. Level 2 has to finish before any of them can be chosen on
evidence rather than on vibes.

| file | what it is |
|---|---|
| `teacher_generation.yaml` | generating a teacher dataset, on a rented GPU |
| `sft_smoke.yaml` | the smallest end-to-end SFT config, for CPU validation |
| `logit_kd_example.yaml` | what a KD config will look like — **the objective is NOT_IMPLEMENTED and will refuse to run** |

The split that makes this affordable:

```
prompts.jsonl -> teacher generation (rented GPU, once) -> sharded JSONL + manifest
                                                                  |
                             student training (free T4) <---------+
```

The teacher and the student never coexist. Generation is expensive and happens once;
training reads a durable artifact and can be repeated cheaply.
