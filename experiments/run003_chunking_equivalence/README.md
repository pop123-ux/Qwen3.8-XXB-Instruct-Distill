# Run 003 — layer-KD chunking equivalence

Evidence that `behavioral_loss_chunked` evaluates the same objective as `behavioral_loss`,
measured on Run 003's own 1536-token calibration batch with the real teacher and student.
Written up in [`docs/LAYER_KD_CHUNKING.md`](../../docs/LAYER_KD_CHUNKING.md).

Harness: `scripts/verify_layer_kd_chunking.py`.

| file | what it is |
| --- | --- |
| `hidden_states.json` / `.log` | **Stage 1.** The objective and its gradient with respect to the hidden states. Gradients bit-identical at all 48 supervised layers; value differs by 6.4e-08 relative, from float32 summation order. |
| `parameters.json` / `.log` | **Stage 2, the result.** Four passes per form, exchangeability test over all 35 even splits. The true split ranks 10 of 35 by mean separation. |
| `parameters_repeats3.json` / `.log` | The same test at three passes per form, 10 splits. Kept because it was run first and agrees; ten splits give too coarse a p-value to rest on. |
| `parameters_biased_criterion.json` / `.log` | Superseded. Compared "between forms" against "within a form" directly, which is biased: the between set has three times the pairs and the maximum of more draws is larger for that reason alone. |
| `parameters_single_pass.json` / `.log` | Superseded. One pass per form plus one repeat; a single pair is too noisy an estimate of a floor that turns out to be ~1e-02. |
| `parameters_uncontrolled_rng.json` / `.log` | Superseded, and wrong. Did not fix the RNG between passes, so LoRA dropout resampled and every `lora_B` gradient disagreed. Kept because it is the reason the harness now restores RNG state, and because a reader comparing the numbers should be able to see what changed. |

The superseded files are retained deliberately. They are iterations of the harness, not
experiments, but the corrections they record are the reason the final numbers can be read
as being about chunking rather than about dropout, sampling or counting.

The `.log` transcripts have had the teacher's `Loading weights` progress frames stripped
(several hundred carriage-return redraws each, no content); nothing else was altered. They
are force-added past the repository's `*.log` ignore rule, as run transcripts are elsewhere.
