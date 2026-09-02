 ▐▛███▛█   Claude Code v2.1.258
▝▜██████▀  Opus 5 · Claude Pro
  ▝▝ ▝▝    /…/.claude/skills/qwen38-distillation-research


❯ history                                                                       

  Ran 7 shell commands (ctrl+o to expand)

● Reading "history" as the project's history — here it is. (If you meant shell
  history, say so.)

  Qwen3.8-27B → 13.01B distillation: history to date

  Phase 1–2 — small-scale scaffolding (Aug 24–26, Colab/T4)
  Levels 1 and 2 on toy-to-94.48M models. The notable event was a T4 OOM whose
  root cause was a missing DeltaNet term in the memory estimator (81de469) — the
  first sign that this architecture's memory model needed real terms, not
  extrapolation. Checkpoint persistence was hardened into a contract: a 
  checkpoint is not persisted until the destination proves it (62ccd43).

  Phase 3 — real teacher, real objective (Aug 26–28)
  Teacher-generation and eval infra, Level 2R on public-domain English, then
  actual teacher→student logit KD (44046f3). The objective was reframed from
  "hit a parameter count" to something measurable (e6c1f77), and a competitor
  audit found two of three claimed 16 GB-class models don't actually fit
  (3f1ea7c).

  The reset (Aug 30)
  Project moved onto the frozen sparse-MoE student, and the expert budget was
  corrected 22.07B → 13.01B (8d291f9) — that correction is what made the student
  fit 16 GB. Final frozen config: qwen38_19b_h5120_l48_moe, 13,008,505,728
  total / 9,611,119,488 active per token, 36 DeltaNet + 12 attention layers.

  Pinning and pre-GPU protocol (Aug 31 – Sep 1)
  Teacher revision pinned at library level to dbdc473d… (superseding 72a217af…),
  pinned downloader, acceptance gate, tokenizer-backed data path.

  Run 001 — canonical KD on the A40 (Sep 2)
  50/50 mixed-KD steps against the real 4-bit Qwen3.8-27B teacher, ~40.6 GB
  observed, 86.8 tok/s, both checkpoints validated at load level. Two findings
  worth carrying forward: the 4-bit student is ~17 GiB, not ~10 (bitsandbytes
  only quantizes nn.Linear; ~4.5B stacked MoE expert params stay bf16), and a
  blocker was cleared where lora/qlora were listed as strategies but trainer.py
  rejected anything but full, and training.optimizer was validated, recorded,
  then ignored in favor of hardcoded torch.optim.AdamW. A
  NameError('param_report') defect ate the pilot's summary.json after training
  completed — metrics survived, the 1-step smoke carries the memory profile.

  Run 002 — memory gate and the pure logit-KD control (Sep 2, latest)
  - seq 2048 calibration: peak 43.014 GiB > 42.0 GiB gate → the 128-step control
    was not started, and sequence length was deliberately not cut to force a
    fit at that point (f5dd3f7).
  - seq 1536 recalibration: 40.401 GiB, passed.
  - 128-step pure logit-KD control at seq 1536: completed, 196,608 tokens, 153.2
    tok/s, peak 40.580 GiB, train loss 7.19 → 1.41, validation 7.204 → 5.033.
    QLoRA, 23,003,136 trainable params (0.18%).

  Where things stand right now
  - HEAD d9570be is 2 commits ahead of origin — f5dd3f7 and d9570be are
    unpushed.
  - Untracked: CLAUDE_CLI_RUNS/Run001.md (the full 140 KB session transcript)
    and .claude/skills/.
  - Ledger has 18 entries including two explicit retraction records for the
    superseded 22.07B accounting.

  No capability claim has been made at any point — everything so far is
  mechanism, memory, and throughput.

✻ Crunched for 1m 5s · done 7:41 PM