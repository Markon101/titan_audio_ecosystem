# Hermes and Titan Text research workflow crosscheck

Scope: read-only inspection on 2026-09-27. The Audio worktree is the writable
project; this note does not edit Titan Text or Hermes configuration, credentials,
sessions, or checkpoints.

## What was checked

- `hermes --version` reports local install `vgit.ca16be5` (2026.9.24), upstream
  `ca16be56`. Its auth and config files were not opened.
- Titan Text branch `exp/ippr-coordinate-channel` is clean at `97f9ec4`.
  Recent commits add literal context-file/stdin transport (`1648306`), a
  semantic context condenser (`5e4fc3d`), and tests/docs for that transport
  (`f85305e`). The skill explicitly warns that passing a filename via literal
  `--context` sends only the path text; `--context-file` sends the contents.
- Titan Text's IPPR coordinate report documents 15 single-shot converged runs
  after invalid chunked resumes were archived. The report says the coordinate
  arm did not meet its preregistered rescue rule (B−A +0.31 pp, p=0.938; B−C
  +4.38 pp, below +8 pp threshold); the raw 15-arm/seed entries exist under
  `runs/ippr_coordinate/analysis_summary.json`. The adaptive-halting report
  narrows a prior positive claim: fixed tau=5 beat adaptive at matched compute,
  and seed-level inference replaced anti-conservative token/pair pooling.
  `reports/raw/qualification_battery/qualification_analysis.json` contains the
  paired results. These are scoped Text findings, not evidence about Audio.
- The halting report mentions GLM corrections only as a sentence; the inspected
  tracked report did not identify a standalone GLM response receipt. Its role
  cannot be independently scored from this audit. This is an evidence gap,
  not a finding that GLM's comments were wrong.

## Concrete review-integrity defect

Hermes's retained IPPR post-run JSON
`~/.hermes/cache/scratch/ippr/postrun_review.json` has three DeepSeek worker
records. Each has `status="ok"`, nonempty answer text, about 6,371–6,376 prompt
tokens, and **`finish_reason="length"` at 2,048 completion tokens**. The
campaign report calls the file "full JSON persisted," which describes
capture of the envelopes; it does not mean the model completed each answer.
The three workers should be treated as partial consultations. The source of
the 2,048-token cap is visible in Titan Text's
`.agents/skills/research-team/scripts/research_coordinator.py` worker call.

`scripts/review_completion_audit.py` now checks the same condition without
retaining answer text. Its local receipt is `hermes_ippr_review_audit.json`:
**0 complete, 3 incomplete**. A review counts as complete only with nonempty
final content, `status="ok"`, and `finish_reason="stop"`; completion still does
not certify scientific correctness.

## Small workflow improvement carried into Audio

Audio's `scripts/audio_research_team.py` uses explicit, hash-recorded
`--context-file` packets. Its current default selects roughly 144 KB of
relevant Audio source and corrections, allows 16,384 output tokens, checks the
returned model and completion reason, requires a factual causal-context
receipt, and refuses full-team synthesis unless every independent role passes.
`team_v2/` retained eight completed independent answers and a synthesis
challenger; `team_v4_preflight/` is a later offline packet check, not another
live review. The new completion-audit tool extends the same gate to external
Hermes/GLM/DeepSeek JSON without copying raw answers or secrets.

For a future Titan Text maintenance pass: make the coordinator's output budget
explicit and large enough for the requested review, mark `finish_reason=length`
as incomplete rather than counting it as a completed worker, and require a
bounded retry with selected evidence before synthesis. Keep the context
condenser's fixed numerical anchors scoped to their run/protocol and reconcile
them with newer negative-result reports before presenting them as canonical.
No Titan Text file was changed here; Audio release and provenance were the
primary deliverable for this turn.
