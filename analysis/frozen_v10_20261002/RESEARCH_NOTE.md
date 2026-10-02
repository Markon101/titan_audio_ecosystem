# Frozen v10 causal audio and temporal credit, 2026-10-02

## Recovery and source

`recover_campaign.py` verifies the six completed matched-exposure runs, their
checkpoint and audio hashes, optimizer counts, schedule bytes, and every
sampled target file/frame. It writes `recovery.json`. The step-58,107 model,
world, AdamW, Morphic state, and metadata in
`analysis/metastable_20261001/frozen_step_58107/` still match their archived
receipt. The mutable `/sdcard/Download/..._cont` alias later advanced to
step 63,966 and is **not** used here. No matched arm was rerun. The previous
matched campaign used two schedule seeds on one model/world, not independent
model seeds; B new families yielded fewer confirmed regions than A under
both schedules, with much more clipping. It did not establish saturation.

## Causal question

Can a specific msfield, GRU, Morphic, feedback, or host-forcing pathway
change **audible organization** over time when the exact mature weights are
frozen? Each arm starts from the same saved TITANM10 state and fixed target
schedule. The runner constructs no optimizer, calls no backward pass, and
writes analysis artifacts to a separate directory. It rehashes canonical
inputs at completion. The exact-RNG follow-up clones the saved controller RNG;
the target and forcing streams remain deterministic paired controls. It
replays realized baseline forcing by rollout offset. This
is a paired frozen counterfactual protocol, not a bitwise reproduction of
ordinary training continuation. No direct reference audio enters the forward
path; corpus audio contributes only post-forward coarse spectral-error
feedback. The `feedback_replay` arm is an artificial scalar replay control.

The opt-in v10 panel includes a no-op clone, coarse-state hold, learned
cross-scale-exchange removal, GRU-state hold, upper Morphic residual bypass,
half-strength structured shear, feedback scalar zero, and feedback scalar
replay. Each is a separate arm. Exchange removal is expected to change field
energy, so any audible effect must be read with that confound. Coarse hold
affects audio from the following chunk. A 16-chunk smoke under schedule
20261002 passed: the no-op clone matched every raw/post-DSP sample and the
final world fingerprint; parent files were unchanged; optimizer steps were
zero. Coarse hold changed the final fine/meso state and audio by very small
amounts (waveform L2 about 1.05e-6). Sixteen chunks are only 1.37 seconds,
so this is an implementation gate, not a musical result.

An initial 256-chunk, 21.845-second **derived-RNG pilot** completed in
`runs/panel_w0_256/`; `audio_readout.json` is calculated by
`analyze_frozen_panel.py`. Its no-op clone and baseline were bitwise identical
in raw/post-DSP audio and final world fingerprint, with unchanged parent
hashes and no optimizer. Coarse hold's full-audio waveform L2 was 0.02877;
the first 64 chunks were 0.00034 and the last 64 were 0.03649. Its final
coarse-state relative distance was only 0.00371, final stereo correlation
remained 0.867, and RMS ratio was 1.0007. This is delayed causal audio
divergence, but could include phase drift and five sampled controller-action
changes; it is not yet proof of better musical organization. GRU hold and
upper Morphic bypass had full-audio L2 0.03486 and 0.03397 respectively;
the latter left persistent field state nearly identical. Exchange removal
had a much larger field displacement and raised coarse RMS from 0.671 to
0.965, so its large audio displacement is energy-confounded. Half-strength
shear is a host-forcing control and also produced large divergence.

Zeroing the target-error scalar produced exactly the baseline audio and
learned field/GRU state over this 256-chunk window, despite baseline sampled error values near
0.4–0.5. Replaying the baseline scalar also reproduced it. This is a bounded
negative result about audible influence in this window: the full world
fingerprint **did** differ because the uncertainty host state changed.
It is not a claim that corpus training
has no effect. The pilot restored all saved ecological state but derived the
controller RNG from the saved RNG hash. The follow-up starts the controller
from the **exact saved RNG state**, and uses a new output directory. The
deterministic forcing stream remains an explicit paired counterfactual
protocol.

Smoke command (relative paths from repository root):

```sh
target/debug/titan --analysis-only --analysis-substrate msfield \
  --base-dir analysis/metastable_20261001/frozen_step_58107 \
  --run-tag v10-msfield-fresh-20260930-02 \
  --corpus-dir analysis/matched_exposure_20261001/runs/corpus/a \
  --corpus-manifest analysis/matched_exposure_20261001/runs/corpus/a_manifest.json \
  --analysis-target-schedule analysis/matched_exposure_20261001/schedule_20261002.json \
  --dynamics-ablation 16 --ablation none --ablation coarse_hold \
  --analysis-stride 4 --analysis-dir analysis/frozen_v10_20261002/runs/smoke_16 \
  --analysis-terminal compact --analysis-no-audio --threads 2
```

## Exact-controller-RNG panel

The exact saved controller RNG is now restored in the frozen runner. The
256-chunk panel in `runs/panel_exact_rng_w0_256/` passed the no-op and parent
non-mutation gates. Coarse hold, GRU hold, and upper Morphic bypass had
full-audio waveform L2 values 0.03420, 0.03381, and 0.03340 relative to
baseline. Their RMS ratios were 0.9986, 0.9996, and 0.9986. The coarse-hold
field displacement remained small (final coarse relative L2 0.00367), and
Morphic bypass left persistent field state almost unchanged while changing
audio. Their full-audio spectral-distribution log RMS distances were only
0.0207, 0.0251, and 0.0219; waveform difference should not be mistaken for
a large timbral or structural difference. Closed-loop controller actions
also changed in some arms.

The pilot's **late onset** for coarse hold did not replicate. Under the exact
controller RNG, first and last 64-chunk waveform L2 were 0.02904 and
0.03651. A second exact-RNG trajectory snapshot after 128 warmup chunks
completed in `runs/panel_exact_rng_w128_256/`. Its coarse-hold first/last
64-chunk L2 were 0.00666/0.03638; full L2 was 0.02816, with RMS ratio
0.9986 and final coarse relative distance 0.00314. GRU hold and upper
Morphic bypass again produced sustained audio differences (full L2
0.02942/0.02753) while coarse-state relative distances stayed near
0.00010/0.000002. Across these two snapshots, specific recurrent/scale
operations reach sound, but onset varies and spectral-distribution changes
remain small. A separate listener must judge whether the differences are
coherent or useful. The feedback-zero arm again had identical audio
and field/GRU state but a different full world fingerprint. The exchange
arm again had large energy/field changes and remains confounded. Neither
panel has blind listening ratings or Suno outputs yet.

`prepare_suno_pilot.py` packaged the exact-RNG intact prime, coarse-hold
prime, a 0.683-second block-order control, and a shared-phase spectral
surrogate into four anonymous 21.845-second PCM16 inputs in
`runs/suno_exact_rng_pilot/`. No Suno API was called. The phase surrogate
preserved each channel's Fourier magnitude and the stereo cross-spectrum
before common edge fade and gain (relative errors below 2e-16). The time
control moved 31 of 32 blocks; its interior block waveform correlation was
1.0, but its boundary jump was 2.46 times the intact input. That seam
confound is recorded in `control_validation.json`; a preference against this
control cannot be attributed solely to lost long-range ordering. Input
hashes, anonymous answer key, receipt sheet, and blind rating sheet are
local. No generations or ratings exist yet.

Exact panel command:

```sh
target/release/titan --analysis-only --analysis-substrate msfield \
  --base-dir analysis/metastable_20261001/frozen_step_58107 \
  --run-tag v10-msfield-fresh-20260930-02 \
  --corpus-dir analysis/matched_exposure_20261001/runs/corpus/a \
  --corpus-manifest analysis/matched_exposure_20261001/runs/corpus/a_manifest.json \
  --analysis-target-schedule analysis/matched_exposure_20261001/schedule_20261002.json \
  --dynamics-ablation 256 --analysis-stride 16 \
  --analysis-dir analysis/frozen_v10_20261002/runs/panel_exact_rng_w0_256 \
  --analysis-terminal quiet --threads 2
```

The second snapshot used the same command with `--analysis-warmup 128`,
`--ablation coarse_hold --ablation gru_hold --ablation morphic_upper_bypass
--ablation feedback_zero`, and output
`analysis/frozen_v10_20261002/runs/panel_exact_rng_w128_256`. The runner
automatically included the independent no-op clone. The fixed target
schedule covered both warmup and evaluation intervals; no validation file
entered target selection.

## Temporal-credit audit

At 4,096 frames and 48,000 Hz, one chunk is 0.085333 seconds. Requested
`--bptt 64` is a 64-chunk optimizer averaging horizon, not one connected
graph. The current default tape is eight chunks (0.683 seconds); the six
matched corpus forks used four chunks (0.341 seconds). At tape boundaries,
fine, meso, coarse, and GRU state detach. On non-core tapes the model forward
also detaches field, GRU, and Morphic outputs before the decoder. Decoder
parameter gradients can still be local on those tapes. Synthesizer phase is
converted to host scalars, and learned decoder smoothing/history tensors
detach each chunk. The target episode lasts 256 chunks (21.845 seconds),
which is a data schedule, not differentiable credit. The long spectral loss
concatenates exactly one full tape; changing tape four to eight also changes
that loss's observation window. A four/eight comparison under the current
objective is therefore a combined credit-and-supervision experiment.

New run metadata exposes these horizons separately and counts nonfinite
loss segments, backward errors, empty optimizer horizons, nonfinite gradient
norms, and optimizer-step errors. A deterministic msfield gradient-lag test
passed: late fine loss reaches an earlier coarse state inside a tape and
stops after a detach. This tests graph connectivity only, not useful musical
credit. Future tape experiments must hold parent, target schedule, chunk
budget, optimizer horizon, core cadence, and loss observation window fixed
before being described as a pure temporal-credit test.
The probe does not measure gradient magnitude after many chunks or prove that
slow-scale parameters learn from delayed audible consequences.

## Gates, resource decision, and next test

All 92 Rust tests passed sequentially, including the gradient-lag test and
existing legacy/experimental schema gates. Both fixed-schedule Python
contract tests passed; Python compilation and `git diff --check` passed.
All frozen arms across both exact-RNG snapshots remained bounded, had no
nonfinite values or clipped audio in their horizon summaries, and wrote no
optimizer update. `freeze_readout.py` checked every stride-sampled target
file/frame against the fixed training schedule and wrote the compact tracked
`RESULTS.json` with output SHA-256s. The large WAVs, traces, and private
Suno answer key remain local under ignored `runs/`.

The four-versus-eight tape training comparison was **not launched**. The
device had about 3.4–3.9 GiB available RAM and only 0.5–0.8 GiB free swap
during this session; the prior eight-chunk training tape peaked near 4.5
GiB RSS. Hermes was also using this device for Titan Text. A matched
comparison would require two sequential 64-chunk-horizon training forks,
and the existing tape-sized long spectral loss makes it a combined
credit-and-supervision test. Run it only after a measured memory gate and
with both arms under the same preregistered objective and schedule. The
frozen causal results and manual blind audio panel are the immediate
decision evidence; neither supports promoting an escape controller.

## Interpretation gates

The analysis is frozen and paired but can still diverge through closed-loop
controller compensation. State displacement, audio displacement, RMS, stereo,
and spectral summaries must be reported separately. A gain or energy shift
alone is not organized audible change. A short effect is not evidence of
long-horizon structure. Manual blind listening and downstream Suno receipts
remain required before any practical-quality claim.
