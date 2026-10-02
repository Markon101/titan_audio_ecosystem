# Frozen v10 host-input clamp, 2026-10-02

## Hypothesis and method

The prior exact-RNG frozen panels showed audio changes from coarse-field hold,
GRU hold, and upper Morphic bypass. Because controller actions changed in
those closed-loop arms, the waveform differences could be mostly host
amplification. This experiment holds the model's per-chunk synthesis control
and forward energy scalar to the corresponding intact baseline values. It
also retains the prior replay of realized radiation, shear, kick, gains, and
fixed train-only target frames. Parameters and morphology stay frozen; there
is no optimizer or backward pass. Controller choice and other host bookkeeping
still run, so the clamp isolates the declared forward inputs, not every
possible ecological feedback route.

Both snapshots start from the archived, read-only TITANM10 step-58,107
model/world/morph/AdamW set. Warmup 0 and 128, each followed by 256 rendered
chunks, match the previous exact-RNG panels. They are two trajectory
snapshots of one mature model/world, not independent model seeds. The source
schedule is identical. No validation file is used to choose conditions or
targets. `PLAN.md` was written before the longer runs.

## Exact controls

The new untouched baseline raw and post-DSP WAVs were byte-identical to the
previous exact-RNG baseline in **both** snapshots. The ordinary independent
no-op and `none_control_energy_replay` each matched their baseline's raw and
post-DSP audio and final world fingerprint exactly. The replay no-op is a
strong check that the new control/energy path is inert when fed its own
baseline values. All canonical input files were rehashed and unchanged at
the end of both runs. Each arm had zero optimizer updates, finite and bounded
field/audio summaries, and zero audio clipping. Every stride-sampled target
file/frame matched the fixed training schedule (85 rows per panel).

## Results

Waveform L2 here is RMS sample difference over the entire 21.845-second
post-DSP WAV. The ratio compares the new host-clamped arm with its prior
closed-loop arm under the same parent and schedule; it is **not** an additive
causal fraction. The time-local spectral column is the median
multiresolution log-spectral distance over stride-sampled chunks. Exact no-op
is zero.

| Warmup | Intervention | Closed-loop L2 | Clamped L2 | Ratio | Clamped local spectral distance | Clamped first/last 64-chunk L2 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Coarse hold | 0.03420 | 0.01347 | 0.394 | 0.2809 | 0.00016 / 0.02319 |
| 0 | GRU hold | 0.03381 | 0.00338 | 0.100 | 0.0668 | 0.00007 / 0.00610 |
| 0 | Upper Morphic bypass | 0.03340 | 0.00328 | 0.098 | 0.1919 | 0.00332 / 0.00327 |
| 128 | Coarse hold | 0.02816 | 0.00931 | 0.331 | 0.1818 | 0.00023 / 0.01603 |
| 128 | GRU hold | 0.02942 | 0.00498 | 0.169 | 0.0785 | 0.00004 / 0.00923 |
| 128 | Upper Morphic bypass | 0.02753 | 0.00332 | 0.121 | 0.1816 | 0.00327 / 0.00337 |

The clamp substantially reduced all three audio differences but did not
remove them. Coarse hold retains the largest and growing late effect in both
snapshots. Upper Morphic bypass leaves a smaller, fairly immediate audio
effect even though its persistent field state is nearly unchanged; this is
consistent with a direct readout contribution. GRU hold retains a small
effect that grows over the window. The closed-loop system strongly amplifies
GRU/Morphic perturbations through the host control/energy route, but this
experiment does not uniquely assign the amplification between control and
energy. A separate control-only versus energy-only fork would do that.

Local spectral differences under the clamp exceed the no-op, but phase drift
and window alignment can also move short-time spectral metrics. They are much
smaller than in the prior closed-loop arms. Neither distance metric measures
coherence, beauty, useful morphing, or Suno conditioning. The existing manual
blind audio pilot has not been rated, and no Suno outputs exist. Promoting an
attractor-escape controller is not justified by these results.

## Better temporal-order listening control

The first manual Suno pilot's fixed 0.683-second block shuffle had boundary
jumps 2.46 times those at corresponding regular positions in the intact
input. A new isolated pilot, `runs/suno_quietcut_pilot/`, chooses each of the
31 internal cut points at the lowest stereo boundary energy within 4,096
frames of its nominal position, then permutes the 32 resulting segments.
All original samples remain in the input and the duration stays exactly
21.845 seconds. Thirty-one of 32 segments moved; median interior segment
waveform correlation after matched processing was 0.999999995. Its boundary
jump is 0.262 times the intact input's regular-boundary jump. It has less
join-click confounding than the fixed-cut control.

The tradeoff is that the time-averaged log-spectral RMS distance from intact
rose from 0.0198 to 0.0408. Both changes are explicitly measured in
`control_validation.json` and tracked `RESULTS.json`. The phase-randomized
surrogate still uses a common phase change across channels and preserves
preprocessing channel magnitudes and stereo cross-spectrum to numerical
precision. This is a better *pilot* control, not a perfect isolation of
long-range order. No file was uploaded and no Suno output or listening
rating exists. Use the new anonymous `uploads/` and its private answer key
for any manual pilot; keep the earlier pack as a documented control audit.

## Reproduction and resource envelope

From the repository root, the warmup-zero command was:

```sh
target/release/titan --analysis-only --analysis-substrate msfield \
  --base-dir analysis/metastable_20261001/frozen_step_58107 \
  --run-tag v10-msfield-fresh-20260930-02 \
  --corpus-dir analysis/matched_exposure_20261001/runs/corpus/a \
  --corpus-manifest analysis/matched_exposure_20261001/runs/corpus/a_manifest.json \
  --analysis-target-schedule analysis/matched_exposure_20261001/schedule_20261002.json \
  --dynamics-ablation 256 \
  --ablation none_control_energy_replay \
  --ablation coarse_hold_control_energy_replay \
  --ablation gru_hold_control_energy_replay \
  --ablation morphic_upper_bypass_control_energy_replay \
  --analysis-stride 16 \
  --analysis-dir analysis/frozen_host_clamp_20261002/runs/w0_256 \
  --analysis-terminal quiet --threads 2
```

For the second snapshot, add `--analysis-warmup 128` and change the output
directory to `runs/w128_256`. The runner inserts the ordinary `none` arm.
`freeze_host_readout.py` verifies baseline parity, no-op equivalence,
schedule frames, parent non-mutation, bounds, and output hashes before
writing tracked `RESULTS.json`. Full audio and sidecars remain in local
ignored `runs/`. The device was checked before each build/run; no concurrent
Titan training ran. Frozen inference used about 0.3 GiB RSS. Free swap fell
very low during the first full window, but available RAM remained above
3.4 GiB; no larger training tape was attempted.
All 92 Rust tests passed sequentially after the opt-in change; Rust format,
Python compilation, and `git diff --check` passed. The two 256-chunk runtime
panels provide the stronger no-op, resume-source, target-frame, and
non-mutation gates.

## Decision

The learned field and Morphic pathways have measurable audible influence
under a paired host-input clamp. Much of the larger closed-loop divergence
comes through host amplification. The next decisive evidence is blind
listening and the already packaged manual Suno pilot; if its controls perform
equally well, internal causal reach alone is insufficient for the practical
objective. A control-only versus energy-only clamp can further localize the
host mediator without new training if that distinction changes an
engineering decision.
