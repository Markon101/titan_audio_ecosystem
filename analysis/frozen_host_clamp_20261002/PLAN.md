# Frozen v10 control/energy replay, 2026-10-02

## Decision question

The first frozen panels showed audible divergence under coarse-state hold,
GRU hold, and upper Morphic bypass. Closed-loop controller actions sometimes
changed, and waveform phase drift can inflate waveform distance without
producing a useful musical change. This follow-up asks whether those learned
pathways still change sound when their two immediate host inputs are fixed.

## Paired design

Use the unchanged archived step-58,107 TITANM10 model/world/morph/optimizer
set and fixed train-only schedule 20261002. No optimizer or backward pass is
constructed. Run the same 256-chunk windows at warmup 0 and 128 as in
`analysis/frozen_v10_20261002/`. In every intervention arm, replay the
baseline's realized stochastic forcing. In the new replay arms, also feed
the baseline's exact per-chunk `SynthesisControl` and forward energy scalar
to the frozen model. Controller choice and ecological host bookkeeping can
still evolve, but they cannot change those replayed forward inputs. This is
an artificial causal control, not an ordinary continuation.

Per snapshot, request:

1. independent ordinary no-op clone, automatically inserted;
2. `none_control_energy_replay`, an exact replay no-op gate;
3. `coarse_hold_control_energy_replay`;
4. `gru_hold_control_energy_replay`;
5. `morphic_upper_bypass_control_energy_replay`.

First verify that the new unmodified baseline audio hashes match the
previous exact-RNG panel's baseline byte for byte, and both no-op arms
match their own baseline raw/post-DSP audio and final world fingerprint.
Then compare full, first-64, and last-64 chunk waveform RMS difference,
spectral distribution, envelope modulation, stereo geometry, field/GRU
distances, action changes, boundedness, clipping, and source hashes. Do not
use validation probes to select conditions. A surviving difference shows
direct or internal-feedback reach to sound under the declared host clamp;
it does not establish coherence or downstream usefulness. A disappearing
difference would point to host control or energy mediation and motivate a
separate control-only versus energy-only test.

The old time-reorder Suno control has seam artifacts and is not part of this
mechanistic decision. Do not promote an attractor-escape mechanism from it.

## Resource gate

Before the optimized build or either run, inspect live processes, available
RAM, swap, and storage. Run sequentially with two analysis threads. The
previous frozen 256-chunk panel used about 0.3 GiB RSS; no new training tape
is launched. Preserve all prior outputs and use new dedicated directories.
