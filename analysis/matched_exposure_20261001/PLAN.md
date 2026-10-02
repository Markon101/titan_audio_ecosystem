# TITAN v10 fixed-schedule longer exposure test

## Why this fork

The additive six-new-family arm never selected a new source in its sampled
target trace. A short guaranteed-exposure test selected two new families and
produced larger gradient/weight updates, but no extra persistent region or
broader measured field/Morphic trajectory in 703 chunks. The next comparison
must give every arm the same exogenous episode slots and source-frame offsets,
extend the budget, and repeat with more than one schedule seed. No
over-residence event justified enabling an escape controller.

This branch starts from the read-only step-58,107 `TITANM10` checkpoint
receipt in `analysis/metastable_20261001/frozen_step_58107_receipt.json`.
Each of six runs receives its **own** byte-verified model, world, AdamW,
Morphic state, and metadata copy. Canonical source files are never targets of
a save. The prior observer calibration remains fixed; no probe or outcome
selects targets or thresholds.

## Inputs and schedule

Three arms use six training aliases, `slot_00.wav` through `slot_05.wav`,
with the same sorted family IDs in every manifest:

| Arm | Audio behind each slot |
| --- | --- |
| A | Existing original training audio |
| C | Mild EQ variant of the corresponding A file |
| B | Genuinely new-family audio, duration-paired to that slot |

The six development and five strict validation files are symlinked from the
same byte-verified original corpus into all arms. Training source SHA-256
identities, alias mapping, manifest hashes, frame counts, and held-out hashes
are in `matched_exposure_receipt.json`. All WAVs are 48 kHz stereo PCM16.

Two schedules, seeds `20261002` and `20261003`, each contain four distinct
256-chunk episodes (1,024 chunks total, approximately 87.4 rendered seconds).
They use different fixed slot orders and start frames; the pair covers all
six source slots. For each seed, A/C/B use the **identical schedule JSON
bytes**. Start frames are chunk aligned and chosen below the shortest file
length at the paired slot. The opt-in `--target-schedule` path selects its
training target by absolute global step and does not consume Titan's runtime
RNG. A scheduled target must be a single-file training slot; development and
validation slots are rejected. At analysis time, sampled `target_file`,
`target_frame`, and `target_chunks_left` must match across arms for a seed.

These are two **schedule seeds on one mature model/world**, not two
independently matured model seeds. That limits generalization claims. The
execution order is seed 20261002 A→C→B, then seed 20261003 C→B→A, one TITAN
process at a time.

## Training budget and memory gate

All six arms request LR `0.00045`, BPTT horizon `64`, core update every tape,
L16/16 at width 512, motif capacity 512, four worker threads, and
measurement-only `--regime-capture --regime-stride 4`. The previous eight-chunk
autograd tape peaked near 4.5 GiB RSS; current device memory is tighter.
The new **opt-in** `--max-autograd-tape 4` caps the actual graph at four
chunks while keeping the 64-chunk optimizer horizon. It changes temporal
credit assignment versus the earlier eight-chunk runs and will be compared
only *within* this matched six-arm test. A bounded smoke run from a disposable
fork must show adequate free RAM and finite checkpoints before the six runs.
If the device cannot sustain the matched setting safely, stop before costly
training and report the resource gate; do not silently lower it for one arm.

## Fixed outcome and interpretation rules

For each schedule seed, compare B with **both** A and C on:

1. Confirmed persistent regions, dwell, transitions, revisits, and
   second-half trajectory participation with the frozen A-observer
   calibration; candidate-region count alone is excluded.
2. Per-scale pooled field, GRU, Morphic activation/delta, decoder, and audio
   view dimensions; source-to-final full field displacement and topology
   diversity; model parameter-group deltas and Adam update count.
3. Motif additions/rejections, activity health, rail/NaN events, gradient
   norms and clipping, and byte-identical fixed development/validation probe
   inputs with their spectral/chroma losses.
4. Final prime WAVs in a blinded listening panel. Beauty, coherence, and
   downstream Suno usefulness require listener ratings and matched uploads;
   objective descriptors do not establish them.

A directional ecology signal requires B to exceed **both** familiar controls
in persistent regions and field-including trajectory expansion in **both**
schedule seeds while remaining bounded and without a material fixed-probe
regression. A decoder/audio-only difference has a narrower interpretation.
High clipping in B makes a null learning response ambiguous. At n=2
schedules, no significance or global capacity claim is planned. If B again
shows update pressure without organized regime growth, the next experiment
should isolate optimizer/credit limitations before turning on anti-attractor
behavior or enlarging the model.
