# TITAN v10 msfield metastable-regime study

## Question and validity boundary

The mature v10 organism may occupy a low-dimensional, recurrent behavioral
manifold while its weights and decoder still retain substantial capacity. A
motif similarity rejection is insufficient evidence of global saturation:
the motif descriptor itself may saturate. The first experiment therefore
measures several internal and audible views without changing training or
ecological decisions. No regime or intervention may be selected with fixed
development or validation probes.

This branch starts from the **latest observed checkpoint at global step
58,107**. Its metadata says that its last continuation began at step 57,259;
the 58,107 set is the selected source for all forks. The original source is
`/sdcard/Download/TITAN_v10_msfield_fresh_20260930_02_cont/`. The five
checkpoint and metadata files were copied byte-for-byte into
`frozen_step_58107/` and made read-only. Their SHA-256 values and source paths
are in `frozen_step_58107_receipt.json`. The frozen set and all run output
directories are excluded from Git. Run forks copy and verify every parent file
again; no run points at the canonical source directory.

The prior experiment notes in `docs/MSFIELD_EXP.md`,
`analysis/msfield_exp_20260928/RESULTS.md`, the fresh-run receipts, and
`METRICS.md` were reviewed before implementation. The working branch was cut
from clean `experiment/v10-msfield-exp` at `c770e0e`; the tracked remote was
verified at the same commit. Hermes is working in a separate session and was
not interrupted.

## Observer implementation

`--regime-capture` is a v10-only opt-in flag. It samples every four global
chunks by default. The capture records pooled signed and RMS spatial maps
from fine, meso, and coarse fields; orthogonal seeded Walsh-Hadamard sketches
of GRU and Morphic L1/L4/L8/L12/L16 activations and deltas; decoder hidden
and control state; and post-DSP stereo, spectral bands, chroma, and temporal
modulation. Internal values are captured before that chunk's optimizer update.
It uses no runtime RNG and never feeds values back into model, optimizer,
world, controller, motif memory, or corpus selection.

`regime_archive.py` is a separate offline observer. It calibrates view scales
and an archive residence distance from a specified observer-only trajectory,
then replays with causal online normalization. It requires consecutive
samples before creating a regime, tracks visits and transitions, computes
trajectory covariance participation ratio and recurrence, and emits a JSONL
open-endedness trace plus graph JSON/DOT. It rejects any capture containing
development or validation fields. Its over-residence label is descriptive;
it cannot trigger an intervention in this phase. The trace marks candidate
interventions and no-op scores as absent, because no branch was run.

The observer uses the first half of its own capture to set the 90th percentile
of local lag-1/2/4 distances as its membership threshold. A time-shuffled
distance distribution tests separability but does not cap that threshold:
close returns are legitimate in a recurrent system. Equal total weight is
given to field, GRU, Morphic, decoder, and audio domains, regardless of how
many subviews each has. Candidate regions require four consecutive samples;
a **confirmed persistent regime** requires the empirically calibrated
long-dwell threshold and mostly healthy visits. The same first half sets a
descriptive trap-score threshold and per-component reference quantiles. A
trap label additionally requires three independent dynamic indicators over a
long confirmation window, so high confidence, motif rejection, and dwell
cannot trigger it by themselves. Claims from the calibration interval are
in-sample. Later comparisons must hold this calibration fixed and use matched
forks from the original frozen checkpoint. `regime_nulls.py` checks whether
independent view shuffles, short-block shuffles, or matched white Gaussian
noise produce as many apparently persistent regimes. If they do, regime count
alone is not evidence of organized discovery.

## Geometry methods and initial observations

`checkpoint_geometry.py` memory-maps SafeTensors and analyzes model matrices
by group and Morphic layer. It reports singular spectra summaries, stable
and participation ranks, spectral entropy, condition number, top-k energy,
Adam moment/update pressure, and adjacent Morphic principal-subspace overlap.
Every reported matrix has entry-shuffled and matched-Gaussian nulls. Optional
capture analysis adds activation-sketch covariance dimension and CKA. The
world exporter reads the checksummed v10 world and writes raw fine/meso/coarse
fields for spatial lag correlation, 2D power, box occupancy, pixel shuffles,
and phase-randomized power-preserving controls. Box occupancy is not a
fractal-dimension estimate at these grid sizes.

At step 58,107, the 82 analyzed matrices span all 16,826,171 parameters
across 21 groups. The first Morphic input matrix has participation rank
143.6 versus entry-shuffled nulls 256.7/255.8; the L16 counterpart has
31.7 versus 255.6/256.2. The GRU recurrent candidate matrix has 205.7
versus 255.3/255.6. This demonstrates matrix structure relative to simple
nulls, not unused capacity or audio quality. Adjacent Morphic top-16 input
subspaces have median squared overlap about 0.177. A single checkpoint cannot
determine which parameter groups are still plastic; matched future deltas and
activations are needed.

The channel-RMS field map shows substantial nearest-neighbor correlation at
all three scales (coarse 0.760, meso 0.890, fine 0.812), while eight pixel
shuffles per scale are near zero. The initial phase-null implementation was
found to mishandle Fourier conjugate symmetry and was corrected before the
eight-null report was finalized. Phase-randomized maps preserve Fourier
power; their box occupancy broadly overlaps the observed maps. Thus the
snapshot supports spatial organization against pixel shuffling but provides
no defensible fractal or higher-order topology claim.

## Completed first observer continuation

The isolated observer-only fork ran from step 58,107 to 59,513 for 1,406
chunks, 120 rendered seconds, and 22 AdamW updates. L16 remained active.
Its 352 descriptor samples are in the deterministic, source-hashed
`observer_capture.jsonl.gz`; the receipt records the binary and metadata
hashes. The source checkpoint hashes were rechecked after the run and remain
unchanged. The final health was 0.782, stagnation 0.272, confidence 0.98;
motifs remained 83/512 despite 88 new candidates. This is descriptive, not
evidence of a completed capacity limit.

The calibrated archive creates four candidate regions, of which three meet
the calibrated long-dwell/health confirmation rule. They occur in sequence
with three transitions and **no revisit** in this 1,406-chunk window. Each
transition falls within an existing target-audio episode rather than exactly
at a sampled target switch, but exogenous corpus forcing remains a confound.
Eight repeats each of independent-view time shuffle, four-sample block
shuffle, and matched white Gaussian noise produce zero confirmed long-lived
regimes. A preliminary four-sample-only archive count was gameable by block
shuffling (four observed versus four to six shuffled) and is preserved only
under the ignored `runs/preliminary_archive_v1/` and `_v2/` directories.
Confirmed dwell is the defensible statistic; raw candidate count is not.

The stricter detector records **zero over-residence events** in this first
window. Its preliminary score had flagged 50 samples mostly because dwell,
motif rejection, and confidence were high, while dynamic novelty, trajectory
dimension, topology, and controller behavior did not corroborate a trap.
That preliminary false-positive interpretation was rejected before any
intervention could be enabled. The model may be developing healthy stable
regions; there is no evidence yet that escape should fire.

Thirty-two-dimensional Morphic activation sketches have raw covariance
participation ranks around 1.1–1.2 and first-difference ranks around 2.1–2.7,
well below independent-channel shuffle nulls. These values describe the
continuing **trained** trajectory, not a frozen source-checkpoint rollout or
full 512D activation rank. Raw adjacent-layer activation CKA is 0.99, but
first-difference layer-delta CKA falls to 0.48–0.70 at representative depth
pairs. Shared coordinates therefore do not justify a layer-diversity penalty.

Across the same 22 optimizer updates, relative parameter L2 changes were
0.000315 for GRU, 0.0115 for msfield, roughly 0.0057–0.0212 across Morphic
layers, and 0.0728 for the temporal decoder. This supports comparatively
settled GRU weights and substantial decoder plasticity over this interval;
it does not establish recoverable unused behavioral capacity.

## Gates and commands

The one-chunk old-binary / new-disabled / new-capture regression from three
identical step-58,107 forks reached step 58,108 in every arm. World and audio
files were byte-identical. The new disabled arm differed in model weights by
at most 1.49e-8 and Adam moments by at most 2.98e-8; the capture-enabled arm
was byte-identical to the old binary for those files too. This supports
behavioral equivalence within normal f32 reduction variation; it does not
prove arbitrary-length bitwise identity. Exact arguments and hashes are in
`regression_gate_result.json`.

Run the bounded observer-only continuation from its verified fork:

```sh
analysis/metastable_20261001/runs/observer_step_58107/titan_run_binary \
  --base-dir analysis/metastable_20261001/runs/observer_step_58107 \
  --corpus-dir /sdcard/Download/OLD_WAVS \
  --corpus-manifest /sdcard/Download/titan_corpus_manifest_v7_sml.json \
  --substrate msfield --run-tag v10-msfield-fresh-20260930-02 \
  --duration 120 --threads 4 --seed 44 --lr 0.00045 \
  --bptt 64 --core-update-every 1 --morph-layers 16 \
  --morph-width 512 --max-morph-depth 16 --motif-capacity 512 \
  --regime-capture --regime-stride 4
```

Before this run, `free -h` showed 4.3 GiB available RAM and 1.2 GiB free
swap; `df -h` showed 27 GiB free storage. The release build was single-worker
and completed. The run itself is one process; Hermes remains separate. Use
the run metadata's `telemetry.regime_capture` path to calibrate and analyze,
and retain that capture with its SHA-256 receipt.

The second measurement-only 120-second continuation reached step 60,919
from an exact step-59,513 snapshot. It added 1,406 chunks, 22 AdamW updates,
and one motif (83→84). Health was 0.775 and stagnation 0.274 at completion.
The fixed archive over both blocks has five confirmed sequential regions,
four transitions, **zero revisits**, and **zero supported trap events**; its
longest region dwell is 944 chunks. Median trajectory participation increased
from 9.52 in the first block to 11.25 in the second. This is evidence against
declaring permanent attractor imprisonment on this measured interval, while
the absence of revisits means these regions are not yet demonstrated as an
ecology of reusable metastable worlds. The first and second raw capture
receipts preserve exact hashes. `archive_resume_gate.json` confirms 703
sample records and byte-equivalent structured outputs/state for uninterrupted
versus JSON save/resume replay. The combined transition graph is also in
SVG/DOT/JSON form.

No attractor escape, diversity regularizer, or fresh lineage has been enabled.
A matched baseline/controller fork remains gated on a detector event that
survives these measurements and nulls. A separate corpus-limit A/B/C
contrast was preregistered in `CORPUS_FORK_PLAN.md` and runs with no behavior
change.

## Additive corpus contrast: exposure failure, not a capacity result

The two-block A/B/C contrast completed 2,812 chunks per arm from the exact
same step-58,107 source, with 44 cumulative new optimizer updates each. A
retained the small corpus (23 train families); B added six genuinely new
families (29 total); C added six mild-EQ variants within existing families
(23 total). Development and strict validation files were byte-identical.
The original small corpus itself has four byte-duplicate pairs, all confined
to a single role and family per pair.

The sampled target traces show **zero B new-family episodes** across both
blocks, but two distinct C added-variant episodes. B therefore fails the
preregistered exposure gate. The confirmed long-lived archive counts are
A/B/C = 5/4/4 with no revisits; these differences cannot be attributed to
new-family audio because B did not encounter it in the sampled trace.
Second-block median trajectory participation was 11.25/11.32/11.81, health
0.777/0.787/0.783, strict validation spectral loss 0.783/0.770/0.776, and
final motif counts 84/83/84. All final prime WAVs had zero clipped samples.
Even without new-family exposure, B's msfield net weight delta and field
state differed from A, demonstrating that the altered target schedule alone
is a material confound. None of these arm differences establishes model
capacity saturation or a benefit from new families.

`ACUTE_EXPOSURE_PLAN.md` specifies a separate, short six-family ecology
replacement to guarantee new-family exposure. It starts again from the
unchanged frozen mature checkpoint and is not pooled with the additive test.

## Guaranteed-exposure follow-up: optimizer response without more regimes

The separate acute A/C/B arms each completed 703 chunks from step 58,107 to
58,810 and 11 AdamW updates. Their sampled target traces contain 3/2/2
distinct training families, so the preregistered exposure gate passes. A
used six existing originals, C used their six mild-EQ variants, and B used
six genuinely new families. All three retained strict, byte-identical
development/validation probes and stayed numerically bounded.

All three arms confirmed **one** long-lived regime and **zero** revisits or
trap events. Median whole-run trajectory participation was A/C/B =
9.00/8.84/8.07; second-half medians were 8.35/9.45/8.08. The new-family arm
did **not** expand the measured field, GRU, or Morphic trajectory dimension
or discover an extra persistent regime in this short window. Its net
msfield weight L2 change was 0.0124 relative to the starting weights versus
0.0090/0.0084 in A/C; temporal decoder change was 0.0421 versus
0.0347/0.0346. GRU change remained small in all arms, though B's 0.0004 was
larger than A/C's roughly 0.0001. Full meso-field state displacement from
the same starting world was 0.487 in B versus 0.631/0.525 in A/C. These
measurements support stronger *update pressure* under new material, not
greater organized regime exploration.

B's median gradient norm was 4.58 versus 2.63/2.32 for A/C, and 36.6% of
sampled optimizer trace rows had clip scale below one versus zero in A/C.
Median health stayed 0.787 versus 0.780/0.778. Strict validation spectral
loss was 0.790 versus 0.802/0.786, while validation chroma loss was about
0.91 in all arms. These one-seed differences are small and not a quality
claim. All three final prime WAVs had zero clipped samples and similar
stereo correlation (roughly +0.87). Their subjective coherence, beauty,
strangeness, and Suno-conditioning usefulness remain untested. A local blind
panel with a private answer key is in the ignored
`runs/acute_listening_panel/` directory.

The new-family result is a **negative immediate-response result at 703
chunks**, not proof of global capacity saturation. The target sources differ
by composition and sound; schedules are not paired across different family
IDs; only two new B families were actually sampled; and eleven optimizer
updates may be too few for latent ecology reorganization. Stronger
counterfactual escapes and a layer-delta regularizer remain gated. The
next discriminating test is longer, exposure-guaranteed, multi-seed training
with target schedules matched where possible and predeclared fixed-probe,
field, and blinded-audio endpoints. If a supported over-residence event
emerges, then an opt-in no-op-versus-structured-escape branch test can be
calibrated and run without touching the frozen parent.

Exact command arrays, run IDs, binary/checkpoint/corpus hashes, and the
source-unchanged verification for all nine completed runs are in
`run_receipts.json`. The full Rust suite passed 90/90 tests; four Python
observer/geometry tests passed; `cargo fmt --check`, `git diff --check`, and
Python compilation passed. The one-chunk disabled-path regression is in
`regression_gate_result.json`; old and new world/audio bytes matched, while
the one new-disabled model/Adam write differed by at most f32-scale
1.49e-8/2.98e-8. No white-noise escape, counterfactual branch rollout,
homeostatic novelty controller, or fresh L1→L16 lineage was run.

## Matched-schedule continuation

The longer two-schedule-seed test was completed on the dedicated
`experiment/v10-matched-exposure` branch. Its exogenous target slots and
source frames matched across A/C/B within each seed. New-family B again
raised weight-update pressure and severe clipping but produced fewer
persistent regions than familiar A under both schedules. See
`analysis/matched_exposure_20261001/RESULTS.md` for controls, hashes, nulls,
and interpretation limits.
