# TITAN Audio Ecosystem v9

TITAN is an adaptive, deterministic neural-cellular-automata audio ecosystem
for sustained CPU training and rendering in Termux. v9 turns its fixed-size CA
storage into a dynamically expanding folded manifold: morphic growth activates
up to four cyclic depth sheets, each with 16 hidden features, while learned
near and far neighborhood rings operate across a non-orientable Klein-bottle
seam. The physical tensors remain phone-sized; logical dimensionality grows
with the organism.

v9 writes independent artifacts and never overwrites v7/v8 state:

- `titan_model_v9.safetensors`
- `titan_optimizer_v9.safetensors`
- `titan_world_v9.bin`
- `titan_morph_state_v9.json`
- `titan_run_metadata_v9.json`

The corpus classification remains in `titan_corpus_manifest_v7.json`. Its
train/development/validation family semantics are data policy rather than model
architecture, so v9 deliberately reuses it.

## Build and run

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --locked
./target/release/titan --base-dir /sdcard/Download --duration 240 --seed 42 --fresh-model
```

Training WAVs live in `OLD_WAVS` under the selected base directory. Use
`./target/release/titan --help` for all checkpoint, reset, thread, learning-rate,
run-tag, BPTT, and core-update options.

Every invocation gives only its finalized audio WAVs a random 12-hex-character
suffix, for example `rust_ecosystem_out_a13f09c2de77.wav` and
`titan_prime_60s_a13f09c2de77.wav`. Both files share the same hash so they remain
associated. Model, optimizer, world, telemetry, and metadata names stay stable
for automatic checkpoint continuation.

To seed v9 from the latest v8 model while retaining the source artifacts:

```bash
./target/release/titan \
  --base-dir /sdcard/Download \
  --import-model /sdcard/Download/titan_model_v8_v8-real-01.safetensors \
  --run-tag v9-manifold-01 \
  --morph-layers 24 \
  --max-morph-depth 24 \
  --motif-capacity 512
```

The near-ring, recurrent, morphic, and decoder tensors with matching shapes are
retained. The new far-ring tensors are initialized deterministically. Because
the spatial rule and world topology changed, v8 world state and Adam moments
are intentionally not resumed. Use `--import-model` only for this first v9 run;
continue normally afterward.

## Architecture

The organism has two 64-state neural cellular automata:

- a 64x64 micro field updated every chunk;
- a 32x32 macro field eligible to evolve on a four-chunk clock.

The state axis is interpreted as four dormant-or-active depth sheets with 16
hidden features per sheet. At MorphicStack depths L01, L04, L07, and L10 the
logical manifold has one, two, three, and four active sheets respectively.
Active sheets have cyclic nearest-depth coupling; inactive sheets are projected
to zero. This makes the field a folded 3D `Klein bottle x S1` manifold once
more than one sheet is active without allocating a dense 64x64x64 spatial
volume.

Each CA has independently learned depthwise 3x3 near and dilation-2 far rings,
followed by learned 64-to-128 and 128-to-64 pointwise feature mixing. The far
ring fades in over L01--L04 rather than appearing abruptly. Horizontal wrap
reverses the vertical coordinate, producing the Klein-bottle seam; vertical
wrap remains periodic. Precomputed deterministic cell-clock masks keep the
existing asynchronous update schedule allocation-free.

Global micro-channel means and a 64-dimensional episodic attention readout feed
a 512-unit GRU. Its output passes through an RMS-normalized, Swish residual
MorphicStack with 12 blocks of width 512 by default. The audible decoder also
reads the fields directly: 64 micro tokens from an 8x8 grid and 16 macro tokens
from a 4x4 grid are projected and attended by recurrent memory.

## Runtime MorphicStack variants

The physical MorphicStack is selected when the process starts without changing
its 512-dimensional external interface:

```bash
./target/release/titan \
  --base-dir /sdcard/Download \
  --morph-layers 16 \
  --morph-width 768 \
  --max-morph-depth 16 \
  --motif-capacity 256
```

`--morph-layers N` constructs 1--64 blocks. `--morph-width N` selects an
internal width from 64 through 4096 in multiples of 64. `--morph-depth N`
explicitly overrides the active depth restored from the world, while
`--max-morph-depth N` remains the adaptive-growth ceiling. Active depth and
physical capacity are separate: dormant blocks occupy checkpoint and optimizer
memory but do not execute in the forward pass.

Motif memory is independently runtime-sized with `--motif-capacity N` from 1
through 4096 (default 64). Increasing it on a resumed world retains every saved
motif. Decreasing it keeps the newest `N` motifs and reports how many oldest
entries were discarded. The selected capacity is recorded in run metadata;
the motifs themselves continue to travel with the world checkpoint.

Resizing preserves block indices. Growing a checkpoint from 12 to 16 blocks
copies blocks 0--11 and their AdamW moments exactly, then appends blocks 12--15
with zero optimizer moments. A new block's output projection is zero-initialized
so activating it cannot immediately change the parent model's output. Widening
copies the old hidden units and zeroes only the new output columns, which is
also function-preserving at migration time.

Use `--import-model` with a distinct `--run-tag` to retain the complete parent
checkpoint set while creating an isolated variant:

```bash
./target/release/titan \
  --base-dir /sdcard/Download \
  --import-model /sdcard/Download/titan_model_v9.safetensors \
  --run-tag m16w768 \
  --morph-layers 16 \
  --morph-width 768
```

This writes `titan_model_v9_m16w768.safetensors` together with matching tagged
optimizer, world, morph-state, and telemetry artifacts. `--model PATH` remains
available when only the model path needs to be selected explicitly.

Shrinking retains the block prefix and the overlapping hidden units. Removing
only inactive tail blocks is output-preserving. Truncating active blocks or
hidden units is necessarily a lossy compression, so the runtime reports the
truncated depth and retaining the parent checkpoint is recommended. Morph-only
resizes keep the CA, recurrent, DSP, and ecological world state compatible.

The resulting context is decoded into 32 temporal control frames per 4,096
samples. Six low-rate residual blocks produce independent time-varying controls
for carrier phase and drive, 32 regional partials, differentiable mid/side
excitation, channel openness, KAN drive, and mid/side dynamics. Linear
interpolation expands those controls to sample rate. Phase-continuous carrier,
FM, auxiliary and modal oscillators remain the primary sound source.

Separate left and right eight-basis KAN-inspired sinusoidal wavefolders remain
active. Global pan is a tightly bounded +/-0.10 residual after independently
decoded left/right and mid/side structure. Its softsign-style map retains a
useful gradient for saturated older checkpoints, and a bounded centering loss
prevents this residual from becoming a permanent unequal-gain-mono shortcut
without consuming the full gradient budget. The learned side-width control
uses the same polynomial-tail map over 0.05--0.50. This keeps existing tensor
shapes checkpoint-compatible while preventing a saturated head from erasing
all independently decoded side structure or forcing anti-correlation. Both
scalar heads receive a parameter-free RMS-normalized hidden state so their
step size cannot scale with deep residual-state magnitude. The first resume
after this control change resets only the pan/width heads' Adam moments; their
learned weights and all other optimizer state remain intact.

## Online training and memory behavior

Target audio is indexed by manifest, not loaded into RAM. A coherent source
episode is seek-decoded in blocks of at most 16 chunks (about 0.5 MiB of stereo
FP32 audio), amortizing file-open and resampling work while preserving constant
memory. Generated Titan audio remains quarantined from the training corpus.

Refresh the manifest after adding WAVs without changing established
train/development/validation assignments:

```bash
./target/release/titan \
  --base-dir /sdcard/Download \
  --refresh-corpus-manifest \
  --manifest-only
```

New variants inherit the safest existing role for their family; new families
default to training, and recognizable Titan outputs are added as `exclude`.
Missing files remain recorded by default so temporarily moved corpus material
does not erase data policy. Add `--prune-missing` to remove those stale entries.

For a deliberate from-scratch reclassification, use
`--rebuild-corpus-manifest --manifest-only`. Rebuild writes the previous
manifest to a timestamped `.bak.*` file before atomically replacing it. A plain
`--manifest-only` creates or repairs the selected manifest and exits. All these
operations respect `--corpus-manifest PATH`; omit `--manifest-only` only when
the same invocation should continue into training.

`--bptt` controls the optimizer's gradient-averaging horizon; the folded CA's
differentiable tape remains capped at 8 chunks on mobile. `--core-update-every N` performs full
CA/GRU/Morphic backward on one of every N tapes and trains the decoder on the
intervening tapes. The default is 4. Use `1` for full end-to-end BPTT on every
tape when maximum adaptation matters more than speed.

After the one-time v8 import above has produced a v9 checkpoint, continue with:

```bash
./target/release/titan \
  --base-dir /sdcard/Download \
  --duration 2500 \
  --threads 7 \
  --bptt 16 \
  --core-update-every 16 \
  --run-tag v9-manifold-01 \
  --max-morph-depth 24 \
  --morph-layers 24 \
  --motif-capacity 512
```

This cadence favors the younger audible decoder. Use `--core-update-every 8`
when a more even decoder/core update balance matters more than throughput. Do
not add `--import-model` or `--fresh-decoder` when continuing the v9 checkpoint.
Keep the same `--run-tag` as the import run: changing or omitting it selects a
different checkpoint namespace and starts a new world when that namespace has
no v9 checkpoint.

The phase profiler reports model forward, target I/O, loss/metrics, backward,
optimizer, output I/O and checkpoint time per completed chunk. These values are
also written to run metadata, so thread counts and architecture changes can be
compared after thermal soak rather than from cold-start speed.

Large topology and scalar trace records are streamed during the run and
converted to their stable CSV formats at finalization. Audio is streamed to a
temporary FP32 file and normalized in a second pass. Full, mutually consistent
model/optimizer/world checkpoints are written every 2,048 chunks and on clean
shutdown, limiting storage traffic from the larger model.

## Supervision and verification

Supervision covers 20 Hz--20 kHz at 1,024-, 4,096-, and tape scales, together
with relative band energy, chroma/pitch salience, onset envelopes, modulation,
multi-lag recurrence, level, chunk seams, and correlation-aware stereo
geometry. Target phase is not supplied to the renderer.

Generated manifests reserve family-disjoint development and validation sets
when the corpus is large enough. User manifests without validation families
remain runnable, but their fallback probe is explicitly marked non-strict in
run metadata and must not be treated as held-out evidence.

Telemetry semantics and their scientific limitations are documented in
[`METRICS.md`](METRICS.md). The implemented equations, evidence status, and
attractor-test criteria are documented in [`math.md`](math.md). Decoder-side
research candidates that were deliberately not implemented in v9 are recorded
in [`DECODER_RESEARCH.md`](DECODER_RESEARCH.md). Verify a build
with:

```bash
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```

`--locked` makes Cargo use the dependency graph already recorded in
`Cargo.lock` and fail rather than silently resolving different versions.

## Frozen scientific instrumentation

Phase 1 adds a sidecar-only scientific harness around Audio v9. It does not
change model tensors, forward equations, training, optimizer behavior, target
scheduling, synthesis, controllers, checkpoint schemas, or the normal runtime
RNG sequence. The normal path reaches no analysis runner; `--analysis-only`
dispatches before directory creation, corpus-manifest repair, optimizer
construction, training writers, or canonical checkpoint saves.

Analysis requires existing v9 model/world files for stateful experiments. It
infers the physical MorphicStack layout from checkpoint tensors, rejects
inexact migration, freezes learned weights, constructs no optimizer, performs
no backward pass, and checks canonical hashes and modification times before
and after the invocation. Reports and artifacts live under a dedicated
`titan_audio_analysis_v1/<analysis-id>` sidecar by default.

```bash
./target/release/titan \
  --base-dir /sdcard/Download \
  --analysis-only \
  --model-stats \
  --frozen-rollout 64,256 \
  --analysis-stride 16 \
  --analysis-seed 424242 \
  --analysis-tag microscope-01
```

Analyses can be combined in one invocation:

- `--model-stats` reports exact tensors, subsystem counts, defensible
  active/reserved MorphicStack parameters, persistent-state sizes, and memory
  estimates.
- `--frozen-rollout LIST` runs a fixed-morphology, frozen-weight cloned world
  to multiple horizons.
- `--dynamics-ablation N` plus repeatable `--ablation NAME` runs typed neural,
  host-controller, memory, feedback, and forcing interventions. Unsafe hooks
  that would alter the protected production forward path are rejected.
- `--perturbation-analysis N`, repeatable `--perturbation NAME`, and
  `--perturbation-scales LIST` measure latent distance and audio/phenotype
  distance separately under deterministic common exogenous forcing.
- `--benchmark` creates deterministic procedural references outside
  `OLD_WAVS` and the corpus manifest. Its core result is descriptor proximity,
  not held-out reconstruction or semantic transfer.
- `--trained-init-control` compares trained/saved, trained/fresh, and
  active-depth-matched initialized/fresh conditions with zero optimizer steps.
- `--separability-analysis` writes transparent state/audio distance matrices.
- `--render-attribution` exports the complete raw-renderer and post-DSP paths.
  No component is called an additive stem unless it passes sum-to-full.

Use `--analysis-help` for the complete surface, including output, stride,
terminal, and mobile-expensive controls. All analysis flags require
`--analysis-only`; training mutation flags are rejected in that mode.

Audio v9 has no target/reference argument in `model.forward`. Corpus audio is
sampled after the current forward and affects training losses and later host
error feedback. Accordingly, reports declare `direct_reference_input: false`.
Synthetic coverage is not Image-style reconstruction, and target switching is
not claimed as direct conditioning. The exact behavior contract, validation
gates, supported interventions, and deferred architecture work are recorded in
[`docs/AUDIO_SCIENTIFIC_INSTRUMENTATION_PLAN.md`](docs/AUDIO_SCIENTIFIC_INSTRUMENTATION_PLAN.md).
