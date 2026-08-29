# Titan Audio Scientific Instrumentation Plan

## Document Status and Source Basis

This is an implementation plan, not an implementation. It defines a behavior-preserving Phase 1 scientific harness around the current Titan Audio organism.

The source audit for this plan used the absolute latest Audio commit available locally and at the live remote branch tip on 2026-08-26:

- repository: `titan_audio_ecosystem`;
- branch: `v7-truthful-resonant-ecology` (the historical branch name does not identify the current model version);
- commit: `202872a13e369aa37901555acd42ab78700ee346`;
- package/model version: Audio v9.0.0, “Morphogenic Manifold”;
- authoritative implementation: `src/main.rs`, with artifact naming in `src/artifacts.rs` and stereo mappings in `src/stereo.rs`;
- current default allocated parameter count, verified by the existing test: 14,885,543 parameters for 12 physical MorphicStack blocks of width 512;
- sibling scientific-harness reference: Titan Image v9 at commit `6ad6941`, especially its isolated analysis, benchmark, report, provenance, terminal, and byte-preservation patterns.

The user initially referred to v8, then explicitly selected the absolute latest commit. Therefore this plan targets Audio v9 and its v9 world evolution. If implementation begins from any other commit, Gate 0 requires a new source audit and an explicit plan-drift review before code is changed.

The repository audit included every tracked source/configuration/documentation file and the current CLI, checkpoint, model construction, training loop, controller loop, renderer, telemetry, tests, and persistence paths. The sibling Image worktree contained an unrelated untracked rendering script; it was inspected read-only and is not part of this plan.

## Executive Summary

Phase 1 will put a microscope around the existing Audio v9 organism. It will add a strictly separate, frozen, read-only analysis path that can load a trained model and world, reconstruct independent analysis worlds, run matched baselines and interventions, and write structured reports and audio/scientific artifacts. It will not change the normal model, dynamics, learning, controller, renderer, random sequence, corpus schedule, checkpoint format, or outputs.

The central architecture rule is:

> Normal training/rendering must never call the analysis runner. Analysis must branch before any normal corpus repair, optimizer construction, canonical output creation, or checkpoint save, and must return without rejoining the training lifecycle.

Phase 1 will provide:

- exact model, run, corpus, and checkpoint provenance;
- total/active/reserved parameter and persistent-state statistics;
- frozen long-rollout measurements at multiple horizons;
- causal subsystem ablations on cloned worlds;
- common-forcing perturbation and recovery experiments;
- deterministic synthetic-audio probes that never enter training;
- trained-versus-initialization controls with active-depth matching;
- output/state separability and attractor-candidate measurements;
- WAV, spectrogram, state-atlas, CSV, JSON, and comparison artifacts;
- honest target-feedback switch experiments where current Audio semantics permit them;
- rich conservative terminal reporting; and
- a behavioral-equivalence gate proving that disabled instrumentation does not change normal execution.

Phase 1 reports operational measurements. It does not claim life, cognition, concepts, semantic understanding, strong emergence, self-healing, or a strange attractor from proxy measurements.

## Current Architecture as Implemented

### Runtime and state

Audio v9 currently implements one stochastic, adaptive, online-trained ecosystem whose relevant state is larger than its learned tensors:

| Area | Current implementation |
|---|---|
| Micro field | 64 channels at 64x64; updated each chunk through a folded-depth `NeuralCAFolded3D`, deterministic cell-clock masks, macro/recurrent modulation, local anti-rail bias, global-mean damping, and clamping |
| Macro field | 64 channels at 32x32; eligible for learned CA advancement every four chunks, probabilistically gated by the host control state, plus structured shear and confinement gain |
| Folded manifold | Fixed 64-channel storage interpreted as up to four active 16-feature sheets; active sheet count derives from MorphicStack depth; dormant channels are projected to zero |
| Recurrent memory | One 512-unit GRU receiving 64 micro channel means plus a 64-dimensional episodic attention readout |
| MorphicStack | 1-64 physically constructed RMSNorm/Swish residual blocks, default 12x512, with a mutable active depth; depth also controls active manifold sheets and far-ring gain |
| Episodic memory | 16 detached 512-value snapshots at a 64-chunk cadence, read through learned 512-to-64 query/key/value projections |
| Motif/“attractor” memory | Host-side bounded `MotifMemory` containing observations and synthesis controls; it is not the same subsystem as learned episodic attention |
| Learned world model | `MonitorHead`: a 522-value state/action input predicts mean and log variance for a 12-value next audio observation |
| Planner/controller | Batched learned action scoring every 8 chunks plus a host model-free bandit, exploration, UCB-like pressure, action-use tax, and escape logic inside `HybridController` |
| Confinement/recovery | `PotentialController`, `AdaptiveDynamics`, macro shear, micro kicks, sparse heavy-tailed radiation, global-mean damping, clamps, energy homeostasis, and a NaN bio-reset |
| Morph policy | Host-side growth/pruning decisions based on mimic pressure, ecology, strict development-probe plateaus, cooldowns, and configured bounds |
| Learned objective mixer | `AudioArbiter` maps 14 host features to seven loss weights; this affects learning, not frozen inference directly |
| Output | A differentiable multirate renderer followed by bounded saturation, a stateful DC blocker, streamed peak normalization, and 16-bit WAV encoding |

One chunk is 4,096 stereo samples at 48 kHz, approximately 85.33 ms. The ordinary loop combines multiple clocks: micro evolution per chunk, macro eligibility every four chunks, planner refresh every eight chunks, novelty every four chunks, motif consideration every 16 chunks, episodic snapshots every 64 chunks, bounded autograd tapes of at most eight chunks, optimizer horizons selected by `--bptt`, and full-core gradient tapes selected by `--core-update-every`.

### Learned subsystems and synthesis families

The real learned subsystem inventory, to be used for statistics and ablations rather than guessed labels, is:

- micro and macro folded NCA near/far depthwise perception and pointwise mixing;
- GRU recurrent memory;
- MorphicStack blocks and norms;
- learned episodic query/key/value projections;
- recurrent-to-micro asymptotic contraction;
- 64-token micro plus 16-token macro spatial-temporal decoder, recurrent cross-attention, six low-rate residual blocks, and a 76-control output head;
- carrier pitch, FM ratio/index, wave morph, auxiliary pitch, regional partial ratio/amplitude/damping, oscillator-gain, pan, and stereo-width heads;
- left/right eight-basis KAN-inspired wavefolders;
- learned `AudioArbiter`; and
- learned `MonitorHead` world model/planner.

The audible synthesis families inside the current renderer are:

- phase-continuous carrier plus FM;
- three auxiliary modal oscillators per channel;
- a 4x8 regional, 32-partial scan synthesis family;
- deterministic mid/side excitation tables gated by learned temporal controls;
- left/right nonlinear wavefolding;
- learned openness and mid/side control;
- a stateful Haas-style side path and bounded learned width;
- bounded residual global pan; and
- the post-render saturation/DC/mastering stages.

Some of these paths combine nonlinearly. Phase 1 must not call every isolated rendering a “stem.” A stem is valid only where an additive component can be exported before a shared nonlinearity and can be recombined exactly. Otherwise the artifact must be labeled an intervention render or leave-one-family-out contrast.

### Target and corpus semantics

The model forward pass has no target-audio argument. In the ordinary loop, `model.forward(...)` runs before `TargetAudioLoader::sample_chunks(...)`. Target audio participates in spectral, chroma, envelope, recurrence, modulation, level, seam, stereo, and related objectives. Target-derived mimic/error state also influences later host ecology through uncertainty, aperture, morph evidence, and control behavior. It does not directly condition the current chunk’s neural forward or synthesis equations.

Consequences:

1. Audio v9 cannot honestly perform Image-style held-out reconstruction at inference.
2. Different target chunks presented to identical cloned pre-forward state must produce identical current-chunk audio. This first-chunk invariance is an important negative control.
3. In a frozen multi-chunk experiment, changing target/error feedback may alter later host-controller behavior. That is a **target-feedback-conditioned ecological response**, not a direct reference-conditioned reconstruction.
4. A target-independent rollout must explicitly remove or hold target-derived host feedback and be labeled as an intervention. It must not be silently called the ordinary autonomous mode.

The current corpus scheduler reads a versioned manifest, sorts filesystem paths, builds family groups in deterministic key order, samples a family then a variant, and follows coherent episodes. The existing `load_or_create_corpus_manifest` may create or repair the manifest. Analysis must not call that mutating path.

### Persistence and optimizer caveat: source-audit correction

Contrary to the provisional caveat in the request, current Audio v9 does persist AdamW state. `PersistentAdamW::save` writes first/second moments, cumulative optimizer update count, global world step, and renderer-control layout version to `titan_optimizer_v9.safetensors`. Startup restores those moments only when:

- a compatible model was loaded;
- a compatible world was loaded;
- neither fresh-model nor fresh-decoder behavior was requested;
- an optimizer file exists; and
- its recorded global step exactly matches the world step.

Missing, rejected, or incompatible moments restart from zero with the existing 32-update learning-rate warmup. Morphic resize has explicit exact/resized/new-moment behavior. The checked runtime metadata also records successful optimizer resume.

The real reproducibility limitation is checkpoint-set transactionality. Model, optimizer, and world files are each written atomically, but the set is published in model → optimizer → world order. A kill between renames can leave the model ahead of the world; the source itself says the next run is then not mathematically bit-exact. Phase 1 will report this condition but will not alter it.

Therefore:

- **Phase 1:** observe/hash/validate model, optimizer, world, morph sidecar, and metadata; never change their schemas or save behavior.
- **Future schema/behavior boundary:** a transaction manifest, generation ID, previous-generation recovery, and exact all-files resume semantics. This is the relevant future “exact optimizer persistence” work; ordinary moment persistence already exists.

## Why Scientific Instrumentation Is Needed

Current telemetry is unusually rich for online operation, but it is produced while weights, world state, controller state, target episodes, morph depth, and optimizer state may all change. That makes causal interpretation difficult. Existing scalar traces do not by themselves answer:

- whether a phenomenon is caused by learned weights, initialized anatomy, carried world state, or controller forcing;
- whether two trajectories are close internally, perceptually close, both, or neither;
- whether a subsystem has a direct neural contribution, a host-control contribution, a training-only contribution, or only correlational telemetry;
- whether apparent recovery is contraction toward a paired baseline or merely bounded output;
- whether outputs separate by condition or collapse into a generic phenotype;
- whether long trajectories are bounded without optimizer drift; or
- whether results reproduce across seeds, threads, checkpoint identities, and corpus bytes.

The harness must isolate these factors without changing the object under study.

## Non-Negotiable Behavioral Equivalence Contract

### Protected behavior

Phase 1 implementation must not change any of the following on the normal path:

- learned parameter tensor names, values, dtypes, shapes, storage, or serialization;
- parameter initialization algorithm, tensor iteration order, seeds, or initialization draws;
- forward-pass equations or floating-point operation order;
- folded CA projection, near/far ring, depth coupling, cell-clock, micro, or macro update equations;
- micro/macro eligibility cadence or probabilistic gating;
- GRU equations or state update;
- MorphicStack equations, residual gains, active-depth semantics, growth/pruning rules, cooldowns, or development evidence;
- episodic-memory snapshot/read behavior;
- motif storage, recurrence, or recall behavior;
- learned world-model loss/prediction behavior;
- planner scoring, bandit update, controller selection, exploration, control smoothing, or rescue contribution;
- active intervention/identification behavior embodied by the current predict–intervene–observe controller loop;
- potential/confinement/recovery, shear, kick, radiation, energy, damping, clamp, or bio-reset behavior;
- objective terms, weights, detached boundaries, arbiter behavior, or validation/development use;
- target selection, family balancing, episode cursor, prefetch, resampling, or corpus ordering;
- AdamW equations, moment migration, gradient accumulation, clipping, learning-rate schedule, warmup, or update cadence;
- BPTT/tape behavior or core/decoder update scheduling;
- oscillator, partial, excitation, wavefolder, stereo, saturation, DC-block, mastering, and WAV equations/state evolution;
- normal training/rendering `RuntimeRng` draws, order, count, or saved RNG state;
- checkpoint paths, contents, compatibility, resume decisions, save cadence, or v9 world evolution; or
- normal terminal/telemetry values except additive startup help text for new flags, if the equivalence gate confirms no runtime effect.

No “cleanup,” refactor, reordering, iterator replacement, allocation optimization, or deduplication is allowed to ride along with Phase 1 if it can affect arithmetic, scheduling, initialization, or random consumption.

### Isolation invariants

1. The normal path must not import or invoke analysis runtime functions after CLI dispatch.
2. With no analysis flags, no analysis state, RNG, buffers, hashes, writers, clones, branches, or metrics are allocated in the hot loop.
3. `--analysis-only` must return before normal manifest create/repair, `PersistentAdamW` construction, training spool creation, normal WAV creation, or any model/world/optimizer/morph/metadata save.
4. Analysis opens canonical checkpoint/corpus inputs read-only. It writes only beneath a dedicated analysis directory.
5. Learned weights are frozen. The analysis runner performs no backward pass and constructs no optimizer. If read-only optimizer statistics are requested, the safetensors file is decoded directly as data.
6. Canonical `WorldCheckpoint` data is immutable. Each condition is reconstructed from an immutable host snapshot and independent tensors.
7. Perturbations, ablations, rollouts, controller changes, and target-feedback changes operate only on clones.
8. Analysis forcing uses cloned/replayed RNG state. Perturbation-direction RNG uses a separately seeded `analysis_rng`. Neither can advance the canonical saved RNG or a later normal continuation.
9. A zero-amplitude perturbation and a `full`/no-ablation condition must be exactly identical to their analysis baseline.
10. Any proposed feature that cannot pass these invariants is deferred from Phase 1.

## What Phase 1 Explicitly Will Not Change

Phase 1 excludes:

- checkpoint/model/world/optimizer schema changes;
- transactional checkpoint redesign;
- optimizer-persistence changes;
- training or corpus-policy changes;
- a new reference-audio input;
- architecture changes;
- objective redesign or new training presets;
- controller retuning;
- renderer improvement;
- MorphicStack implementation changes;
- a broad split/rewrite of `main.rs`;
- an Image-style spatial recurrent interface;
- macro-resolution changes;
- any claim that analysis improves model quality; and
- automatic continuation from an analysis clone.

Analysis may duplicate or wrap current operations in analysis-only code, but it may not silently become the production implementation.

## Proposed Analysis Architecture

### Lifecycle hook

The exact safe hook is in `main()` after argument parsing and analysis-specific validation, but before the current `create_dir_all(&base_dir)`, manifest refresh/rebuild/create/repair, wake-lock-dependent training setup, target-loader construction, optimizer construction, or output writers.

The dispatch should have this shape conceptually:

```text
parse existing arguments plus AnalysisRequest
validate mutual exclusions and analysis bounds
if analysis_only:
    analysis::run_read_only(request, resolved paths)
    return
existing normal main lifecycle continues unchanged
```

Do not make the normal lifecycle call a generalized “runner” extracted from the monolith in Phase 1. That refactor would enlarge the equivalence surface.

### Proposed files

Add isolated modules rather than reorganizing the organism:

| File | Responsibility |
|---|---|
| `src/analysis/mod.rs` | Analysis-only orchestrator, feature ordering, terminal banner, final report assembly |
| `src/analysis/config.rs` | `AnalysisRequest`, flag parsing helpers, validation, canonical config serialization |
| `src/analysis/load.rs` | Read-only checkpoint-set resolution, model/world loading, physical MorphicStack inference, canonical input pre/post hashes |
| `src/analysis/state.rs` | Immutable `AnalysisOrigin`, per-condition `AnalysisWorld`, deep tensor reconstruction, DSP/host state restore |
| `src/analysis/step.rs` | Frozen ecology stepper, forcing record/replay, full-condition parity surface; no optimizer/backward code |
| `src/analysis/metrics.rs` | State/audio/spectral/stereo distances, streaming accumulators, validity flags, recurrence candidates |
| `src/analysis/rollout.rs` | Multiple-horizon frozen rollouts and fixed-anatomy/ecology-faithful policies |
| `src/analysis/ablation.rs` | Typed neural, host-controller, forcing, synthesis, and post-DSP interventions |
| `src/analysis/perturbation.rs` | Deterministic perturbation generation, paired recovery traces, amplitude sweeps |
| `src/analysis/benchmark.rs` | Procedural audio definitions and evaluator/feedback probes that never join the corpus |
| `src/analysis/attribution.rs` | Exact additive exports where possible and clearly labeled intervention renders otherwise |
| `src/analysis/report.rs` | Versioned serde schema, atomic JSON/CSV writes, warnings, artifact index |
| `src/provenance.rs` | Streaming SHA-256 or equally explicit byte identity, code/corpus/checkpoint manifests; no corpus-policy behavior |
| `src/diagnostics.rs` | Parameter/state counts and memory estimates from actual names/shapes |

If a pure-Rust SHA-256 implementation is selected to avoid a dependency change, keep it small, isolated, covered by standard known-answer tests, and use it only for provenance. Do not reuse the existing FNV-style world checksum as a claim of collision-resistant identity.

`src/main.rs` changes should be limited to module declarations, additive flag capture/help, early analysis dispatch, and the smallest visibility changes required by child modules. Do not move existing model or loop code merely to make the modules aesthetically cleaner.

### Frozen stepper parity strategy

Some meaningful ablations require intervention points inside the monolithic forward path. The production `ComplexAudioEcosystem::forward` must remain the normal oracle. Phase 1 should add a separate analysis-only forward/step surface with typed intervention hooks and prove that its `Full` configuration matches the production forward from identical cloned inputs and runtime state.

Preferred order:

1. Lock a compact production-forward golden case before adding hooks.
2. Implement analysis state reconstruction and a no-intervention analysis step.
3. Compare every returned tensor and every mutated DSP/runtime field against production forward.
4. Only then add intervention branches to the analysis-only surface.

If sharing an internal implementation would require changing the normal arithmetic path, prefer bounded duplication plus parity tests in Phase 1. A later golden-tested modularization can remove duplication. If no-ablation parity cannot be proven, do not ship causal ablations; retain observational rollouts that can use the unchanged production method.

### Clone and memory policy

`WorldCheckpoint` already serializes the main world tensors, `RuntimeRng`, recurrent state, DSP phases/history, episodic slots, novelty history, controllers, motifs, monitor history, adaptive dynamics, and host state. `AnalysisOrigin` should retain this immutable decoded representation plus checkpoint hashes.

Do not hold every condition’s 15-17 million parameter model in memory. Use one immutable loaded `VarMap` of frozen weights and execute conditions serially. Reconstruct a fresh model/runtime object and analysis world for each condition, sharing immutable parameter tensors where Candle safely permits it. Stream baseline summaries/audio and compact forcing records; retain full tensors only for the active condition and comparison points.

## Proposed CLI Surface

All Phase 1 experiment flags require `--analysis-only`. They are additive and must be rejected, not ignored, on the normal path.

| Flag | Semantics |
|---|---|
| `--analysis-only` | Enter the read-only frozen harness and guarantee optimizer steps = 0 |
| `--analysis-dir PATH` | Dedicated output root; default `<base-dir>/titan_audio_analysis_v1/<analysis-id>` |
| `--analysis-tag NAME` | Human-readable artifact suffix; must not change checkpoint selection |
| `--analysis-seed N` | RNG for perturbation directions, synthetic definitions, and fresh-world controls; separate from cloned world forcing RNG |
| `--analysis-stride N` | State/audio observation stride in chunks, default 16; horizons still evolve every chunk |
| `--model-stats` | Emit parameter, active/reserve, state-size, optimizer-file, and memory reports |
| `--frozen-rollout LIST` | Comma-separated horizons, for example `64,256,1024`; one baseline evolution reaches the maximum and records all requested boundaries |
| `--analysis-morph-policy fixed|ecology` | `fixed` holds cloned active depth; `ecology` permits existing clone-only morph decisions and labels required target feedback; default `fixed` |
| `--dynamics-ablation N` | Run selected cloned subsystem interventions for N chunks |
| `--ablation NAME` | Repeatable typed intervention selector; `--dynamics-ablation` with none selects the safe core suite |
| `--perturbation-analysis N` | Run paired common-forcing perturbation/recovery experiments for N chunks |
| `--perturbation NAME` | Repeatable perturbation selector |
| `--perturbation-scales LIST` | Normalized amplitudes, default `0.0001,0.001,0.01,0.1` where meaningful |
| `--benchmark` | Run deterministic procedural audio evaluator and target-feedback probes; never add signals to training |
| `--trained-init-control` | Run capacity-matched trained-versus-deterministic-initialization controls |
| `--separability-analysis` | Compare outputs/states across the invocation’s declared conditions |
| `--target-feedback-switch N` | Run A→B, A→A, and B→B cloned feedback histories with switch at chunk N; never call this direct conditioning |
| `--render-attribution` | Export full baseline plus approved synthesis intervention renders/exact stems |
| `--analysis-expensive` | Enable optional high-cost recurrence grids, dense spectrograms, larger perturbation direction counts, or supplementary embeddings |
| `--analysis-no-audio` | Suppress WAV/PNG artifacts while retaining metrics |
| `--analysis-terminal compact|rich|quiet` | Control analysis reporting without changing normal terminal output |

Additional rules:

- Reject `--analysis-only` with `--fresh-model`, `--fresh-world`, `--fresh-decoder`, `--import-model`, manifest mutation flags, `--lr`, or any request to continue/save training.
- `--model`, `--state`, `--base-dir`, `--run-tag`, and `--corpus-manifest` may identify read-only inputs. Existing `--run-tag` selects checkpoint companions; `--analysis-tag` names outputs.
- Infer physical MorphicStack block count/width from checkpoint tensor names/shapes and cross-check optional run metadata. Never silently load a checkpoint into default capacity.
- Reject `--morph-depth` as a startup mutation in analysis. Depth changes belong to a named cloned intervention or the declared clone-only ecology policy.
- Require at least one requested analysis, or default to `--model-stats --frozen-rollout 256` and print that expansion before work begins.
- Bound all horizons to a documented mobile-safe maximum (initially 16,384 chunks only with `--analysis-expensive`; default validation limit 4,096).

Example:

```sh
./target/release/titan \
  --base-dir /sdcard/Download \
  --model /sdcard/Download/titan_model_v9.safetensors \
  --state /sdcard/Download/titan_world_v9.bin \
  --analysis-only \
  --model-stats \
  --frozen-rollout 64,256,1024 \
  --perturbation-analysis 256 \
  --dynamics-ablation 128 \
  --benchmark \
  --trained-init-control \
  --analysis-seed 424242 \
  --analysis-stride 16 \
  --analysis-tag microscope-01
```

The startup banner must state `FROZEN WEIGHTS`, `OPTIMIZER STEPS: 0`, the exact conditioning label, and canonical input paths before the first experiment.

## Proposed Report / Artifact Layout

Use a sidecar-only analysis schema. Do not add fields to model, world, optimizer, morph, corpus, or normal run metadata.

```text
titan_audio_analysis_v1/<analysis-id>/
├── analysis_report.json
├── provenance.json
├── model_stats.json
├── artifact_index.json
├── run.log
├── rollouts/
│   ├── frozen_rollout.csv
│   ├── frozen_rollout_summary.json
│   ├── baseline.wav
│   └── baseline_spectrogram.png
├── ablations/
│   ├── comparisons.csv
│   ├── summary.json
│   ├── <condition>.wav
│   └── <condition>_spectrogram.png
├── perturbations/
│   ├── traces.csv
│   ├── summary.json
│   └── <perturbation>_<scale>.wav
├── benchmark/
│   ├── definitions.json
│   ├── metrics.csv
│   ├── summary.json
│   └── references/*.wav
├── separability/
│   ├── distance_matrix.csv
│   ├── summary.json
│   └── comparison_montage.png
└── state/
    ├── micro_atlas_<condition>_<step>.png
    ├── macro_atlas_<condition>_<step>.png
    └── memory_heatmap_<condition>_<step>.png
```

`analysis-id` should be a deterministic short digest of checkpoint identity, normalized analysis configuration, and analysis seed, optionally prefixed by `--analysis-tag`. Do not consume `RuntimeRng` or `thread_rng` for names. Atomic temporary-file-and-rename writes are required. Default behavior must refuse to overwrite a completed analysis directory; an explicit analysis-only overwrite flag may be considered, but must never target canonical artifacts.

## Detailed Feature Plan

### 1. Analysis-only mode

Implement an explicit `AnalysisRequest` and early lifecycle branch.

Load sequence:

1. Resolve model/world/morph/run-metadata/optimizer companions without changing their paths.
2. Hash canonical inputs before decoding.
3. Require a v9 model and valid v9 world for stateful analyses. Model-stats-only may operate on a model without a world, but the report must say so.
4. Infer physical MorphicStack architecture from model tensor shapes; validate active depth from the world and sidecar.
5. Construct the model deterministically, load exact tensors, and reject non-exact non-morphic migration. Analysis is not an import/migration operation.
6. Decode the world into immutable `AnalysisOrigin`.
7. Load the corpus manifest through a new read-only parser. Never create, repair, refresh, rebuild, or reorder it.
8. Create only the analysis output directory.
9. Run all requested analyses serially from fresh clones.
10. Hash canonical inputs again and fail the non-mutation gate if any byte changed.

The terminal and report must always contain:

- `weights_frozen: true`;
- `backward_passes: 0`;
- `optimizer_constructed: false`;
- `optimizer_steps: 0`;
- `canonical_world_written: false`;
- `canonical_checkpoint_bytes_unchanged: true|false`; and
- the declared conditioning and morph policies.

### 2. Exact run and corpus provenance

Create two separate records.

**Build/checkpoint provenance**

- package name/version;
- compile-time Git commit/dirty/release/target architecture/Rust flags;
- live source Git commit and dirty status if safely obtainable, with an explicit warning if it differs from the binary provenance;
- normalized invocation and analysis configuration;
- model/world/optimizer/morph/run-metadata paths, byte sizes, modification times as annotations, and SHA-256 hashes;
- decoded world version/global step/seed/active depth;
- model tensor inventory hash based on sorted name, dtype, shape, and bytes;
- optimizer global-step/update-count/layout metadata if present, decoded read-only;
- checkpoint-set consistency result and limitation that Audio v9 has no transactional set manifest; and
- analysis report schema/build identity.

**Corpus provenance**

- manifest path, bytes, schema, generator, and SHA-256;
- every entry in manifest order: relative filename, role, family, declared provenance;
- every present WAV in scheduler order: byte hash, size, sample rate, channels, bits/sample, sample format, source frames, 48 kHz output frames, and supported/skipped status;
- counts and ordered family/variant membership for train/development/validation/exclude;
- strict/fallback development and validation status;
- unlisted/quarantined/missing entries without modifying policy;
- the exact derived scheduler statement `uniform_family_then_uniform_variant` and coherent episode length; and
- one aggregate corpus identity hash over normalized ordered records.

Recording provenance must not change corpus identity or scheduling semantics. Hashing may read bytes and cost time; it may be disabled only through an explicit `--analysis-fast-provenance` warning mode, never silently.

### 3. Model statistics

Generate sorted per-tensor and per-subsystem statistics:

- tensor name, shape, dtype, element count, byte count, min/max/RMS/zero fraction/nonfinite count;
- total allocated learned parameters;
- per-subsystem counts using actual prefixes;
- physical and active MorphicStack depth/width;
- exact allocated and inactive parameters for wholly inactive MorphicStack blocks;
- far-ring parameters marked inactive only when their scalar gain is exactly zero;
- dormant manifold parameter slices reported only where structural reachability is provable from layout; otherwise label shared tensors as allocated/executed rather than guessing an active count;
- active manifold sheets/features and fixed storage channels;
- 512 recurrent dimension, 64 episodic readout dimension, 16 episodic slots, 48 novelty slots, motif capacity/occupancy, 12 observation values, and 10 controller actions;
- temporal decoder token/control/frame dimensions and 32 regional partials;
- learned world-model, planner-related, arbiter, and episodic parameter counts;
- persistent non-learned state element/byte counts by micro, macro, recurrent, DSP, episodic, novelty, motifs, controllers, monitors, and host runtime;
- model/world/optimizer file sizes;
- estimated FP32 weights, two Adam moments, active world, one analysis clone, and worst selected experiment working set; and
- an explicit distinction among learned parameters, persistent non-learned state, deterministic constants/tables, and host controller state.

Do not infer scientific importance from parameter count.

### 4. Frozen long-rollout analysis

Support multiple horizons in one invocation by evolving once to the maximum horizon and recording requested boundaries plus stride samples.

Provide two honest policies:

- `fixed`: weights and active morphology are fixed; existing recurrent, controller, forcing, DSP, episodic, motif, and compact host state evolve on a clone. Target-derived host feedback is either retained or removed according to an explicit conditioning label.
- `ecology`: weights are frozen but existing clone-only morph decisions may occur under their current rules. This is not fixed-anatomy analysis and must report every depth event.

Default the rollout to fixed anatomy and `normal_target_feedback`, because “frozen weights” does not make target feedback disappear. Offer `target_independent_feedback_ablation` only as a separately named intervention.

At each stride and requested horizon measure:

- micro/macro mean, RMS, min/max, L1 movement, relative L2 movement, spatial variance, channel RMS range, near-bound fraction, clamp fraction, nonfinite count, and dormant-channel leakage;
- recurrent `hidden_mem` and refined MorphicStack output RMS/movement;
- episodic slot count, slot age distribution, attention entropy/max weight if exposed analysis-only, snapshot/eviction counts;
- motif occupancy, store/reject/recall/recurrence statistics;
- MorphicStack physical/active depth, manifold sheets, far-ring gain, and clone-only events;
- controller action, planner/bandit proposals, prediction error/calibration/confidence, reward, action occupancy, and switch counts;
- potential, temperature drives, energy, radiation amplitude/probability, realized radiation/kick/shear events, escape strength, gains, and reset warnings;
- world-model mean/log-variance error and calibration;
- raw renderer audio and post-saturation/DC audio RMS, peak, crest, DC, clipping, nonfinite count, and chunk-boundary jump;
- multiresolution log-spectral change, spectral envelope, centroid/rolloff/flatness, band ratios, temporal envelope/onsets/modulation, and ultrasonic ratio;
- side/mid ratio, correlation-aware width, side-energy width, interchannel correlation, channel level ratio, pan and width-control saturation; and
- elapsed time, peak RSS if available, and bytes written.

Approximate recurrence must use compact normalized state/audio signatures, a minimum temporal separation (Theiler window), nearest prior distance, and a `recurrence_distance_valid` flag. A low distance is an `approximate_cycle_candidate`, never proof of a limit cycle or attractor. Exact finite-state recurrence may be separately reported only when full serialized state fingerprints match.

### 5. Causal subsystem ablations

Every ablation starts from the same immutable origin as `full`, uses frozen weights and matched analysis forcing, and reports both internal and output distance from `full`. Conditions run serially.

Classify interventions by subsystem type.

**Learned neural dynamics**

| Name | Exact analysis-only intervention |
|---|---|
| `micro_nca_hold` | Retain prior projected micro field instead of the learned micro CA advance; preserve other declared forcing separately |
| `macro_nca_hold` | Retain prior macro field at eligible update points instead of the learned macro CA advance |
| `gru_hold` | Carry prior 512-state recurrent memory instead of the GRU update |
| `morphic_bypass` | Feed `next_hidden` directly to downstream consumers instead of applying active MorphicStack residual blocks |
| `episodic_read_zero` | Replace only the 64-value learned episodic readout with zeros while retaining slot storage/cadence |
| `episodic_empty` | Start with no slots and prevent snapshots; label as a state-plus-path intervention, not only a neural ablation |
| `memory_to_micro_zero` | Zero only the asymptotic recurrent contraction contribution to micro modulation |
| `temporal_decoder_neutral` | Replace temporal decoder controls with documented neutral values; this is a renderer-network intervention |

**Learned world-model/training components**

- `planner_model_zero`: neutralize cached learned model scores while leaving bandit/controller mechanics active.
- `monitor_prediction_zero`: remove learned world-model contribution and report the downstream controller effect.
- `arbiter_bypass`: expected to have no output effect in zero-optimizer frozen analysis because the arbiter mixes training losses. Treat a nonzero effect as a harness bug. Do not market this null as lack of training importance.

**Host controller/memory/ecology**

- motif recall disabled while motif observation remains measured;
- model-free bandit contribution neutralized while planner remains;
- planner contribution neutralized while bandit remains;
- controller held to the baseline action/control tape (direct neural-effect protocol);
- controller allowed to react under common exogenous forcing (closed-loop total-effect protocol);
- escape/rescue blend disabled;
- potential amplitude gains set to identity;
- structured shear disabled;
- micro kick disabled;
- radiation disabled;
- macro stochastic force gate replayed versus disabled;
- energy modulation held at the baseline tape; and
- bio-reset must never be intentionally triggered as an ablation; nonfinite conditions terminate and report failure.

The current source has no standalone object named “active system identification.” Its operational analogue is the learned next-observation monitor plus planner, model-free reward update, and controller interventions. Report those components separately; do not invent an `ActiveSystemId` neural module.

**Synthesis/DSP interventions**

- carrier/FM family;
- auxiliary modal family;
- regional scan partial family;
- deterministic learned-gated excitation family;
- KAN wavefolding;
- openness/amplitude controls;
- side/width/Haas path;
- residual pan;
- saturation; and
- DC blocker.

Where a family is removed before a shared wavefolder or normalization, label the result a nonlinear intervention render. Export exact additive stems only where a sum-to-full numerical test passes.

For every record include intervention definition, subsystem class, horizon, forcing/control protocol, baseline identity, initial equality result, internal distances, audio distances, descriptor changes, and warnings about compensatory closed-loop behavior.

### 6. Perturbation and recovery analysis

Implement perturbations on cloned state with an independent `analysis_rng`. Apply only to active manifold channels unless a dormant-channel integrity test is explicitly requested.

Required perturbations:

- normalized Gaussian micro noise;
- normalized Gaussian macro noise;
- localized micro patch erase or attenuation;
- localized macro patch erase or attenuation;
- active-channel masking by deterministic channel set;
- spatial-region perturbation without channel masking;
- recurrent-memory Gaussian noise and coordinate masking;
- episodic-slot perturbation, slot masking, and empty-buffer intervention where feasible;
- bounded energy/radiation/temperature/controller-state perturbations, each separately labeled as host-state experiments; and
- amplitude sweep at declared normalized scales.

The paired baseline and perturbed world must receive common forcing. Implement compact replay by recording scalar random decisions and the pre-draw `ChaCha8Rng` state/metadata needed to regenerate large kick/radiation tensors, rather than retaining every tensor in RAM. Perturbation RNG must never share this stream.

Two protocols are required:

- **open-loop/common-control:** replay baseline actions/controls and exogenous forcing, isolating intrinsic state response;
- **closed-loop/common-forcing:** replay exogenous forcing but allow the cloned controller to react, measuring the complete compensatory ecology.

At time zero and each stride compute component-wise state distance and audio distance from the paired unperturbed trajectory. Use `recovery_ratio(t) = distance(t) / max(distance(0), epsilon)`. Define time-to-half only when the initial distance is above a declared numerical floor and the ratio stays at or below 0.5 for at least two observations. Otherwise write `null` with a reason.

Terminology rules:

- decreasing latent distance = latent contraction/recovery toward the paired baseline;
- small audio distance with persistent latent distance = phenotype robustness;
- growing distance = perturbation amplification/divergence;
- bounded distance is not recovery;
- output similarity is not state recovery;
- neither result is automatically self-healing.

An optional common-noise, repeated-renormalization tangent protocol may estimate finite-time conditional expansion. Without renormalization, multiple directions, and a stated fitting interval, do not label a slope a Lyapunov exponent.

### 7. Deterministic synthetic-audio benchmark

Generate signals in memory and optionally export reference WAVs. Never copy them into `OLD_WAVS`, never add them to the manifest, and never call corpus mutation code.

Minimum deterministic suite:

- pure tones across low/mid/high bands;
- harmonic and inharmonic stacks;
- linear and logarithmic chirps;
- impulses and decaying impulse responses;
- amplitude- and frequency-modulated tones;
- rhythmic impulse trains and envelopes;
- polyrhythms;
- filtered white/pink-like deterministic noise;
- stereo pan, width, antiphase, decorrelated, and rotating-motion cases;
- nested temporal modulation; and
- branching/multiscale rhythmic patterns.

Every definition records generator version, seed, exact formula/parameters, sample rate, duration, and byte hash.

Current-architecture benchmark modes must be separated:

1. **Descriptor coverage (allowed):** compare frozen generated audio against synthetic references using transparent descriptors. This measures coverage/proximity, not transfer or reconstruction.
2. **Target-feedback response (allowed, explicitly conditioned):** use a procedural signal only in the same target-derived error/host-feedback role that corpus audio currently occupies, with zero optimizer steps. The current chunk must remain invariant; later differences measure host ecological response to error feedback.
3. **Held-out reconstruction/conditioning (not available):** deferred until a future reference-audio pathway exists.

The benchmark must include a target-label permutation/null test and the identical-first-chunk test. If the model appears to distinguish procedural targets before target-derived feedback can causally reach the controller, the harness is leaking reference information.

### 8. Trained versus near-initialization controls

Use a factorial control design to avoid conflating weights, carried world, and active capacity:

| Condition | Weights | World | Active/physical capacity |
|---|---|---|---|
| `trained_saved_world` | checkpoint | saved clone | checkpoint exact |
| `trained_fresh_world` | checkpoint | deterministic fresh analysis world | checkpoint exact |
| `init_depth_matched_fresh_world` | deterministic v9 initialization | identical fresh world seed | same physical blocks, width, active depth, manifold sheets, and rings as trained |
| `init_native_depth_fresh_world` | deterministic v9 initialization | identical fresh world seed | native L01; descriptive secondary control only |

The primary learned-versus-initialized comparison is `trained_fresh_world` versus `init_depth_matched_fresh_world`. Construct both with the same architecture, active depth, seeds, forcing/control protocol, horizon, and zero optimizer updates. Activating initialized blocks is an analysis condition, not a MorphicStack implementation change.

Separately compare `trained_saved_world` with `trained_fresh_world` to measure carried ecology/state. Do not attribute that difference to training alone.

If a genuine near-initialization checkpoint is supplied, require its own provenance and step count. Phase 1 must not create it by taking optimizer steps during evaluation.

### 9. Output separability and attractor analysis

Conditions available today include seeds/fresh worlds, saved versus fresh ecology, trained versus initialized weights, ablations, perturbations, controller policies, synthetic error-feedback streams, and corpus error-feedback episodes. Direct target-conditioned output families are not available.

Use transparent pairwise distance families:

- waveform L1/L2/correlation only for sample-aligned common-state comparisons;
- multiresolution log-spectral distance;
- coarse spectral-envelope and band-energy distance;
- temporal-envelope/onset/modulation/rhythm distance;
- centroid, rolloff, flatness, crest, RMS, and clipping differences;
- stereo correlation, side/mid, level balance, correlation-aware width, and pan differences;
- state distances for micro, macro, recurrent, episodic summary, controller, and DSP state; and
- trajectory distances such as mean/maximum pairwise distance, nearest-neighbor condition, within-condition drift, and between/within ratio.

Report distance matrices and clustering summaries without naming clusters as concepts. Operational outcomes:

- low between-condition and low within-condition distance: generic stable phenotype candidate;
- low between-condition but high drift: shared drifting family;
- high between-condition and low within-condition: differentiated stable phenotype families;
- high between- and within-condition: differentiated but unstable/drifting trajectories.

Attractor analysis should estimate occupation summaries, recurrence candidates, dwell/transition times among descriptor-defined regimes, and convergence/separation across progressively longer horizons. One finite run cannot distinguish separate basins from extremely slow mixing. Optional pretrained audio embeddings may be supplementary only, versioned and clearly non-ground-truth; Phase 1 core must not require a new model or network dependency.

### 10. State and audio attribution artifacts

Export:

- baseline, ablation, perturbation, and control WAVs using the existing 48 kHz stereo convention;
- explicit raw-renderer and post-DSP variants where useful;
- spectrogram PNGs with fixed scale/color bounds within a comparison;
- per-condition waveform/envelope/spectral/stereo summary JSON and CSV;
- micro/macro state atlases using stable channel/page layout and fixed value range;
- channel RMS/near-bound heatmaps;
- recurrent and episodic-memory heatmaps;
- controller/action timelines;
- compact comparison montages; and
- exact additive stems only where sum-to-full validation passes.

Analysis audio naming must be deterministic and must not use the normal random audio filename helper. Do not call the normal prime-selection/prompt path unless it is intentionally reimplemented as an analysis artifact with an analysis-only name.

### 11. Target-feedback switch and hysteresis

True Image-style target switching is not supported. There is no target/reference tensor in Audio `model.forward`.

Phase 1 may implement the following narrower experiment:

1. Clone one origin into A→A, A→B, and B→B conditions.
2. Feed deterministic target/error feature streams A and B only to the existing loss/uncertainty feedback calculations; take no optimizer steps.
3. Use identical current-world state and common forcing through the switch.
4. Verify current-chunk output equality at a switch because the target is sampled after forward.
5. Measure later controller, aperture, forcing, state, and audio divergence.
6. Compare post-switch A→B against B→B at matched time. Persistent difference is an operational prehistory/hysteresis candidate in the host ecology, not reference memory or reconstruction.

Reports must use `conditioning: target_error_feedback`, `direct_reference_input: false`, and `first_possible_effect_chunk` rather than “target-conditioned synthesis.”

### 12. Analysis terminal output

Provide compact/rich/quiet analysis modes without changing normal terminal output. Rich output should show:

- analysis schema/id and output directory;
- build, model, world, optimizer-file, corpus, and checkpoint hashes/steps;
- `FROZEN WEIGHTS` and `OPTIMIZER STEPS = 0` prominently;
- cloned-world RNG versus independent perturbation RNG seeds;
- physical/active morphology and conditioning policy;
- current analysis and condition;
- horizon, offset, stride, progress, elapsed time, and estimated remaining work;
- state boundedness/movement, recurrent movement, output RMS/peak/spectral delta/stereo correlation;
- baseline/intervention distances and validity flags;
- nonfinite, clipping, checksum, corpus, or semantic warnings; and
- artifact paths on completion.

Use operational labels such as `STATE CONTRACTING`, `STATE DISTANCE PERSISTING`, `OUTPUT ROBUST / STATE NOT RECOVERED`, `APPROXIMATE RECURRENCE CANDIDATE`, and `TARGET-ERROR-FEEDBACK CONDITIONED`. Never print `SELF-HEALED`, `LIFE`, `COGNITION`, `CONCEPT`, or `STRANGE ATTRACTOR` as an automatic conclusion.

### 13. Report schema

Use an Audio-specific sidecar schema with shared Image-style top-level ideas where they fit:

```json
{
  "schema": {"name": "titan_audio_analysis", "version": 1, "modality": "audio"},
  "analysis": {
    "id": "...",
    "invocation": [],
    "started_unix_ms": 0,
    "finished_unix_ms": 0,
    "weights_frozen": true,
    "backward_passes": 0,
    "optimizer_constructed": false,
    "optimizer_steps": 0,
    "analysis_seed": 0,
    "stride_chunks": 16
  },
  "identity": {
    "build": {},
    "model": {},
    "world": {},
    "checkpoint_step": 0,
    "checkpoint_set_consistency": "consistent|warning|invalid"
  },
  "corpus_provenance": {},
  "model_statistics": {},
  "configuration": {
    "conditioning": "normal_target_feedback|target_independent_feedback_ablation|target_error_feedback",
    "direct_reference_input": false,
    "morph_policy": "fixed|ecology",
    "forcing_protocol": "native_cloned_rng|recorded_common_forcing"
  },
  "conditions": [],
  "rollouts": [],
  "ablations": [],
  "perturbations": [],
  "benchmarks": [],
  "separability": {},
  "artifacts": [],
  "warnings": [],
  "interpretation_constraints": [],
  "non_mutation": {
    "canonical_before": {},
    "canonical_after": {},
    "unchanged": true
  }
}
```

Every metric record should include condition ID, clone origin, rollout offset, absolute checkpoint age, analysis developmental age where defined, units, validity, and reason when invalid. JSON must encode missing/invalid metrics as `null` plus a reason, never NaN or infinity. CSV columns require schema/version and stable headers. Artifact index entries include path, media type, SHA-256, condition, start/end offset, and whether mastering/post-processing was applied.

## Behavioral Equivalence Test Strategy

### Gate 0: freeze the pre-instrumentation oracle

Before any Rust edit, record:

- commit and clean/dirty status;
- `Cargo.lock`, compiler, target flags, and thread count;
- current `--help` output;
- sorted learned tensor names/shapes;
- default deterministic initialization tensor fingerprints;
- a compact fixed-corpus, fixed-seed, single-thread normal run; and
- a protected resume continuation from copied checkpoint artifacts.

Use a tiny deterministic WAV fixture generated outside the training binary and a scratch base directory under Termux’s writable temporary root. Preserve the baseline binary or run the baseline commit in a separate worktree. Do not put phone production checkpoints into the test repository.

### Gate 1: disabled-instrumentation equivalence

Run baseline and candidate binaries with the same normal command, seed, corpus bytes/order, starting checkpoint/world/optimizer copies, thread count, compiler/profile, and duration. No analysis flags.

Compare:

- model tensors name-by-name and value-by-value;
- optimizer moment tensors, global-step stamp, and cumulative update count;
- decoded world tensors and every serialized host/DSP/controller field;
- saved RNG state and several subsequent cloned draws;
- target file/frame/chunks-left trace;
- optimizer update boundaries and values;
- planner/bandit/selected action trace;
- morph depth/events/manifold/ring trace;
- controller/potential/radiation/shear/kick trace;
- normalized PCM WAV sample bytes where deterministic;
- topology and scalar traces after removing only run ID/timestamps/path names;
- final global step and checkpoint-resume result; and
- fresh and resumed cases.

Single-thread CPU execution is the bit-identical authority. Multi-thread production equivalence should also be tested with the same thread count. If a parallel reduction prevents bit identity, first prove that the source operation and scheduling are unchanged, then use declared tight tolerances no looser than necessary: default candidate maxima are state/parameter absolute error ≤ 1e-7 and relative error ≤ 1e-6, identical discrete events/RNG/corpus selection, and identical 16-bit PCM. Any tolerance relaxation requires a written numerical cause and approval; it is not an automatic escape hatch.

### Gate 2: analysis non-mutation

For model, optimizer, world, morph state, corpus manifest, normal metrics, normal metadata, and normal WAVs:

1. hash bytes;
2. run every analysis in one invocation;
3. hash again;
4. require exact equality and unchanged modification times where the filesystem supports them.

Also prove:

- a later normal continuation after analysis is identical to a continuation that skipped analysis;
- analysis with a missing manifest refuses or runs model-only stats without creating one;
- no normal temp/spool filenames are created;
- no optimizer object/step/save path is reached; and
- analysis failure leaves canonical inputs unchanged and produces either an atomic complete report or a clearly marked incomplete analysis directory.

### Gate 3: frozen-runner correctness

- production `forward` versus analysis `Full` from cloned state: exact returned tensors and mutated runtime fields;
- `Full` versus no-ablation: exact trajectory;
- zero perturbation versus paired baseline: exact trajectory;
- identical seed/config/checkpoint: identical JSON after excluding timestamps/elapsed time, identical WAVs, and identical state fingerprints;
- different `analysis_seed`: perturbation/benchmark direction changes without changing the unperturbed baseline;
- common-forcing replay regenerates exactly the baseline stochastic tensors/events; and
- first target-feedback switch chunk remains audio-identical.

### Gate 4: report and scientific-validity tests

- schema round-trip and golden JSON keys;
- no NaN/infinity serialization;
- valid/invalid flags for recurrence, half-recovery, waveform alignment, and target effects;
- known audio metric fixtures (mono, panned mono, antiphase, tone, silence, impulse);
- procedural benchmark known hashes;
- exact active-depth parameter classification fixtures;
- ablation type labels and baseline presence;
- additive stem recombination where claimed; and
- conservative terminology lint over terminal/report strings.

### Gate 5: mobile performance

- disabled normal-path median chunks/s within measurement noise and no new hot-loop allocations; target ≤1% median regression across repeated thermal-stable runs;
- model-stats peak memory bounded to one tensor readback at a time;
- rollout memory independent of horizon except streamed files/compact signatures;
- serial conditions by default;
- focused subset execution; and
- Ctrl-C produces a valid partial analysis report and never saves a clone as canonical state.

## Android / Termux Performance Considerations

- Default to CPU and current thread controls; record effective threads.
- Hash files in streaming blocks.
- Compute spectra at `--analysis-stride` unless dense output is explicitly requested.
- Reuse FFT plans and fixed windows.
- Stream CSV/JSONL rows and WAV frames; atomically finalize summaries.
- Keep only the active clone, compact baseline stride records, and compact forcing replay metadata in RAM.
- Serialize ablation/perturbation conditions; do not retain one model per condition.
- Bound spectrogram resolution and montage count by default.
- Make state atlases optional at long horizons.
- Provide cheap `stats`, `rollout`, `perturb`, `ablation`, `benchmark`, and `separability` subsets.
- Estimate requested memory/time before starting and warn or require `--analysis-expensive` when the plan exceeds a conservative phone budget.
- Acquiring a wake lock is acceptable only inside explicit analysis execution and must not affect normal RNG/state. Release it on every exit path.

## Compatibility and Versioning

Phase 1 requires no model-format or world-format change.

- Existing v9 model/world/optimizer/morph files load unchanged.
- No new fields are embedded in canonical checkpoints.
- Analysis reports use independent schema `titan_audio_analysis` v1.
- Analysis artifact names are versioned separately from telemetry schema v10.
- CLI changes are additive.
- A report reader must tolerate additive fields within analysis schema v1.
- Analysis must refuse v8 worlds and must not import/migrate a model.
- If physical morphology cannot be inferred exactly or conflicts with metadata/world, stateful analysis fails safely rather than resizing.
- A future transactional checkpoint set, optimizer-generation manifest, reference pathway, or model anatomy change is a separate compatibility boundary.

## Implementation Order

1. **Phase 0 — oracle and audit lock:** recheck HEAD/live remote, record baseline commands/artifacts, capture single-thread fresh/resume golden results, and document exact tensor prefix mappings.
2. **Phase 1A — early dispatch and read-only loading:** add CLI request parsing, early return, companion resolution, read-only world/model loading, canonical hash guard, and analysis directory handling. Ship no rollouts until the non-mutation test passes.
3. **Phase 1B — provenance and model stats:** implement corpus/checkpoint/code provenance, tensor/subsystem counts, state sizes, memory estimates, schema, terminal identity banner, and atomic report writers.
4. **Phase 1C — frozen baseline parity:** implement `AnalysisOrigin`, clone reconstruction, analysis full step, native cloned RNG, forcing replay, no-ablation parity, and fixed-anatomy rollout.
5. **Phase 1D — long-rollout metrics/artifacts:** add streaming state/audio metrics, recurrence candidates, WAV/spectrogram/state atlases, multiple horizons, and optional ecology morph policy.
6. **Phase 1E — perturbations:** add zero/noise/patch/channel/memory/host-state variants, common-control and closed-loop protocols, recovery/phenotype distinction, and scale sweeps.
7. **Phase 1F — causal ablations:** add learned-neural hooks first, then host controller/forcing interventions, then nonlinear synthesis intervention renders. Each hook requires full/no-op parity tests.
8. **Phase 1G — controls and benchmarks:** add capacity-matched trained/init comparisons, deterministic procedural suite, descriptor coverage, target-feedback response, switch/hysteresis, and separability matrices.
9. **Phase 1H — hardening:** combined multi-analysis invocation, Ctrl-C partial reports, performance bounds, terminology lint, README/METRICS documentation, full locked tests/Clippy/release build, and final equivalence run.
10. **Later modularization only:** after Phase 1 is accepted, consider extracting production runtime components from `main.rs` in small golden-equivalent stages. Do not make that refactor a Phase 1 prerequisite.

## Validation Gates

Phase 1 is not complete unless every applicable gate is green:

| Gate | Pass condition |
|---|---|
| Source lock | Implementation still targets audited v9 or has an approved drift update |
| Normal behavior | Fresh and resumed normal commands meet the equivalence thresholds with no analysis flags |
| RNG | Normal saved RNG and subsequent draws are unchanged; analysis/perturb RNGs are isolated |
| Corpus | Normal ordering/sampling is unchanged; analysis never mutates manifest/corpus |
| Checkpoint | Canonical model/world/optimizer/morph/metadata bytes unchanged by analysis |
| Frozen | No backward, optimizer construction, optimizer step, or checkpoint save occurs |
| Clone | Every intervention begins from an independently reconstructed origin |
| Baseline | Full/no-op and zero-perturbation trajectories match exactly |
| Reports | Schema-valid, finite, provenance-complete, validity-aware, atomically written |
| Semantics | Every result declares direct reference input, target feedback, morph, forcing, and controller policies |
| Performance | Focused defaults fit Termux/CPU memory; normal disabled performance is essentially unchanged |
| Quality | `cargo fmt --all -- --check`, `cargo test --locked`, strict locked Clippy, and locked release build pass |

Any failure in normal equivalence, RNG, checkpoint non-mutation, baseline parity, or optimizer-zero enforcement is release-blocking. A failed optional expensive analysis may be deferred, but it may not weaken these gates.

## Risks and Failure Modes

| Risk | Required mitigation |
|---|---|
| Analysis accidentally calls manifest repair | Separate read-only parser; hash/mtime pre/post test; early dispatch before current loader |
| Analysis overwrites normal artifacts | Dedicated root and deterministic analysis names; path containment test; canonical hash guard |
| Constructor silently resizes morphology | Infer exact model shapes, cross-check world/metadata, reject ambiguity |
| “Frozen” path still updates optimizer | Do not construct optimizer; static call-path review plus zero-step report/test |
| Clone shares mutable tensor/runtime state | Deep reconstruction tests; mutate one clone and fingerprint origin/sibling |
| Analysis RNG shifts forcing | Clone/replay world RNG; independent perturbation RNG; continuation-equivalence test |
| No-ablation runner drifts from production | Full tensor/runtime parity gate; defer ablations if parity fails |
| Common-noise experiment uses independent noise | Forcing replay metadata and exact regeneration tests |
| Target is misrepresented as forward conditioning | Schema fields and first-chunk invariance; terminology lint |
| Host controller ablation is called neural causality | Typed subsystem class in every condition/report |
| Output robustness is called state recovery | Separate latent and phenotype metrics/labels |
| Morph depth confounds trained/init result | Capacity/depth-matched primary control plus native-depth secondary control |
| Inactive parameter count is overstated | Count only provably inactive slices; label estimates/unknown shared tensors |
| Nonlinear component export is called a stem | Sum-to-full test or label as intervention render |
| Recurrence is called an attractor | Minimum separation, validity flags, progressive horizons, candidate language |
| Full-state recording exhausts phone RAM | Streaming metrics, signatures, state atlases at sparse checkpoints, serial conditions |
| Checkpoint set is already inconsistent | Read-only step/hash consistency warnings; fail stateful analysis when model/world incompatibility is detectable |
| Parallel FP nondeterminism hides a regression | Single-thread bitwise oracle; same-thread production test; minimal justified tolerances only |
| Analysis code becomes a second drifting model | Production-forward parity tests and explicit future plan to modularize only after golden coverage |

## Deferred Work / Future Architecture Experiments

The following may be scientifically valuable, but they change schema, function, learning, conditioning, or architecture and are **not part of Phase 1**:

- transactional checkpoint generations and exact model/world/optimizer resume after interrupted publication;
- any optimizer moment/layout/persistence redesign beyond observing current v9 files;
- a true held-out audio reference/probe pathway;
- an explicit reference-fidelity sweep down to target-independent zero;
- a static-chunk developmental probe;
- a streaming held-out temporal probe;
- an Image-style regional-token recurrent interface;
- recurrent spatial writeback into Audio CA fields;
- a genuinely lower-resolution macro field;
- zero-contract/function-preserving MorphicBlock activation changes;
- reconstruction, grounded-emergent, and free-ecology research presets;
- stronger output-level grounded/emergent decomposition;
- objective redesign, new weights, or corpus scheduling changes;
- renderer/DSP changes including new residual, codec, or anti-aliasing paths;
- shared cross-modal Titan substrate or architecture convergence with Image; and
- broad production modularization of `main.rs` without its own staged golden-equivalence plan.

Future exact-resume work must acknowledge that Audio v9 already persists AdamW moments. Its boundary is atomic checkpoint-set identity/recovery and any new schema needed to guarantee the exact resumed trajectory after process interruption.

## Scientific Language Policy

Reports and documentation must distinguish measurement from interpretation:

- persistent dynamics are persistent dynamics, not life;
- positive or growing perturbation distance is divergence/amplification, not strong emergence;
- boundedness is boundedness, not homeostasis;
- latent contraction toward a baseline may be called recovery only under the stated metric;
- output similarity with latent divergence is phenotype robustness, not self-healing;
- benchmark proximity is descriptor proximity, not semantic understanding;
- recurrent memory and approximate recurrence are not cognition;
- clusters/attractors are operational regimes or candidates, not concepts;
- a host-controller intervention is not a learned-neural ablation; and
- a frozen target-feedback response is not direct target-conditioned synthesis.

Larger claims require replicated seeds/checkpoints, preregistered criteria, suitable controls, objective audio evidence, and when relevant blinded listening.

## Assumptions and Unresolved Questions

### Locked assumptions for implementation

1. The implementation baseline is commit `202872a13e369aa37901555acd42ab78700ee346`; re-audit on drift.
2. Phase 1 remains Audio v9 sidecar instrumentation with no canonical schema change.
3. The default authoritative deterministic equivalence environment is one CPU thread with identical build flags; normal multi-thread configurations receive an additional same-thread-count comparison.
4. Physical MorphicStack layout will be inferred from actual checkpoint tensor names/shapes, not defaults or filenames.
5. The default frozen rollout holds active morphology fixed and retains ordinary target-derived host feedback, with both facts printed and serialized.
6. Core metrics use transparent signal processing already implementable with current dependencies; no pretrained embedding is required.
7. Cryptographic provenance uses an isolated streaming implementation or an already available platform capability without adding a Cargo dependency or changing model behavior.
8. External phone checkpoints are test inputs, not repository fixtures; tests use protected copies/scratch artifacts.

### Questions to resolve during Phase 0 without changing scope

1. Which exact normal command/corpus snapshot should be retained as the long-term phone equivalence oracle in addition to the portable tiny fixture?
2. What maximum default horizon/condition count stays below the target phone’s practical peak-RSS and thermal budget? Measure before fixing defaults.
3. Does Candle remain bit-identical for the chosen multi-thread configuration on the device? If not, record the exact reduction source before approving tolerances.
4. Which current run metadata should be treated as the companion of a checkpoint when timestamps disagree? Phase 1 should hash/report all candidates and require explicit selection rather than guess.
5. Which perturbation normalization is most interpretable for micro/macro fields at the mature checkpoint: relative RMS, relative L2, or both? The report can carry both; one must be declared primary before comparisons.
6. What minimum recurrence separation and descriptor thresholds are empirically above numerical/noise floors? Calibrate on null controls; do not hard-code scientific conclusions from Image thresholds.
7. Should the optional ecology-faithful rollout permit clone-only morph events by default in a separate preset, or remain opt-in? This plan defaults it to opt-in.

None of these questions authorizes a model, training, controller, renderer, objective, or checkpoint-format change.

## Definition of Done

Phase 1 is done only when an implementer can demonstrate all of the following:

1. `--analysis-only` loads an exact v9 model/world read-only, reports frozen weights and zero optimizer steps, runs multiple requested analyses, and writes only a dedicated sidecar directory.
2. Provenance identifies build, invocation, checkpoint files, source/corpus files and order, hashes, formats, seed, model version, global step, physical/active morphology, and conditioning policy.
3. Model stats report exact total and subsystem counts, defensible active/reserve counts, persistent state sizes, and memory estimates.
4. Frozen rollouts cover multiple horizons and report boundedness, movement, recurrent/memory/controller state, output/spectral/stereo stability, nonfinites, and recurrence candidates.
5. Causal ablations use a full baseline, typed subsystem categories, cloned state, matched forcing, and both internal/audio distances.
6. Perturbation sweeps separate latent recovery from phenotype robustness and use mathematically valid half-recovery fields.
7. The procedural benchmark is deterministic, provenance-complete, excluded from training, and honest about descriptor coverage versus target-feedback response versus deferred direct conditioning.
8. Trained/init controls match architecture and active depth in the primary comparison and separate learned weights from carried world state.
9. Separability artifacts quantify differentiated/stable/drifting/generic phenotype candidates without semantic overclaiming.
10. WAVs, spectrograms, state atlases/heatmaps, JSON, CSV, and comparison artifacts are indexed and hashed; any claimed stems recombine exactly.
11. Target-switch reports state that Audio has no direct reference input and identify the first possible feedback-mediated effect.
12. Terminal output is clear, progress-aware, artifact-aware, conditioning-aware, and scientifically conservative.
13. Analysis reports validate against schema v1, contain no nonfinite JSON numbers, and record all interpretation constraints.
14. Canonical model, optimizer, world, morph, manifest, training telemetry, metadata, and audio bytes are identical before and after analysis.
15. Fresh and resumed normal runs with instrumentation disabled pass the behavioral-equivalence gate, including RNG, optimizer, controller, morph, audio, telemetry, and checkpoint comparisons.
16. Normal-path performance is essentially unchanged, focused analysis fits Android/Termux/CPU constraints, and expensive work is opt-in.
17. Existing checkpoints remain compatible and no canonical format version changes.
18. Locked tests, strict locked Clippy, formatting, and locked release build pass.
19. README/METRICS documentation describes the new flags, schema, operational terminology, and exact target-conditioning limitation.
20. No deferred architecture item has entered Phase 1 under an instrumentation label.

Only after all of these pass should Phase 1 be described as implemented. Empirical claims about trained behavior remain separate from software completion.

## Phase 1 Implementation Validation Record

Phase 1 was implemented against the locked source commit documented above.
The protected production diff in `src/main.rs` is limited to module declarations,
one early analysis-flag dispatch, and additive help text. `Cargo.toml`,
`Cargo.lock`, model/world/optimizer formats, and every production forward,
training, controller, synthesis, and persistence equation remain unchanged.

The Termux validation oracle used a deterministic 48 kHz stereo WAV, one CPU
thread, seed 4242, a one-chunk fresh run, and a one-chunk exact-resume run. The
pre-instrumentation release binary was preserved outside the repository before
the first Rust edit.

Validation results:

- final locked test suite: 76 passed, 0 failed;
- strict locked Clippy with all targets: passed with `-D warnings`;
- formatting and whitespace gates: passed;
- native aarch64-Android locked fat-LTO release build: passed;
- analysis-only model/world/corpus loading: passed without manifest creation or repair;
- canonical pre/post model, optimizer, world, morph, metadata, telemetry, manifest,
  and normal-audio identity checks: unchanged, including modification times;
- analysis-followed-by-continuation versus continuation-without-analysis:
  serialized world/RNG and rendered WAV were bit-identical;
- final fresh pre/post instrumentation oracle: serialized world/RNG and WAV
  were bit-identical; learned parameters had maximum absolute delta
  `7.450580596923828e-9`, optimizer tensors `2.9802322387695312e-8`, and zero
  elements exceeded `1e-7 + 1e-6 * max(abs(a), abs(b))`;
- pre-instrumentation versus candidate exact-resume oracle: serialized
  world/RNG and WAV were bit-identical; learned parameters had maximum absolute
  delta `7.450580596923828e-9`, optimizer tensors `1.1920928955078125e-7`, and
  zero elements exceeded the same combined tolerance; and
- same candidate binary, same checkpoint, separate resume processes exhibited
  the same last-bit reduction variability while world/RNG and audio remained
  bit-identical. This locates the non-bitwise optimizer bytes in CPU floating
  reduction/code-generation variability rather than analysis state or RNG
  leakage.

The measured fresh-run wall time remained within ordinary thermal/process
noise for the phone. Analysis conditions and perturbation scales execute
serially; condition artifacts are finalized and dropped before the next clone.

Two planned intervention families remain explicitly unavailable rather than
weakening the behavior contract: internal MorphicStack/renderer family hooks
that could not be added without changing the production forward surface, and
an exact A-to-B target-error-feedback tape still embedded in the protected
training objective. Reports mark these as deferred and do not claim additive
stems, direct target conditioning, or Image-style held-out reconstruction.
