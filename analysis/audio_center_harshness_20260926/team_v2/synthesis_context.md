# Shared Audio evidence snapshot

## EVIDENCE research/audio_context.md:1-83
SHA256 d311772b95faed9be7ea91e20000080e8151f99f188909ed00bdf8ff2a2d0183

1: # Audio research context contract
2: 
3: This is the shared core for every Audio research-team role. Read the selected
4: source excerpts as evidence. Report factual reconstruction and supported
5: conclusions, not hidden reasoning. A larger context does not certify an answer.
6: 
7: 1. **Decision:** Diagnose harshness and center bias in the latest `v9-long-01`
8:    exported audio; choose a reversible experiment before changing learned
9:    behavior. Useful sound and subsequent human use are the practical goals.
10: 2. **Modality:** This is Titan AUDIO, a CPU Rust NCA/GRU/DDSP ecosystem. It is
11:    neither Titan Text nor an audio-conditioned DiT. No Suno/API experiment is
12:    authorized or available in this investigation.
13: 3. **Forward inputs:** Fields, recurrent/episodic memory, oscillator phase,
14:    prior controls, timing, energy, and host synthesis controls. `forward` has
15:    NO text prompt, target audio, prime, or external reference argument.
16: 4. **Causal order:** Current world -> forward audio -> target sampling/loss ->
17:    optimizer and later host feedback. Current target cannot change that same
18:    precomputed chunk; prior targets can affect weights and later ecology.
19: 5. **Prompt direction:** `suno_priming_prompt*.txt` is written AFTER rendering
20:    and prime extraction. Its descriptive words cannot cause the run. It may
21:    influence a later external music system, which is outside this experiment.
22: 6. **Clocks:** 4096 samples at 48 kHz per chunk; macro eligibility every four
23:    chunks; bounded eight-chunk tapes; latest BPTT 64, full core every four
24:    tapes. Render duration is not wall time or optimizer update count.
25: 7. **Anatomy:** Folded micro/macro NCA, GRU, residual MorphicStack, episodic
26:    and host motif memories, learned low-rate controls, oscillators, separate
27:    wavefolders, mid/side stage, bounded pan. Latest active depth is 16.
28: 8. **Optimizer:** AdamW moments are persisted. The three latest WAVs form one
29:    sequential continuation lineage, not independent seeds or matched arms.
30: 9. **Initialization/resume:** The untagged parent remains separate from the
31:    tagged run. Exact checkpoint/model/world/moment hashes identify inputs;
32:    hashes alone establish neither sound quality nor mathematical consistency.
33: 10. **Corpus:** Family roles and generated-name exclusions matter. Two local
34:     manifest entries have quarantine provenance but non-exclude roles;
35:     production excludes their filenames. Report declared and candidate roles
36:     separately; do not infer they entered gradients from manifest roles alone.
37: 11. **Targets:** Streaming family/variant episodes, sampled after forward.
38:     No inference-time reconstruction target exists. Target mix and online
39:     learning confound comparisons between elapsed portions of a run.
40: 12. **Losses:** Multiscale magnitude, envelope, band, chroma, pitch, onset,
41:     recurrence, modulation, low-band, stereo geometry/correlation/balance,
42:     plus ecology. Read actual weights in selected code. The low-band term
43:     measures the first log band; it is not our whole 20-200 Hz measurement.
44: 13. **Stereo controls:** Side derives from L-R before a delayed scaled side
45:     stage. Learned width is bounded 0.05..0.50; global pan +/-0.10. Pan-center
46:     loss penalizes bounded pan; it does not directly reward correlation=1.
47:     Increasing side gain cannot synthesize missing side information.
48: 14. **DSP/mastering:** Loss-visible tanh, later gain/saturation/DC blocker,
49:     and whole-file peak normalization. A sample peak below 0 dBFS does not
50:     exclude nonlinear saturation or perceptual harshness.
51: 15. **Prime selection:** Approximately 60 seconds selected using field entropy,
52:     activity health, complexity, stagnation. No listener rating or direct
53:     stereo/harshness ranking. Selection can amplify a within-run outlier.
54: 16. **Metrics:** The supplied first survey uses MID-channel sampled FFT power
55:     fractions. Changes in fraction can reflect denominator or stereo changes.
56:     Measure absolute and total L+R band power before claiming bass recovery.
57:     Correlation and side/mid energy describe stereo, not musical usefulness.
58: 17. **Observations:** Read the measurement packet. The listener reports
59:     harsh/centered sound. Neither the PI nor agents have performed controlled
60:     blinded listening yet; no perceptual improvement is established.
61: 18. **Historical boundaries:** Image smoothing reduced some state high-frequency
62:     energy without establishing better output/recovery. Renderer proposal
63:     documents contain unimplemented ideas; they are not available modules.
64: 19. **Inference limits:** Offline waveform filtering tests signal processing.
65:     It is not a proxy experiment for latent-field diffusion, and neither is
66:     a trained diffusion/flow decoder. A zero-output identity branch proves
67:     bypass behavior only. Latent field entropy cannot be recovered from a WAV.
68: 20. **Preservation/budget:** Canonical checkpoints, corpus, existing WAVs, normal
69:     renderer and training stay intact. An offline test may create copied WAVs
70:     and sidecar receipts. Use paired intervals and record level-matching gains.
71: 21. **Missing evidence:** Perceptual ratings, independent training replicas,
72:     pre-side-scale renderer taps, frequency-specific loss gradients, detailed
73:     target mix attribution, and downstream priming outcomes. Flag missing
74:     information explicitly; do not invent controls, inputs, or measurements.
75: 
76: Each review must include an object `context_receipt` with these exact factual
77: fields: `project: "titan_audio_ecosystem"`, `direct_reference_input: false`,
78: `prompt_is_post_render_output: true`, `waveform_filter_tests_latent_diffusion:
79: false`, `canonical_mutation_allowed: false`. Give `summary`, `evidence`
80: (citing source and line numbers), `hypotheses`, `missing_evidence`,
81: `discriminating_experiment`, and `recommendation`. Label speculation. Keep
82: different causal mechanisms separable. These receipt facts are a minimum
83: comprehension gate; the PI independently checks the substantive answer.

## EVIDENCE analysis/audio_center_harshness_20260926/deepseek_packet.md:1-52
SHA256 bb0cf2c1123513ce152aff78fe74d5ab03823bb60986804fe4a73cbc7206972a

1: # Bounded Audio v9 output review packet, 2026-09-26
2: 
3: Question: A listener reports the latest v9 continuation as harsh and biased
4: toward the center. Consider only small, opt-in interventions that preserve the
5: existing trained checkpoint and legacy renderer. The user is interested in
6: diffusion, but no diffusion mechanism has been selected. No downstream Suno/API
7: access is planned. Treat metrics as diagnostics; audible quality requires
8: listening.
9: 
10: Source: local `titan_audio_ecosystem` at commit `6b9f680`; production model,
11: training, and renderer code are unchanged in the working tree. Current tagged
12: lineage `v9-long-01` finished at global step 81,371 / AdamW update 3,116,
13: active morphic depth 16; last 600-second invocation used 7 threads, BPTT 64,
14: core update every 4 tapes. The untagged parent ended at step 60,278 / update
15: 2,786. All WAVs here are finalized 48 kHz stereo PCM16. Latest run is stopped.
16: 
17: Measurements from existing 60-second prime WAVs (no resynthesis). Correlation
18: is centered Pearson L/R; side/mid is 10log10(mean(((L-R)/2)^2) /
19: mean(((L+R)/2)^2)); balance is 10log10(E_L/E_R). Spectral fractions estimate
20: mean mid-channel Hann-window FFT power from 64 evenly spaced 8192-sample
21: windows, normalized over 20-20,000 Hz. They are approximate and not
22: perceptual scores.
23: 
24: | Prime | L/R corr | side/mid dB | L/R balance dB | 20-200 Hz | 2-6 kHz | 6-20 kHz |
25: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
26: | untagged parent | 0.603 | -6.01 | +1.00 | 20.8% | 33.6% | 3.0% |
27: | tagged first continuation | 0.314 | -2.81 | +0.79 | 21.9% | 32.4% | 3.2% |
28: | tagged second continuation | 0.900 | -12.42 | +1.21 | 3.0% | 46.4% | 8.3% |
29: | tagged latest continuation | 0.991 | -23.31 | -0.13 | 2.1% | 47.3% | 6.2% |
30: 
31: The latest full 600-second WAV is not uniformly centered: non-overlapping
32: 60-second intervals have L/R correlation 0.740-0.992; minutes 5-6 have
33: 0.990-0.992 and side/mid about -23 dB. The current 60-second prime is from a
34: very center-heavy interval. Its selection score uses field entropy, activity
35: health, structured complexity, and stagnation, not audible width or harshness.
36: 
37: Latest training trace (704 sampled rows, steps 74,340-81,370): median output
38: stereo correlation 0.963 versus target correlation 0.862; median truthful
39: width 0.027; decoder width control 0.120; width raw -0.946; global pan 0.0999
40: at its +0.10 limit; decoder side control 1.0. The trace's `ultrasonic_ratio`
41: median is 0.0965, but this is a pre-master guardrail, not a harshness score.
42: The latest generated prompt says `focused center image` and `bright glassy
43: upper spectrum`. The target mix and loss history may confound attribution.
44: 
45: Important boundaries: Audio v9 does not take target audio into the current
46: forward pass; target losses affect learning and later host feedback. The
47: checkpoint set and existing audio files must remain byte-for-byte intact.
48: Prior Image experiments found fixed diffusion could suppress high-band state
49: energy without establishing better reconstruction; do not equate smoothing
50: with quality. Distinguish signal-domain smoothing, latent-field diffusion, and
51: a learned spectral/diffusion decoder. Prefer a discriminating frozen or sidecar
52: test before any training fork.

## EVIDENCE docs/AUDIO_SCIENTIFIC_INSTRUMENTATION_PLAN.md:46-144
SHA256 667dc3f0858009c2a3e5044344f5f760854372e007dda1492ce7615f838cda14

46: ## Current Architecture as Implemented
47: 
48: ### Runtime and state
49: 
50: Audio v9 currently implements one stochastic, adaptive, online-trained ecosystem whose relevant state is larger than its learned tensors:
51: 
52: | Area | Current implementation |
53: |---|---|
54: | Micro field | 64 channels at 64x64; updated each chunk through a folded-depth `NeuralCAFolded3D`, deterministic cell-clock masks, macro/recurrent modulation, local anti-rail bias, global-mean damping, and clamping |
55: | Macro field | 64 channels at 32x32; eligible for learned CA advancement every four chunks, probabilistically gated by the host control state, plus structured shear and confinement gain |
56: | Folded manifold | Fixed 64-channel storage interpreted as up to four active 16-feature sheets; active sheet count derives from MorphicStack depth; dormant channels are projected to zero |
57: | Recurrent memory | One 512-unit GRU receiving 64 micro channel means plus a 64-dimensional episodic attention readout |
58: | MorphicStack | 1-64 physically constructed RMSNorm/Swish residual blocks, default 12x512, with a mutable active depth; depth also controls active manifold sheets and far-ring gain |
59: | Episodic memory | 16 detached 512-value snapshots at a 64-chunk cadence, read through learned 512-to-64 query/key/value projections |
60: | Motif/“attractor” memory | Host-side bounded `MotifMemory` containing observations and synthesis controls; it is not the same subsystem as learned episodic attention |
61: | Learned world model | `MonitorHead`: a 522-value state/action input predicts mean and log variance for a 12-value next audio observation |
62: | Planner/controller | Batched learned action scoring every 8 chunks plus a host model-free bandit, exploration, UCB-like pressure, action-use tax, and escape logic inside `HybridController` |
63: | Confinement/recovery | `PotentialController`, `AdaptiveDynamics`, macro shear, micro kicks, sparse heavy-tailed radiation, global-mean damping, clamps, energy homeostasis, and a NaN bio-reset |
64: | Morph policy | Host-side growth/pruning decisions based on mimic pressure, ecology, strict development-probe plateaus, cooldowns, and configured bounds |
65: | Learned objective mixer | `AudioArbiter` maps 14 host features to seven loss weights; this affects learning, not frozen inference directly |
66: | Output | A differentiable multirate renderer followed by bounded saturation, a stateful DC blocker, streamed peak normalization, and 16-bit WAV encoding |
67: 
68: One chunk is 4,096 stereo samples at 48 kHz, approximately 85.33 ms. The ordinary loop combines multiple clocks: micro evolution per chunk, macro eligibility every four chunks, planner refresh every eight chunks, novelty every four chunks, motif consideration every 16 chunks, episodic snapshots every 64 chunks, bounded autograd tapes of at most eight chunks, optimizer horizons selected by `--bptt`, and full-core gradient tapes selected by `--core-update-every`.
69: 
70: ### Learned subsystems and synthesis families
71: 
72: The real learned subsystem inventory, to be used for statistics and ablations rather than guessed labels, is:
73: 
74: - micro and macro folded NCA near/far depthwise perception and pointwise mixing;
75: - GRU recurrent memory;
76: - MorphicStack blocks and norms;
77: - learned episodic query/key/value projections;
78: - recurrent-to-micro asymptotic contraction;
79: - 64-token micro plus 16-token macro spatial-temporal decoder, recurrent cross-attention, six low-rate residual blocks, and a 76-control output head;
80: - carrier pitch, FM ratio/index, wave morph, auxiliary pitch, regional partial ratio/amplitude/damping, oscillator-gain, pan, and stereo-width heads;
81: - left/right eight-basis KAN-inspired wavefolders;
82: - learned `AudioArbiter`; and
83: - learned `MonitorHead` world model/planner.
84: 
85: The audible synthesis families inside the current renderer are:
86: 
87: - phase-continuous carrier plus FM;
88: - three auxiliary modal oscillators per channel;
89: - a 4x8 regional, 32-partial scan synthesis family;
90: - deterministic mid/side excitation tables gated by learned temporal controls;
91: - left/right nonlinear wavefolding;
92: - learned openness and mid/side control;
93: - a stateful Haas-style side path and bounded learned width;
94: - bounded residual global pan; and
95: - the post-render saturation/DC/mastering stages.
96: 
97: Some of these paths combine nonlinearly. Phase 1 must not call every isolated rendering a “stem.” A stem is valid only where an additive component can be exported before a shared nonlinearity and can be recombined exactly. Otherwise the artifact must be labeled an intervention render or leave-one-family-out contrast.
98: 
99: ### Target and corpus semantics
100: 
101: The model forward pass has no target-audio argument. In the ordinary loop, `model.forward(...)` runs before `TargetAudioLoader::sample_chunks(...)`. Target audio participates in spectral, chroma, envelope, recurrence, modulation, level, seam, stereo, and related objectives. Target-derived mimic/error state also influences later host ecology through uncertainty, aperture, morph evidence, and control behavior. It does not directly condition the current chunk’s neural forward or synthesis equations.
102: 
103: Consequences:
104: 
105: 1. Audio v9 cannot honestly perform Image-style held-out reconstruction at inference.
106: 2. Different target chunks presented to identical cloned pre-forward state must produce identical current-chunk audio. This first-chunk invariance is an important negative control.
107: 3. In a frozen multi-chunk experiment, changing target/error feedback may alter later host-controller behavior. That is a **target-feedback-conditioned ecological response**, not a direct reference-conditioned reconstruction.
108: 4. A target-independent rollout must explicitly remove or hold target-derived host feedback and be labeled as an intervention. It must not be silently called the ordinary autonomous mode.
109: 
110: The current corpus scheduler reads a versioned manifest, sorts filesystem paths, builds family groups in deterministic key order, samples a family then a variant, and follows coherent episodes. The existing `load_or_create_corpus_manifest` may create or repair the manifest. Analysis must not call that mutating path.
111: 
112: ### Persistence and optimizer caveat: source-audit correction
113: 
114: Contrary to the provisional caveat in the request, current Audio v9 does persist AdamW state. `PersistentAdamW::save` writes first/second moments, cumulative optimizer update count, global world step, and renderer-control layout version to `titan_optimizer_v9.safetensors`. Startup restores those moments only when:
115: 
116: - a compatible model was loaded;
117: - a compatible world was loaded;
118: - neither fresh-model nor fresh-decoder behavior was requested;
119: - an optimizer file exists; and
120: - its recorded global step exactly matches the world step.
121: 
122: Missing, rejected, or incompatible moments restart from zero with the existing 32-update learning-rate warmup. Morphic resize has explicit exact/resized/new-moment behavior. The checked runtime metadata also records successful optimizer resume.
123: 
124: The real reproducibility limitation is checkpoint-set transactionality. Model, optimizer, and world files are each written atomically, but the set is published in model → optimizer → world order. A kill between renames can leave the model ahead of the world; the source itself says the next run is then not mathematically bit-exact. Phase 1 will report this condition but will not alter it.
125: 
126: Therefore:
127: 
128: - **Phase 1:** observe/hash/validate model, optimizer, world, morph sidecar, and metadata; never change their schemas or save behavior.
129: - **Future schema/behavior boundary:** a transaction manifest, generation ID, previous-generation recovery, and exact all-files resume semantics. This is the relevant future “exact optimizer persistence” work; ordinary moment persistence already exists.
130: 
131: ## Why Scientific Instrumentation Is Needed
132: 
133: Current telemetry is unusually rich for online operation, but it is produced while weights, world state, controller state, target episodes, morph depth, and optimizer state may all change. That makes causal interpretation difficult. Existing scalar traces do not by themselves answer:
134: 
135: - whether a phenomenon is caused by learned weights, initialized anatomy, carried world state, or controller forcing;
136: - whether two trajectories are close internally, perceptually close, both, or neither;
137: - whether a subsystem has a direct neural contribution, a host-control contribution, a training-only contribution, or only correlational telemetry;
138: - whether apparent recovery is contraction toward a paired baseline or merely bounded output;
139: - whether outputs separate by condition or collapse into a generic phenotype;
140: - whether long trajectories are bounded without optimizer drift; or
141: - whether results reproduce across seeds, threads, checkpoint identities, and corpus bytes.
142: 
143: The harness must isolate these factors without changing the object under study.
144: 

## EVIDENCE src/main.rs:70-160
SHA256 e64f5fc3d1664f860c66e275f9662c4f78a1506c1a2f8d666df62a40d8af4453

70:     fn acquire() -> Self {
71:         let _ = std::process::Command::new("termux-wake-lock").status();
72:         Self
73:     }
74: }
75: impl Drop for WakeLockGuard {
76:     fn drop(&mut self) {
77:         let _ = std::process::Command::new("termux-wake-unlock").status();
78:     }
79: }
80: 
81: // --- ECOSYSTEM CONSTANTS ---
82: const SAMPLE_RATE: u32 = 48000;
83: const DURATION_SECONDS: f32 = 240.0;
84: const CHUNK_SIZE: usize = 4096;
85: 
86: // Folded field: fixed 64-state storage on a 64x64 Klein surface = 262,144
87: // scalars, interpreted as up to four cyclic depth sheets. This is deliberately
88: // larger than v3's 96x512 ring while --threads and --bptt remain available as
89: // thermal/RAM controls for other devices.
90: const GRID_H: usize = 64;
91: const GRID_W: usize = 64;
92: const MACRO_H: usize = 32;
93: const MACRO_W: usize = 32;
94: const MACRO_UPDATE_EVERY: u64 = 4;
95: const CA_CHANNELS: usize = 64;
96: const CA_HIDDEN: usize = 128; // conv hidden channels
97: const CA_UPDATE_PROB: f32 = 0.7185; // deterministic asynchronous cell clock
98: const CA_MANIFOLD_MAX_DEPTH: usize = 4;
99: const CA_FEATURE_CHANNELS: usize = CA_CHANNELS / CA_MANIFOLD_MAX_DEPTH;
100: const CA_MANIFOLD_LAYERS_PER_SHEET: usize = 3;
101: const CA_FAR_RING_DILATION: usize = 2;
102: const CA_FAR_RING_MAX_GAIN: f64 = 0.35;
103: const CA_DEPTH_LAPLACIAN_GAIN: f64 = 0.18;
104: 
105: const MEMORY_DIM: usize = 512;
106: const BPTT_WINDOW: usize = 16;
107: const CORE_UPDATE_EVERY: usize = 4;
108: // The folded near/far CA graph is substantially larger than the v8 graph.
109: // Retain at most eight differentiable chunks on mobile; longer requested
110: // horizons still average detached segment gradients, so --bptt 16/32/64 does
111: // not multiply the live graph. This cap is a memory limit, not a horizon limit.
112: const MAX_AUTOGRAD_TAPE_CHUNKS: usize = 8;
113: const SPEC_BINS: usize = 96;
114: // Eight smooth basis functions are enough for a learned waveshaper at 48 kHz.
115: // The former 64-basis pointwise nonlinearity generated strong foldback images
116: // near Nyquist; those are discretization artifacts, not organism complexity.
117: const KAN_BASIS_FUNCTIONS: usize = 8;
118: 
119: // Scan-synth (the 2D field made directly audible)
120: const SCAN_PARTIALS: usize = 32; // 4x8 regional agents -> partial amplitudes
121:                                  // Traverse only a few field columns per audio chunk. The old full-width scan
122:                                  // retriggered at 11.7 Hz and directly produced a small-engine amplitude buzz.
123: const SCAN_COLUMNS_PER_CHUNK: usize = 4;
124: const REGION_ROWS: usize = 4;
125: const REGION_COLS: usize = 8;
126: const REGION_COUNT: usize = REGION_ROWS * REGION_COLS;
127: const REGION_H: usize = GRID_H / REGION_ROWS;
128: const REGION_W: usize = GRID_W / REGION_COLS;
129: 
130: // Channel-aware, low-rate audio decoder. Most parameters execute only on
131: // 32 control frames per 4,096-sample chunk, rather than at waveform rate.
132: const DECODER_MICRO_ROWS: usize = 8;
133: const DECODER_MICRO_COLS: usize = 8;
134: const DECODER_MACRO_ROWS: usize = 4;
135: const DECODER_MACRO_COLS: usize = 4;
136: const DECODER_TOKEN_DIM: usize = 256;
137: const DECODER_CONTROL_FRAMES: usize = 32;
138: const DECODER_WIDTH: usize = 224;
139: const DECODER_EXPANSION: usize = 448;
140: const DECODER_BLOCKS: usize = 6;
141: const DECODER_GLOBAL_CONTROLS: usize = 12;
142: const DECODER_CONTROL_COUNT: usize = DECODER_GLOBAL_CONTROLS + SCAN_PARTIALS * 2;
143: const EXCITATION_TABLE_LEN: usize = CHUNK_SIZE * 16;
144: const TARGET_PREFETCH_CHUNKS: usize = 16;
145: const CELL_CLOCK_MASKS: usize = 8;
146: 
147: // Recursive self-model / hybrid control
148: const OBS_DIM: usize = 12;
149: const ACTION_COUNT: usize = 10;
150: const PLAN_EVERY: usize = 8;
151: const WORLD_VERSION: u32 = 9;
152: const WORLD_MAGIC: [u8; 8] = *b"TITANW9\0";
153: const WORLD_SAVE_EVERY: usize = 2048;
154: const TRACE_SCHEMA_VERSION: u32 = 10;
155: const OPTIMIZER_RENDERER_CONTROL_VERSION: i64 = 1;
156: const TRACE_EVERY: usize = 10;
157: const BUILD_COMMIT: &str = env!("TITAN_GIT_COMMIT");
158: const BUILD_DIRTY: &str = env!("TITAN_GIT_DIRTY");
159: const MOTIF_DEFAULT_CAPACITY: usize = 64;
160: const MOTIF_RUNTIME_MAX_CAPACITY: usize = 4096;

## EVIDENCE src/main.rs:235-260
SHA256 4368c9fd28c73f7194b88f8918f9e5498bcb86328b04a2b7cf28f1625d3f1f15

235: // dominate every ecological objective. v8's 0.31 under-grounded the renderer;
236: // the experimental v9 value 0.05 over-corrected. This middle value keeps most
237: // of v9's adaptive source emphasis while retaining generative autonomy.
238: const RESONANT_AUTONOMY: f32 = 0.15;
239: const GRAD_NORM_MAX: f32 = 5.0; // global-norm ceiling applied directly to gradients before AdamW
240: const RADIATION_REFERENCE_WINDOW: f32 = BPTT_WINDOW as f32;
241: const LOCAL_RAIL_START: f32 = 0.70;
242: const LOCAL_RAIL_STRENGTH: f64 = 0.08;
243: // Damp only the spatial/channel DC mode.
244: const GLOBAL_MEAN_DAMP: f64 = 0.035;
245: // Width without persistent anti-phase dominance.
246: const MAX_STEREO_SIDE_GAIN: f32 = 1.25;
247: // The regional field is already the primary stereo map. Global pan is only a
248: // residual coordinate, kept smooth and modest so it cannot replace spatial
249: // structure with a fixed interchannel level difference.
250: // The completed v9 run was slightly narrower and more correlated than its
251: // targets. Modestly emphasize target-relative side and correlation geometry;
252: // do not raise the learned-width rail, which can create anti-correlation.
253: const STEREO_SIDE_LOSS_WEIGHT: f64 = 0.30;
254: const STEREO_CORRELATION_LOSS_WEIGHT: f64 = 0.95;
255: const STEREO_LEVEL_LOSS_WEIGHT: f64 = 0.45;
256: const TWO_PI: f32 = 2.0 * std::f32::consts::PI;
257: 
258: const ENERGY_HOMEO_RATE: f32 = 0.025; // energy bowl strength applied directly to the scalar
259: 
260: const LARGE_D_DIM: usize = 512;

## EVIDENCE src/main.rs:4225-4315
SHA256 a0bf7531594e80c972691c104ae596219650e5e153a6209866b4aae23f3584ab

4225: struct KANLayer {
4226:     basis_fn: usize,
4227:     w: Tensor,
4228:     mod_proj: Linear,
4229:     freqs: Tensor,
4230:     tilt: Tensor,
4231: }
4232: impl KANLayer {
4233:     fn new(basis_fn: usize, vb: VBV) -> Result<Self> {
4234:         let w = vb.get_with_hints(
4235:             (basis_fn,),
4236:             "weights",
4237:             candle_nn::Init::Randn {
4238:                 mean: 0.0,
4239:                 stdev: 0.1,
4240:             },
4241:         )?;
4242:         let mod_proj = candle_nn::linear(MEMORY_DIM, basis_fn, vb.pp("mod_proj"))?;
4243:         let freq_vec: Vec<f32> = (1..=basis_fn).map(|i| i as f32).collect();
4244:         let freqs = Tensor::from_vec(freq_vec, (1, basis_fn), vb.device())?;
4245:         // A sinusoidal basis' derivative grows with harmonic index. 1/sqrt(k)
4246:         // therefore let the derivative energy grow with k; 1/k^1.25 makes the
4247:         // series and its practical finite-band slope well behaved while still
4248:         // leaving every basis trainable.
4249:         let tilt_vec: Vec<f32> = (1..=basis_fn)
4250:             .map(|i| 1.0 / (i as f32).powf(1.25))
4251:             .collect();
4252:         let tilt = Tensor::from_vec(tilt_vec, (1, basis_fn), vb.device())?;
4253:         Ok(Self {
4254:             basis_fn,
4255:             w,
4256:             mod_proj,
4257:             freqs,
4258:             tilt,
4259:         })
4260:     }
4261:     fn forward(&self, x: &Tensor, mem: &Tensor) -> CResult<Tensor> {
4262:         let (d0, d1) = x.dims2()?;
4263:         let delta_w = self
4264:             .mod_proj
4265:             .forward(mem)?
4266:             .tanh()?
4267:             .reshape((self.basis_fn,))?;
4268:         let active_w = self
4269:             .w
4270:             .add(&delta_w.affine(0.08, 0.0)?)?
4271:             .reshape((1, self.basis_fn))?
4272:             .broadcast_mul(&self.tilt)?;
4273:         let xf = x.reshape((d0 * d1, 1))?;
4274:         let basis = xf.broadcast_mul(&self.freqs)?.sin()?;
4275:         let summed = basis.broadcast_mul(&active_w)?.sum(D::Minus1)?;
4276:         summed
4277:             .reshape((d0, d1))?
4278:             .affine(1.0 / (self.basis_fn as f64).sqrt(), 0.0)
4279:     }
4280: }
4281: 
4282: struct MorphicStack {
4283:     layers: Vec<candle_nn::Sequential>,
4284:     norms: Vec<RmsNorm>,
4285:     active_depth: usize,
4286: }
4287: impl MorphicStack {
4288:     fn new(dim: usize, width: usize, max_depth: usize, vb: VBV) -> Result<Self> {
4289:         let mut layers = Vec::new();
4290:         let mut norms = Vec::new();
4291:         for i in 0..max_depth {
4292:             norms.push(candle_nn::rms_norm(dim, 1e-5, vb.pp(format!("norm{}", i)))?);
4293:             let seq = candle_nn::seq()
4294:                 .add(candle_nn::linear(dim, width, vb.pp(format!("l{}_1", i)))?)
4295:                 .add(candle_nn::Activation::Swish)
4296:                 .add(candle_nn::linear(width, dim, vb.pp(format!("l{}_2", i)))?)
4297:                 .add(candle_nn::Activation::Swish);
4298:             layers.push(seq);
4299:         }
4300:         Ok(Self {
4301:             layers,
4302:             norms,
4303:             active_depth: MORPH_START_DEPTH,
4304:         })
4305:     }
4306:     fn forward(&self, x: &Tensor) -> CResult<Tensor> {
4307:         let mut out = x.clone();
4308:         for i in 0..self.active_depth {
4309:             let residual = self.layers[i]
4310:                 .forward(&self.norms[i].forward(&out)?)?
4311:                 .affine(morphic_residual_gain(i), 0.0)?;
4312:             out = out.add(&residual)?;
4313:         }
4314:         Ok(out)
4315:     }

## EVIDENCE src/main.rs:5107-5255
SHA256 75b0075357ad8791e10102b888002961c294206fe12675c6e30da739050ce5c0

5107:     #[allow(clippy::too_many_arguments)]
5108:     fn forward(
5109:         &mut self,
5110:         micro: &Tensor,
5111:         macro_t: &Tensor,
5112:         mem: &Tensor,
5113:         epi_out: &Tensor,
5114:         phases: [f32; 4], // [carrier_l, carrier_r, mod_l, mod_r] — host f32
5115:         theta_prev: f32,  // theta from last step's batched readout (1-step lag)
5116:         theta_prev2: f32,
5117:         force: bool,
5118:         absolute_step: u64,
5119:         train_ecology: bool,
5120:         energy: f32,
5121:         control: &SynthesisControl,
5122:     ) -> Result<ForwardOut> {
5123:         let dev = micro.device();
5124:         let [pc_l, pc_r, pm_l, pm_r] = phases;
5125:         let manifold_depth = self.manifold_depth();
5126:         let far_ring_gain = self.far_ring_gain();
5127: 
5128:         let mut next_macro = macro_t.clone();
5129:         if force {
5130:             // Sign-symmetric local anti-rail restoring field (the global-mean
5131:             // amplitude barrier lives in the PotentialController).
5132:             let field = local_rail_bias(macro_t)?;
5133:             next_macro = self
5134:                 .macro_ca
5135:                 .forward(
5136:                     macro_t,
5137:                     None,
5138:                     Some(&field),
5139:                     self.macro_clocks.get(absolute_step),
5140:                     manifold_depth,
5141:                     far_ring_gain,
5142:                 )?
5143:                 .tanh()?
5144:                 .affine(0.95, 0.0)?;
5145:         }
5146:         let next_macro = damp_global_mean(&next_macro)?.clamp(-1.0f32, 1.0f32)?;
5147:         let macro_act = next_macro.abs()?.mean_all()?;
5148:         let metab = macro_act.affine(5.0, 0.0)?.clamp(0.01f32, 1.0f32)?;
5149:         let inv_metab = metab.affine(-1.0, 1.0)?;
5150:         let contracted_mem = self.asymptotic_contraction.forward(mem)?;
5151:         let macro_ch = next_macro.mean(D::Minus1)?.mean(D::Minus1)?; // (1, C)
5152:         let macro_mod = contracted_mem.add(&macro_ch)?;
5153:         let micro_field = local_rail_bias(micro)?;
5154:         let raw_next_micro = self.micro_ca.forward(
5155:             micro,
5156:             Some(&macro_mod),
5157:             Some(&micro_field),
5158:             self.micro_clocks.get(absolute_step),
5159:             manifold_depth,
5160:             far_ring_gain,
5161:         )?;
5162:         let next_micro = micro
5163:             .broadcast_mul(&inv_metab)?
5164:             .add(&raw_next_micro.broadcast_mul(&metab)?)?
5165:             .clamp(-1.0f32, 1.0f32)?;
5166:         let next_micro = damp_global_mean(&next_micro)?.clamp(-1.0f32, 1.0f32)?;
5167: 
5168:         // GRU input = channel features ++ episodic attention readout.
5169:         let core_micro_feats = next_micro.mean(D::Minus1)?.mean(D::Minus1)?; // (1, C)
5170:         let gru_in = Tensor::cat(&[&core_micro_feats, epi_out], 1)?; // (1, C + EPI_DIM)
5171:         let next_hidden = self.gru_memory.forward(&gru_in, mem)?;
5172:         let refined_hidden = self.morphic.forward(&next_hidden)?;
5173: 
5174:         // Most horizons adapt the audible decoder only. Detaching here keeps
5175:         // forward ecology exact while preventing the large CA graph from
5176:         // participating in backward. Periodic full horizons retain end-to-end
5177:         // credit assignment.
5178:         let (next_micro, next_macro, next_hidden, refined_hidden) = if train_ecology {
5179:             (next_micro, next_macro, next_hidden, refined_hidden)
5180:         } else {
5181:             (
5182:                 next_micro.detach(),
5183:                 next_macro.detach(),
5184:                 next_hidden.detach(),
5185:                 refined_hidden.detach(),
5186:             )
5187:         };
5188:         let movement_t = next_micro.sub(&micro.detach())?.abs()?.mean_all()?;
5189:         let micro_feats = next_micro.mean(D::Minus1)?.mean(D::Minus1)?; // (1, C)
5190:         let pop_l = micro_feats.narrow(1, 0, 1)?.reshape(())?;
5191:         let pop_r = micro_feats.narrow(1, 1, 1)?.reshape(())?;
5192:         let paired = micro_feats.reshape((CA_CHANNELS / 2, 2))?;
5193:         let pair_sums = paired.sum(0)?; // (2,)
5194:         let temporal_controls =
5195:             self.temporal_decoder
5196:                 .forward(&next_micro, &next_macro, &refined_hidden)?;
5197: 
5198:         let fm_ratios = self
5199:             .fm_mod_ratio
5200:             .forward(&refined_hidden)?
5201:             .affine(4.0, 0.0)?;
5202:         let fm_indices = self
5203:             .fm_mod_index
5204:             .forward(&refined_hidden)?
5205:             .affine(2.0, 0.0)?;
5206:         let ratio_l = fm_ratios.narrow(1, 0, 1)?.reshape(())?;
5207:         let ratio_r = fm_ratios.narrow(1, 1, 1)?.reshape(())?;
5208:         let idx_l = fm_indices.narrow(1, 0, 1)?.reshape(())?;
5209:         let idx_r = fm_indices.narrow(1, 1, 1)?.reshape(())?;
5210:         // Learned in log-Hz. The former abs(parameter) geometry had a cusp at
5211:         // zero and let the carrier fall into sub-audio engine rates. Energy is
5212:         // an amplitude budget, not a pitch control, so it no longer scales Hz.
5213:         let b_l = smooth_bounded_frequency(&self.base_freq_l.reshape(())?.exp()?, 32.0, 880.0)?;
5214:         let b_r = smooth_bounded_frequency(&self.base_freq_r.reshape(())?.exp()?, 32.0, 880.0)?;
5215: 
5216:         // The recurrent state now chooses pitch in log-frequency space. This
5217:         // is continuous (no scale snap or teacher waveform path), bounded to
5218:         // four octaves around the learned base, and gently perturbed by the
5219:         // spatial field so topology remains causally audible.
5220:         let carrier_pitch = self.carrier_pitch_head.forward(&refined_hidden)?;
5221:         let pitch_l = carrier_pitch
5222:             .narrow(1, 0, 1)?
5223:             .reshape(())?
5224:             .affine(std::f64::consts::LN_2 * 2.0, 0.0)?
5225:             .exp()?;
5226:         let pitch_r = carrier_pitch
5227:             .narrow(1, 1, 1)?
5228:             .reshape(())?
5229:             .affine(std::f64::consts::LN_2 * 2.0, 0.0)?
5230:             .exp()?;
5231: 
5232:         let energy_factor = energy.clamp(0.15, 1.0);
5233:         let target_l_raw = b_l
5234:             .mul(&pitch_l)?
5235:             .add(&pop_l.affine(36.0, 0.0)?)?
5236:             .add(&movement_t.affine(18.0, 0.0)?)?;
5237:         let target_r_raw = b_r
5238:             .mul(&pitch_r)?
5239:             .add(&pop_r.affine(36.0, 0.0)?)?
5240:             .add(&movement_t.affine(-18.0, 0.0)?)?;
5241:         let target_l = smooth_bounded_frequency(&target_l_raw, 24.0, 4000.0)?;
5242:         let target_r = smooth_bounded_frequency(&target_r_raw, 24.0, 4000.0)?;
5243: 
5244:         let g = FREQ_GLIDE_SPEED as f64;
5245:         let cur_l = target_l.affine(g, self.current_freq_l as f64 * (1.0 - g))?;
5246:         let cur_r = target_r.affine(g, self.current_freq_r as f64 * (1.0 - g))?;
5247:         let mod_f_l = cur_l.mul(&ratio_l)?.clamp(0.0f32, 4000.0f32)?;
5248:         let mod_f_r = cur_r.mul(&ratio_r)?.clamp(0.0f32, 4000.0f32)?;
5249:         let omega_m_l = mod_f_l.affine(TWO_PI as f64, 0.0)?;
5250:         let omega_m_r = mod_f_r.affine(TWO_PI as f64, 0.0)?;
5251:         let ph_m_l = self
5252:             .t_steps
5253:             .broadcast_mul(&omega_m_l)?
5254:             .affine(1.0, pm_l as f64)?;
5255:         let ph_m_r = self

## EVIDENCE src/main.rs:5290-5648
SHA256 f0cce3a0268988f71bc8d76dfe4de0ae88aefaf63c4fbc653c9d535233dd0221

5290:                     .reshape((CHUNK_SIZE,))?
5291:                     .affine(0.35, 0.0)?,
5292:             )?;
5293: 
5294:         let morphs = self.wave_morph_head.forward(&refined_hidden)?;
5295:         let morph_l = morphs.narrow(1, 0, 1)?.reshape(())?;
5296:         let morph_r = morphs.narrow(1, 1, 1)?.reshape(())?;
5297:         let oscillator_gains = self.oscillator_gain_head.forward(&refined_hidden)?;
5298:         let carrier_gain_l = oscillator_gains
5299:             .narrow(1, 0, 1)?
5300:             .reshape(())?
5301:             .affine(1.15, 0.05)?;
5302:         let carrier_gain_r = oscillator_gains
5303:             .narrow(1, 1, 1)?
5304:             .reshape(())?
5305:             .affine(1.15, 0.05)?;
5306: 
5307:         let mut audio_l = morph_wave(&ph_c_l, &morph_l)?.broadcast_mul(&carrier_gain_l)?;
5308:         let mut audio_r = morph_wave(&ph_c_r, &morph_r)?.broadcast_mul(&carrier_gain_r)?;
5309:         let auxiliary_pitch = self.auxiliary_pitch_head.forward(&refined_hidden)?;
5310:         let modal_bases = [2.03f64, 3.01, 5.07];
5311:         let mut aux_pairs = Vec::with_capacity(3);
5312:         for (j, base_ratio) in modal_bases.into_iter().enumerate() {
5313:             let ratio_l = auxiliary_pitch
5314:                 .narrow(1, j, 1)?
5315:                 .reshape(())?
5316:                 .affine(std::f64::consts::LN_2, base_ratio.ln())?
5317:                 .exp()?;
5318:             let ratio_r = auxiliary_pitch
5319:                 .narrow(1, j + 3, 1)?
5320:                 .reshape(())?
5321:                 .affine(std::f64::consts::LN_2, base_ratio.ln())?
5322:                 .exp()?;
5323:             aux_pairs.push((
5324:                 cur_l.mul(&ratio_l)?.clamp(24.0f32, 12000.0f32)?,
5325:                 cur_r.mul(&ratio_r)?.clamp(24.0f32, 12000.0f32)?,
5326:             ));
5327:         }
5328:         let mut aux_freqs_l = Vec::with_capacity(3);
5329:         let mut aux_freqs_r = Vec::with_capacity(3);
5330:         for (j, (f_l, f_r)) in aux_pairs.into_iter().enumerate() {
5331:             let p_l = self
5332:                 .t_steps
5333:                 .broadcast_mul(&f_l.affine(TWO_PI as f64, 0.0)?)?
5334:                 .affine(1.0, self.aux_phase_l[j] as f64)?
5335:                 .add(&theta_curve)?;
5336:             let p_r = self
5337:                 .t_steps
5338:                 .broadcast_mul(&f_r.affine(TWO_PI as f64, 0.0)?)?
5339:                 .affine(1.0, self.aux_phase_r[j] as f64)?
5340:                 .add(&theta_curve)?;
5341:             let aux_gain_l = oscillator_gains
5342:                 .narrow(1, 2 + j, 1)?
5343:                 .reshape(())?
5344:                 .affine(0.43, 0.02)?;
5345:             let aux_gain_r = oscillator_gains
5346:                 .narrow(1, 5 + j, 1)?
5347:                 .reshape(())?
5348:                 .affine(0.43, 0.02)?;
5349:             audio_l = audio_l.add(&morph_wave(&p_l, &morph_l)?.broadcast_mul(&aux_gain_l)?)?;
5350:             audio_r = audio_r.add(&morph_wave(&p_r, &morph_r)?.broadcast_mul(&aux_gain_r)?)?;
5351:             aux_freqs_l.push(f_l.reshape((1,))?);
5352:             aux_freqs_r.push(f_r.reshape((1,))?);
5353:         }
5354:         let aux_l_refs: Vec<&Tensor> = aux_freqs_l.iter().collect();
5355:         let aux_r_refs: Vec<&Tensor> = aux_freqs_r.iter().collect();
5356:         let aux_freqs_l = Tensor::cat(&aux_l_refs, 0)?;
5357:         let aux_freqs_r = Tensor::cat(&aux_r_refs, 0)?;
5358: 
5359:         // --- REGIONAL SPECTRAL FIELD ---
5360:         // The previous row/column projection discarded most 2-D topology.  A
5361:         // 4x8 regional readout now drives thirty-two independently panned partial
5362:         // agents.  The ratios continuously interpolate between harmonic and
5363:         // inharmonic modal sets; local temporal change adds micro-detuning.
5364:         let field_cm = next_micro.mean(1)?; // (1, H, W)
5365:         let region_grid = field_cm
5366:             .reshape((REGION_ROWS, REGION_H, REGION_COLS, REGION_W))?
5367:             .mean(D::Minus1)?
5368:             .mean(1)?; // (4,8)
5369:         let region_activity = region_grid.reshape((REGION_COUNT,))?;
5370:         let delta_cm = next_micro.sub(micro)?.mean(1)?;
5371:         let region_change = delta_cm
5372:             .reshape((REGION_ROWS, REGION_H, REGION_COLS, REGION_W))?
5373:             .abs()?
5374:             .mean(D::Minus1)?
5375:             .mean(1)?
5376:             .reshape((REGION_COUNT,))?;
5377: 
5378:         let amps = region_activity
5379:             .affine(0.5, 0.5)?
5380:             .add(&region_change.affine(0.55, 0.0)?)?
5381:             .relu()?
5382:             .reshape((SCAN_PARTIALS, 1))?;
5383:         let tilt = self
5384:             .scan_brightness
5385:             .affine(control.spectral_tilt as f64, 1.0)?
5386:             .clamp(0.18f32, 1.82f32)?;
5387:         let learned_amps = self
5388:             .partial_amplitude_head
5389:             .forward(&refined_hidden)?
5390:             .reshape((SCAN_PARTIALS, 1))?
5391:             .affine(1.6, 0.20)?;
5392:         let damping = self
5393:             .partial_damping_head
5394:             .forward(&refined_hidden)?
5395:             .reshape((SCAN_PARTIALS, 1))?;
5396:         let amps = amps.broadcast_mul(&tilt)?.broadcast_mul(&learned_amps)?;
5397:         let amp_sum = amps.sum_all()?.affine(1.0, 1e-4)?;
5398:         let amps_n = amps.broadcast_div(&amp_sum)?;
5399: 
5400:         let inh = control.inharmonicity.clamp(0.0, 1.0);
5401:         let learned_ratio = self
5402:             .partial_ratio_head
5403:             .forward(&refined_hidden)?
5404:             .reshape((SCAN_PARTIALS, 1))?
5405:             .affine(0.35, 0.0)?
5406:             .exp()?;
5407:         let ratios = self
5408:             .scan_harmonic
5409:             .affine((1.0 - inh) as f64, 0.0)?
5410:             .add(&self.scan_inharmonic.affine(inh as f64, 0.0)?)?
5411:             .add(
5412:                 &region_change
5413:                     .reshape((REGION_COUNT, 1))?
5414:                     .broadcast_mul(&self.scan_detune)?
5415:                     .affine((0.7 + 0.8 * inh) as f64, 0.0)?,
5416:             )?
5417:             .broadcast_mul(&learned_ratio)?;
5418:         // Damping is modal attenuation rather than an external filter: every
5419:         // audible partial remains an explicit solution of the learned
5420:         // oscillator bank. Higher modes may decay more strongly, closing the
5421:         // bright-comb shortcut while preserving phase continuity.
5422:         let modal_decay = damping.broadcast_mul(&ratios)?.affine(-0.10, 0.0)?.exp()?;
5423:         let amps_n = amps_n.broadcast_mul(&modal_decay)?;
5424: 
5425:         // Keep one global column envelope as a slow, coherent breath while the
5426:         // regional agents retain spatially independent spectra.
5427:         let cols = field_cm.mean(1)?.reshape((GRID_W, 1))?;
5428:         let offset = self.scan_column_offset % GRID_W;
5429:         let cols = if offset == 0 {
5430:             cols
5431:         } else {
5432:             Tensor::cat(
5433:                 &[
5434:                     &cols.narrow(0, offset, GRID_W - offset)?,
5435:                     &cols.narrow(0, 0, offset)?,
5436:                 ],
5437:                 0,
5438:             )?
5439:         };
5440:         let env = self
5441:             .scan_interp
5442:             .matmul(&cols)?
5443:             .affine(0.30, 0.70)?
5444:             .clamp(0.18f32, 1.25f32)?
5445:             .reshape((1, CHUNK_SIZE))?;
5446:         let scan_freqs_l = ratios.broadcast_mul(&cur_l)?.clamp(24.0f32, 18000.0f32)?;
5447:         let scan_freqs_r = ratios.broadcast_mul(&cur_r)?.clamp(24.0f32, 18000.0f32)?;
5448:         let scan_phase_l = Tensor::from_vec(self.scan_phase_l.to_vec(), (SCAN_PARTIALS, 1), dev)?;
5449:         let scan_phase_r = Tensor::from_vec(self.scan_phase_r.to_vec(), (SCAN_PARTIALS, 1), dev)?;
5450:         let ph_l = scan_freqs_l
5451:             .broadcast_mul(&self.t_steps)?
5452:             .affine(TWO_PI as f64, 0.0)?
5453:             .broadcast_add(&scan_phase_l)?;
5454:         let ph_r = scan_freqs_r
5455:             .broadcast_mul(&self.t_steps)?
5456:             .affine(TWO_PI as f64, 0.0)?
5457:             .broadcast_add(&scan_phase_r)?;
5458:         let partial_mod_l = temporal_controls
5459:             .narrow(0, DECODER_GLOBAL_CONTROLS, SCAN_PARTIALS)?
5460:             .affine(0.65, 1.0)?;
5461:         let partial_mod_r = temporal_controls
5462:             .narrow(0, DECODER_GLOBAL_CONTROLS + SCAN_PARTIALS, SCAN_PARTIALS)?
5463:             .affine(0.65, 1.0)?;
5464:         let partials_l = ph_l
5465:             .sin()?
5466:             .broadcast_mul(&amps_n)?
5467:             .broadcast_mul(&self.scan_pan_l)?
5468:             .mul(&partial_mod_l)?;
5469:         let partials_r = ph_r
5470:             .sin()?
5471:             .broadcast_mul(&amps_n)?
5472:             .broadcast_mul(&self.scan_pan_r)?
5473:             .mul(&partial_mod_r)?;
5474:         let scan_l = partials_l.sum(0)?.reshape((1, CHUNK_SIZE))?.mul(&env)?;
5475:         let scan_r = partials_r.sum(0)?.reshape((1, CHUNK_SIZE))?.mul(&env)?;
5476:         let scan_gain_l = oscillator_gains
5477:             .narrow(1, 8, 1)?
5478:             .reshape(())?
5479:             .affine(0.75, 0.05)?;
5480:         let scan_gain_r = oscillator_gains
5481:             .narrow(1, 9, 1)?
5482:             .reshape(())?
5483:             .affine(0.75, 0.05)?;
5484:         audio_l = audio_l.add(&scan_l.broadcast_mul(&scan_gain_l)?.reshape((CHUNK_SIZE,))?)?;
5485:         audio_r = audio_r.add(&scan_r.broadcast_mul(&scan_gain_r)?.reshape((CHUNK_SIZE,))?)?;
5486: 
5487:         // Learned broad-band and transient excitation is inside the
5488:         // differentiable renderer. The deterministic tables carry no target
5489:         // information; the temporal decoder must learn when and how strongly
5490:         // each mid/side component is audible.
5491:         let excitation_mid = ring_window(&self.excitation_mid, self.excitation_offset, CHUNK_SIZE)?;
5492:         let excitation_side =
5493:             ring_window(&self.excitation_side, self.excitation_offset, CHUNK_SIZE)?;
5494:         let noise_mid = excitation_mid.mul(
5495:             &temporal_controls
5496:                 .narrow(0, 6, 1)?
5497:                 .reshape((CHUNK_SIZE,))?
5498:                 .affine(0.055, 0.0)?,
5499:         )?;
5500:         let noise_side = excitation_side.mul(
5501:             &temporal_controls
5502:                 .narrow(0, 7, 1)?
5503:                 .reshape((CHUNK_SIZE,))?
5504:                 .affine(0.045, 0.0)?,
5505:         )?;
5506:         audio_l = audio_l.add(&noise_mid)?.add(&noise_side)?;
5507:         audio_r = audio_r.add(&noise_mid)?.sub(&noise_side)?;
5508:         audio_l = audio_l.mul(
5509:             &temporal_controls
5510:                 .narrow(0, 0, 1)?
5511:                 .reshape((CHUNK_SIZE,))?
5512:                 .affine(0.35, 1.0)?,
5513:         )?;
5514:         audio_r = audio_r.mul(
5515:             &temporal_controls
5516:                 .narrow(0, 1, 1)?
5517:                 .reshape((CHUNK_SIZE,))?
5518:                 .affine(0.35, 1.0)?,
5519:         )?;
5520:         let fold_drive_l = temporal_controls
5521:             .narrow(0, 10, 1)?
5522:             .reshape((1, CHUNK_SIZE))?
5523:             .affine(0.30, 1.0)?;
5524:         let fold_drive_r = temporal_controls
5525:             .narrow(0, 11, 1)?
5526:             .reshape((1, CHUNK_SIZE))?
5527:             .affine(0.30, 1.0)?;
5528: 
5529:         let audio_l = self
5530:             .wavefolder_l
5531:             .forward(&audio_l.unsqueeze(0)?.mul(&fold_drive_l)?, &refined_hidden)?
5532:             .reshape((1, CHUNK_SIZE))?;
5533:         let audio_r = self
5534:             .wavefolder_r
5535:             .forward(&audio_r.unsqueeze(0)?.mul(&fold_drive_r)?, &refined_hidden)?
5536:             .reshape((1, CHUNK_SIZE))?;
5537: 
5538:         let open_t = refined_hidden
5539:             .abs()?
5540:             .mean_all()?
5541:             .affine(5.0, 0.0)?
5542:             .add(&movement_t)?
5543:             .clamp(0.4f32, 1.0f32)?
5544:             .affine(energy_factor as f64, 0.0)?;
5545:         let open_curve = self.ramp_param(&open_t, &self.prev_openness)?;
5546:         let open_l = open_curve
5547:             .add(
5548:                 &temporal_controls
5549:                     .narrow(0, 8, 1)?
5550:                     .reshape((CHUNK_SIZE,))?
5551:                     .affine(0.25, 0.0)?,
5552:             )?
5553:             .clamp(0.10f32, 1.25f32)?;
5554:         let open_r = open_curve
5555:             .add(
5556:                 &temporal_controls
5557:                     .narrow(0, 9, 1)?
5558:                     .reshape((CHUNK_SIZE,))?
5559:                     .affine(0.25, 0.0)?,
5560:             )?
5561:             .clamp(0.10f32, 1.25f32)?;
5562:         let audio_l = audio_l.broadcast_mul(&open_l.unsqueeze(0)?)?;
5563:         let audio_r = audio_r.broadcast_mul(&open_r.unsqueeze(0)?)?;
5564: 
5565:         let mid = audio_l
5566:             .add(&audio_r)?
5567:             .affine(0.5, 0.0)?
5568:             .mul(&temporal_controls.narrow(0, 4, 1)?.affine(0.30, 1.0)?)?;
5569:         let side = audio_l
5570:             .sub(&audio_r)?
5571:             .affine(0.5, 0.0)?
5572:             .mul(&temporal_controls.narrow(0, 5, 1)?.affine(0.55, 1.0)?)?;
5573:         // These two scalar heads sit after a deep residual stack. Normalize
5574:         // their shared input without adding parameters so a single Adam step
5575:         // cannot turn a large hidden-state norm into a rail-to-rail jump.
5576:         let renderer_control_hidden = refined_hidden.broadcast_div(
5577:             &refined_hidden
5578:                 .sqr()?
5579:                 .mean_keepdim(D::Minus1)?
5580:                 .affine(1.0, 1e-5)?
5581:                 .sqrt()?,
5582:         )?;
5583:         let pan_raw = self
5584:             .spatial_panner
5585:             .forward(&renderer_control_hidden)?
5586:             .reshape(())?;
5587:         let pan_t = soft_global_pan(&pan_raw)?;
5588:         // Haas width from LAST step's pan (host mirror, updated by the batched
5589:         // readback) — removes the one remaining per-step to_scalar sync. Width
5590:         // is a slow spatial parameter; the 85 ms lag is inaudible.
5591:         let width_val = stereo_side_gain(self.last_pan, control.width_mult);
5592:         let width_raw = self
5593:             .stereo_width_head
5594:             .forward(&renderer_control_hidden)?
5595:             .reshape(())?;
5596:         let learned_width = soft_width_control(&width_raw)?;
5597:         let side_wide = side
5598:             .affine(width_val as f64, 0.0)?
5599:             .broadcast_mul(&learned_width)?;
5600:         let side_history = Tensor::cat(&[&self.prev_haas_side, &side_wide], 1)?;
5601:         let side_delayed = side_history.narrow(1, 0, CHUNK_SIZE)?;
5602:         let audio_l = mid.add(&side_delayed)?;
5603:         let audio_r = mid.sub(&side_delayed)?;
5604: 
5605:         let gain_l = pan_t.affine(-0.5, 0.5)?.sqrt()?;
5606:         let gain_r = pan_t.affine(0.5, 0.5)?.sqrt()?;
5607:         let gain_curve_l = self.ramp_param(&gain_l, &self.prev_gain_l)?;
5608:         let gain_curve_r = self.ramp_param(&gain_r, &self.prev_gain_r)?;
5609:         let audio_l = audio_l
5610:             .broadcast_mul(&gain_curve_l.unsqueeze(0)?)?
5611:             .affine(1.414, 0.0)?;
5612:         let audio_r = audio_r
5613:             .broadcast_mul(&gain_curve_r.unsqueeze(0)?)?
5614:             .affine(1.414, 0.0)?;
5615: 
5616:         let stereo = Tensor::cat(&[&audio_l, &audio_r], 0)?.reshape((2, CHUNK_SIZE))?;
5617: 
5618:         self.prev_fm_idx_l = idx_l.detach();
5619:         self.prev_fm_idx_r = idx_r.detach();
5620:         self.prev_openness = open_t.detach();
5621:         self.prev_gain_l = gain_l.detach();
5622:         self.prev_gain_r = gain_r.detach();
5623:         self.prev_haas_side = side_wide.narrow(1, CHUNK_SIZE - 16, 16)?.detach();
5624:         self.scan_column_offset = (self.scan_column_offset + SCAN_COLUMNS_PER_CHUNK) % GRID_W;
5625:         self.excitation_offset = (self.excitation_offset + CHUNK_SIZE) % EXCITATION_TABLE_LEN;
5626: 
5627:         Ok(ForwardOut {
5628:             stereo,
5629:             next_micro,
5630:             next_macro,
5631:             next_hidden,
5632:             refined_hidden,
5633:             movement_t,
5634:             cur_freq_l: cur_l.reshape((1,))?,
5635:             cur_freq_r: cur_r.reshape((1,))?,
5636:             mod_freq_l: mod_f_l.reshape((1,))?,
5637:             mod_freq_r: mod_f_r.reshape((1,))?,
5638:             pan: pan_t.reshape((1,))?,
5639:             pan_raw: pan_raw.reshape((1,))?,
5640:             side_control: temporal_controls
5641:                 .narrow(0, 7, 1)?
5642:                 .abs()?
5643:                 .mean_all()?
5644:                 .reshape((1,))?,
5645:             width_control: learned_width.reshape((1,))?,
5646:             width_raw: width_raw.reshape((1,))?,
5647:             pair_sums,
5648:             aux_freqs_l,

## EVIDENCE src/main.rs:7020-7270
SHA256 979f392857dce0252b76b7ed4232a9c622c085dc4fd58eefc1559aecbe05eb6d

7020:         smoothed_control = current_control;
7021:         let current_predictor_input = predictor_input(
7022:             &hidden_mem.detach(),
7023:             current_action,
7024:             current_control,
7025:             &device,
7026:         )?
7027:         .detach();
7028:         let force_probability = (0.18
7029:             + aperture * 0.48
7030:             + 0.12 * (current_control.shear_mult - 1.0).max(0.0)
7031:             + 0.10 * controller.meta.surprise()
7032:             + 0.32 * escape_strength)
7033:             .clamp(0.05, 0.98);
7034:         let force_macro = absolute_step.is_multiple_of(MACRO_UPDATE_EVERY)
7035:             && rng.gen_range(0.0f32..1.0) < force_probability;
7036: 
7037:         // Episodic attention readout from the current memory (before this step's GRU).
7038:         let epi_out = episodic.read(&hidden_mem, &device)?;
7039: 
7040:         let forward_started = Instant::now();
7041:         let out = model.forward(
7042:             &micro_tape,
7043:             &macro_tape,
7044:             &hidden_mem,
7045:             &epi_out,
7046:             phases,
7047:             theta_prev,
7048:             theta_prev2,
7049:             force_macro,
7050:             absolute_step,
7051:             train_ecology_tape,
7052:             energy_state,
7053:             &current_control,
7054:         )?;
7055:         phase_profiler.model_forward += forward_started.elapsed();
7056:         let ForwardOut {
7057:             stereo: stereo_chunk,
7058:             next_micro,
7059:             next_macro,
7060:             next_hidden,
7061:             refined_hidden,
7062:             movement_t,
7063:             cur_freq_l,
7064:             cur_freq_r,
7065:             mod_freq_l,
7066:             mod_freq_r,
7067:             pan,
7068:             pan_raw,
7069:             side_control,
7070:             width_control,
7071:             width_raw,
7072:             pair_sums,
7073:             aux_freqs_l,
7074:             aux_freqs_r,
7075:             scan_freqs_l,
7076:             scan_freqs_r,
7077:             region_activity,
7078:             region_change,
7079:         } = out;
7080: 
7081:         let loss_started = Instant::now();
7082: 
7083:         let synergy_tensor = calculate_cross_layer_synergy_tensor(&next_micro, &next_macro)?;
7084:         let memory_delta = next_hidden.sub(&hidden_mem)?;
7085:         let tape_delta = next_micro.sub(&micro_tape)?;
7086:         let trans_var = var_all(&memory_delta)?
7087:             .add(&var_all(&tape_delta)?)?
7088:             .affine(1.0, 1e-4)?;
7089:         let cont_entropy_t = trans_var
7090:             .log()
7091:             .map_err(anyhow::Error::msg)?
7092:             .affine(0.5, 0.0)?;
7093:         let empowerment_t = cont_entropy_t
7094:             .affine(1.0, 7.0)?
7095:             .clamp(0.0f32, 5.0f32)?
7096:             .mul(&movement_t.affine(1.0, 1.0)?)?;
7097:         let coarse_micro = decimate2_2d(&next_micro)?;
7098:         let rg_loss = coarse_micro.sub(&next_macro.detach())?.sqr()?.mean_all()?;
7099: 
7100:         // --- MIN-OF-K TARGET SELECTION ---
7101:         // K candidate chunks; the coarse mimic picks the NEAREST, so the model
7102:         // matches a mode of the target set instead of the blur of all modes.
7103:         let audio_for_loss = stereo_chunk.tanh()?;
7104:         let out_spec_l = spec_proj.log_mag(&audio_for_loss.narrow(0, 0, 1)?)?;
7105:         let out_spec_r = spec_proj.log_mag(&audio_for_loss.narrow(0, 1, 1)?)?;
7106:         let target_started = Instant::now();
7107:         let targets = target_loader.sample_chunks(TARGET_K, &mut rng, &device)?;
7108:         phase_profiler.target_load += target_started.elapsed();
7109:         let tgt_specs = spec_proj
7110:             .log_mag(&targets.reshape((TARGET_K * 2, CHUNK_SIZE))?)?
7111:             .detach(); // (K*2, bins)
7112:         let best_k = if TARGET_K == 1 {
7113:             // The manifest sampler intentionally uses one independently
7114:             // selected family. Avoid building and synchronizing a vacuous
7115:             // nearest-candidate graph.
7116:             0
7117:         } else {
7118:             let out_lr = Tensor::cat(&[&out_spec_l, &out_spec_r], 0)?.detach();
7119:             let mut dists = Vec::with_capacity(TARGET_K);
7120:             for k in 0..TARGET_K {
7121:                 dists.push(
7122:                     tgt_specs
7123:                         .narrow(0, k * 2, 2)?
7124:                         .sub(&out_lr)?
7125:                         .sqr()?
7126:                         .mean_all()?
7127:                         .reshape((1,))?,
7128:                 );
7129:             }
7130:             let dist_refs: Vec<&Tensor> = dists.iter().collect();
7131:             let dist_v = Tensor::cat(&dist_refs, 0)?.to_vec1::<f32>()?;
7132:             dist_v
7133:                 .iter()
7134:                 .enumerate()
7135:                 .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
7136:                 .map(|(i, _)| i)
7137:                 .unwrap_or(0)
7138:         };
7139:         target_loader.commit_selection(best_k);
7140:         let target_chunk = targets.narrow(0, best_k, 1)?.reshape((2, CHUNK_SIZE))?;
7141:         let tgt_spec = tgt_specs.narrow(0, best_k * 2, 2)?;
7142: 
7143:         let out_spec = Tensor::cat(&[&out_spec_l, &out_spec_r], 0)?; // (2, bins), grad-carrying
7144:         let mimic_coarse = robust_distance(&out_spec.sub(&tgt_spec)?, 0.03)?;
7145:         let out_bands = log_band_energy(&out_spec)?;
7146:         let target_bands = log_band_energy(&tgt_spec)?.detach();
7147:         let band_loss = robust_distance(&out_bands.sub(&target_bands)?, 0.05)?;
7148:         let out_chroma = chroma_proj.features(&audio_for_loss)?;
7149:         let target_chroma = chroma_proj.features(&target_chunk)?.detach();
7150:         let chroma_loss = robust_distance(&out_chroma.sub(&target_chroma)?, 0.05)?;
7151:         let out_pitch_salience = out_chroma.abs()?.max(D::Minus1)?.mean_all()?;
7152:         let target_pitch_salience = target_chroma.abs()?.max(D::Minus1)?.mean_all()?;
7153:         let pitch_salience_loss =
7154:             robust_distance(&out_pitch_salience.sub(&target_pitch_salience)?, 0.05)?;
7155:         let out_fine = spec_proj_fine.log_mag(&audio_for_loss.reshape((8, 1024))?)?;
7156:         let tgt_fine = spec_proj_fine
7157:             .log_mag(&target_chunk.reshape((8, 1024))?)?
7158:             .detach();
7159:         let mimic_fine = robust_distance(&out_fine.sub(&tgt_fine)?, 0.03)?;
7160:         // Frame energy is phase-invariant but time-aligned. It gives the GRU
7161:         // an honest gradient for pulse, accents, rests, and phrase dynamics
7162:         // that a single chunk-wide spectrum cannot represent.
7163:         let out_env = audio_for_loss
7164:             .reshape((2, 16, CHUNK_SIZE / 16))?
7165:             .sqr()?
7166:             .mean(D::Minus1)?
7167:             .affine(1.0, 1e-5)?
7168:             .sqrt()?;
7169:         let target_env = target_chunk
7170:             .reshape((2, 16, CHUNK_SIZE / 16))?
7171:             .sqr()?
7172:             .mean(D::Minus1)?
7173:             .affine(1.0, 1e-5)?
7174:             .sqrt()?
7175:             .detach();
7176:         let envelope_loss = out_env.sub(&target_env)?.sqr()?.mean_all()?;
7177:         let onset_loss = robust_distance(
7178:             &onset_curve(&out_env)?.sub(&onset_curve(&target_env)?)?,
7179:             0.02,
7180:         )?;
7181:         let mono_out_bands = out_bands.mean(0)?.reshape((1, BAND_COUNT))?;
7182:         let mono_target_bands = target_bands.mean(0)?.reshape((1, BAND_COUNT))?;
7183:         let mono_out_chroma = out_chroma.mean(0)?.reshape((1, 12))?;
7184:         let mono_target_chroma = target_chroma.mean(0)?.reshape((1, 12))?;
7185:         let recurrence_loss = target_relative_recurrence_loss(
7186:             &mono_out_bands,
7187:             &mono_target_bands,
7188:             &mono_out_chroma,
7189:             &mono_target_chroma,
7190:             &feature_history,
7191:             &device,
7192:         )?;
7193:         let mono_out_env = out_env.mean(0)?.reshape((1, 16))?;
7194:         let mono_target_env = target_env.mean(0)?.reshape((1, 16))?;
7195:         let modulation_loss = target_relative_modulation_loss(
7196:             &mono_out_env,
7197:             &mono_target_env,
7198:             &feature_history,
7199:             &modulation_proj,
7200:             &device,
7201:         )?;
7202:         let output_low_band_ratio = first_band_energy_ratio(&out_bands)?;
7203:         let target_low_band_ratio = first_band_energy_ratio(&target_bands)?;
7204:         let low_band_loss = robust_distance(
7205:             &output_low_band_ratio
7206:                 .affine(1.0, 1e-4)?
7207:                 .log()?
7208:                 .sub(&target_low_band_ratio.affine(1.0, 1e-4)?.log()?)?,
7209:             0.05,
7210:         )?;
7211:         let output_stereo = stereo_geometry(&audio_for_loss)?;
7212:         let target_stereo = stereo_geometry(&target_chunk)?;
7213:         let output_side_ratio = output_stereo.side_mid_log_ratio;
7214:         let target_side_ratio = target_stereo.side_mid_log_ratio.detach();
7215:         let output_stereo_corr = output_stereo.correlation;
7216:         let target_stereo_corr = target_stereo.correlation.detach();
7217:         let output_stereo_level_ratio = output_stereo.level_log_ratio;
7218:         let target_stereo_level_ratio = target_stereo.level_log_ratio.detach();
7219:         let side_geometry_loss =
7220:             robust_distance(&output_side_ratio.sub(&target_side_ratio)?, 0.05)?;
7221:         let correlation_loss =
7222:             robust_distance(&output_stereo_corr.sub(&target_stereo_corr)?, 0.03)?;
7223:         let stereo_level_loss = robust_distance(
7224:             &output_stereo_level_ratio.sub(&target_stereo_level_ratio)?,
7225:             0.05,
7226:         )?;
7227:         let pan_center_loss = pan_center_loss(&pan.reshape(())?)?;
7228:         // A side/mid ratio alone admits the panned-mono shortcut L=a*x,
7229:         // R=b*x. Correlation remains +1 in that family, so jointly matching
7230:         // correlation and channel balance makes that degeneracy observable.
7231:         let stereo_balance_loss = side_geometry_loss
7232:             .affine(STEREO_SIDE_LOSS_WEIGHT, 0.0)?
7233:             .add(&correlation_loss.affine(STEREO_CORRELATION_LOSS_WEIGHT, 0.0)?)?
7234:             .add(&stereo_level_loss.affine(STEREO_LEVEL_LOSS_WEIGHT, 0.0)?)?;
7235:         let (development_best, development_mean, development_chroma) = development_bank.scores(
7236:             &audio_for_loss.detach(),
7237:             &out_spec.detach(),
7238:             &out_chroma.detach(),
7239:         )?;
7240:         let (validation_best, validation_mean, validation_chroma) = validation_bank.scores(
7241:             &audio_for_loss.detach(),
7242:             &out_spec.detach(),
7243:             &out_chroma.detach(),
7244:         )?;
7245:         let mimic_loss = mimic_coarse
7246:             .add(&mimic_fine.affine(0.5, 0.0)?)?
7247:             .add(&envelope_loss.affine(0.35, 0.0)?)?
7248:             .add(&band_loss.affine(0.45, 0.0)?)?
7249:             .add(&chroma_loss.affine(0.25, 0.0)?)?
7250:             .add(&pitch_salience_loss.affine(0.08, 0.0)?)?
7251:             .add(&onset_loss.affine(0.25, 0.0)?)?
7252:             .add(&recurrence_loss.affine(0.25, 0.0)?)?
7253:             .add(&modulation_loss.affine(0.20, 0.0)?)?
7254:             .add(&low_band_loss.affine(0.50, 0.0)?)?
7255:             .add(&stereo_balance_loss)?
7256:             .add(&pan_center_loss.affine(PAN_CENTER_LOSS_WEIGHT, 0.0)?)?;
7257: 
7258:         if feature_history.len() >= FEATURE_HISTORY_CHUNKS {
7259:             feature_history.pop_front();
7260:         }
7261:         feature_history.push_back(DetachedFeatureFrame {
7262:             output_band: mono_out_bands.detach(),
7263:             target_band: mono_target_bands.detach(),
7264:             output_chroma: mono_out_chroma.detach(),
7265:             target_chroma: mono_target_chroma.detach(),
7266:             output_envelope: mono_out_env.detach(),
7267:             target_envelope: mono_target_env.detach(),
7268:         });
7269: 
7270:         // Explicitly supervise the seam between adjacent chunks. The prior

## EVIDENCE src/main.rs:7780-7995
SHA256 74638905037a6e90dcec5f0a5a466ee5f521e703fe68f05a5c4b3dbf31013980

7780:         let cur_loss_vec = [
7781:             current_var_val,
7782:             mimic_drift_n,
7783:             movement_loss_val,
7784:             roughness_loss_val,
7785:             rg_v,
7786:             self_model_loss_val,
7787:             empowerment_loss_val,
7788:         ];
7789:         let improvement: Vec<f32> = (0..7)
7790:             .map(|i| (prev_loss_vec[i] - cur_loss_vec[i]).clamp(-1.0, 1.0))
7791:             .collect();
7792:         prev_loss_vec = cur_loss_vec;
7793:         let improvement_t = Tensor::new(improvement, &device)?;
7794:         let arb_progress_loss = w_graph
7795:             .reshape((7,))?
7796:             .mul(&improvement_t)?
7797:             .sum_all()?
7798:             .affine(-0.5, 0.0)?;
7799: 
7800:         let end_of_run = step == total_chunks - 1;
7801:         let tape_boundary =
7802:             steps_in_tape + 1 >= tape_chunks || steps_since_update + 1 >= bptt_window || end_of_run;
7803:         // A tape-scale view supplies the missing temporal receptive field.
7804:         // Partial final tapes remain covered by the two short scales and seam
7805:         // loss, while complete tapes use the precomputed long projector.
7806:         let trajectory_loss = if tape_boundary && audio_sequence.len() == tape_chunks {
7807:             let out_refs: Vec<&Tensor> = audio_sequence.iter().collect();
7808:             let target_refs: Vec<&Tensor> = target_sequence.iter().collect();
7809:             let out_long = Tensor::cat(&out_refs, 1)?;
7810:             let target_long = Tensor::cat(&target_refs, 1)?.detach();
7811:             let out_long_spec = spec_proj_long.log_mag(&out_long)?;
7812:             let target_long_spec = spec_proj_long.log_mag(&target_long)?.detach();
7813:             Some(robust_distance(
7814:                 &out_long_spec.sub(&target_long_spec)?,
7815:                 0.03,
7816:             )?)
7817:         } else {
7818:             None
7819:         };
7820: 
7821:         // --- TOTAL LOSS ---
7822:         // Source evidence is an invariant, not a preference the learned
7823:         // arbiter may switch off. The arbiter modulates additional emphasis
7824:         // around a non-zero grounding floor.
7825:         let source_weight = 0.80 + 0.35 * lw[1] * (1.0 - RESONANT_AUTONOMY);
7826:         let mut total_loss = mimic_loss.affine(source_weight as f64, 0.0)?;
7827:         total_loss = total_loss.add(&level_loss.affine((lw[0] * 0.75) as f64, 0.0)?)?;
7828:         total_loss = total_loss.add(&saturation_loss.affine(2.0, 0.0)?)?;
7829:         total_loss = total_loss.add(&movement_loss.affine((lw[2] * 0.20) as f64, 0.0)?)?;
7830:         let anti_weld_weight = 0.06 + 0.20 * adaptive_dynamics.stagnation;
7831:         let regional_weld_weight = 0.04 + 0.12 * adaptive_dynamics.stagnation;
7832:         total_loss = total_loss.add(&movement_floor_loss.affine(anti_weld_weight as f64, 0.0)?)?;
7833:         total_loss =
7834:             total_loss.add(&regional_floor_loss.affine(regional_weld_weight as f64, 0.0)?)?;
7835:         total_loss = total_loss.add(&roughness_loss.affine(lw[3] as f64, 0.0)?)?;
7836:         total_loss = total_loss.add(&boundary_loss.affine(0.50, 0.0)?)?;
7837:         if let Some(long) = trajectory_loss {
7838:             total_loss = total_loss.add(&long.affine(0.35, 0.0)?)?;
7839:         }
7840:         total_loss = total_loss.add(&reg_loss.affine(0.002, 0.0)?)?;
7841:         total_loss = total_loss.add(&rg_loss.affine((0.15 * lw[4].max(0.2)) as f64, 0.0)?)?;
7842:         total_loss =
7843:             total_loss.add(&self_model_loss.affine((0.30 * lw[5].max(0.2)) as f64, 0.0)?)?;
7844:         total_loss = total_loss.add(&empowerment_loss.affine((0.10 * lw[6]) as f64, 0.0)?)?;
7845:         // Coupling band, released by the controller's over-coupling/heat signal.
7846:         let synergy_w = (SYNERGY_BAND_W * (1.0 - 0.7 * pot.couple_release)).max(0.1f32);
7847:         total_loss = total_loss.add(&synergy_loss.affine(synergy_w as f64, 0.0)?)?;
7848:         // Entropy BONUS (v3 minimized it — see AudioArbiter): adding neg_entropy
7849:         // with a positive coefficient maximizes mixing entropy.
7850:         total_loss = total_loss.add(&neg_entropy.affine(0.05, 0.0)?)?;
7851:         total_loss = total_loss.add(&arb_progress_loss)?;
7852:         if let Some(nl) = novelty_loss {
7853:             let novelty_weight = NOVELTY_W * adaptive_dynamics.stagnation as f64;
7854:             total_loss = total_loss.add(&nl.affine(novelty_weight, 0.0)?)?;
7855:         }
7856: 
7857:         tape_loss = Some(match tape_loss.take() {
7858:             None => total_loss,
7859:             Some(w) => w.add(&total_loss)?,
7860:         });
7861:         steps_in_tape += 1;
7862:         steps_since_update += 1;
7863: 
7864:         // `sigma` is movement persistence, not a calibrated branching ratio.
7865:         // Retain it as evidence, but do not shape LR around an unproved sigma=1
7866:         // singularity.
7867:         let crit_gain = 1.0f32;
7868:         let phi_gate = 1.0 / (1.0 + phi);
7869:         let curiosity_lr_gain = 1.0 + curiosity_factor as f64 * LR_CURIOSITY_MAX;
7870:         latest_lr_gain = crit_gain as f64 * pot.lr_heat * phi_gate as f64 * curiosity_lr_gain;
7871:         tape_lr_gain_sum += latest_lr_gain;
7872: 
7873:         if tape_boundary {
7874:             if let Some(w) = tape_loss.take() {
7875:                 let segment_steps = steps_in_tape;
7876:                 let segment_mean = w.affine(1.0 / segment_steps as f64, 0.0)?;
7877:                 match segment_mean.to_scalar::<f32>() {
7878:                     Ok(loss_val) if loss_val.is_finite() && tape_lr_gain_sum.is_finite() => {
7879:                         let backward_started = Instant::now();
7880:                         let backward_result = segment_mean.backward();
7881:                         phase_profiler.backward += backward_started.elapsed();
7882:                         match backward_result {
7883:                             Ok(mut segment_grads) => {
7884:                                 // Store a weighted SUM across bounded tape segments.
7885:                                 // Dividing once at the optimizer boundary makes -w 64
7886:                                 // a 64-chunk gradient average without a 64-chunk graph.
7887:                                 let segment_weight = segment_steps as f64;
7888:                                 if let Some(grads) = accumulated_grads.as_mut() {
7889:                                     for var in varmap.all_vars() {
7890:                                         if let Some(g) = segment_grads.remove(var.as_tensor()) {
7891:                                             let weighted = g.affine(segment_weight, 0.0)?.detach();
7892:                                             let merged = match grads.remove(var.as_tensor()) {
7893:                                                 Some(previous) => previous.add(&weighted)?.detach(),
7894:                                                 None => weighted,
7895:                                             };
7896:                                             grads.insert(var.as_tensor(), merged);
7897:                                         }
7898:                                     }
7899:                                 } else {
7900:                                     for var in varmap.all_vars() {
7901:                                         if let Some(g) = segment_grads.remove(var.as_tensor()) {
7902:                                             segment_grads.insert(
7903:                                                 var.as_tensor(),
7904:                                                 g.affine(segment_weight, 0.0)?.detach(),
7905:                                             );
7906:                                         }
7907:                                     }
7908:                                     accumulated_grads = Some(segment_grads);
7909:                                 }
7910:                                 accumulated_steps += segment_steps;
7911:                                 accumulated_lr_gain_sum += tape_lr_gain_sum;
7912:                             }
7913:                             Err(e) => println!(
7914:                                 "! WARNING: backward failed: {} — dropping tape segment.",
7915:                                 e
7916:                             ),
7917:                         }
7918:                     }
7919:                     _ => println!("! WARNING: Non-finite loss detected. Dropping tape segment."),
7920:                 }
7921:             }
7922:             steps_in_tape = 0;
7923:             tape_lr_gain_sum = 0.0;
7924:             audio_sequence.clear();
7925:             target_sequence.clear();
7926:             prev_audio_tail = None;
7927:             prev_target_tail = None;
7928:         }
7929: 
7930:         let update_boundary = steps_since_update >= bptt_window || end_of_run;
7931:         if update_boundary {
7932:             if let Some(mut grads) = accumulated_grads.take() {
7933:                 if accumulated_steps > 0 {
7934:                     let mean_scale = 1.0 / accumulated_steps as f64;
7935:                     let mut sq = Tensor::zeros((), DType::F32, &device)?;
7936:                     for var in varmap.all_vars() {
7937:                         if let Some(g) = grads.get(var.as_tensor()) {
7938:                             sq = sq.add(&g.affine(mean_scale, 0.0)?.sqr()?.sum_all()?)?;
7939:                         }
7940:                     }
7941:                     let gnorm = sq.to_scalar::<f32>().unwrap_or(f32::INFINITY).sqrt();
7942:                     if gnorm.is_finite() {
7943:                         // True global-norm clipping happens before AdamW sees the
7944:                         // accumulated mean, so a hot segment cannot poison moments.
7945:                         let clip_scale = (GRAD_NORM_MAX / gnorm.max(1e-6)).min(1.0) as f64;
7946:                         let optimizer_scale = mean_scale * clip_scale;
7947:                         latest_grad_norm = gnorm;
7948:                         latest_clip_scale = clip_scale as f32;
7949:                         for var in varmap.all_vars() {
7950:                             if let Some(g) = grads.remove(var.as_tensor()) {
7951:                                 grads.insert(
7952:                                     var.as_tensor(),
7953:                                     g.affine(optimizer_scale, 0.0)?.detach(),
7954:                                 );
7955:                             }
7956:                         }
7957:                         let mean_lr_gain = accumulated_lr_gain_sum / accumulated_steps as f64;
7958:                         let moment_warmup = (0.20
7959:                             + 0.80 * (optimizer.cumulative_updates() + 1) as f64 / 32.0)
7960:                             .min(1.0);
7961:                         optimizer.set_learning_rate(target_lr * mean_lr_gain * moment_warmup);
7962:                         let optimizer_started = Instant::now();
7963:                         let _ = optimizer.step(&grads);
7964:                         phase_profiler.optimizer += optimizer_started.elapsed();
7965:                         optimizer_update_count += 1;
7966:                     } else {
7967:                         println!("! WARNING: non-finite grad norm — skipping horizon.");
7968:                     }
7969:                 }
7970:             }
7971:             steps_since_update = 0;
7972:             accumulated_steps = 0;
7973:             accumulated_lr_gain_sum = 0.0;
7974:         }
7975: 
7976:         if tape_boundary {
7977:             micro_tape = next_micro.detach();
7978:             macro_tape = next_macro.detach();
7979:             hidden_mem = next_hidden.detach();
7980:         } else {
7981:             micro_tape = next_micro;
7982:             macro_tape = next_macro;
7983:             hidden_mem = next_hidden;
7984:         }
7985: 
7986:         // Radiation is an ecological event, not an optimizer event. Convert
7987:         // the old default-window probability to an equivalent per-chunk
7988:         // hazard so changing --bptt no longer changes organism dynamics.
7989:         let radiation_window_probability = (RADIATE_PROB
7990:             + curiosity_factor * 0.04
7991:             + (current_control.kick_mult - 1.0).max(0.0) * 0.03
7992:             + controller.meta.surprise() * 0.02
7993:             + escape_strength * 0.12)
7994:             .clamp(0.0, 0.30);
7995:         let radiation_probability =

## EVIDENCE src/main.rs:8048-8150
SHA256 c12c789d7278477b8b85621728f3c66aae2efa1ba027e0519cd21b23d64abe85

8048:         // --- TRUTHFUL POST PATH ---
8049:         // The prior system let discrete controller actions inject spectral noise, modal
8050:         // ringing, and FDN echo *after* the differentiable source loss. That
8051:         // created an acoustic shortcut: the controller could buy entropy that
8052:         // the organism could neither predict nor learn to synthesize. Those
8053:         // effects and states are gone; the audible path is the learned renderer
8054:         // followed only by bounded saturation and a stateful DC blocker.
8055:         let output_started = Instant::now();
8056:         let audio_normalized_vec = audio_normalized.to_vec2::<f32>()?;
8057:         let mut audio_l = audio_normalized_vec[0].clone();
8058:         let mut audio_r = audio_normalized_vec[1].clone();
8059:         for i in 0..CHUNK_SIZE {
8060:             audio_l[i] = (audio_l[i] * 0.92).tanh();
8061:             audio_r[i] = (audio_r[i] * 0.92).tanh();
8062: 
8063:             // Stateful DC blocker is part of the observed signal path, so the
8064:             // recursive controller hears the same waveform later mastered.
8065:             let xl = audio_l[i];
8066:             let yl = xl - dc_x1_l + dc_pole * dc_y1_l;
8067:             dc_x1_l = xl;
8068:             dc_y1_l = yl;
8069:             audio_l[i] = yl;
8070:             let xr = audio_r[i];
8071:             let yr = xr - dc_x1_r + dc_pole * dc_y1_r;
8072:             dc_x1_r = xr;
8073:             dc_y1_r = yr;
8074:             audio_r[i] = yr;
8075:             raw_peak = raw_peak.max(yl.abs()).max(yr.abs());
8076:         }
8077: 
8078:         let post = spectral_mon.analyze(
8079:             &audio_l,
8080:             &audio_r,
8081:             movement,
8082:             synergy_val,
8083:             field_entropy,
8084:             sigma,
8085:         );
8086:         let s_sig = &post.json;
8087:         uncertainty.update(
8088:             s_sig,
8089:             &m_sig,
8090:             mimic_drift_n,
8091:             synergy_val,
8092:             empowerment_val,
8093:             &controller.meta,
8094:         );
8095:         phi = uncertainty.phi;
8096: 
8097:         let region_change_mean = region_change_host.iter().sum::<f32>() / REGION_COUNT as f32;
8098:         let observation_delta = last_observation
8099:             .as_ref()
8100:             .map(|prev| post.observation.distance(prev))
8101:             .unwrap_or(0.08);
8102:         let structured_complexity = post.observation.structured_complexity();
8103:         adaptive_dynamics.observe(
8104:             movement,
8105:             region_change_mean,
8106:             observation_delta,
8107:             structured_complexity,
8108:             sigma,
8109:             controller.meta.confidence,
8110:         );
8111:         let recurrence = motifs.recurrence(&post.observation, absolute_step);
8112:         let reward = post.observation.reward_against(
8113:             last_observation.as_ref(),
8114:             recurrence,
8115:             &adaptive_dynamics,
8116:             controller.meta.confidence,
8117:             controller.action_age,
8118:         );
8119:         controller.bandit.update(current_action, reward);
8120:         adaptive_dynamics.observe_reward(reward);
8121:         if absolute_step.is_multiple_of(MOTIF_EVERY as u64) {
8122:             motifs.maybe_store(
8123:                 &post.observation,
8124:                 current_control,
8125:                 absolute_step,
8126:                 motif_capacity,
8127:                 &adaptive_dynamics,
8128:                 &mut motif_diagnostics,
8129:             );
8130:         }
8131:         last_observation = Some(post.observation.clone());
8132:         pending_predictor_input = Some(current_predictor_input);
8133: 
8134:         for i in 0..CHUNK_SIZE {
8135:             let byte = i * 8;
8136:             raw_chunk_bytes[byte..byte + 4].copy_from_slice(&audio_l[i].to_le_bytes());
8137:             raw_chunk_bytes[byte + 4..byte + 8].copy_from_slice(&audio_r[i].to_le_bytes());
8138:         }
8139:         raw_audio_writer.write_all(&raw_chunk_bytes)?;
8140:         phase_profiler.output_io += output_started.elapsed();
8141:         chunk_scores.push(
8142:             field_entropy * (0.25 + 0.50 * adaptive_dynamics.activity_health)
8143:                 + structured_complexity * 0.75
8144:                 - adaptive_dynamics.stagnation * 0.25,
8145:         );
8146:         completed_chunks += 1;
8147:         evolved_chunks = step + 1;
8148: 
8149:         if step % TRACE_EVERY == 0 {
8150:             let sample_index = trace_rows;

## EVIDENCE src/main.rs:8448-8542
SHA256 79ae91e8c6399b91f60f198a4b20ffaa106beb662ac7f054c77736626c2dfa34

8448:     // ---- STREAMED MASTERING + PRIME EXTRACTION ----
8449:     raw_audio_writer.flush()?;
8450:     drop(raw_audio_writer);
8451:     let norm = 0.891 / raw_peak.max(1e-6);
8452:     println!(
8453:         "Mastering: streamed DC-blocked peak {:.3} normalized to -1 dBFS (gain {:.2}x).",
8454:         raw_peak, norm
8455:     );
8456: 
8457:     let total_frames = completed_chunks * CHUNK_SIZE;
8458:     let rendered_seconds = total_frames as f32 / SAMPLE_RATE as f32;
8459:     let prime_secs = 60.0f32.min(rendered_seconds.max(CHUNK_SIZE as f32 / SAMPLE_RATE as f32));
8460:     let win = ((SAMPLE_RATE as f32 * prime_secs / CHUNK_SIZE as f32) as usize)
8461:         .max(1)
8462:         .min(chunk_scores.len().max(1));
8463:     let (best_start, best_sum) = if !chunk_scores.is_empty() {
8464:         let mut run: f32 = chunk_scores.iter().take(win).sum();
8465:         let mut best_start = 0usize;
8466:         let mut best_sum = run;
8467:         if chunk_scores.len() > win {
8468:             for start in 1..=(chunk_scores.len() - win) {
8469:                 run += chunk_scores[start + win - 1] - chunk_scores[start - 1];
8470:                 if run > best_sum {
8471:                     best_sum = run;
8472:                     best_start = start;
8473:                 }
8474:             }
8475:         }
8476:         (best_start, best_sum)
8477:     } else {
8478:         (0usize, 0.0f32)
8479:     };
8480:     let prime_start_frame = best_start * CHUNK_SIZE;
8481:     let prime_end_frame = ((best_start + win) * CHUNK_SIZE).min(total_frames);
8482:     let fade = 2048usize.min(prime_end_frame.saturating_sub(prime_start_frame) / 4);
8483: 
8484:     let spec = hound::WavSpec {
8485:         channels: 2,
8486:         sample_rate: SAMPLE_RATE,
8487:         bits_per_sample: 16,
8488:         sample_format: hound::SampleFormat::Int,
8489:     };
8490:     let output_path = hashed_audio_path(
8491:         &base_dir,
8492:         "rust_ecosystem_out",
8493:         run_tag.as_deref(),
8494:         &audio_file_hash,
8495:     );
8496:     let prime_path = hashed_audio_path(
8497:         &base_dir,
8498:         &format!("titan_prime_{}s", prime_secs as u32),
8499:         run_tag.as_deref(),
8500:         &audio_file_hash,
8501:     );
8502:     let mut output_writer = hound::WavWriter::create(&output_path, spec)?;
8503:     let mut prime_writer = hound::WavWriter::create(&prime_path, spec)?;
8504:     let mut raw_reader = BufReader::with_capacity(1 << 20, File::open(&raw_audio_path)?);
8505:     let mut audio_frames =
8506:         Vec::with_capacity(prime_end_frame.saturating_sub(prime_start_frame) * 2);
8507:     let mut bytes = [0u8; 4];
8508:     for frame in 0..total_frames {
8509:         raw_reader.read_exact(&mut bytes)?;
8510:         let left = f32::from_le_bytes(bytes) * norm;
8511:         raw_reader.read_exact(&mut bytes)?;
8512:         let right = f32::from_le_bytes(bytes) * norm;
8513:         output_writer.write_sample((left * 32767.0).clamp(-32768.0, 32767.0) as i16)?;
8514:         output_writer.write_sample((right * 32767.0).clamp(-32768.0, 32767.0) as i16)?;
8515:         if frame >= prime_start_frame && frame < prime_end_frame {
8516:             let g = if fade == 0 {
8517:                 1.0
8518:             } else if frame < prime_start_frame + fade {
8519:                 (frame - prime_start_frame) as f32 / fade as f32
8520:             } else if frame >= prime_end_frame.saturating_sub(fade) {
8521:                 (prime_end_frame - frame) as f32 / fade as f32
8522:             } else {
8523:                 1.0
8524:             };
8525:             let pl = left * g;
8526:             let pr = right * g;
8527:             prime_writer.write_sample((pl * 32767.0).clamp(-32768.0, 32767.0) as i16)?;
8528:             prime_writer.write_sample((pr * 32767.0).clamp(-32768.0, 32767.0) as i16)?;
8529:             audio_frames.push(pl);
8530:             audio_frames.push(pr);
8531:         }
8532:     }
8533:     output_writer.finalize()?;
8534:     prime_writer.finalize()?;
8535:     let _ = std::fs::remove_file(&raw_audio_path);
8536:     println!("Audio saved to {}", output_path);
8537:     println!(
8538:         "Priming segment: chunks {}..{} (avg score {:.2}) -> {}",
8539:         best_start,
8540:         best_start + win,
8541:         best_sum / win.max(1) as f32,
8542:         prime_path

## EVIDENCE src/main.rs:8665-8698
SHA256 45531e115cf92af234df7cabe8e8cd446e4489a173251afa689b37d5fa32a7fe

8665:     let avg_synergy = trace_synergy_sum / n_trace;
8666:     let avg_temp = trace_temp_sum / n_trace;
8667:     let avg_sigma = if trace_rows > 0 {
8668:         trace_sigma_sum / n_trace
8669:     } else {
8670:         1.0
8671:     };
8672:     let avg_pi = trace_pi_sum / n_trace;
8673:     let avg_field_h = if field_entropy_n > 0 {
8674:         field_entropy_sum / field_entropy_n as f64
8675:     } else {
8676:         0.0
8677:     };
8678:     let dom_phase = semantic.dominant_phase();
8679:     let dom_archetype = semantic.dominant_archetype();
8680:     let final_depth = model.depth();
8681: 
8682:     let prompt = format!(
8683:         "Style: {}, {}, {}, {}. Texture: {}. Tempo: {}. Tone: {}. Space: {}. Field: {} regime · {} archetype · depth L{:02}. [Phi: {:.2}, Sigma: {:.3}, Temp: {:.2}, PI-proxy: {:.2}, Aperture: {:.2}, Synergy: {:.2}, Field-Entropy: {:.2}b, Energy: {:.2}, Model-Confidence raw/effective: {:.2}/{:.2}, Ecology-Health: {:.2}, Stagnation: {:.2}, Motifs: {}/{}, Seed: {}]",
8684:         if avg_phi > 0.85 { "Hyper-Resonant" } else { "Chaotic" },
8685:         if avg_aperture > 0.5 { "Evolving" } else { "Stable" },
8686:         if total_complexity > 500.0 { "Dense" } else { "Minimal" },
8687:         if avg_sigma > 0.97 && avg_sigma < 1.03 { "Critical-Edge Emergence" } else { "Information-Theoretic Glitch" },
8688:         if avg_synergy > 0.6 { "Crystalline-Autonomous" } else if avg_phi > 0.55 { "Organic" } else { "Grit" },
8689:         tempo_txt, tone, width_word, dom_phase, dom_archetype, final_depth,
8690:         avg_phi, avg_sigma, avg_temp, avg_pi, avg_aperture, avg_synergy, avg_field_h,
8691:         energy_state, controller.meta.confidence, adaptive_dynamics.effective_model_weight,
8692:         adaptive_dynamics.activity_health, adaptive_dynamics.stagnation,
8693:         motifs.entries.len(), motif_capacity, seed
8694:     );
8695:     println!("\n=== GENERATIVE PRIMING PROMPT ===\n{}", prompt);
8696:     let prompt_path = artifact_path(&base_dir, "suno_priming_prompt", "txt", run_tag.as_deref());
8697:     std::fs::write(&prompt_path, &prompt)?;
8698: 

## EVIDENCE src/stereo.rs:1-51
SHA256 25fc62745b54f6c5fab7effa27bc4c4bc8a503f6f489cb20bc95d4998fbf2485

1: use candle_core::{Result, Tensor};
2: 
3: pub const GLOBAL_PAN_LIMIT: f64 = 0.10;
4: pub const PAN_CENTER_LOSS_WEIGHT: f64 = 0.10;
5: pub const WIDTH_CONTROL_MIN: f64 = 0.05;
6: pub const WIDTH_CONTROL_MAX: f64 = 0.50;
7: pub const WIDTH_CONTROL_SOFTNESS: f64 = 1.0;
8: 
9: pub fn side_gain(last_pan: f32, width_mult: f32, maximum: f32) -> f32 {
10:     ((1.0 + last_pan.abs() * 0.8) * width_mult.clamp(0.5, 1.6)).clamp(0.5, maximum)
11: }
12: 
13: pub fn soft_global_pan(raw: &Tensor) -> Result<Tensor> {
14:     // z / sqrt(1 + z^2) is smooth and bounded like tanh, but its gradient
15:     // decays polynomially instead of exponentially when an old checkpoint has
16:     // driven the residual pan head far into saturation.
17:     let scale = raw.sqr()?.affine(1.0, 1.0)?.sqrt()?;
18:     raw.div(&scale)?.affine(GLOBAL_PAN_LIMIT, 0.0)
19: }
20: 
21: pub fn pan_center_loss(pan: &Tensor) -> Result<Tensor> {
22:     // Penalize only the bounded audible residual. An earlier penalty on the
23:     // raw coordinate could dominate the complete loss when an old checkpoint
24:     // had already driven its pan head far into saturation.
25:     pan.sqr()
26: }
27: 
28: pub fn soft_width_control(raw: &Tensor) -> Result<Tensor> {
29:     // Preserve the width head's tensor layout while replacing its saturating
30:     // sigmoid. The normalized head input keeps the useful operating region
31:     // near zero, while the polynomial tail retains a recovery gradient and a
32:     // 0.50 ceiling prevents learned side gain from forcing anti-correlation.
33:     let unit = raw.div(
34:         &raw.sqr()?
35:             .affine(1.0, WIDTH_CONTROL_SOFTNESS * WIDTH_CONTROL_SOFTNESS)?
36:             .sqrt()?,
37:     )?;
38:     let half_range = (WIDTH_CONTROL_MAX - WIDTH_CONTROL_MIN) * 0.5;
39:     let midpoint = (WIDTH_CONTROL_MAX + WIDTH_CONTROL_MIN) * 0.5;
40:     unit.affine(half_range, midpoint)
41: }
42: 
43: pub fn regional_pan_position(partial: usize, columns: usize) -> f32 {
44:     let column = partial % columns;
45:     column as f32 / (columns - 1) as f32 * 2.0 - 1.0
46: }
47: 
48: pub fn correlation_aware_width(side_energy_width: f32, stereo_corr: f32) -> f32 {
49:     let incoherence = ((1.0 - stereo_corr.clamp(-1.0, 1.0)) * 0.5).sqrt();
50:     (side_energy_width * incoherence).clamp(0.0, 1.0)
51: }

## EVIDENCE METRICS.md:45-158
SHA256 2de340d138553c34cba16d19dae52bb8d331a6c9a4a0f2d3c9abbf73d588ac1d

45: ## Measurements and estimators
46: 
47: - `sigma` is a mean-centered, multi-lag propagation-slope estimate over recent
48:   movement. It is not a literal physical branching ratio. Its accompanying
49:   `criticality_confidence` discounts low-variance, non-stationary, short, or
50:   cross-lag-inconsistent windows.
51: - `phi` is a bounded audio-structure proxy derived from observable signal
52:   statistics. It is not Integrated Information Theory's Phi.
53: - `pi_proxy` summarizes multi-lag predictive structure. It is not causal proof
54:   of information integration.
55: - `empowerment` is a transition-variance heuristic coupled to movement. It is
56:   not channel-capacity empowerment.
57: - `novelty_dmin` is distance to recent, level-normalized log-spectral shapes.
58:   It detects timbral change, not semantic or compositional novelty.
59: - `carrier_freq_l`, `carrier_freq_r`, and `carrier_beat_hz` expose low-frequency
60:   beating directly. `mimic_coarse`, `mimic_fine`, `band_loss`, `chroma_loss`,
61:   `onset_loss`, `modulation_loss`, `recurrence_loss`, `boundary_loss`, and
62:   `level_loss` separate source-grounding terms instead of collapsing them into
63:   one score.
64: - `output_low_band_ratio` and `target_low_band_ratio` compare the first
65:   supervised 20 Hz log band. They expose the sub-bass/RMS shortcut directly.
66: - `output_side_mid_log_ratio` and `target_side_mid_log_ratio` expose stereo
67:   side dominance. `decoder_stereo_corr`/`target_stereo_corr` and the two
68:   `stereo_level_log_ratio` fields close the panned-mono loophole: channel gain
69:   imbalance can no longer masquerade as spatial width. `stereo_balance_loss`
70:   combines all three relations without mixing target audio into the renderer.
71:   `stereo_side_geometry_loss`, `stereo_correlation_loss`, and
72:   `stereo_level_loss` expose its unweighted components. `decoder_pan` is the
73:   bounded global residual after the 4x8 field-to-stereo map. v8.1 limits it to
74:   +/-0.10. `decoder_pan_raw` exposes saturation before that map, while
75:   `pan_center_loss` measures squared energy in the bounded audible residual.
76:   It deliberately does not grow with an arbitrarily large raw coordinate.
77:   `decoder_side_control`, `decoder_width_control`, and `decoder_width_raw`
78:   distinguish collapsed side excitation, an active width rail, and saturation
79:   in the underlying width head. A version marker makes the first v8.1 resume
80:   reset only the changed pan/width controls' incompatible Adam moments.
81: - `development_best_spectral`, `development_mean_spectral`, and
82:   `development_mean_chroma` use fixed, gradient-excluded development families.
83:   `development_score`, `development_plateau_ready`, and
84:   `development_relative_improvement` make the morphic-growth gate auditable.
85: - `validation_best_spectral`, `validation_mean_spectral`, and
86:   `validation_mean_chroma` score every emitted chunk against fixed probes.
87:   `validation_is_strict` in run metadata must be true before these are called
88:   held-out results. A training-family fallback remains available for tiny or
89:   manually incomplete corpora, but is only a within-run diagnostic. Validation
90:   metrics never select a training target or an architecture transition.
91: - `target_file`, `target_frame`, and `target_chunks_left` identify the coherent
92:   source episode in force at each sample. This makes temporal-supervision bugs
93:   and corpus bias auditable.
94: - `ultrasonic_ratio` is the fraction of pre-master magnitude above 20 kHz. It
95:   is an aliasing/foldback guardrail, not a musical brightness score.
96: 
97: ## Controllers
98: 
99: - `V` is a designed control potential over operating-state summaries, not
100:   thermodynamic free energy.
101: - `temp` controls perturbation, plasticity, and exploration. It is an adaptive
102:   control variable, not physical temperature.
103: - `energy` is a bounded synthesis/control budget.
104: - `temp_stuck`, `temp_subcritical`, `temp_curiosity`, `temp_stagnation`, and
105:   `temp_motion` are the additive drives of the temperature target before its
106:   final clamp and EMA. They are the first place to inspect persistent heat.
107: - `controlled_shear_rms` is the requested and phase-normalized RMS of the
108:   structured macro-field forcing.
109: - `field_signed_mean` detects sign bias in the micro field;
110:   `field_rail_excess` is the mean amount by which cell magnitude exceeds 0.9.
111:   A weak global-mean damper removes only 3.5% of the field's DC mode per chunk,
112:   so local signed structure remains free while population drift is bounded.
113: - `radiation_probability` is the per-chunk sparse-radiation hazard and is
114:   independent of the selected BPTT window.
115: - `grad_norm` is the requested-horizon mean gradient norm before global
116:   clipping and `clip_scale` is the factor applied before AdamW updates its
117:   moments. Horizons above 8 accumulate bounded, detached tape segments.
118: - `optimizer_updates` is the persisted cumulative AdamW step count;
119:   `optimizer_updates_run` is the current process count and `optimizer_resumed`
120:   states whether compatible moments were restored. `optimizer_migration` in
121:   run metadata separates exact, resized, and newly initialized moment pairs.
122: - `core_update_every_tapes` in run metadata is the two-timescale optimization
123:   cadence. Decoder weights receive every optimizer horizon; CA, GRU, episodic,
124:   and MorphicStack gradients are retained only on the scheduled full tapes.
125: - `phase_profile` reports average milliseconds per completed chunk for model
126:   forward, target loading, loss/metrics, backward, optimizer, output I/O, and
127:   checkpoint work. Target-loading time is a measured subset of loss/metrics,
128:   not an additional mutually exclusive bucket.
129: - `side_energy_width` is the old RMS channel-difference measure. It can be high
130:   for panned mono and is retained as a diagnostic, not called true width.
131:   `width` multiplies it by interchannel incoherence, so perfectly correlated
132:   unequal-gain channels measure near zero. `stereo_corr` is the final post-DC,
133:   pre-master normalized correlation. Strongly negative values warn of mono
134:   cancellation; strongly positive values warn of mono collapse.
135: - `morph_frozen` and `morph_max_depth` state the run's structural policy.
136:   Growth additionally requires a strict development split and a completed
137:   plateau window; validation metrics never participate in that decision.
138:   Run metadata additionally records the physically constructed `morph_layers`
139:   and internal `morph_width`; these determine parameter and optimizer size but
140:   do not change the MorphicStack's 512-dimensional external interface.
141: - `manifold_depth` is the number of active 16-feature CA sheets derived from
142:   morph depth. `active_spatial_rings` is one at L01 and two once the dilated
143:   far ring has nonzero gain. `far_ring_gain` exposes its gradual fade-in. These
144:   are architecture controls, not estimators of intrinsic physical dimension.
145: - `motif_capacity` in run metadata is the selected host-memory limit (1--4096,
146:   default 64). `motifs_active` is the retained entry count at finalization.
147:   Resume-time growth retains every motif; shrinking retains the newest entries
148:   and discards the oldest first.
149: - `field_entropy` is the channel-archetype entropy in bits for the current
150:   micro field. It is not the entropy of the rendered waveform.
151: - `crit_gain` is fixed at 1.0 in v9. `sigma` remains useful evidence, but no
152:   learning-rate singularity is applied until calibration establishes a real
153:   critical surface rather than merely naming one.
154: 
155: Claims about improved sound should be supported by repeated seeded runs,
156: ablation comparisons, objective audio measurements, and blinded listening—not
157: by these internal metrics alone.
158: 


## ADVISORY REVIEW independent-investigator; not empirical evidence
{
  "conclusions": [
    "The causal audio pipeline in Titan Audio v9 is: world state (micro field, macro field, GRU hidden, episodic readout, oscillator phases, host controls) -> model.forward -> stereo waveform chunk -> tanh + stateful DC blocker -> streamed peak normalization to -1 dBFS -> PCM16 WAV. Target audio is sampled AFTER forward and only enters the loss/optimizer and later host ecology; it cannot condition the same chunk.",
    "The listener's 'harsh and centered' report is consistent with the supplied measurements: the latest tagged continuation has L/R correlation 0.991, side/mid -23.31 dB, and 2-6 kHz fraction 47.3% vs 33.6% for the untagged parent, with 20-200 Hz dropping from 20.8% to 2.1%. These are diagnostics, not perceptual scores.",
    "The center bias is plausibly a within-run outlier amplified by prime selection: the full 600 s WAV is not uniformly centered (per-minute correlation 0.740-0.992), and the 60 s prime is drawn from a very center-heavy interval (minutes 5-6, corr 0.990-0.992, side/mid ~-23 dB). The prime selection score uses field entropy, activity health, structured complexity, and stagnation, not audible width or harshness.",
    "The prompt file (suno_priming_prompt*.txt) is written after rendering and prime extraction (src/main.rs:8682-8697), so its descriptive words cannot have caused the run. It is a post-render output.",
    "The forward pass has no text prompt, target audio, prime, or external reference argument (research/audio_context.md:13-15; docs/AUDIO_SCIENTIFIC_INSTRUMENTATION_PLAN.md:101).",
    "Offline waveform filtering tests signal processing only; they are not a proxy for latent-field diffusion, and a trained diffusion/flow decoder is also not a proxy (research/audio_context.md:64-67).",
    "Canonical checkpoints, corpus, existing WAVs, and the normal renderer/training must remain intact; only copied WAVs and sidecar receipts are permitted for offline tests (research/audio_context.md:68-70)."
  ],
  "evidence": [
    "research/audio_context.md:13-15 — forward has NO text prompt, target audio, prime, or external reference argument.",
    "research/audio_context.md:16-18 — causal order: current world -> forward audio -> target sampling/loss -> optimizer and later host feedback; current target cannot change the same precomputed chunk.",
    "research/audio_context.md:19-21 — suno_priming_prompt*.txt is written AFTER rendering and prime extraction; its words cannot cause the run.",
    "research/audio_context.md:44-47 — side derives from L-R before a delayed scaled side stage; learned width bounded 0.05..0.50; global pan +/-0.10; pan-center loss penalizes bounded pan, does not directly reward correlation=1.",
    "research/audio_context.md:48-50 — loss-visible tanh, later gain/saturation/DC blocker, whole-file peak normalization; sample peak below 0 dBFS does not exclude nonlinear saturation or perceptual harshness.",
    "research/audio_context.md:51-53 — prime selection ~60 s using field entropy, activity health, complexity, stagnation; no listener rating or direct stereo/harshness ranking; selection can amplify a within-run outlier.",
    "research/audio_context.md:64-67 — offline waveform filtering tests signal processing; not a proxy for latent-field diffusion; zero-output identity branch proves bypass only; latent field entropy cannot be recovered from a WAV.",
    "research/audio_context.md:68-70 — canonical checkpoints, corpus, existing WAVs, normal renderer and training stay intact; offline test may create copied WAVs and sidecar receipts; use paired intervals and record level-matching gains.",
    "analysis/audio_center_harshness_20260926/deepseek_packet.md:24-29 — measurement table: untagged parent corr 0.603, side/mid -6.01, 20-200 Hz 20.8%, 2-6 kHz 33.6%; latest tagged corr 0.991, side/mid -23.31, 20-200 Hz 2.1%, 2-6 kHz 47.3%.",
    "analysis/audio_center_harshness_20260926/deepseek_packet.md:31-35 — full 600 s WAV not uniformly centered; per-minute corr 0.740-0.992; minutes 5-6 corr 0.990-0.992, side/mid ~-23 dB; prime is from a very center-heavy interval; selection score uses field entropy, activity health, structured complexity, stagnation.",
    "analysis/audio_center_harshness_20260926/deepseek_packet.md:37-43 — training trace: median output corr 0.963 vs target 0.862; median truthful width 0.027; decoder width control 0.120; width raw -0.946; global pan 0.0999 at +0.10 limit; decoder side control 1.0; ultrasonic_ratio median 0.0965 is a pre-master guardrail, not a harshness score.",
    "docs/AUDIO_SCIENTIFIC_INSTRUMENTATION_PLAN.md:101 — model.forward runs before TargetAudioLoader::sample_chunks; target does not directly condition the current chunk's neural forward or synthesis equations.",
    "docs/AUDIO_SCIENTIFIC_INSTRUMENTATION_PLAN.md:105-108 — Audio v9 cannot honestly perform Image-style held-out reconstruction at inference; first-chunk invariance is an important negative control; target-feedback-conditioned ecological response is not direct reference-conditioned reconstruction.",
    "src/main.rs:82-84 — SAMPLE_RATE 48000, CHUNK_SIZE 4096.",
    "src/main.rs:94 — MACRO_UPDATE_EVERY = 4.",
    "src/main.rs:112 — MAX_AUTOGRAD_TAPE_CHUNKS = 8.",
    "src/main.rs:246 — MAX_STEREO_SIDE_GAIN = 1.25.",
    "src/main.rs:253-255 — STEREO_SIDE_LOSS_WEIGHT 0.30, STEREO_CORRELATION_LOSS_WEIGHT 0.95, STEREO_LEVEL_LOSS_WEIGHT 0.45.",
    "src/main.rs:5108-5122 — forward signature: micro, macro_t, mem, epi_out, phases, theta_prev, theta_prev2, force, absolute_step, train_ecology, energy, control. No target/reference argument.",
    "src/main.rs:5565-5603 — mid = (L+R)/2 scaled by temporal_controls[4]; side = (L-R)/2 scaled by temporal_controls[5]; side_wide = side * width_val * learned_width; side_delayed from prev_haas_side; audio_l = mid + side_delayed; audio_r = mid - side_delayed.",
    "src/main.rs:5583-5596 — pan_raw from spatial_panner; pan_t = soft_global_pan(pan_raw); width_val = stereo_side_gain(last_pan, control.width_mult); learned_width = soft_width_control(width_raw).",
    "src/main.rs:5605-5614 — gain_l/gain_r from pan_t; audio scaled by 1.414.",
    "src/main.rs:7103-7107 — audio_for_loss = stereo_chunk.tanh(); targets = target_loader.sample_chunks(...) AFTER forward.",
    "src/main.rs:7211-7234 — stereo_geometry on output and target; side_geometry_loss, correlation_loss, stereo_level_loss; stereo_balance_loss combines them with the three weights.",
    "src/main.rs:7227 — pan_center_loss(&pan).",
    "src/main.rs:8059-8076 — post path: tanh(x*0.92) then stateful DC blocker; raw_peak tracked.",
    "src/main.rs:8451-8455 — norm = 0.891 / raw_peak; streamed DC-blocked peak normalized to -1 dBFS.",
    "src/main.rs:8463-8481 — prime selection: sliding window over chunk_scores; best_start, best_sum; prime_start_frame = best_start * CHUNK_SIZE.",
    "src/main.rs:8510-8528 — output and prime written as PCM16 with norm applied; prime fade 2048 samples.",
    "src/main.rs:8682-8697 — prompt formatted from run statistics and written to suno_priming_prompt txt AFTER rendering.",
    "src/stereo.rs:3-7 — GLOBAL_PAN_LIMIT 0.10, PAN_CENTER_LOSS_WEIGHT 0.10, WIDTH_CONTROL_MIN 0.05, WIDTH_CONTROL_MAX 0.50.",
    "src/stereo.rs:13-19 — soft_global_pan: z/sqrt(1+z^2) scaled by GLOBAL_PAN_LIMIT.",
    "src/stereo.rs:21-26 — pan_center_loss: pan.sqr().",
    "src/stereo.rs:28-41 — soft_width_control: unit * half_range + midpoint, bounded 0.05..0.50.",
    "METRICS.md:64-76 — output_low_band_ratio and target_low_band_ratio compare the first supervised 20 Hz log band; decoder_pan bounded +/-0.10; pan_center_loss measures squared energy in the bounded audible residual.",
    "METRICS.md:94-95 — ultrasonic_ratio is the fraction of pre-master magnitude above 20 kHz; aliasing/foldback guardrail, not a musical brightness score.",
    "METRICS.md:129-134 — side_energy_width can be high for panned mono; width multiplies by interchannel incoherence; stereo_corr is final post-DC, pre-master normalized correlation."
  ],
  "uncertainties": [
    "No controlled blinded listening has been performed; perceptual improvement or degradation is not established (research/audio_context.md:58-60).",
    "The supplied spectral fractions are MID-channel sampled FFT power fractions; changes in fraction can reflect denominator or stereo changes. Absolute and total L+R band power have not been reported (research/audio_context.md:54-57).",
    "The three latest WAVs form one sequential continuation lineage, not independent seeds or matched arms (research/audio_context.md:28-29).",
    "Target mix and online learning confound comparisons between elapsed portions of a run (research/audio_context.md:37-39).",
    "The low-band loss term measures the first log band; it is not the whole 20-200 Hz measurement (research/audio_context.md:40-43).",
    "The training trace's ultrasonic_ratio is a pre-master guardrail, not a harshness score (deepseek_packet.md:40-41).",
    "Whether the harshness is caused by learned weights, initialized anatomy, carried world state, or controller forcing is not separable from existing scalar traces (docs/AUDIO_SCIENTIFIC_INSTRUMENTATION_PLAN.md:133-141).",
    "Pre-side-scale renderer taps, frequency-specific loss gradients, detailed target mix attribution, and downstream priming outcomes are missing (research/audio_context.md:71-74).",
    "The exact contribution of the wavefolder, scan synthesis, excitation noise, and tanh saturation to perceived harshness is not measured.",
    "Whether the center bias is primarily from the learned width control, the global pan, the side excitation, or the target-relative stereo losses is not isolated.",
    "The prime selection window is chosen by chunk_scores that do not include audible width or harshness; the selected interval may be unrepresentative of the full run."
  ],
  "proposed_experiments": [
    {
      "name": "Paired-interval offline stereo and band-power audit",
      "purpose": "Separate center bias from harshness and from prime-selection outlier effects without touching canonical files.",
      "method": "Copy the latest full 600 s WAV and the untagged parent WAV to a sidecar directory. Compute per-60 s non-overlapping interval: L/R Pearson correlation, side/mid dB, L/R balance dB, absolute L+R band power in 20-200 Hz, 2-6 kHz, 6-20 kHz, and total power. Record level-matching gains. Compare the prime interval against the full-run distribution.",
      "controls": "Use identical windowing and FFT parameters for both files; report absolute power, not only fractions; do not modify originals.",
      "expected_discriminator": "If the prime interval is an extreme outlier in correlation and side/mid relative to the full run, center bias is partly a selection artifact. If harshness (2-6 kHz absolute power) is elevated across the full run, it is not only a prime-selection artifact.",
      "cost": "Low; offline, no training, no model load."
    },
    {
      "name": "First-chunk invariance negative control",
      "purpose": "Confirm that target audio does not condition the current chunk's forward pass.",
      "method": "Clone identical pre-forward state, run model.forward with two different target chunks presented to the loss, and compare the emitted current-chunk waveform. Repeat across several steps.",
      "controls": "Hold all world state, phases, controls, and RNG identical; vary only the target chunk.",
      "expected_discriminator": "Identical current-chunk audio confirms no direct reference conditioning. Any difference would contradict the documented causal order and require source re-audit.",
      "cost": "Low to moderate; requires harness access, no training."
    },
    {
      "name": "Side-path ablation on copied WAVs",
      "purpose": "Estimate how much of the center bias is attributable to the side path versus the mid path.",
      "method": "On copied WAVs, apply a mid/side decomposition and re-render with side gain set to 0, to the observed value, and to a modestly increased value. Measure correlation, side/mid, and 2-6 kHz absolute power for each.",
      "controls": "Level-match all variants; use paired intervals; keep originals untouched.",
      "expected_discriminator": "If increasing side gain reduces correlation and does not increase 2-6 kHz harshness, center bias and harshness are separable. If harshness tracks side gain, the side path contributes to harshness.",
      "cost": "Low; offline signal processing only."
    },
    {
      "name": "Post-path stage ablation on copied WAVs",
      "purpose": "Separate tanh saturation, DC blocker, and peak normalization contributions to harshness.",
      "method": "On copied WAVs, bypass tanh, bypass DC blocker, and bypass normalization individually and in combination; measure 2-6 kHz absolute power, THD-like residual, and correlation.",
      "controls": "Level-match; use paired intervals; do not modify originals.",
      "expected_discriminator": "If harshness drops when tanh is bypassed, nonlinear saturation is a contributor. If it persists, the harshness originates earlier in the renderer.",
      "cost": "Low; offline signal processing only."
    },
    {
      "name": "Prime-selection sensitivity re-scoring",
      "purpose": "Test whether the prime selection score is biased toward center-heavy intervals.",
      "method": "Recompute chunk_scores for the full run and correlate them with per-interval correlation and 2-6 kHz power. Identify whether high-scoring intervals are systematically more centered or brighter.",
      "controls": "Use the same score formula as src/main.rs:8141-8145; do not change the selection.",
      "expected_discriminator": "A positive correlation between score and center bias would indicate the selection criterion favors centered intervals. No correlation would weaken the selection-artifact hypothesis.",
      "cost": "Low; requires trace data or recomputation from the full WAV."
    }
  ],
  "context_receipt": {
    "project": "titan_audio_ecosystem",
    "direct_reference_input": false,
    "prompt_is_post_render_output": true,
    "waveform_filter_tests_latent_diffusion": false,
    "canonical_mutation_allowed": false,
    "summary": "Titan Audio v9 is a CPU Rust NCA/GRU/DDSP ecosystem. Forward takes world state and host controls, not text or target audio. Target audio is sampled after forward and affects loss, optimizer, and later host ecology. The prompt file is written after rendering. Offline waveform filtering tests signal processing, not latent-field diffusion. Canonical checkpoints, corpus, and existing WAVs must remain intact. The listener's harsh/centered report is consistent with measured correlation 0.991, side/mid -23.31 dB, and 2-6 kHz fraction 47.3% in the latest tagged prime, but no blinded listening has been done.",
    "evidence": [
      "research/audio_context.md:13-15",
      "research/audio_context.md:16-18",
      "research/audio_context.md:19-21",
      "research/audio_context.md:44-47",
      "research/audio_context.md:48-50",
      "research/audio_context.md:51-53",
      "research/audio_context.md:64-67",
      "research/audio_context.md:68-70",
      "analysis/audio_center_harshness_20260926/deepseek_packet.md:24-29",
      "analysis/audio_center_harshness_20260926/deepseek_packet.md:31-35",
      "analysis/audio_center_harshness_20260926/deepseek_packet.md:37-43",
      "docs/AUDIO_SCIENTIFIC_INSTRUMENTATION_PLAN.md:101",
      "docs/AUDIO_SCIENTIFIC_INSTRUMENTATION_PLAN.md:105-108",
      "src/main.rs:82-84",
      "src/main.rs:94",
      "src/main.rs:112",
      "src/main.rs:246",
      "src/main.rs:253-255",
      "src/main.rs:5108-5122",
      "src/main.rs:5565-5603",
      "src/main.rs:5583-5596",
      "src/main.rs:5605-5614",
      "src/main.rs:7103-7107",
      "src/main.rs:7211-7234",
      "src/main.rs:7227",
      "src/main.rs:8059-8076",
      "src/main.rs:8451-8455",
      "src/main.rs:8463-8481",
      "src/main.rs:8510-8528",
      "src/main.rs:8682-8697",
      "src/stereo.rs:3-7",
      "src/stereo.rs:13-19",
      "src/stereo.rs:21-26",
      "src/stereo.rs:28-41",
      "METRICS.md:64-76",
      "METRICS.md:94-95",
      "METRICS.md:129-134"
    ],
    "hypotheses": [
      "The center bias is partly a prime-selection artifact: the 60 s prime is drawn from a very center-heavy interval (minutes 5-6, corr 0.990-0.992, side/mid ~-23 dB) while the full run ranges 0.740-0.992.",
      "The harshness is partly attributable to the post-render tanh saturation and/or the wavefolder and scan synthesis, but this is not isolated by existing measurements.",
      "The learned width control and global pan are near their bounds (width raw -0.946, pan 0.0999 at +0.10 limit), which may contribute to center bias, but the causal share is not measured.",
      "The target-relative stereo losses (side 0.30, correlation 0.95, level 0.45) may have driven the model toward high correlation, but the training trace shows output corr 0.963 vs target 0.862, so the target itself is not fully centered."
    ],
    "missing_evidence": [
      "Perceptual ratings and blinded listening.",
      "Independent training replicas or matched arms.",
      "Pre-side-scale renderer taps.",
      "Frequency-specific loss gradients.",
      "Detailed target mix attribution.",
      "Downstream priming outcomes.",
      "Absolute and total L+R band power.",
      "Per-interval harshness measurements across the full run."
    ],
    "discriminating_experiment": "Paired-interval offline stereo and band-power audit on copied WAVs: compute per-60 s correlation, side/mid, balance, and absolute L+R band power in 20-200 Hz, 2-6 kHz, 6-20 kHz, and total for the latest full 600 s WAV and the untagged parent. Compare the prime interval against the full-run distribution. This separates prime-selection outlier effects from run-wide harshness and center bias without touching canonical files.",
    "recommendation": "Run the paired-interval offline audit first. It is the cheapest measurement that can separate prime-selection artifact from run-wide center bias and harshness. Do not change learned behavior, checkpoints, corpus, or existing WAVs. Treat all metrics as diagnostics; audible quality requires blinded listening."
  }
}


## ADVISORY REVIEW falsification-arbiter; not empirical evidence
{
  "conclusions": [
    "The listener report ('harsh and center-biased') is consistent with the supplied measurements: the latest tagged continuation has L/R correlation 0.991, side/mid -23.31 dB, and 2-6 kHz fraction 47.3% vs 33.6% for the untagged parent, while 20-200 Hz fraction fell from 20.8% to 2.1%.",
    "The center bias is at least partly a prime-selection artifact, not necessarily a whole-run property: the full 600 s WAV has per-minute correlation 0.740-0.992, and the selected 60 s prime comes from a 0.990-0.992 interval. Prime selection uses field entropy, activity health, complexity, and stagnation, with no width or harshness term.",
    "Two separable mechanisms can produce the observed center bias: (A) learned renderer/decoder state (width head, side control, pan head, Haas side path) has drifted toward mono; (B) the target mix and stereo losses (STEREO_SIDE_LOSS_WEIGHT 0.30, STEREO_CORRELATION_LOSS_WEIGHT 0.95, STEREO_LEVEL_LOSS_WEIGHT 0.45, PAN_CENTER_LOSS_WEIGHT 0.10) plus the target episode in force at that interval drove the learned controls there. These are not mutually exclusive and the supplied evidence cannot separate them.",
    "Harshness is not directly measured. The 2-6 kHz fraction rise and 6-20 kHz fraction rise are consistent with a bright/glassy upper spectrum, but the generated prompt text ('bright glassy upper spectrum') is written after rendering and cannot cause the run; it is a description, not a cause.",
    "The 'ultrasonic_ratio' median 0.0965 is a pre-master aliasing guardrail, not a harshness score; it does not establish or exclude perceptual harshness.",
    "The post-render path (tanh at 0.92, stateful DC blocker, whole-file peak normalization to -1 dBFS) is loss-visible and can add nonlinear saturation even when sample peak is below 0 dBFS, so a peak-normalized file does not exclude saturation-driven harshness.",
    "The three latest WAVs are a sequential continuation lineage, not independent seeds or matched arms, so cross-WAV comparisons confound elapsed training, target mix, and world state."
  ],
  "evidence": [
    "analysis/audio_center_harshness_20260926/deepseek_packet.md:24-29 — measurement table: untagged parent corr 0.603 / side-mid -6.01 / 20-200 Hz 20.8% / 2-6 kHz 33.6%; latest tagged corr 0.991 / side-mid -23.31 / 20-200 Hz 2.1% / 2-6 kHz 47.3%.",
    "analysis/audio_center_harshness_20260926/deepseek_packet.md:31-35 — full 600 s WAV is not uniformly centered; minutes 5-6 have corr 0.990-0.992; prime selection uses field entropy, activity health, complexity, stagnation, not width or harshness.",
    "analysis/audio_center_harshness_20260926/deepseek_packet.md:37-43 — training trace: median output corr 0.963 vs target 0.862; median truthful width 0.027; decoder width control 0.120; width raw -0.946; global pan 0.0999 at +0.10 limit; decoder side control 1.0; ultrasonic_ratio median 0.0965 is a pre-master guardrail.",
    "analysis/audio_center_harshness_20260926/team_v2/context_packet.md:44-47 — side derives from L-R before a delayed scaled side stage; learned width bounded 0.05..0.50; global pan +/-0.10; pan-center loss penalizes bounded pan and does not directly reward correlation=1; increasing side gain cannot synthesize missing side information.",
    "analysis/audio_center_harshness_20260926/team_v2/context_packet.md:48-50 — loss-visible tanh, later gain/saturation/DC blocker, whole-file peak normalization; sample peak below 0 dBFS does not exclude nonlinear saturation or perceptual harshness.",
    "analysis/audio_center_harshness_20260926/team_v2/context_packet.md:51-53 — prime selection ~60 s using field entropy, activity health, complexity, stagnation; no listener rating or direct stereo/harshness ranking; selection can amplify a within-run outlier.",
    "analysis/audio_center_harshness_20260926/team_v2/context_packet.md:54-57 — supplied first survey uses MID-channel sampled FFT power fractions; changes in fraction can reflect denominator or stereo changes; measure absolute and total L+R band power before claiming bass recovery.",
    "analysis/audio_center_harshness_20260926/team_v2/context_packet.md:28-29 — three latest WAVs form one sequential continuation lineage, not independent seeds or matched arms.",
    "analysis/audio_center_harshness_20260926/team_v2/context_packet.md:19-21 — prompt is written after rendering; descriptive words cannot cause the run; prompt direction is post-render output.",
    "src/main.rs:5529-5603 — wavefolder, openness, mid/side, Haas side path, learned width, and pan are applied in the differentiable renderer before the post path.",
    "src/main.rs:8059-8076 — post path: tanh(0.92*x), stateful DC blocker, raw_peak tracked; then src/main.rs:8451 normalizes to 0.891/raw_peak.",
    "src/main.rs:7227-7256 — pan_center_loss and stereo_balance_loss (side 0.30, correlation 0.95, level 0.45) are part of mimic_loss; pan_center_loss weight PAN_CENTER_LOSS_WEIGHT.",
    "src/stereo.rs:3-7, 9-11, 21-26, 28-41 — GLOBAL_PAN_LIMIT 0.10, PAN_CENTER_LOSS_WEIGHT 0.10, WIDTH_CONTROL_MIN 0.05, WIDTH_CONTROL_MAX 0.50; side_gain depends on last_pan and width_mult; pan_center_loss is pan^2; soft_width_control is bounded polynomial.",
    "src/main.rs:8463-8481 — prime selection is a sliding-window argmax over chunk_scores; no width or harshness term.",
    "src/main.rs:8141-8145 — chunk_scores = field_entropy*(0.25+0.50*activity_health) + structured_complexity*0.75 - stagnation*0.25.",
    "METRICS.md:129-134 — side_energy_width can be high for panned mono; width multiplies by interchannel incoherence; stereo_corr is post-DC pre-master normalized correlation.",
    "METRICS.md:94-95 — ultrasonic_ratio is an aliasing/foldback guardrail, not a musical brightness score.",
    "METRICS.md:155-157 — claims about improved sound require repeated seeded runs, ablation comparisons, objective audio measurements, and blinded listening."
  ],
  "uncertainties": [
    "No controlled blinded listening has been performed; perceptual harshness and center bias are listener-reported, not measured.",
    "The supplied spectral fractions are MID-channel sampled FFT power fractions; they can change with denominator or stereo changes and are not absolute L+R band power. Bass 'loss' is not established without absolute and total L+R band power.",
    "The training trace is a sampled summary (704 rows, steps 74,340-81,370); it does not identify which target episode was in force during the selected prime interval.",
    "The target mix attribution is missing; the target episode in force at the prime interval is not reported in the supplied evidence.",
    "Pre-side-scale renderer taps are missing, so the contribution of the learned width head vs the Haas side path vs the mid/side stage cannot be separated from the supplied WAVs alone.",
    "Frequency-specific loss gradients are missing; the relative contribution of the 2-6 kHz band to the total loss is not reported.",
    "The checkpoint set transactionality caveat (model -> optimizer -> world publish order) means the latest run may not be mathematically bit-exact with its recorded step; this is a reproducibility limitation, not a sound-quality claim.",
    "Whether the harshness is dominated by the learned renderer, the post-render tanh/DC/normalization, or the target-driven loss is not separable from the supplied evidence.",
    "The 'ultrasonic_ratio' median 0.0965 is pre-master and is not a harshness score; it neither supports nor refutes harshness.",
    "No independent training replicas or matched arms exist; the three WAVs are a sequential lineage."
  ],
  "hypotheses": [
    {
      "id": "H1",
      "claim": "CLAIM A: The center bias is primarily a prime-selection artifact. The full 600 s run is not uniformly centered (per-minute corr 0.740-0.992), and the 60 s prime was selected from a 0.990-0.992 interval by a score that ignores width and harshness.",
      "mechanism": "Prime selection argmax over chunk_scores (field entropy, activity health, structured complexity, stagnation) picks a center-heavy interval; the listener hears that interval.",
      "predictions": [
        "Recomputing the prime with a width-aware or harshness-aware score selects a different interval with materially lower correlation and lower 2-6 kHz fraction.",
        "The full 600 s WAV has intervals with correlation well below 0.99 and side/mid well above -23 dB."
      ]
    },
    {
      "id": "H2",
      "claim": "CLAIM B: The center bias is primarily a learned-renderer drift. The width head, side control, pan head, and Haas side path have moved toward mono, and the prime merely samples that state.",
      "mechanism": "Training trace shows decoder width control 0.120 (near the 0.05 floor), width raw -0.946, global pan 0.0999 at the +0.10 limit, decoder side control 1.0; median truthful width 0.027 vs target 0.862 correlation.",
      "predictions": [
        "A frozen renderer replay from the same checkpoint with the same world state reproduces the center-heavy output even when the prime window is chosen differently.",
        "Intervening on the width/pan controls (sidecar, no weight change) changes correlation and side/mid materially."
      ]
    },
    {
      "id": "H3",
      "claim": "CLAIM C: The harshness is primarily post-render nonlinearity (tanh at 0.92, DC blocker, peak normalization to -1 dBFS) rather than the learned renderer's spectral content.",
      "mechanism": "Loss-visible tanh and whole-file peak normalization can add saturation and intermodulation even when sample peak is below 0 dBFS.",
      "predictions": [
        "An offline bypass of the post-render tanh/DC/normalization on a copied WAV reduces measured high-band intermodulation and is judged less harsh in blinded listening.",
        "The pre-master raw signal has lower 2-6 kHz fraction than the mastered WAV."
      ]
    },
    {
      "id": "H4",
      "claim": "CLAIM D: The harshness is primarily target-driven: the target episode in force during the prime interval has a bright upper spectrum and high correlation, and the stereo/correlation losses pulled the learned controls toward that target.",
      "mechanism": "STEREO_CORRELATION_LOSS_WEIGHT 0.95 and STEREO_SIDE_LOSS_WEIGHT 0.30 push output correlation toward target correlation; if the target is bright and correlated, the output follows.",
      "predictions": [
        "The target episode in force at the prime interval has high correlation and high 2-6 kHz fraction.",
        "Holding the renderer frozen and swapping the target episode changes the learned controls' trajectory in later chunks (target-feedback-conditioned ecological response), not the current chunk."
      ]
    },
    {
      "id": "H5",
      "claim": "CLAIM E: The center bias and harshness are not separable from the supplied evidence because the prime is a within-run outlier and the three WAVs are a sequential lineage, not matched arms.",
      "mechanism": "Confounds: elapsed training, target mix, world state, and prime selection all vary together.",
      "predictions": [
        "No single offline test on the existing WAVs can attribute the effect to one mechanism.",
        "A matched-arm or frozen-replay experiment is required."
      ]
    }
  ],
  "why_they_differ": [
    "H1 vs H2 differ on locus: H1 attributes the observed center bias to the selection window (a measurement/selection artifact), H2 attributes it to the learned renderer state (a model property). They differ because the prime is a 60 s window chosen by a score that ignores width, while the full run is heterogeneous; the supplied evidence shows both a heterogeneous full run and a drifted width/pan trace, so both are consistent with the same WAV.",
    "H2 vs H4 differ on cause of the learned drift: H2 treats the drift as intrinsic to the renderer/optimizer trajectory, H4 treats it as target-driven via the stereo/correlation losses. They differ because the forward pass has no target argument, but target losses affect weights and later host feedback; the current chunk cannot be changed by the current target, but later chunks can.",
    "H3 vs H2/H4 differ on stage: H3 locates harshness in the post-render nonlinearity, H2/H4 locate it in the learned renderer's spectral content. They differ because the post path is loss-visible and applied after the differentiable renderer, so both can contribute and the supplied WAVs are post-master.",
    "H5 is a meta-claim: it says the supplied evidence is insufficient to separate H1-H4 because the comparisons are confounded by elapsed training, target mix, world state, and prime selection."
  ],
  "discriminating_experiment": {
    "name": "Frozen-replay prime-window and post-path ablation on copied WAVs plus a width-aware prime re-selection",
    "purpose": "Separate prime-selection artifact (H1) from learned-renderer drift (H2) from post-render nonlinearity (H3), using only offline operations on copied WAVs and sidecar receipts; no canonical mutation, no training fork.",
    "inputs": [
      "The latest full 600 s WAV and its 60 s prime (copies only).",
      "The latest checkpoint set (model, optimizer, world, morph sidecar, metadata) read-only, hashed.",
      "The training trace rows for steps 74,340-81,370 (read-only)."
    ],
    "procedure": [
      "Step 0 (control): Hash the checkpoint set and the WAVs; record SHA256. Confirm no canonical mutation.",
      "Step 1 (prime re-selection): Recompute the 60 s prime using the existing chunk_scores, and separately using a width-aware score (e.g., add a term proportional to side/mid dB and a penalty on 2-6 kHz fraction). Compare the two selected intervals' correlation, side/mid, and 2-6 kHz fraction. This tests H1.",
      "Step 2 (post-path ablation): On a copy of the full WAV, produce three offline variants: (a) bypass the tanh(0.92*x) stage only; (b) bypass the DC blocker only; (c) bypass both and re-normalize to the same -1 dBFS. Measure 2-6 kHz fraction, 6-20 kHz fraction, and intermodulation proxies on each. This tests H3.",
      "Step 3 (frozen renderer replay): From the same checkpoint and world state, render a short frozen segment (no optimizer step, no target sampling) with the width head and pan head held at their recorded values, and a second segment with the width head clamped to a higher value (e.g., 0.30) and pan clamped to 0.0. Compare correlation and side/mid. This tests H2.",
      "Step 4 (target attribution, read-only): Identify the target episode in force during the prime interval from the trace's target_file/target_frame fields. Report its correlation and 2-6 kHz fraction. This tests H4 without changing the forward pass.",
      "Step 5 (blinded listening): Present the original prime, the width-aware re-selected prime, and the post-path-bypassed variant to at least two listeners in randomized order with level matching; record harshness and center-bias ratings."
    ],
    "decision_criteria": [
      "H1 supported if the width-aware re-selected prime has correlation <= 0.90 and side/mid >= -12 dB, and the original prime interval is confirmed to be a within-run outlier (per-minute corr >= 0.99).",
      "H2 supported if frozen replay with recorded width/pan reproduces correlation >= 0.98 and side/mid <= -20 dB, and clamping width to 0.30 / pan to 0.0 reduces correlation by >= 0.10 and raises side/mid by >= 6 dB.",
      "H3 supported if bypassing the post-render tanh and/or DC blocker reduces the 2-6 kHz fraction by >= 5 percentage points or reduces an intermodulation proxy by >= 3 dB, and blinded listeners rate the bypassed variant less harsh by a pre-registered margin.",
      "H4 supported if the target episode in force at the prime interval has correlation >= 0.95 and 2-6 kHz fraction >= 45%, and the trace shows the stereo/correlation losses were active (non-zero) during that interval.",
      "H5 supported if none of the above reach their thresholds, or if the effects are within measurement noise across the three WAVs."
    ],
    "inconclusive_outcome": "If the width-aware re-selection changes the prime but the new prime still has correlation >= 0.95 and side/mid <= -18 dB, AND frozen replay with recorded width/pan does not reproduce the center bias (correlation < 0.95), AND post-path bypass does not change the 2-6 kHz fraction by >= 5 percentage points, AND the target episode is not identifiable from the trace, then the experiment is inconclusive: the supplied evidence cannot separate H1-H4, and a matched-arm or independent-seed experiment is required.",
    "controls": [
      "Level-match all compared variants to the same integrated loudness before listening.",
      "Use paired intervals (same time window) for all comparisons.",
      "Record SHA256 of every copied WAV and sidecar receipt.",
      "Do not call load_or_create_corpus_manifest (mutating path).",
      "Do not modify canonical checkpoints, corpus, existing WAVs, renderer, or training.",
      "Report absolute and total L+R band power, not only MID-channel fractions."
    ],
    "expected_information_gain": "High: the experiment separates selection artifact from renderer drift from post-path nonlinearity using only offline operations, and its outcomes map directly onto the competing claims. It does not require a training fork or a diffusion mechanism."
  },
  "recommendation": "Run the frozen-replay prime-window and post-path ablation first. It is reversible, uses copied WAVs and sidecar receipts, preserves the checkpoint and renderer, and its outcomes discriminate H1-H4. Do not change learned behavior or introduce a diffusion mechanism before this test. Treat the listener report as a hypothesis to be tested, not as established perceptual improvement or degradation.",
  "context_receipt": {
    "project": "titan_audio_ecosystem",
    "direct_reference_input": false,
    "prompt_is_post_render_output": true,
    "waveform_filter_tests_latent_diffusion": false,
    "canonical_mutation_allowed": false
  }
}


## ADVISORY REVIEW experiment-designer; not empirical evidence
{
  "conclusions": [
    "The smallest reproducible comparison that can separate 'harshness' from 'center bias' is an offline, frozen-checkpoint waveform intervention on the existing 60-second prime WAV, not a training fork. The evidence packet already contains a center-heavy prime (L/R corr 0.991, side/mid -23.31 dB) and a 2-6 kHz fraction of 47.3%, so both reported symptoms are present in the same artifact and can be manipulated independently.",
    "Two mechanisms are separable by construction: (a) harshness is plausibly tied to the 2-6 kHz / 6-20 kHz energy and to post-render nonlinearity (tanh at 0.92 drive, wavefolder, peak normalization), while (b) center bias is tied to side/mid ratio and correlation. A mid/side gain change moves (b) without touching (a); a band-limited gain or a soft saturator moves (a) without touching (b).",
    "Level matching is mandatory and must be done on the same loudness basis (integrated LUFS or RMS over the identical interval) before any A/B, because the prime is peak-normalized to -1 dBFS and any gain change will otherwise be heard as a loudness difference.",
    "Transient handling must be explicit: the renderer has a Haas-style delayed side path (16-sample history) and a stateful DC blocker, so any offline filter that is not phase-matched to the original will alter transient arrival and can be mistaken for width or harshness change.",
    "The strongest single discriminating experiment is a 2x2 factorial on the prime: {mid/side gain: unchanged vs side-boosted to target -6 dB side/mid} x {high-band: unchanged vs 2-6 kHz shelved down 3 dB}, all level-matched, with a null control (identity copy) and a mono-collapse control. This yields four conditions plus two controls in one listening session."
  ],
  "evidence": [
    "context_packet.md lines 7-9: decision is to diagnose harshness and center bias in v9-long-01 and choose a reversible experiment before changing learned behavior.",
    "context_packet.md lines 13-14 and 46-47: side derives from L-R before a delayed scaled side stage; learned width bounded 0.05..0.50; global pan +/-0.10; increasing side gain cannot synthesize missing side information.",
    "context_packet.md lines 48-50: loss-visible tanh, later gain/saturation/DC blocker, whole-file peak normalization; a sample peak below 0 dBFS does not exclude nonlinear saturation or perceptual harshness.",
    "context_packet.md lines 51-53: prime selection uses field entropy, activity health, complexity, stagnation; no listener rating or direct stereo/harshness ranking; selection can amplify a within-run outlier.",
    "context_packet.md lines 64-67: offline waveform filtering tests signal processing, not latent-field diffusion; latent field entropy cannot be recovered from a WAV.",
    "deepseek_packet.md lines 24-29: latest prime L/R corr 0.991, side/mid -23.31 dB, 2-6 kHz 47.3%, 6-20 kHz 6.2%; untagged parent 0.603 / -6.01 dB / 33.6% / 3.0%.",
    "deepseek_packet.md lines 31-35: latest full 600 s is not uniformly centered; minutes 5-6 have corr 0.990-0.992 and side/mid about -23 dB; the 60 s prime is from a very center-heavy interval.",
    "deepseek_packet.md lines 37-43: training trace median output corr 0.963 vs target 0.862; median truthful width 0.027; decoder width control 0.120; width raw -0.946; global pan 0.0999 at its +0.10 limit; decoder side control 1.0; ultrasonic_ratio median 0.0965 is a pre-master guardrail, not a harshness score.",
    "src/main.rs 8059-8076: per-sample tanh at 0.92 drive then stateful DC blocker; raw_peak tracked after DC blocker.",
    "src/main.rs 8451-8455: whole-file peak normalization to 0.891 (-1 dBFS).",
    "src/main.rs 5591-5603: width_val from last_pan and control.width_mult; side_wide = side * width_val * learned_width; 16-sample Haas delay on side.",
    "src/stereo.rs 3-11, 28-41: GLOBAL_PAN_LIMIT 0.10; WIDTH_CONTROL_MIN 0.05, MAX 0.50; side_gain clamps width_mult to 0.5..1.6 and result to 0.5..maximum.",
    "src/main.rs 5529-5536: per-channel learned wavefolder applied after fold_drive scaling.",
    "src/main.rs 7103: audio_for_loss = stereo_chunk.tanh() — the loss sees a tanh'd signal, so harshness in the mastered WAV is not fully represented in the training objective.",
    "METRICS.md 129-134: side_energy_width can be high for panned mono; width multiplies by interchannel incoherence; stereo_corr is post-DC pre-master normalized correlation.",
    "METRICS.md 155-157: claims about improved sound require repeated seeded runs, ablation comparisons, objective measurements, and blinded listening."
  ],
  "uncertainties": [
    "No perceptual ratings exist yet (context_packet.md line 58-60); the listener report is a single unblinded observation.",
    "The prime is a selected 60 s window from a non-uniform 600 s render; any conclusion from the prime may not generalize to the full file or to other runs.",
    "The 2-6 kHz fraction is a mid-channel FFT power fraction; it can change because of denominator or stereo changes, not only because of absolute high-band energy (context_packet.md lines 54-56). Absolute L+R band power must be measured before attributing harshness to the high band.",
    "Whether the perceived harshness is dominated by the tanh/wavefolder nonlinearity, by the 2-6 kHz balance, or by transient density is not established by the supplied metrics.",
    "Whether the perceived center bias is dominated by side/mid collapse, by correlation near 1, or by the prime selection is not established.",
    "The Haas 16-sample delay and DC blocker make offline mid/side edits non-commutative with the original chain; a naive side boost may not reproduce what a renderer-side width change would sound like.",
    "No independent training replicas or matched arms exist (context_packet.md lines 28-29); this experiment cannot attribute the symptoms to training dynamics.",
    "Compute budget is unspecified; the proposal below assumes a single workstation and a single listening session."
  ],
  "proposed_experiments": [
    {
      "name": "E1: 2x2 offline factorial on the existing prime, level-matched, blinded",
      "purpose": "Separate harshness (high-band) from center bias (side/mid) with the smallest number of conditions that still yields a discriminating result.",
      "inputs": [
        "Existing 60 s prime WAV for v9-long-01 (read-only; copy to a scratch directory).",
        "Existing 60 s prime WAV for the untagged parent as a reference anchor (read-only copy)."
      ],
      "conditions": [
        "C0 null: byte-identical copy of the prime (control for file handling and for listener bias).",
        "C1 mono-collapse: L=R=(L+R)/2, level-matched (control for 'center bias' perception; should sound maximally centered).",
        "C2 side-boost only: mid/side matrix, raise side by +17 dB to bring side/mid from -23.31 dB to about -6 dB (matching the untagged parent), level-matched.",
        "C3 high-shelf only: 2-6 kHz shelving filter, -3 dB, Q=0.7, linear-phase or minimum-phase with matched group delay, level-matched.",
        "C4 both: C2 and C3 applied together, level-matched."
      ],
      "controls": [
        "C0 identity copy verifies the pipeline is transparent.",
        "C1 mono-collapse verifies the listener can detect a known center extreme.",
        "A second null: prime passed through the same mid/side matrix with side gain = 1.0 and the same shelf with 0 dB, to control for filter coloration."
      ],
      "level_matching": "Measure integrated LUFS (ITU-R BS.1770) over the full 60 s for each condition; apply a single static gain to each condition so all conditions match the C0 LUFS within +/-0.1 LU. Record the applied gain in a sidecar receipt. Do not use peak normalization for matching.",
      "transient_handling": "Use linear-phase FIR for the shelf so group delay is constant and can be compensated by a fixed integer-sample shift applied identically to all conditions. For the mid/side matrix, use the same 16-sample side delay as the renderer (src/main.rs 5600-5603) so the side path is phase-aligned with the original. Document the exact delay in the receipt.",
      "measurements_per_condition": [
        "L/R Pearson correlation (centered).",
        "side/mid in dB using the packet's definition (deepseek_packet.md lines 18-19).",
        "L/R balance in dB.",
        "Absolute L+R band power in 20-200 Hz, 200-2000 Hz, 2-6 kHz, 6-20 kHz (not fractions).",
        "Fractional band power for comparability with the packet.",
        "Crest factor and 95th-percentile short-term RMS to characterize transient density.",
        "Integrated LUFS and true peak."
      ],
      "listening_protocol": [
        "Blinded, randomized order, at least 5 listeners if available; otherwise the same listener across 3 sessions on different days.",
        "Each trial: A/B between two conditions, forced choice on (i) which is harsher, (ii) which is wider, (iii) preference.",
        "Anchor trials: C0 vs C1 and C0 vs C0 to estimate false-positive rate.",
        "Report per-condition mean and 95% CI for each question."
      ],
      "decision_criteria": [
        "If C3 (high-shelf only) is judged harsher than C0 by a majority and C2 is not, harshness is dominated by the 2-6 kHz band, not by side/mid.",
        "If C2 (side-boost only) is judged wider than C0 by a majority and C3 is not, center bias is dominated by side/mid collapse, not by high-band content.",
        "If C4 is judged both less harsh and wider than C0, the two mechanisms are separable and additive in this artifact.",
        "If C2 is judged harsher than C0, side-boost is introducing its own harshness (e.g., Haas comb or decorrelated high band) and the renderer-side width change is not a safe fix.",
        "If C0 vs C0 is not at chance, the protocol is invalid and must be repeated."
      ],
      "compute_bound": "Offline only. No training, no renderer invocation, no checkpoint mutation. Estimated cost: minutes of CPU for filtering plus one listening session.",
      "reversibility": "All outputs are copies in a scratch directory with sidecar receipts; canonical WAVs and checkpoints untouched."
    },
    {
      "name": "E2: Absolute-band-power audit of the full 600 s render",
      "purpose": "Test whether the prime's 2-6 kHz fraction reflects absolute high-band energy or a denominator shift, and whether the center bias is uniform or localized.",
      "inputs": [
        "Existing full 600 s WAV for v9-long-01 (read-only)."
      ],
      "method": "Compute non-overlapping 60 s windows; for each, report absolute L+R band power in 20-200 Hz, 200-2000 Hz, 2-6 kHz, 6-20 kHz, plus correlation, side/mid, balance, and integrated LUFS. Compare the prime window to the rest.",
      "decision_criteria": [
        "If the prime window's absolute 2-6 kHz power is not higher than the median window, the 'harsh' prime is a selection artifact and the fix is in prime selection, not in the renderer.",
        "If the prime window's side/mid is an outlier relative to the median, the same conclusion applies to center bias.",
        "If both are outliers, the smallest reversible fix is a prime-selection change (add a width/harshness guard to the selection score), not a renderer change."
      ],
      "compute_bound": "Offline FFT over 600 s; minutes of CPU.",
      "reversibility": "Read-only; no files modified."
    },
    {
      "name": "E3 (only if E1 and E2 point to the renderer): frozen-checkpoint sidecar render with width control clamped",
      "purpose": "Test whether the center bias is caused by the learned width head sitting near its low rail (width raw -0.946, width control 0.120) rather than by missing side information.",
      "inputs": [
        "Frozen checkpoint set for v9-long-01 (read-only).",
        "A sidecar renderer invocation that overrides control.width_mult to a fixed value (e.g., 1.0 and 1.6) without changing weights."
      ],
      "method": "Render a short bounded tape (e.g., 8 chunks) with the same seed and world state, once at width_mult=1.0 and once at width_mult=1.6. Compare side/mid and correlation. This is an intervention render, not a stem (docs/AUDIO_SCIENTIFIC_INSTRUMENTATION_PLAN.md lines 96-97).",
      "decision_criteria": [
        "If side/mid rises materially with width_mult and correlation falls, the width head is the binding constraint and a renderer-side control change is the smallest fix.",
        "If side/mid does not rise, the side signal itself is collapsed upstream (decoder side control 1.0 but side content absent) and the fix must be in the decoder or loss, not in the width control."
      ],
      "compute_bound": "One short render per condition; bounded by the existing 8-chunk autograd tape cap (src/main.rs 112).",
      "reversibility": "Sidecar outputs only; canonical checkpoint and WAVs untouched."
    }
  ],
  "context_receipt": {
    "project": "titan_audio_ecosystem",
    "direct_reference_input": false,
    "prompt_is_post_render_output": true,
    "waveform_filter_tests_latent_diffusion": false,
    "canonical_mutation_allowed": false
  },
  "summary": "Propose a bounded, offline, level-matched 2x2 factorial on the existing v9-long-01 prime (side-boost x high-shelf) with null and mono-collapse controls, blinded listening, and absolute-band-power audit of the full 600 s render. This separates harshness from center bias with the fewest conditions and no training fork. A frozen-checkpoint sidecar render is proposed only as a follow-up if the offline results point to the renderer.",
  "hypotheses": [
    "H1 (harshness = high-band balance): the 2-6 kHz fraction of 47.3% in the prime reflects elevated absolute high-band energy, and a -3 dB shelf there reduces perceived harshness.",
    "H2 (harshness = nonlinearity): the tanh at 0.92 drive and the learned wavefolder generate intermodulation in 2-6 kHz; a shelf will not fully remove it, and a lower drive would be needed.",
    "H3 (center bias = side collapse): side/mid -23.31 dB and corr 0.991 reflect a collapsed side signal, and a side boost restores perceived width.",
    "H4 (center bias = prime selection): the prime is a center-heavy outlier from a non-uniform render, and the fix is in prime selection, not in the renderer.",
    "H5 (interaction): side-boost introduces its own harshness via the Haas comb or decorrelated high band, so the two symptoms are not independent."
  ],
  "missing_evidence": [
    "Perceptual ratings from any listener, blinded or not.",
    "Absolute L+R band power for the prime and for the full 600 s render.",
    "Pre-side-scale renderer taps to attribute side collapse to the decoder vs the width control.",
    "Frequency-specific loss gradients to attribute the 2-6 kHz balance to a specific objective term.",
    "Detailed target mix attribution for the latest run.",
    "Independent training replicas or matched arms.",
    "Downstream priming outcomes (out of scope here)."
  ],
  "discriminating_experiment": "E1: 2x2 offline factorial (side-boost x high-shelf) on the existing prime, level-matched to integrated LUFS, with identity and mono-collapse controls, blinded A/B with forced choice on harshness, width, and preference. Decision rule: harshness tracks the shelf, width tracks the side boost, and any interaction (side boost increasing harshness) is itself the finding. E2 (absolute-band-power audit of the full 600 s) is run in parallel to test whether the prime is a selection outlier.",
  "recommendation": "Run E1 and E2 first. Do not fork training, do not modify the renderer, and do not change prime selection until E1 and E2 report. If E1 shows the shelf reduces harshness and the side boost restores width without introducing harshness, the smallest reversible fix is a renderer-side control change (width_mult and a high-band tilt), validated by E3. If E2 shows the prime is an outlier, the smallest fix is a prime-selection guard, which is cheaper and more reversible than any renderer change."
}


## ADVISORY REVIEW dynamics-agent; not empirical evidence
{
  "conclusions": [
    "The three candidate mechanisms are separable in principle but not by any single existing artifact: (a) waveform/signal-domain heat diffusion is an offline filter on the finalized WAV; (b) latent-field diffusion would act on the micro/macro CA field or the 512-d recurrent/MorphicStack state before the decoder; (c) a learned decoder is the existing 64+16-token spatial-temporal decoder plus control heads (src/main.rs:130-143, 5194-5196).",
    "Only (a) is testable with the current checkpoint and renderer without training: it is a copied-WAV, sidecar-receipt offline experiment (context_packet.md:64-70). It cannot test (b) or (c) because latent field entropy is not recoverable from a WAV (context_packet.md:67) and a trained diffusion/flow decoder is not a proxy for latent-field diffusion (context_packet.md:65-66).",
    "Mechanism (b) is not currently implemented as a module; renderer proposal documents are unimplemented ideas (context_packet.md:62-63). Any latent-diffusion claim therefore requires a new module and a training fork, which is outside the reversible-experiment budget.",
    "Mechanism (c) is already present and is the most likely locus of the reported harshness/center bias: the decoder's temporal controls drive mid/side (src/main.rs:5565-5572), wavefolder drive (5520-5536), openness (5538-5563), and the learned width head (5591-5599). The trace shows decoder width control 0.120, width raw -0.946, decoder side control 1.0, global pan 0.0999 at its +0.10 limit (deepseek_packet.md:37-41).",
    "Preservation risks differ by mechanism. Offline waveform filtering risks only creating copied WAVs and sidecars (context_packet.md:68-70). Latent-field diffusion risks changing the object under study and cannot be validated against the frozen checkpoint. A learned decoder change risks altering trained behavior and invalidating the v9-long-01 lineage (deepseek_packet.md:45-47).",
    "The reported center bias is at least partly a selection artifact: the 60-second prime is drawn from a very center-heavy interval (minutes 5-6, correlation 0.990-0.992, side/mid about -23 dB) while the full 600-second WAV ranges 0.740-0.992 (deepseek_packet.md:31-35). Prime selection uses field entropy, activity health, complexity, stagnation, not audible width or harshness (context_packet.md:51-53).",
    "Harshness is not established by any supplied metric. ultrasonic_ratio is a pre-master foldback guardrail, not a harshness score (deepseek_packet.md:40-41; METRICS.md:94-95). No controlled blinded listening has been performed (context_packet.md:58-60)."
  ],
  "evidence": [
    "context_packet.md:64-67 — offline waveform filtering tests signal processing; it is not a proxy for latent-field diffusion; a trained diffusion/flow decoder is also not a proxy; latent field entropy cannot be recovered from a WAV.",
    "context_packet.md:68-70 — preservation/budget: canonical checkpoints, corpus, existing WAVs, normal renderer and training stay intact; offline test may create copied WAVs and sidecar receipts; use paired intervals and record level-matching gains.",
    "context_packet.md:62-63 — renderer proposal documents contain unimplemented ideas; they are not available modules.",
    "context_packet.md:51-53 — prime selection uses field entropy, activity health, complexity, stagnation; no listener rating or direct stereo/harshness ranking; selection can amplify a within-run outlier.",
    "context_packet.md:58-60 — listener reports harsh/centered sound; no controlled blinded listening yet; no perceptual improvement established.",
    "deepseek_packet.md:24-29 — measurement table: untagged parent corr 0.603, side/mid -6.01; tagged latest corr 0.991, side/mid -23.31, 20-200 Hz 2.1%, 2-6 kHz 47.3%.",
    "deepseek_packet.md:31-35 — latest full 600-second WAV is not uniformly centered; minutes 5-6 have corr 0.990-0.992 and side/mid about -23 dB; current prime is from a very center-heavy interval.",
    "deepseek_packet.md:37-41 — trace: median output corr 0.963 vs target 0.862; median truthful width 0.027; decoder width control 0.120; width raw -0.946; global pan 0.0999 at +0.10 limit; decoder side control 1.0; ultrasonic_ratio median 0.0965 is a pre-master guardrail.",
    "deepseek_packet.md:45-47 — Audio v9 does not take target audio into the current forward pass; checkpoint set and existing audio files must remain byte-for-byte intact.",
    "src/main.rs:130-143 — decoder is low-rate (32 control frames per 4096-sample chunk), 76-control output head; DECODER_CONTROL_COUNT = 12 + 32*2.",
    "src/main.rs:5194-5196 — temporal_decoder.forward(next_micro, next_macro, refined_hidden) produces temporal_controls.",
    "src/main.rs:5520-5536 — fold_drive_l/r from temporal_controls[10,11] feed wavefolder_l/r.",
    "src/main.rs:5565-5572 — mid and side are formed from audio_l/audio_r and scaled by temporal_controls[4] and [5].",
    "src/main.rs:5591-5599 — width_val = stereo_side_gain(last_pan, control.width_mult); learned_width = soft_width_control(width_raw); side_wide = side * width_val * learned_width.",
    "src/main.rs:5600-5603 — Haas side path: side_history cat of prev_haas_side and side_wide; side_delayed = first CHUNK_SIZE; audio_l = mid + side_delayed; audio_r = mid - side_delayed.",
    "src/main.rs:8059-8076 — post path: tanh(0.92*x) then stateful DC blocker; raw_peak tracked.",
    "src/main.rs:8451-8455 — streamed mastering: norm = 0.891 / raw_peak; whole-file peak normalization to -1 dBFS.",
    "src/main.rs:8463-8482 — prime selection by sliding-window sum of chunk_scores; fade 2048 samples.",
    "src/stereo.rs:3-7 — GLOBAL_PAN_LIMIT 0.10; WIDTH_CONTROL_MIN 0.05; WIDTH_CONTROL_MAX 0.50.",
    "src/stereo.rs:9-11 — side_gain = ((1 + |last_pan|*0.8) * width_mult.clamp(0.5,1.6)).clamp(0.5, maximum).",
    "src/stereo.rs:28-41 — soft_width_control maps raw to [0.05, 0.50] with polynomial tail.",
    "src/stereo.rs:48-51 — correlation_aware_width multiplies side_energy_width by incoherence sqrt((1-corr)/2).",
    "METRICS.md:94-95 — ultrasonic_ratio is a pre-master aliasing/foldback guardrail, not a musical brightness score.",
    "METRICS.md:129-134 — side_energy_width can be high for panned mono; width multiplies by incoherence; stereo_corr is post-DC, pre-master.",
    "METRICS.md:155-157 — claims about improved sound require repeated seeded runs, ablation comparisons, objective audio measurements, and blinded listening."
  ],
  "uncertainties": [
    "No perceptual rating exists for any of the three mechanisms; harshness and center bias are listener reports only (context_packet.md:58-60).",
    "The supplied spectral fractions are MID-channel sampled FFT power fractions; changes can reflect denominator or stereo changes, so absolute and total L+R band power must be measured before claiming bass recovery (context_packet.md:54-57).",
    "The three latest WAVs form one sequential continuation lineage, not independent seeds or matched arms (context_packet.md:28-29), so between-WAV differences confound elapsed training, target mix, and online learning (context_packet.md:37-39).",
    "Target mix and loss history may confound attribution of the center bias (deepseek_packet.md:42-43).",
    "It is not established whether the harshness is caused by the wavefolder, the scan partials, the tanh saturation, the DC blocker, or the peak normalization; no frequency-specific loss gradients or pre-side-scale renderer taps are available (context_packet.md:71-74).",
    "Latent-field diffusion has no implemented module; any claim about it is speculation until a module exists and is tested against a frozen baseline.",
    "A learned decoder change cannot be evaluated against the frozen checkpoint without a training fork, which is outside the reversible-experiment budget (deepseek_packet.md:45-47).",
    "The trace's decoder width control 0.120 and width raw -0.946 are consistent with a width head near its lower operating region, but the trace is a sampled summary (704 rows) and does not by itself prove the audible width is collapsed.",
    "The global pan at 0.0999 is at the +0.10 limit, but pan_center_loss penalizes only the bounded audible residual (src/stereo.rs:21-26), so the pan head can sit at the rail without a strong gradient to return."
  ],
  "proposed_experiments": [
    {
      "id": "E1_waveform_filter_paired",
      "mechanism_tested": "signal-domain smoothing / waveform heat diffusion",
      "design": "Offline, on copied WAVs only. Apply a small family of low-pass or heat-kernel filters (e.g., one-pole, Gaussian, or explicit heat equation with small diffusion coefficient) to the latest prime and to the full 600-second WAV. Use paired non-overlapping 60-second intervals. Record level-matching gains and sidecar receipts. Do not modify canonical files.",
      "controls": [
        "Identity filter (zero diffusion) as negative control.",
        "Level-matched gain-only copy as a second negative control.",
        "Same filter applied to the untagged parent prime for a cross-lineage reference."
      ],
      "measurements": [
        "Absolute and total L+R band power in 20-200 Hz, 2-6 kHz, 6-20 kHz before and after.",
        "L/R correlation, side/mid dB, L/R balance dB before and after.",
        "Peak and RMS before and after; note that whole-file peak normalization is applied at render time (src/main.rs:8451-8455), so post-filter renormalization must be reported."
      ],
      "what_it_can_conclude": "Whether signal-domain smoothing changes the measured stereo and band statistics in the direction of the listener report.",
      "what_it_cannot_conclude": "It cannot test latent-field diffusion or a learned decoder (context_packet.md:64-67). It cannot establish perceptual improvement without blinded listening (METRICS.md:155-157)."
    },
    {
      "id": "E2_prime_selection_audit",
      "mechanism_tested": "selection artifact vs. renderer behavior",
      "design": "Recompute the prime selection score over the full 600-second WAV and report the score, correlation, side/mid, and band fractions for every non-overlapping 60-second interval. Compare the selected interval to the distribution.",
      "controls": [
        "Report the same statistics for the untagged parent and the two earlier tagged continuations."
      ],
      "measurements": [
        "Per-interval L/R correlation, side/mid dB, balance dB, 20-200 Hz, 2-6 kHz, 6-20 kHz.",
        "Selection score components (field entropy, activity health, structured complexity, stagnation) if recoverable from the trace."
      ],
      "what_it_can_conclude": "Whether the reported center bias is concentrated in the selected prime interval or is present across the run.",
      "what_it_cannot_conclude": "It cannot attribute the bias to a specific renderer subsystem."
    },
    {
      "id": "E3_frozen_renderer_tap_contrast",
      "mechanism_tested": "learned decoder contribution to center bias and harshness",
      "design": "On a frozen checkpoint, run a short render with instrumentation that records, per chunk, the pre-side-scale side energy, the learned width control, the wavefolder drive, the openness curve, and the mid/side split before and after the Haas path. This is an observation-only intervention; it must not change weights or the canonical renderer.",
      "controls": [
        "Same run with the width head output held at its trace median (0.120) to test sensitivity.",
        "Same run with the wavefolder drive held at 1.0 to test sensitivity."
      ],
      "measurements": [
        "Per-chunk side energy before and after width scaling.",
        "Per-chunk wavefolder drive and openness.",
        "Per-chunk mid/side ratio and correlation."
      ],
      "what_it_can_conclude": "Whether the center bias is produced by the learned width head, the side excitation, or the Haas path.",
      "what_it_cannot_conclude": "It cannot test latent-field diffusion, and it cannot establish perceptual improvement."
    },
    {
      "id": "E4_latent_diffusion_module_gate",
      "mechanism_tested": "latent-field diffusion",
      "design": "Not runnable now. A latent-diffusion experiment requires a new module that operates on the micro/macro field or the 512-d recurrent state before the decoder, plus a training fork. It must be gated on a written module specification and a frozen-baseline comparison.",
      "controls": [
        "Frozen baseline with no diffusion.",
        "Zero-output identity branch to prove bypass behavior only (context_packet.md:66-67)."
      ],
      "measurements": [
        "Field entropy, movement, and decoder control statistics before and after.",
        "Absolute and total L+R band power, correlation, side/mid."
      ],
      "what_it_can_conclude": "Nothing until the module exists and is trained.",
      "what_it_cannot_conclude": "It cannot be inferred from a WAV filter or from a trained diffusion/flow decoder (context_packet.md:64-67)."
    },
    {
      "id": "E5_blinded_listening",
      "mechanism_tested": "perceptual harshness and center bias",
      "design": "Paired, level-matched, blinded A/B between the latest prime and a reference (untagged parent prime or a filtered copy). Randomize order and require a stated preference and a harshness rating.",
      "controls": [
        "Level-matched gain-only copy as a null.",
        "Repeat with a second listener if available."
      ],
      "measurements": [
        "Preference, harshness rating, perceived width rating."
      ],
      "what_it_can_conclude": "Whether the listener report replicates under controlled conditions.",
      "what_it_cannot_conclude": "It cannot attribute the percept to a specific mechanism without the instrumented experiments above."
    }
  ],
  "context_receipt": {
    "project": "titan_audio_ecosystem",
    "direct_reference_input": false,
    "prompt_is_post_render_output": true,
    "waveform_filter_tests_latent_diffusion": false,
    "canonical_mutation_allowed": false
  },
  "summary": "Waveform heat diffusion, latent-field diffusion, and a learned decoder are three distinct mechanisms. Only waveform filtering is testable offline on copied WAVs; it cannot test latent-field diffusion or a learned decoder. The learned decoder is already present and is the most plausible locus of the reported center bias, but the current prime is drawn from a very center-heavy interval, so selection is a confound. No perceptual improvement is established. Proposed experiments are offline, paired, and preservation-safe; the latent-diffusion experiment is gated on a module that does not yet exist.",
  "hypotheses": [
    "H1 (supported by measurement): the reported center bias is partly a prime-selection artifact; the selected interval is among the most correlated in the run (deepseek_packet.md:31-35).",
    "H2 (plausible, not established): the learned width head and side excitation are contributing to the center bias; the trace shows width control 0.120 and width raw -0.946 (deepseek_packet.md:37-41).",
    "H3 (speculation): the harshness is associated with the wavefolder drive, scan partials, or the tanh saturation; no frequency-specific loss gradients or pre-side-scale taps are available to test this (context_packet.md:71-74).",
    "H4 (speculation): latent-field diffusion would reduce high-band state energy; this is not testable from a WAV and no module exists (context_packet.md:62-67)."
  ],
  "missing_evidence": [
    "Perceptual ratings under controlled blinded listening (context_packet.md:58-60, 71-74).",
    "Independent training replicas or matched arms; the three WAVs are one sequential lineage (context_packet.md:28-29).",
    "Pre-side-scale renderer taps and frequency-specific loss gradients (context_packet.md:71-74).",
    "Detailed target mix attribution (context_packet.md:71-74).",
    "Absolute and total L+R band power; the supplied fractions are MID-channel sampled FFT power fractions (context_packet.md:54-57).",
    "A written specification and implementation of any latent-diffusion module (context_packet.md:62-63)."
  ],
  "recommendation": "Run E1 (offline waveform filter on copied WAVs with paired intervals and level-matching receipts) and E2 (prime-selection audit) first; both are preservation-safe and can be completed without touching the checkpoint or renderer. Run E3 (frozen renderer tap contrast) only if instrumentation can be added without changing weights or canonical outputs. Do not attempt E4 (latent diffusion) until a module specification exists. Treat all conclusions as provisional until E5 (blinded listening) is performed."
}


## ADVISORY REVIEW rust-audit-agent; not empirical evidence
{"conclusions": ["The harshness/center-bias report is plausibly a selection artifact as much as a synthesis artifact: the 60-second prime is chosen by a scalar score that contains no stereo-width or harshness term, so a center-heavy, bright interval can be selected even if the full 600-second render is not uniformly centered (context_packet.md:31-35; main.rs:8141-8145, 8463-8479).", "The renderer's stereo path is structurally biased toward high L/R correlation: side is derived from L-R, then scaled by a bounded learned width (0.05..0.50) and a bounded side gain, and the mid/side recombination plus bounded pan cannot synthesize side information that the pre-mid/side signal lacks (main.rs:5565-5603; stereo.rs:5-11, 28-41).", "The pan head is at its +0.10 rail in the latest trace (deepseek_packet.md:39), and pan_center_loss penalizes only the bounded residual pan (stereo.rs:21-26), so the loss does not directly reward correlation=1; the observed correlation is more consistent with a collapsed side path than with pan alone.", "The training objective explicitly rewards matching target side/mid, correlation, and level (main.rs:7211-7234), but the learned width control is capped at 0.50 (stereo.rs:6, 28-41) and the side gain is capped (stereo.rs:9-11), so the optimizer cannot close a large target-side gap by widening alone; it can instead reduce correlation by other means or accept the gap.", "The prompt file is written after rendering and prime extraction (main.rs:8682-8697), so its descriptive words ('focused center image', 'bright glassy upper spectrum') cannot have caused the current run's audio; they are post-hoc descriptions (context_packet.md:19-21).", "The waveform-filter / diffusion framing is a category error for this codebase: forward has no target-audio argument (context_packet.md:13-15; AUDIO_SCIENTIFIC_INSTRUMENTATION_PLAN.md:101), and offline waveform filtering is not a proxy for latent-field diffusion (context_packet.md:64-67).", "Several loss terms are computed on detached targets and on tanh'd audio (main.rs:7103-7111, 7144-7176), which is correct for target detachment but means the 'harshness' seen post-master is only partially represented in the differentiable objective; the post path adds tanh saturation, a DC blocker, and whole-file peak normalization (main.rs:8059-8076, 8451-8514).", "The prime selection score (field_entropy * activity_health + structured_complexity * 0.75 - stagnation * 0.25) is a within-run argmax over a sliding window (main.rs:8141-8145, 8463-8479); it can amplify a within-run outlier and has no listener rating or stereo/harshness ranking (context_packet.md:51-53).", "The latest lineage is a sequential continuation, not independent seeds or matched arms (context_packet.md:28-29), so comparisons between the untagged parent and tagged continuations confound optimizer state, world state, and target mix."], "evidence": ["Prime selection: main.rs:8141-8145 defines chunk_scores; main.rs:8463-8479 selects best_start by max sliding-window sum; main.rs:8480-8482 maps to frames and applies a 2048-sample fade. No stereo or harshness term appears.", "Stereo path: main.rs:5565-5572 computes mid and side from audio_l/audio_r with learned scalars; main.rs:5591-5603 applies width_val and learned_width to side, then a 16-sample Haas delay, then recombines. stereo.rs:5-11 bounds width and side gain; stereo.rs:28-41 bounds learned width to 0.05..0.50.", "Pan: stereo.rs:3 sets GLOBAL_PAN_LIMIT=0.10; stereo.rs:13-19 soft_global_pan; stereo.rs:21-26 pan_center_loss penalizes pan.sqr() only. deepseek_packet.md:39 reports global pan 0.0999 at its +0.10 limit.", "Losses: main.rs:7103-7111 tanh and target detach; main.rs:7144-7176 mimic/band/chroma/envelope; main.rs:7185-7226 recurrence, modulation, low-band, stereo geometry; main.rs:7227-7234 stereo_balance_loss weights (0.30 side, 0.95 correlation, 0.45 level); main.rs:7245-7256 mimic_loss assembly.", "Post path: main.rs:8059-8076 tanh(0.92*x) and stateful DC blocker; main.rs:8451-8454 peak normalization to 0.891; main.rs:8508-8531 PCM16 write and prime fade.", "Prompt timing: main.rs:8682-8697 formats and writes suno_priming_prompt after rendering; context_packet.md:19-21 states it is written after rendering and prime extraction.", "Forward inputs: context_packet.md:13-15 and AUDIO_SCIENTIFIC_INSTRUMENTATION_PLAN.md:101 state forward has no target-audio argument; target sampling occurs after forward (main.rs:7041-7054 then 7107).", "Lineage: context_packet.md:28-29 states the three latest WAVs form one sequential continuation lineage, not independent seeds or matched arms.", "Metrics caveat: METRICS.md:64-76 and context_packet.md:54-57 warn that spectral fractions can reflect denominator or stereo changes and that correlation/side-mid describe stereo, not musical usefulness."], "uncertainties": ["Whether the audible harshness is dominated by the renderer's high-band content (scan partials, wavefolder, FM) or by the post-master tanh/DC/peak-normalization chain is not separable from the supplied evidence; the trace's ultrasonic_ratio is a pre-master guardrail, not a harshness score (deepseek_packet.md:40-41; METRICS.md:94-95).", "The exact contribution of the target mix to the latest continuation's correlation is unknown; target mix and online learning confound comparisons between elapsed portions of a run (context_packet.md:37-39).", "Whether the 60-second prime's center-heavy interval is representative of the full 600-second render is only partially addressed: deepseek_packet.md:31-33 reports non-overlapping 60-second intervals with correlation 0.740-0.992, but the full per-interval table is not in the supplied evidence.", "The learned width control's actual value in the latest run is reported as 0.120 (deepseek_packet.md:39), well below the 0.50 ceiling, so the cap is not the binding constraint; why the optimizer did not push width higher is not established by the supplied lines.", "Whether the side path's 16-sample Haas delay (main.rs:5600-5601) contributes to perceived harshness or center bias is not measured in the supplied evidence.", "No blinded listening has been performed (context_packet.md:58-60), so perceptual improvement or degradation is not established."], "proposed_experiments": ["Paired-interval prime audit (no training, no renderer change): for the existing latest WAV, compute per-60-second-interval L/R correlation, side/mid dB, and 2-6 kHz / 6-20 kHz fractions; compare the selected prime interval against the full-file distribution. This tests whether the report is a selection artifact. Cite main.rs:8463-8479 for the selection rule.", "Frozen side-path probe (no weight change): render a short segment with the existing checkpoint, then offline apply a controlled side-gain multiplier to the pre-master signal and re-master with the same tanh/DC/peak chain (main.rs:8059-8076, 8451-8514). Measure correlation and side/mid. This tests whether the center bias is recoverable by side gain alone or requires new side information.", "Loss-gradient attribution (frozen, single chunk): instrument the stereo_balance_loss components (main.rs:7219-7234) on a fixed chunk and report per-component gradient norms to the width head, pan head, and side excitation controls. This tests which term is actually driving the observed correlation.", "Target-mix control: hold the model and world state fixed, vary only the target episode selection (main.rs:7107-7140) across a small set of frozen chunks, and measure output correlation and band fractions. This tests whether the center bias tracks the target mix.", "Prime-selection counterfactual: recompute the prime window using an alternative score that includes a stereo-width term, and compare the selected interval to the current one. This tests whether the current score systematically prefers center-heavy intervals.", "Post-path ablation (offline, copied WAVs only): apply the tanh/DC/peak chain with and without the tanh stage to the same pre-master signal and measure 2-6 kHz / 6-20 kHz fractions. This tests whether the post path contributes to the reported harshness. Do not modify canonical files (context_packet.md:68-70)."], "context_receipt": {"project": "titan_audio_ecosystem", "direct_reference_input": false, "prompt_is_post_render_output": true, "waveform_filter_tests_latent_diffusion": false, "canonical_mutation_allowed": false}}


## ADVISORY REVIEW statistical-agent; not empirical evidence
{
  "context_receipt": {
    "project": "titan_audio_ecosystem",
    "direct_reference_input": false,
    "prompt_is_post_render_output": true,
    "waveform_filter_tests_latent_diffusion": false,
    "canonical_mutation_allowed": false
  },
  "conclusions": [
    {
      "id": "C1",
      "claim": "The supplied spectral table is a MID-channel-only, fraction-normalized estimator and cannot by itself support claims about absolute band energy, bass recovery, or stereo width.",
      "support": "The packet states spectral fractions estimate mean mid-channel Hann-window FFT power normalized over 20-20,000 Hz (deepseek_packet.md:19-22). The context contract warns that fraction changes can reflect denominator or stereo changes and instructs measuring absolute and total L+R band power before claiming bass recovery (audio_context.md:54-57).",
      "confidence": "high"
    },
    {
      "id": "C2",
      "claim": "The three tagged WAVs are a sequential continuation lineage, not independent seeds or matched arms; cross-prime comparisons are confounded by online learning, target mix, and carried state.",
      "support": "audio_context.md:28-29 states the three latest WAVs form one sequential continuation lineage, not independent seeds or matched arms. audio_context.md:37-39 notes target mix and online learning confound comparisons between elapsed portions of a run. deepseek_packet.md:12-15 gives the step/update counts showing monotone progression.",
      "confidence": "high"
    },
    {
      "id": "C3",
      "claim": "The 60-second prime is selected by an internal score (field entropy, activity health, structured complexity, stagnation) with no listener rating or stereo/harshness ranking, so it can amplify a within-run outlier.",
      "support": "audio_context.md:51-53 and deepseek_packet.md:33-35. The scoring formula is visible in src/main.rs:8141-8145 and the sliding-window selection in src/main.rs:8459-8481.",
      "confidence": "high"
    },
    {
      "id": "C4",
      "claim": "The latest prime is drawn from a center-heavy interval of a non-uniform 600-second render; the full render is not uniformly centered.",
      "support": "deepseek_packet.md:31-34 reports non-overlapping 60-second intervals with L/R correlation 0.740-0.992 and minutes 5-6 at 0.990-0.992 with side/mid about -23 dB.",
      "confidence": "high"
    },
    {
      "id": "C5",
      "claim": "The listener report of harshness and center bias is a single uncontrolled observation; no blinded listening has been performed, so no perceptual improvement or perceptual cause is established.",
      "support": "audio_context.md:58-60 states neither PI nor agents have performed controlled blinded listening yet and no perceptual improvement is established. deepseek_packet.md:7-8 instructs treating metrics as diagnostics and requiring listening for audible quality.",
      "confidence": "high"
    },
    {
      "id": "C6",
      "claim": "Offline waveform filtering tests signal processing only; it is not a proxy for latent-field diffusion, and latent field entropy cannot be recovered from a WAV.",
      "support": "audio_context.md:64-67. deepseek_packet.md:48-51 distinguishes signal-domain smoothing, latent-field diffusion, and a learned spectral/diffusion decoder.",
      "confidence": "high"
    },
    {
      "id": "C7",
      "claim": "The prompt file is written after rendering and prime extraction and cannot have caused the run; it may only influence a later external system outside this experiment.",
      "support": "audio_context.md:19-21. The write occurs at src/main.rs:8695-8697, after the mastering/prime block at src/main.rs:8448-8542.",
      "confidence": "high"
    },
    {
      "id": "C8",
      "claim": "The forward pass has no text prompt, target audio, prime, or external reference argument; target sampling occurs after forward.",
      "support": "audio_context.md:13-18. src/main.rs:7041-7054 calls model.forward before src/main.rs:7107 samples targets. docs/AUDIO_SCIENTIFIC_INSTRUMENTATION_PLAN.md:101 states the same ordering.",
      "confidence": "high"
    },
    {
      "id": "C9",
      "claim": "AdamW moments are persisted and resumed under strict compatibility conditions; the real reproducibility limitation is checkpoint-set transactionality, not absence of moment persistence.",
      "support": "docs/AUDIO_SCIENTIFIC_INSTRUMENTATION_PLAN.md:112-124. audio_context.md:28-29 confirms persisted moments.",
      "confidence": "high"
    },
    {
      "id": "C10",
      "claim": "The low-band loss term supervises only the first log band and is not the same as the 20-200 Hz measurement used in the survey table.",
      "support": "audio_context.md:40-43. METRICS.md:64-65 describes output_low_band_ratio as the first supervised 20 Hz log band. src/main.rs:7202-7210 shows first_band_energy_ratio usage.",
      "confidence": "high"
    },
    {
      "id": "C11",
      "claim": "Pan-center loss penalizes the bounded audible residual pan and does not directly reward correlation equal to one.",
      "support": "audio_context.md:44-47. src/stereo.rs:21-26 implements pan_center_loss as squared bounded pan. src/main.rs:7227 applies it to the bounded pan tensor.",
      "confidence": "high"
    },
    {
      "id": "C12",
      "claim": "Increasing side gain cannot synthesize missing side information; the side path derives from L-R before a delayed scaled side stage.",
      "support": "audio_context.md:44-47. src/main.rs:5569-5603 shows side derived from audio_l minus audio_r, scaled by width and delayed Haas history.",
      "confidence": "high"
    },
    {
      "id": "C13",
      "claim": "A sample peak below 0 dBFS does not exclude nonlinear saturation or perceptual harshness, because loss-visible tanh, later gain/saturation/DC blocking, and whole-file peak normalization are in the path.",
      "support": "audio_context.md:48-50. src/main.rs:8059-8076 applies tanh and a stateful DC blocker; src/main.rs:8451 normalizes to -1 dBFS.",
      "confidence": "high"
    },
    {
      "id": "C14",
      "claim": "The trace's ultrasonic_ratio is a pre-master aliasing/foldback guardrail, not a harshness score.",
      "support": "deepseek_packet.md:40-41. METRICS.md:94-95 states the same.",
      "confidence": "high"
    },
    {
      "id": "C15",
      "claim": "The latest training trace shows the model narrower and more correlated than its targets, with global pan at its +0.10 limit and decoder width control at 0.120.",
      "support": "deepseek_packet.md:37-40. The pan limit is defined in src/stereo.rs:3 and applied in src/stereo.rs:13-19.",
      "confidence": "medium",
      "note": "Trace values are reported by the packet; the underlying trace file was not supplied for independent recomputation."
    }
  ],
  "evidence": [
    {
      "path": "analysis/audio_center_harshness_20260926/team_v2/context_packet.md",
      "lines": "1-83",
      "role": "Shared contract; receipt fields, causal order, prime selection, metric caveats, inference limits."
    },
    {
      "path": "analysis/audio_center_harshness_20260926/deepseek_packet.md",
      "lines": "1-52",
      "role": "Measurement table, run identity, trace summary, boundaries."
    },
    {
      "path": "docs/AUDIO_SCIENTIFIC_INSTRUMENTATION_PLAN.md",
      "lines": "46-144",
      "role": "Architecture inventory, target semantics, optimizer persistence and transactionality."
    },
    {
      "path": "src/main.rs",
      "lines": "70-160",
      "role": "Constants: sample rate, chunk size, macro cadence, tape cap, decoder controls."
    },
    {
      "path": "src/main.rs",
      "lines": "235-260",
      "role": "Stereo loss weights and side-gain ceiling."
    },
    {
      "path": "src/main.rs",
      "lines": "5107-5255",
      "role": "Forward signature and oscillator/pitch path; confirms no reference argument."
    },
    {
      "path": "src/main.rs",
      "lines": "5290-5648",
      "role": "Renderer: scan partials, excitation, wavefolders, mid/side, pan, width, Haas delay."
    },
    {
      "path": "src/main.rs",
      "lines": "7020-7270",
      "role": "Forward-before-target ordering, min-of-K target selection, loss terms."
    },
    {
      "path": "src/main.rs",
      "lines": "7780-7995",
      "role": "Loss aggregation, tape accumulation, gradient clipping, optimizer step."
    },
    {
      "path": "src/main.rs",
      "lines": "8048-8150",
      "role": "Post path: tanh, DC blocker, raw peak, chunk score formula."
    },
    {
      "path": "src/main.rs",
      "lines": "8448-8542",
      "role": "Streamed mastering, prime window selection, fade, WAV writing."
    },
    {
      "path": "src/main.rs",
      "lines": "8665-8698",
      "role": "Prompt construction and write after rendering."
    },
    {
      "path": "src/stereo.rs",
      "lines": "1-51",
      "role": "Pan limit, width bounds, soft pan/width, pan-center loss, correlation-aware width."
    },
    {
      "path": "METRICS.md",
      "lines": "45-158",
      "role": "Estimator definitions and explicit non-claims."
    }
  ],
  "uncertainties": [
    {
      "id": "U1",
      "item": "The supplied table reports only mid-channel fractions; absolute L+R band power, total energy, and per-band side energy are not provided.",
      "impact": "Bass-recovery and harshness claims cannot be evaluated from the table alone."
    },
    {
      "id": "U2",
      "item": "No perceptual ratings, no blinded listening, and no independent training replicas are available.",
      "impact": "Audible quality and causal attribution to training remain unestablished."
    },
    {
      "id": "U3",
      "item": "Pre-side-scale renderer taps and frequency-specific loss gradients are missing.",
      "impact": "Cannot separate whether center bias originates in the learned renderer, the side stage, the pan head, or the target distribution."
    },
    {
      "id": "U4",
      "item": "Detailed target mix attribution is missing.",
      "impact": "Cannot rule out that the target corpus itself is center-heavy or bright, which would confound any harshness attribution."
    },
    {
      "id": "U5",
      "item": "The trace values in deepseek_packet.md:37-41 were not independently recomputed from the trace file in this review.",
      "impact": "Trace-derived conclusions are medium confidence."
    },
    {
      "id": "U6",
      "item": "The 600-second render's interval statistics are reported but the underlying WAV was not supplied for recomputation.",
      "impact": "Non-uniformity of the render is accepted as reported, not verified."
    },
    {
      "id": "U7",
      "item": "Whether the two quarantine-provenance manifest entries entered gradients is unknown from manifest roles alone.",
      "impact": "Corpus contamination cannot be confirmed or excluded from the supplied evidence."
    },
    {
      "id": "U8",
      "item": "Downstream priming outcomes are missing.",
      "impact": "Any effect of the prompt on an external system is outside this experiment and unmeasured."
    }
  ],
  "proposed_experiments": [
    {
      "id": "E1",
      "name": "Absolute and stereo-resolved band power re-measurement",
      "purpose": "Replace fraction-only mid-channel estimates with defensible units.",
      "design": "On the existing prime WAVs and on non-overlapping 60-second intervals of the full render, compute per-band absolute power in dBFS for L, R, mid, and side over fixed bands (for example 20-200 Hz, 200-2000 Hz, 2-6 kHz, 6-20 kHz). Report total L+R power and side/mid ratio per band. Use identical windowing and normalization across files.",
      "controls": "Same window count, window length, and band edges for every file; report the estimator explicitly.",
      "expected_discriminator": "If bass fraction fell because total energy rose rather than because low-band power fell, absolute low-band power will be flat or higher.",
      "limits": "Signal-domain measurement only; does not establish perceptual harshness."
    },
    {
      "id": "E2",
      "name": "Prime-selection sensitivity audit",
      "purpose": "Quantify how much the reported center bias is an artifact of prime selection.",
      "design": "Recompute the chunk-score sliding window over the full render and report the distribution of candidate 60-second windows by L/R correlation and side/mid. Compare the selected window against the median and against the widest window.",
      "controls": "Use the same score formula as src/main.rs:8141-8145 without modification.",
      "expected_discriminator": "If the selected prime is an extreme outlier, its correlation and side/mid will sit far from the window distribution median.",
      "limits": "Does not change the renderer or checkpoint."
    },
    {
      "id": "E3",
      "name": "Frozen first-chunk invariance negative control",
      "purpose": "Confirm the forward pass has no reference input.",
      "design": "Clone identical pre-forward state and run forward with two different target chunks presented to the loader. Compare the emitted current-chunk audio byte-for-byte.",
      "controls": "Identical model, world, optimizer, RNG seed, and pre-forward state.",
      "expected_discriminator": "Identical current-chunk audio supports the no-reference-input claim; any difference would contradict it.",
      "limits": "Tests the current chunk only; later host feedback may still differ."
    },
    {
      "id": "E4",
      "name": "Sidecar renderer-tap capture",
      "purpose": "Localize center bias to a stage.",
      "design": "On a copied WAV and a sidecar receipt, capture pre-side-scale, post-side-scale, pre-pan, and post-pan signals for the same interval. Report side/mid and correlation at each tap.",
      "controls": "No change to canonical checkpoints, corpus, or existing WAVs; outputs are copies with receipts.",
      "expected_discriminator": "If side/mid is already low before the side stage, the collapse originates upstream; if it drops only after the side stage or pan, the stage is implicated.",
      "limits": "Requires renderer instrumentation that may not currently exist; treat as a proposal, not a completed measurement."
    },
    {
      "id": "E5",
      "name": "Level-matched paired-interval comparison",
      "purpose": "Make any A/B comparison defensible.",
      "design": "Select paired intervals of equal length from the untagged parent and the latest tagged continuation, match integrated loudness, and record the applied gain in a sidecar receipt. Report absolute band power and stereo geometry for each pair.",
      "controls": "Record level-matching gains; do not alter canonical files.",
      "expected_discriminator": "If differences persist after level matching, they are not explained by gain alone.",
      "limits": "Still not a perceptual test."
    },
    {
      "id": "E6",
      "name": "Blinded listening protocol",
      "purpose": "Establish or refute the perceptual claim.",
      "design": "Present level-matched, order-randomized, labeled-hidden excerpts to multiple listeners with a fixed rating instrument covering harshness and center bias. Report inter-rater agreement.",
      "controls": "Randomized order, hidden labels, level matching, documented exclusion rules.",
      "expected_discriminator": "Consistent ratings above chance support the perceptual claim; chance-level ratings do not.",
      "limits": "Requires listeners and a protocol not currently available."
    },
    {
      "id": "E7",
      "name": "Target-mix attribution audit",
      "purpose": "Determine whether center bias or brightness is inherited from targets.",
      "design": "Compute the same stereo geometry and band-power statistics on the target episodes actually sampled during the run, using the recorded target_file and target_frame fields.",
      "controls": "Use the recorded target identity, not a re-sampled corpus.",
      "expected_discriminator": "If targets are themselves center-heavy or bright, the model's geometry may be a faithful match rather than a defect.",
      "limits": "Depends on trace fields being complete."
    }
  ],
  "defensible_units_and_limits": {
    "units": [
      "Band power in dBFS per channel and per mid/side, with explicit band edges and window parameters.",
      "L/R Pearson correlation over a stated interval.",
      "Side/mid as 10*log10(mean(side^2)/mean(mid^2)) with side=(L-R)/2 and mid=(L+R)/2, as defined in deepseek_packet.md:18-19.",
      "L/R balance as 10*log10(E_L/E_R).",
      "Integrated loudness for level matching, with the applied gain recorded."
    ],
    "limits": [
      "Fraction-only mid-channel spectra cannot support absolute-energy or stereo claims.",
      "Internal metrics (sigma, phi, pi_proxy, empowerment, novelty_dmin, field_entropy, ultrasonic_ratio) are proxies or guardrails, not perceptual or causal measures (METRICS.md:45-95, 149-153).",
      "Correlation and side/mid describe stereo geometry, not musical usefulness (audio_context.md:57).",
      "Offline waveform filtering is not a proxy for latent-field diffusion (audio_context.md:64-67).",
      "A zero-output identity branch proves bypass behavior only (audio_context.md:66-67).",
      "Sample peak below 0 dBFS does not exclude saturation or harshness (audio_context.md:48-50).",
      "Single-run, single-prime observations cannot establish a training-caused effect."
    ]
  },
  "conflicts": [
    {
      "claim_a": "The provisional request caveat suggested AdamW state may not be persisted.",
      "claim_b": "Current Audio v9 persists AdamW moments and resumes them under strict compatibility conditions.",
      "why_they_differ": "The caveat predates or misreads the current implementation; the source-audit correction in docs/AUDIO_SCIENTIFIC_INSTRUMENTATION_PLAN.md:112-124 documents PersistentAdamW::save and restore conditions.",
      "discriminating_experiment": "Inspect titan_optimizer_v9.safetensors and the run metadata optimizer_resumed and optimizer_migration fields after a controlled restart; confirm moments and cumulative update count match the pre-restart values."
    },
    {
      "claim_a": "The latest continuation sounds harsh and centered (listener report).",
      "claim_b": "The latest prime is drawn from a center-heavy interval, and the full render is not uniformly centered.",
      "why_they_differ": "The listener may have heard the prime or a center-heavy portion; the report and the prime selection are not independent of each other.",
      "discriminating_experiment": "Blinded, level-matched listening across multiple non-overlapping intervals of the full render, including the widest and the most center-heavy windows, with hidden labels."
    }
  ],
  "recommendation": "Treat the current evidence as diagnostic only. Before any training fork, complete E1, E2, E3, and E5 on copies with sidecar receipts, keeping canonical checkpoints, corpus, and existing WAVs byte-for-byte intact. Do not equate signal-domain smoothing with quality, and do not treat waveform filtering as a latent-diffusion proxy. Any perceptual claim requires E6.",
  "speculation_label": "Items marked as hypotheses or as proposals in proposed_experiments are untested. Trace-derived values in C15 are accepted as reported and were not independently recomputed."
}


## ADVISORY REVIEW ideation-agent; not empirical evidence
{"conclusions": {"summary": "The supplied evidence describes a center-biased, harsh-leaning renderer whose center collapse is consistent with at least three separable mechanisms (learned width/pan rails, loss geometry, and post-render nonlinearity). The evidence does not establish which dominates. The cheapest discriminating tests are offline, checkpoint-preserving, and sidecar-only, so they cannot certify perceptual improvement.", "key_points": ["Latest tagged prime shows L/R correlation 0.991 and side/mid -23.31 dB, plus 2-6 kHz at 47.3% and 6-20 kHz at 6.2%, consistent with mono-like center and upper-mid emphasis.", "The 600 s file is not uniformly centered: minutes 5-6 are 0.990-0.992 correlation, so the 60 s prime is a selected within-run extreme. Prime selection uses field entropy, activity health, complexity, and stagnation, not width or harshness (context lines 51-54; main.rs 8463-8481).", "Learned width is bounded 0.05..0.50 and pan +/-0.10; increasing side gain cannot synthesize missing side information (context lines 44-47; stereo.rs 3-11).", "Stereo loss combines side/mid, correlation, and level terms with weights 0.30/0.95/0.45; pan-center loss is separate and smaller (main.rs 253-255, 7227-7234; stereo.rs 4).", "Loss-visible tanh, then later tanh(0.92 x) and a DC blocker, then whole-file peak normalization to 0.891; a sub-0 dBFS peak does not exclude saturation (main.rs 8059-8076, 8451).", "The forward pass has no target-audio argument; target sampling occurs after forward (main.rs 7041-7107; context lines 13-16)."], "bounded_purpose_fit": "All proposed tests are offline signal or frozen-rollout diagnostics that preserve checkpoints, corpus, and the normal renderer. None is a proxy for latent-field diffusion."}, "evidence": [{"source": "analysis/audio_center_harshness_20260926/deepseek_packet.md", "lines": "24-35", "claim": "Table shows latest prime corr 0.991, side/mid -23.31 dB, 2-6 kHz 47.3%; full-file minutes 5-6 are 0.990-0.992."}, {"source": "analysis/audio_center_harshness_20260926/deepseek_packet.md", "lines": "37-43", "claim": "Training trace median output corr 0.963 vs target 0.862; width control 0.120; pan 0.0999 at limit; decoder side control 1.0; ultrasonic_ratio is a guardrail, not a harshness score."}, {"source": "research/audio_context.md", "lines": "44-47", "claim": "Side derives from L-R before delayed scaled side stage; width bounded 0.05..0.50; pan +/-0.10; scaling side cannot create missing side information."}, {"source": "research/audio_context.md", "lines": "48-50", "claim": "Loss-visible tanh, later saturation/DC/peak normalization; sub-0 dBFS peak does not exclude saturation or harshness."}, {"source": "src/stereo.rs", "lines": "3-11, 21-41", "claim": "Pan limit 0.10, pan-center weight 0.10, width range 0.05..0.50, side_gain depends on last_pan and width_mult."}, {"source": "src/main.rs", "lines": "253-255, 7227-7234", "claim": "Stereo side/correlation/level weights 0.30/0.95/0.45 combined into stereo_balance_loss; pan_center_loss weighted separately."}, {"source": "src/main.rs", "lines": "8059-8076, 8451-8455", "claim": "Post path applies tanh(x*0.92), stateful DC blocker, then peak normalization to 0.891."}, {"source": "src/main.rs", "lines": "8463-8481", "claim": "Prime is the highest-scoring 60 s window by chunk_scores (field entropy, complexity, activity, stagnation)."}, {"source": "src/main.rs", "lines": "7041-7107", "claim": "model.forward runs before target_loader.sample_chunks; forward has no target argument."}], "uncertainties": ["No blinded listening has been performed; perceptual harshness and center bias are unverified (context lines 58-60).", "Target mix and online learning confound comparisons between elapsed portions of the run (context lines 38-39).", "The 60 s prime is a selected within-run outlier; metrics on it may not represent the run.", "Whether the harshness is spectral (upper-mid emphasis), nonlinear (post-render saturation), or statistical (center collapse reducing interchannel decorrelation) is not separable from the supplied metrics alone.", "Frequency-specific loss gradients and pre-side-scale renderer taps are missing (context line 72)."], "hypotheses": {"conventional": [{"id": "H1", "claim": "CLAIM A: Center bias is a learned-control rail effect. The width head saturates low and the pan head sits at its +0.10 limit, so the side path contributes little and mid dominates. Harshness is a secondary consequence of correlated channels plus upper-mid partials.", "falsifier": "If a frozen rollout with width_mult forced to 1.0 and pan raw zeroed still shows corr > 0.95 and side/mid < -18 dB, the rail story is insufficient.", "cheapest_discriminating_experiment": "Frozen single-chunk probe: load the latest checkpoint, hold world state and RNG fixed, run forward with (a) width_mult=1.0 and last_pan=0, (b) width_mult=0.5, (c) width_mult=1.6. Compare side/mid and L/R correlation of the emitted chunk. No training, no checkpoint writes."}, {"id": "H2", "claim": "CLAIM B: Harshness is dominated by the post-render nonlinearity chain (loss-visible tanh, then tanh(0.92x), then peak normalization), which adds harmonics when the pre-master signal is hot in the 2-6 kHz region.", "falsifier": "If offline bypass of the second tanh and DC blocker on the existing WAV does not reduce measured 2-6 kHz energy relative to 20-20 kHz while preserving level, the nonlinearity is not the dominant contributor.", "cheapest_discriminating_experiment": "Offline sidecar test on copied WAVs: apply (a) exact inverse of peak normalization, (b) atanh of the second-stage tanh on the pre-normalized signal if recoverable, (c) DC-blocker bypass. Compare 2-6 kHz and 6-20 kHz absolute band power, level-matched. Label as signal-domain only."}], "unconventional": [{"id": "H3", "claim": "CLAIM C (speculative, retained): The perceived center bias is partly an artifact of the prime-selection objective. chunk_scores rewards field entropy and structured complexity, which can be maximized by coherent high-energy mid-channel motion; the 60 s prime therefore oversamples center-heavy, upper-mid-dense intervals even if the full 600 s file has more width elsewhere.", "falsifier": "If re-ranking the same run's chunks by an objective that includes measured side/mid and 2-6 kHz balance selects a window whose correlation and band fractions are statistically indistinguishable from the current prime, the selection-objective story fails.", "cheapest_discriminating_experiment": "Offline re-selection: from the existing full 600 s WAV, compute per-60 s window correlation, side/mid, and 2-6 kHz fraction using the same estimator family as the packet, then rank windows by (a) current chunk_scores proxy if recoverable from trace, (b) side/mid, (c) 2-6 kHz balance. Compare the top windows. This tests selection bias only; it does not test the renderer."}], "why_they_differ": "H1 places the cause in learned control heads; H2 places it in the fixed post-render chain; H3 places it in the evaluation/selection pipeline. They make different predictions about what a frozen forward probe, an offline signal test, and a re-ranking test would each show.", "discriminating_experiment": "Run the three experiments above in order. H1 and H2 are separable because H1 changes the emitted chunk before mastering while H2 changes only the post path. H3 is separable because it changes only which window is reported, not the audio. None requires training or checkpoint mutation."}, "missing_evidence": ["Blinded listening ratings for harshness and center bias.", "Pre-side-scale renderer taps to separate side generation from side scaling.", "Frequency-specific loss gradients for the 2-6 kHz region.", "Per-chunk side/mid and correlation trace across the full 600 s file.", "Independent training replicas to separate checkpoint effects from run-specific state."], "context_receipt": {"project": "titan_audio_ecosystem", "direct_reference_input": false, "prompt_is_post_render_output": true, "waveform_filter_tests_latent_diffusion": false, "canonical_mutation_allowed": false}, "proposed_experiments": [{"id": "E1_frozen_width_pan_probe", "tests": "H1", "method": "Frozen single-chunk forward with fixed world state and RNG; vary width_mult in {0.5, 1.0, 1.6} and last_pan in {0.0, +0.10, -0.10}. Measure side/mid and L/R correlation on the emitted chunk before mastering.", "cost": "Low; one process, no training, no file writes beyond copied outputs.", "success_criterion": "If side/mid changes by >6 dB across width_mult while correlation stays >0.95, H1 is weakened. If correlation tracks width_mult strongly, H1 is supported.", "controls": "Identical world state, seed, chunk index, and target-free forward; report level-matched gains."}, {"id": "E2_offline_post_chain_bypass", "tests": "H2", "method": "On copied WAVs, apply level-matched variants: (a) as-is, (b) second-stage tanh removed via atanh where the pre-normalized signal is recoverable, (c) DC-blocker bypass. Compare absolute 2-6 kHz and 6-20 kHz band power and crest factor.", "cost": "Low; offline only, sidecar receipts.", "success_criterion": "If 2-6 kHz fraction drops materially with (b) while total L+R power is matched, H2 is supported.", "controls": "Level-match each variant to the original RMS; report gain applied; do not overwrite canonical WAVs."}, {"id": "E3_prime_reselection_audit", "tests": "H3", "method": "From the existing 600 s WAV, compute per-60 s correlation, side/mid, and 2-6 kHz fraction with the packet's estimator family. Compare top windows under current-score proxy versus side/mid and 2-6 kHz objectives.", "cost": "Low; analysis only.", "success_criterion": "If the current-selection window is a statistical outlier on correlation and 2-6 kHz relative to the alternative objectives, H3 is supported.", "controls": "Use the same window length and estimator; report both selected windows and their metrics."}], "recommendation": "Run E1 and E3 first (cheapest, checkpoint-preserving), then E2 if H2 remains live. Treat all results as diagnostics, not perceptual proof. Do not raise the learned-width rail or change loss weights before E1 clarifies whether the rail is the binding constraint. Label H3 and any diffusion-adjacent idea as speculation until a frozen or sidecar test discriminates it."}


## ADVISORY REVIEW skeptical-agent; not empirical evidence
{
  "conclusions": [
    "The reported harshness and center bias are most plausibly explained by mundane, non-exotic mechanisms already visible in the supplied evidence: (a) the 60-second prime is selected by an internal score that does not measure width or harshness, so it can land on a center-heavy, bright interval; (b) the latest continuation's measured L/R correlation (0.991) and side/mid (-23.31 dB) are near mono-collapse, consistent with a center-heavy prime; (c) the audible path ends in tanh saturation plus DC blocking plus whole-file peak normalization, so a sample peak below 0 dBFS does not exclude nonlinear harshness; (d) the learned width control is bounded to 0.05..0.50 and the global pan to +/-0.10, so the renderer cannot synthesize missing side information.",
    "The strongest unsupported leap in the packet is treating the generated prompt text ('focused center image', 'bright glassy upper spectrum') as a cause of the run. The source writes the prompt after rendering and prime extraction (src/main.rs:8682-8697), so it is a post-render description, not an input.",
    "A second unsupported leap is treating waveform-domain filtering or a trained diffusion/flow decoder as a proxy for latent-field diffusion. The context contract explicitly separates these (research/audio_context.md:64-67).",
    "A third unsupported leap is inferring that the whole 600-second WAV is uniformly centered. The packet itself reports non-overlapping 60-second intervals with correlation 0.740-0.992 (deepseek_packet.md:31-35), so the prime is a within-run outlier, not a global property.",
    "The 'center collapse' is better described as a prime-selection artifact plus a bounded-width renderer, not as proof that the model lost stereo capability. The full-run interval spread contradicts uniform collapse.",
    "The 'harshness' claim is not established by the supplied metrics. ultrasonic_ratio is explicitly a pre-master aliasing guardrail, not a harshness score (deepseek_packet.md:40-41; METRICS.md:94-95). No blinded listening has been performed (audio_context.md:58-60).",
    "Practical priming is a downstream, out-of-scope concern here: the prompt file may influence a later external music system, but that is outside this experiment (audio_context.md:19-21)."
  ],
  "evidence": [
    {
      "claim": "Prompt is written after rendering and prime extraction.",
      "source": "src/main.rs:8682-8697",
      "detail": "prompt is formatted and written to suno_priming_prompt after the mastering/prime block at 8448-8542."
    },
    {
      "claim": "Prime selection uses field entropy, activity health, structured complexity, stagnation; not width or harshness.",
      "source": "src/main.rs:8141-8145 and 8463-8479",
      "detail": "chunk_scores combine field_entropy, activity_health, structured_complexity, stagnation; the prime window maximizes the sum of these scores."
    },
    {
      "claim": "Latest continuation is near mono-collapse in the prime interval.",
      "source": "analysis/audio_center_harshness_20260926/deepseek_packet.md:24-35",
      "detail": "tagged latest continuation: L/R corr 0.991, side/mid -23.31 dB; minutes 5-6 have 0.990-0.992 and side/mid about -23 dB."
    },
    {
      "claim": "Full run is not uniformly centered.",
      "source": "analysis/audio_center_harshness_20260926/deepseek_packet.md:31-35",
      "detail": "non-overlapping 60-second intervals have L/R correlation 0.740-0.992."
    },
    {
      "claim": "Width and pan are bounded; side gain cannot synthesize missing side information.",
      "source": "src/stereo.rs:3-11, 28-41; research/audio_context.md:44-47",
      "detail": "GLOBAL_PAN_LIMIT 0.10; WIDTH_CONTROL_MIN 0.05, WIDTH_CONTROL_MAX 0.50; side_gain clamps to maximum."
    },
    {
      "claim": "Audible path ends in tanh saturation, DC blocker, and whole-file peak normalization.",
      "source": "src/main.rs:8059-8076 and 8448-8455",
      "detail": "per-sample tanh(x*0.92), stateful DC blocker, then norm = 0.891 / raw_peak."
    },
    {
      "claim": "ultrasonic_ratio is a guardrail, not a harshness score.",
      "source": "METRICS.md:94-95 and deepseek_packet.md:40-41",
      "detail": "explicitly labeled pre-master aliasing/foldback guardrail."
    },
    {
      "claim": "No blinded listening has been performed.",
      "source": "research/audio_context.md:58-60",
      "detail": "listener reports harsh/centered; no controlled blinded listening yet."
    },
    {
      "claim": "Waveform filtering is not a proxy for latent-field diffusion.",
      "source": "research/audio_context.md:64-67",
      "detail": "explicit inference limit."
    },
    {
      "claim": "Target audio does not enter the current forward pass.",
      "source": "research/audio_context.md:13-18; docs/AUDIO_SCIENTIFIC_INSTRUMENTATION_PLAN.md:101-108",
      "detail": "forward has no target argument; target sampling occurs after forward."
    },
    {
      "claim": "Training trace shows near-collapse and pan at its limit.",
      "source": "analysis/audio_center_harshness_20260926/deepseek_packet.md:37-43",
      "detail": "median output correlation 0.963 vs target 0.862; median truthful width 0.027; global pan 0.0999 at +0.10 limit; decoder side control 1.0."
    }
  ],
  "uncertainties": [
    "No blinded listening has been performed, so the perceptual harshness claim is unverified.",
    "The supplied spectral fractions are mid-channel only and normalized; changes can reflect denominator or stereo changes rather than absolute band power (research/audio_context.md:54-57).",
    "The target mix and online learning confound comparisons between elapsed portions of a run (research/audio_context.md:37-39).",
    "The three latest WAVs are one sequential continuation lineage, not independent seeds or matched arms (research/audio_context.md:28-29).",
    "Per-frequency loss gradients, pre-side-scale renderer taps, and detailed target-mix attribution are missing (research/audio_context.md:71-74).",
    "It is not established whether the harshness is dominated by tanh saturation, wavefolder behavior, scan-partial brightness, or the target distribution; the supplied metrics do not separate these.",
    "It is not established whether the center bias is dominated by prime selection, learned width rail, pan rail, or target correlation; the trace shows several of these simultaneously.",
    "The prompt text is descriptive and post-hoc; any downstream priming effect is outside this experiment and unmeasured."
  ],
  "proposed_experiments": [
    {
      "name": "Prime-selection control (decisive for center-collapse attribution)",
      "design": "From the existing finalized 600-second WAV, extract multiple non-overlapping 60-second windows (including the current prime interval and a low-correlation interval). Compute L/R correlation, side/mid, and absolute L+R band power per window. Compare the current prime interval against the full-run distribution.",
      "prediction_if_artifact": "The current prime is an extreme outlier in correlation/side-mid relative to the run distribution.",
      "prediction_if_real_collapse": "All or most windows are similarly center-heavy.",
      "controls": "Use the same measurement code and window length; report absolute and total L+R band power, not only fractions; do not resynthesize."
    },
    {
      "name": "Level-matched A/B of prime vs non-prime interval",
      "design": "Create copied WAVs of the prime interval and a matched non-prime interval, level-matched to the same RMS. Blind the listener to which is which.",
      "prediction_if_artifact": "Harshness/centering ratings track the prime interval, not the run as a whole.",
      "controls": "Record level-matching gains in sidecar receipts; keep canonical files byte-for-byte intact."
    },
    {
      "name": "Post-path ablation (signal-domain only)",
      "design": "On copied WAVs, bypass or reduce the tanh saturation and DC blocker in an offline signal-domain test, level-matched to the original.",
      "prediction_if_harshness_is_post_path": "Reducing tanh drive reduces perceived harshness at matched level.",
      "prediction_if_harshness_is_upstream": "No perceptual change.",
      "controls": "This tests signal processing only; it is not a proxy for latent-field diffusion (research/audio_context.md:64-67)."
    },
    {
      "name": "Stereo-geometry diagnostic on the full run",
      "design": "Compute per-window correlation, side/mid, and balance across the full 600 seconds using the same estimators as the packet.",
      "prediction_if_artifact": "Wide spread with the prime at one extreme.",
      "prediction_if_real_collapse": "Narrow spread near correlation 1."
    },
    {
      "name": "Prompt-causality negative control",
      "design": "Confirm by source inspection and, if desired, by a no-prompt-file run that the prompt text is not read by forward or by the renderer.",
      "prediction": "Removing or altering the prompt file does not change the rendered audio for the same checkpoint and seed.",
      "controls": "Do not mutate canonical checkpoints or corpus."
    },
    {
      "name": "Target-mix attribution audit",
      "design": "From existing traces, tabulate target_file/target_frame/target_chunks_left against output correlation and band fractions for the prime interval.",
      "prediction_if_target_confound": "Prime-interval center bias co-occurs with a specific target family or episode.",
      "controls": "Read-only; do not call load_or_create_corpus_manifest (docs/AUDIO_SCIENTIFIC_INSTRUMENTATION_PLAN.md:110)."
    }
  ],
  "context_receipt": {
    "project": "titan_audio_ecosystem",
    "direct_reference_input": false,
    "prompt_is_post_render_output": true,
    "waveform_filter_tests_latent_diffusion": false,
    "canonical_mutation_allowed": false
  }
}
