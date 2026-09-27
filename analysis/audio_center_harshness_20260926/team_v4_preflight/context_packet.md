# Shared Audio evidence snapshot

## EVIDENCE research/audio_context.md:1-91
SHA256 78f30022a16dfafdcae97beedd7990a17b85efc698a2a92ca960ace6ecf7544a

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
64:     Gemini's original Audio E2/E3 diagnostics are superseded: their telemetry
65:     describes a later 9.56-second continuation, not the 599.98-second WAV.
66: 19. **Inference limits:** Offline waveform filtering tests signal processing.
67:     It is not a proxy experiment for latent-field diffusion, and neither is
68:     a trained diffusion/flow decoder. A zero-output identity branch proves
69:     bypass behavior only. Latent field entropy cannot be recovered from a WAV.
70: 20. **Preservation/budget:** Canonical checkpoints, corpus, existing WAVs, normal
71:     renderer and training stay intact. An offline test may create copied WAVs
72:     and sidecar receipts. Use paired intervals and record level-matching gains.
73: 21. **Missing evidence:** Perceptual ratings, independent training replicas,
74:     pre-side-scale renderer taps, frequency-specific loss gradients, detailed
75:     target mix attribution, and downstream priming outcomes. Flag missing
76:     information explicitly; do not invent controls, inputs, or measurements.
77: 
78: The working tree may contain an opt-in spectral-loss and prime-score candidate.
79: Check its current flags, metadata profile, tests, and actual binary before
80: interpreting a new run. The eight independent team reviews and synthesis
81: challenger in `analysis/audio_center_harshness_20260926/team_v2/` used the
82: earlier immutable packet; they are advisory and do not validate later edits.
83: 
84: Each review must include an object `context_receipt` with these exact factual
85: fields: `project: "titan_audio_ecosystem"`, `direct_reference_input: false`,
86: `prompt_is_post_render_output: true`, `waveform_filter_tests_latent_diffusion:
87: false`, `canonical_mutation_allowed: false`. Give `summary`, `evidence`
88: (citing source and line numbers), `hypotheses`, `missing_evidence`,
89: `discriminating_experiment`, and `recommendation`. Label speculation. Keep
90: different causal mechanisms separable. These receipt facts are a minimum
91: comprehension gate; the PI independently checks the substantive answer.

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

## EVIDENCE analysis/audio_center_harshness_20260926/diagnostics/CORRECTION.md:1-15
SHA256 cfc76f585af39a4834c5407e30993aa0c081be58228f5e28f8c7424f87dee2fa

1: # Correction to the 2026-09-26 Gemini audio diagnostic
2: 
3: `diagnostic_report.json` and `RESEARCH_DIAGNOSTIC_FINDINGS.md` are preserved as the original investigation, but their E2, E3, prime-origin, and causal conclusions are superseded. Use `diagnostic_report.corrected.json` for the checked measurements. Reproduce it with:
4: 
5: ```sh
6: python scripts/audio_diagnostic_experiments.py
7: ```
8: 
9: The immutable `9bcf582d4b13` WAV has 28,798,976 frames (599.9787 seconds). The available tagged trace has only 12 sampled rows, steps 81,371–81,481, and matches a later run whose metadata says 112 chunks, 458,752 frames, and 9.5573 seconds. The archived metadata file named for `9bcf582d4b13` also contains those later run fields. Therefore E2 cannot attribute target, width register, or oscillator behavior to the 600-second WAV. E3 cannot reconstruct its prime-selection score from that trace; it also omitted the `+ observation.width() * 0.40` score term in `src/main.rs`. The corrected report marks E2 and E3 unavailable.
10: 
11: An exact match of the prime's unfaded PCM16 interior places it at frames 17,031,168–19,910,656, or 354.816–414.805 seconds. The original report's 359.02–419.01-second localization was wrong. The original E1 used integer division and omitted the final 540–599.9787-second interval. That interval has left/right correlation 0.8841, side/mid power −11.95 dB, and a 2–6 kHz power fraction of 47.32%. The earlier 300–360-second fixed window is more centered than the extracted prime by these metrics; the “single most mono-centered window” claim is unsupported.
12: 
13: The 2–6 kHz fraction is a descriptive band fraction from sampled 4096-frame FFTs. It has no matched target or listening control, so it cannot establish that the renderer's learned weights caused perceived harshness. E4 measures distortion from `tanh(0.92 x)` on synthetic 1-kHz pure sines. The pre-tanh renderer waveform was not saved, so those values cannot measure distortion added to this recording or exclude the post path as a contributor.
14: 
15: The existing EQ sidecars are reproducible transforms of their listed inputs: their output SHA-256 hashes match the receipts. The −3 dB bell lowers the 2–6 kHz fraction and leaves broad stereo correlation and side/mid ratio similar in these files, while lowering RMS by about 10%. These measurements do not establish that musical detail, transients, or downstream priming quality were preserved. Listen to level-matched raw and processed clips before adopting the transform.

## EVIDENCE analysis/audio_center_harshness_20260926/multi_arm_sidecar_package/PROVENANCE_CORRECTION.md:1-22
SHA256 3cfecfa62f61dc2a502d9b2ec1d751681bed1376392d3650c651a2a24f086169

1: # Provenance correction for this package
2: 
3: The WAV files in this directory match their recorded SHA-256 output hashes.
4: The package's `receipt.json` must not be used to claim checkpoint or run
5: lineage. It associates a 28,798,976-frame, 599.9787-second source WAV
6: (`22091e4f6fceddd7b61e304bf6b9ddb0d03b98596365346f688cfd3591082185`)
7: with a copied metadata file reporting only 112 chunks, 458,752 frames,
8: 9.5573 seconds, and end step 81,483. That metadata belongs to a later short
9: continuation. A matching original metadata snapshot for the long WAV was not
10: available to this audit.
11: 
12: The seven clips remain valid *audio derivatives* of the input paths and hashes
13: listed in the receipt. They are historical, non-blind candidates. Two named
14: "wide" candidates also include EQ, so comparisons with the legacy prime
15: change both source interval and filter. The original EQ copies were attenuated
16: by about 0.9-1.0 dB RMS, which further confounds unadjusted listening.
17: 
18: `scripts/prime_package.py` now checks rendered duration and frame count before
19: associating a run metadata file with a full WAV; it rejects this mismatch.
20: Use the corrected diagnostic report and newly level-matched listening panel
21: for comparisons. This note preserves the original receipt and files rather
22: than silently rewriting their history.

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

## EVIDENCE src/main.rs:2945-3050
SHA256 1e4218f8422ef9021ff76df9818947e30e36819bc5b21ef5d224ae7d8207ec4b

2945:     let distance_to_ceiling = lower.affine(-1.0, ceiling)?;
2946:     smooth_lower_bound(&distance_to_ceiling, 0.0, 64.0)?.affine(-1.0, ceiling)
2947: }
2948: 
2949: // --- SPECTRAL PROJECTOR ---
2950: pub(crate) struct SpectralProjector {
2951:     window: Tensor, // pre-unsqueezed (1, n)
2952:     cos_m: Tensor,
2953:     sin_m: Tensor,
2954:     dft_scale: f64,
2955:     deemphasis_weights: Option<Tensor>, // present only for the opt-in loss
2956: }
2957: impl SpectralProjector {
2958:     pub(crate) fn new(device: &Device) -> CResult<Self> {
2959:         Self::new_with(CHUNK_SIZE, SPEC_BINS, device)
2960:     }
2961:     pub(crate) fn new_with(n: usize, bins: usize, device: &Device) -> CResult<Self> {
2962:         let mut win = Vec::with_capacity(n);
2963:         for i in 0..n {
2964:             win.push(0.5 - 0.5 * (TWO_PI * i as f32 / (n as f32 - 1.0)).cos());
2965:         }
2966:         // Twenty hertz closes the former 20--40 Hz supervision gap. A carrier
2967:         // could previously satisfy broadband RMS almost entirely below the
2968:         // first supervised bin, producing the observed lawnmower fundamental.
2969:         let f_lo = 20.0f32;
2970:         // The old 8 kHz ceiling made harsh energy above the loss bandwidth
2971:         // effectively free. Keep the logarithmic resolution but supervise the
2972:         // full audible band below Nyquist.
2973:         let f_hi = 20000.0f32;
2974:         let mut cos_v = vec![0.0f32; n * bins];
2975:         let mut sin_v = vec![0.0f32; n * bins];
2976:         for k in 0..bins {
2977:             let frac = k as f32 / (bins as f32 - 1.0);
2978:             let omega = TWO_PI * f_lo * (f_hi / f_lo).powf(frac) / SAMPLE_RATE as f32;
2979:             for i in 0..n {
2980:                 cos_v[i * bins + k] = (omega * i as f32).cos();
2981:                 sin_v[i * bins + k] = (omega * i as f32).sin();
2982:             }
2983:         }
2984:         Ok(Self {
2985:             window: Tensor::new(win, device)?.unsqueeze(0)?,
2986:             cos_m: Tensor::from_vec(cos_v, (n, bins), device)?,
2987:             sin_m: Tensor::from_vec(sin_v, (n, bins), device)?,
2988:             dft_scale: 2.0 / n as f64,
2989:             deemphasis_weights: None,
2990:         })
2991:     }
2992:     fn with_deemphasis(mut self) -> CResult<Self> {
2993:         let bins = self.cos_m.dims()[1];
2994:         let mut weights_v = vec![1.0f32; bins];
2995:         // Experimental 2-6 kHz spectral-error de-emphasis. Weight creation is
2996:         // deliberately outside the legacy projector constructor.
2997:         let f_lo = 20.0f32;
2998:         let f_hi = 20000.0f32;
2999:         let center_freq = 3500.0f32;
3000:         let sigma = 0.55f32;
3001:         let attenuation = 0.20f32;
3002:         for (k, weight) in weights_v.iter_mut().enumerate() {
3003:             let frac = k as f32 / (bins as f32 - 1.0);
3004:             let freq = f_lo * (f_hi / f_lo).powf(frac);
3005:             let log_ratio = (freq / center_freq).ln();
3006:             let bell = (-0.5 * (log_ratio / sigma).powi(2)).exp();
3007:             *weight = 1.0 - attenuation * bell;
3008:         }
3009:         let mean_w = weights_v.iter().sum::<f32>() / bins as f32;
3010:         for weight in &mut weights_v {
3011:             *weight /= mean_w;
3012:         }
3013:         self.deemphasis_weights =
3014:             Some(Tensor::from_vec(weights_v, (1, bins), self.cos_m.device())?);
3015:         Ok(self)
3016:     }
3017:     #[cfg(test)]
3018:     fn weights(&self) -> &Tensor {
3019:         self.deemphasis_weights.as_ref().expect("enabled in test")
3020:     }
3021:     pub(crate) fn spectral_loss(
3022:         &self,
3023:         pred: &Tensor,
3024:         target: &Tensor,
3025:         epsilon: f64,
3026:     ) -> CResult<Tensor> {
3027:         let delta = pred.sub(target)?;
3028:         match &self.deemphasis_weights {
3029:             Some(weights) => weighted_robust_distance(&delta, weights, epsilon),
3030:             None => robust_distance(&delta, epsilon),
3031:         }
3032:     }
3033:     pub(crate) fn log_mag(&self, x: &Tensor) -> CResult<Tensor> {
3034:         let xw = x.broadcast_mul(&self.window)?;
3035:         // Normalize before the log. The old absolute epsilon became
3036:         // effectively microscopic for long windows, making spectral nulls
3037:         // dominate the gradient while global clipping hid the imbalance.
3038:         let re = xw.matmul(&self.cos_m)?.affine(self.dft_scale, 0.0)?;
3039:         let im = xw.matmul(&self.sin_m)?.affine(self.dft_scale, 0.0)?;
3040:         re.sqr()?
3041:             .add(&im.sqr()?)?
3042:             .affine(1.0, 1e-5)?
3043:             .log()?
3044:             .affine(0.5, 0.0)
3045:     }
3046: }
3047: 
3048: fn log_band_energy(log_spectrum: &Tensor) -> CResult<Tensor> {
3049:     let dims = log_spectrum.dims();
3050:     let batch: usize = dims[..dims.len() - 1].iter().product();

## EVIDENCE src/main.rs:4225-4315
SHA256 0e09a393d780c7a84867ac3a338a9234e97113899eed46db847839db4f7b8b33

4225:     ]
4226: }
4227: 
4228: fn predictor_input(
4229:     hidden: &Tensor,
4230:     action: ControlAction,
4231:     control: SynthesisControl,
4232:     device: &Device,
4233: ) -> CResult<Tensor> {
4234:     let control = Tensor::from_vec(
4235:         control_features(action, control).to_vec(),
4236:         (1, ACTION_COUNT),
4237:         device,
4238:     )?;
4239:     Tensor::cat(&[hidden, &control], 1)
4240: }
4241: 
4242: fn predicted_interest(values: &[f32], ecology: &AdaptiveDynamics) -> f32 {
4243:     if values.len() < OBS_DIM {
4244:         return 0.0;
4245:     }
4246:     let obs = AudioObservation {
4247:         values: std::array::from_fn(|i| values[i].clamp(0.0, 1.0)),
4248:     };
4249:     let recovery = ecology.stagnation
4250:         * (0.42 * obs.values[7]
4251:             + 0.24 * obs.values[3]
4252:             + 0.20 * obs.values[11]
4253:             + 0.14 * obs.values[10]);
4254:     obs.structured_complexity() + recovery
4255: }
4256: 
4257: fn plan_action_scores(
4258:     head: &MonitorHead,
4259:     hidden: &Tensor,
4260:     device: &Device,
4261:     ecology: &AdaptiveDynamics,
4262:     smoothed_control: SynthesisControl,
4263:     control_slew: f32,
4264: ) -> Result<[f32; ACTION_COUNT]> {
4265:     let refs: Vec<&Tensor> = (0..ACTION_COUNT).map(|_| hidden).collect();
4266:     let hidden_batch = Tensor::cat(&refs, 0)?;
4267:     let mut control_rows = Vec::with_capacity(ACTION_COUNT * ACTION_COUNT);
4268:     for action in ControlAction::ALL {
4269:         let candidate_control =
4270:             smoothed_control.blend(SynthesisControl::for_action(action), control_slew);
4271:         control_rows.extend_from_slice(&control_features(action, candidate_control));
4272:     }
4273:     let action_batch = Tensor::from_vec(control_rows, (ACTION_COUNT, ACTION_COUNT), device)?;
4274:     let input = Tensor::cat(&[&hidden_batch, &action_batch], 1)?;
4275:     let (mean, log_var) = head.forward(&input)?;
4276:     let means = mean.to_vec2::<f32>()?;
4277:     let vars = log_var.to_vec2::<f32>()?;
4278:     let mut scores = [0.0f32; ACTION_COUNT];
4279:     for i in 0..ACTION_COUNT {
4280:         let uncertainty = vars[i].iter().map(|v| v.exp()).sum::<f32>() / OBS_DIM as f32;
4281:         scores[i] = predicted_interest(&means[i], ecology) - 0.08 * uncertainty.sqrt();
4282:     }
4283:     Ok(scores)
4284: }
4285: 
4286: struct KANLayer {
4287:     basis_fn: usize,
4288:     w: Tensor,
4289:     mod_proj: Linear,
4290:     freqs: Tensor,
4291:     tilt: Tensor,
4292: }
4293: impl KANLayer {
4294:     fn new(basis_fn: usize, vb: VBV) -> Result<Self> {
4295:         let w = vb.get_with_hints(
4296:             (basis_fn,),
4297:             "weights",
4298:             candle_nn::Init::Randn {
4299:                 mean: 0.0,
4300:                 stdev: 0.1,
4301:             },
4302:         )?;
4303:         let mod_proj = candle_nn::linear(MEMORY_DIM, basis_fn, vb.pp("mod_proj"))?;
4304:         let freq_vec: Vec<f32> = (1..=basis_fn).map(|i| i as f32).collect();
4305:         let freqs = Tensor::from_vec(freq_vec, (1, basis_fn), vb.device())?;
4306:         // A sinusoidal basis' derivative grows with harmonic index. 1/sqrt(k)
4307:         // therefore let the derivative energy grow with k; 1/k^1.25 makes the
4308:         // series and its practical finite-band slope well behaved while still
4309:         // leaving every basis trainable.
4310:         let tilt_vec: Vec<f32> = (1..=basis_fn)
4311:             .map(|i| 1.0 / (i as f32).powf(1.25))
4312:             .collect();
4313:         let tilt = Tensor::from_vec(tilt_vec, (1, basis_fn), vb.device())?;
4314:         Ok(Self {
4315:             basis_fn,

## EVIDENCE src/main.rs:5107-5255
SHA256 239959189db9d03d3a48a5e648c958039d290a5ab5358386a9de287e0be197c3

5107:                 excitation_table(0xE8C1_7A72, true),
5108:                 (EXCITATION_TABLE_LEN,),
5109:                 dev,
5110:             )?,
5111:             current_freq_l: BASE_FREQ_L,
5112:             current_freq_r: BASE_FREQ_R,
5113:             last_pan: 0.0,
5114:             prev_fm_idx_l: Tensor::new(0.0f32, dev)?,
5115:             prev_fm_idx_r: Tensor::new(0.0f32, dev)?,
5116:             prev_openness: Tensor::new(0.7f32, dev)?,
5117:             prev_gain_l: Tensor::new(0.707f32, dev)?,
5118:             prev_gain_r: Tensor::new(0.707f32, dev)?,
5119:             aux_phase_l: [0.0; 3],
5120:             aux_phase_r: [0.0; 3],
5121:             scan_phase_l: phase_vec.clone().try_into().unwrap(),
5122:             scan_phase_r: phase_vec
5123:                 .into_iter()
5124:                 .map(|p| (-p).rem_euclid(TWO_PI))
5125:                 .collect::<Vec<_>>()
5126:                 .try_into()
5127:                 .unwrap(),
5128:             scan_column_offset: 0,
5129:             excitation_offset: 0,
5130:             prev_haas_side: Tensor::zeros((1, 16), DType::F32, dev)?,
5131:         })
5132:     }
5133:     fn depth(&self) -> usize {
5134:         self.morphic.depth()
5135:     }
5136:     fn morph_capacity(&self) -> usize {
5137:         self.morphic.capacity()
5138:     }
5139:     fn manifold_depth(&self) -> usize {
5140:         manifold_depth_for_morph_depth(self.depth())
5141:     }
5142:     fn active_spatial_rings(&self) -> usize {
5143:         usize::from(far_ring_gain_for_morph_depth(self.depth()) > 0.0) + 1
5144:     }
5145:     fn far_ring_gain(&self) -> f64 {
5146:         far_ring_gain_for_morph_depth(self.depth())
5147:     }
5148:     fn project_manifold(&self, state: &Tensor) -> CResult<Tensor> {
5149:         project_manifold_state(state, self.manifold_depth())
5150:     }
5151:     fn set_depth(&mut self, d: usize) {
5152:         self.morphic.set_depth(d);
5153:     }
5154:     fn grow(&mut self) -> bool {
5155:         self.morphic.grow()
5156:     }
5157:     fn prune(&mut self) -> bool {
5158:         self.morphic.prune()
5159:     }
5160: 
5161:     fn ramp_param(&self, new_val: &Tensor, prev_val: &Tensor) -> CResult<Tensor> {
5162:         let delta = new_val.sub(prev_val)?;
5163:         self.ramp
5164:             .broadcast_mul(&delta.reshape((1,))?)?
5165:             .broadcast_add(prev_val)
5166:     }
5167: 
5168:     #[allow(clippy::too_many_arguments)]
5169:     fn forward(
5170:         &mut self,
5171:         micro: &Tensor,
5172:         macro_t: &Tensor,
5173:         mem: &Tensor,
5174:         epi_out: &Tensor,
5175:         phases: [f32; 4], // [carrier_l, carrier_r, mod_l, mod_r] — host f32
5176:         theta_prev: f32,  // theta from last step's batched readout (1-step lag)
5177:         theta_prev2: f32,
5178:         force: bool,
5179:         absolute_step: u64,
5180:         train_ecology: bool,
5181:         energy: f32,
5182:         control: &SynthesisControl,
5183:     ) -> Result<ForwardOut> {
5184:         let dev = micro.device();
5185:         let [pc_l, pc_r, pm_l, pm_r] = phases;
5186:         let manifold_depth = self.manifold_depth();
5187:         let far_ring_gain = self.far_ring_gain();
5188: 
5189:         let mut next_macro = macro_t.clone();
5190:         if force {
5191:             // Sign-symmetric local anti-rail restoring field (the global-mean
5192:             // amplitude barrier lives in the PotentialController).
5193:             let field = local_rail_bias(macro_t)?;
5194:             next_macro = self
5195:                 .macro_ca
5196:                 .forward(
5197:                     macro_t,
5198:                     None,
5199:                     Some(&field),
5200:                     self.macro_clocks.get(absolute_step),
5201:                     manifold_depth,
5202:                     far_ring_gain,
5203:                 )?
5204:                 .tanh()?
5205:                 .affine(0.95, 0.0)?;
5206:         }
5207:         let next_macro = damp_global_mean(&next_macro)?.clamp(-1.0f32, 1.0f32)?;
5208:         let macro_act = next_macro.abs()?.mean_all()?;
5209:         let metab = macro_act.affine(5.0, 0.0)?.clamp(0.01f32, 1.0f32)?;
5210:         let inv_metab = metab.affine(-1.0, 1.0)?;
5211:         let contracted_mem = self.asymptotic_contraction.forward(mem)?;
5212:         let macro_ch = next_macro.mean(D::Minus1)?.mean(D::Minus1)?; // (1, C)
5213:         let macro_mod = contracted_mem.add(&macro_ch)?;
5214:         let micro_field = local_rail_bias(micro)?;
5215:         let raw_next_micro = self.micro_ca.forward(
5216:             micro,
5217:             Some(&macro_mod),
5218:             Some(&micro_field),
5219:             self.micro_clocks.get(absolute_step),
5220:             manifold_depth,
5221:             far_ring_gain,
5222:         )?;
5223:         let next_micro = micro
5224:             .broadcast_mul(&inv_metab)?
5225:             .add(&raw_next_micro.broadcast_mul(&metab)?)?
5226:             .clamp(-1.0f32, 1.0f32)?;
5227:         let next_micro = damp_global_mean(&next_micro)?.clamp(-1.0f32, 1.0f32)?;
5228: 
5229:         // GRU input = channel features ++ episodic attention readout.
5230:         let core_micro_feats = next_micro.mean(D::Minus1)?.mean(D::Minus1)?; // (1, C)
5231:         let gru_in = Tensor::cat(&[&core_micro_feats, epi_out], 1)?; // (1, C + EPI_DIM)
5232:         let next_hidden = self.gru_memory.forward(&gru_in, mem)?;
5233:         let refined_hidden = self.morphic.forward(&next_hidden)?;
5234: 
5235:         // Most horizons adapt the audible decoder only. Detaching here keeps
5236:         // forward ecology exact while preventing the large CA graph from
5237:         // participating in backward. Periodic full horizons retain end-to-end
5238:         // credit assignment.
5239:         let (next_micro, next_macro, next_hidden, refined_hidden) = if train_ecology {
5240:             (next_micro, next_macro, next_hidden, refined_hidden)
5241:         } else {
5242:             (
5243:                 next_micro.detach(),
5244:                 next_macro.detach(),
5245:                 next_hidden.detach(),
5246:                 refined_hidden.detach(),
5247:             )
5248:         };
5249:         let movement_t = next_micro.sub(&micro.detach())?.abs()?.mean_all()?;
5250:         let micro_feats = next_micro.mean(D::Minus1)?.mean(D::Minus1)?; // (1, C)
5251:         let pop_l = micro_feats.narrow(1, 0, 1)?.reshape(())?;
5252:         let pop_r = micro_feats.narrow(1, 1, 1)?.reshape(())?;
5253:         let paired = micro_feats.reshape((CA_CHANNELS / 2, 2))?;
5254:         let pair_sums = paired.sum(0)?; // (2,)
5255:         let temporal_controls =

## EVIDENCE src/main.rs:5290-5648
SHA256 1bdac2527c281cb75bbfdb678941c260c9c3ec6c8311c6807e90c4639be430c9

5290:             .affine(std::f64::consts::LN_2 * 2.0, 0.0)?
5291:             .exp()?;
5292: 
5293:         let energy_factor = energy.clamp(0.15, 1.0);
5294:         let target_l_raw = b_l
5295:             .mul(&pitch_l)?
5296:             .add(&pop_l.affine(36.0, 0.0)?)?
5297:             .add(&movement_t.affine(18.0, 0.0)?)?;
5298:         let target_r_raw = b_r
5299:             .mul(&pitch_r)?
5300:             .add(&pop_r.affine(36.0, 0.0)?)?
5301:             .add(&movement_t.affine(-18.0, 0.0)?)?;
5302:         let target_l = smooth_bounded_frequency(&target_l_raw, 24.0, 4000.0)?;
5303:         let target_r = smooth_bounded_frequency(&target_r_raw, 24.0, 4000.0)?;
5304: 
5305:         let g = FREQ_GLIDE_SPEED as f64;
5306:         let cur_l = target_l.affine(g, self.current_freq_l as f64 * (1.0 - g))?;
5307:         let cur_r = target_r.affine(g, self.current_freq_r as f64 * (1.0 - g))?;
5308:         let mod_f_l = cur_l.mul(&ratio_l)?.clamp(0.0f32, 4000.0f32)?;
5309:         let mod_f_r = cur_r.mul(&ratio_r)?.clamp(0.0f32, 4000.0f32)?;
5310:         let omega_m_l = mod_f_l.affine(TWO_PI as f64, 0.0)?;
5311:         let omega_m_r = mod_f_r.affine(TWO_PI as f64, 0.0)?;
5312:         let ph_m_l = self
5313:             .t_steps
5314:             .broadcast_mul(&omega_m_l)?
5315:             .affine(1.0, pm_l as f64)?;
5316:         let ph_m_r = self
5317:             .t_steps
5318:             .broadcast_mul(&omega_m_r)?
5319:             .affine(1.0, pm_r as f64)?;
5320:         let idx_curve_l = self.ramp_param(&idx_l, &self.prev_fm_idx_l)?;
5321:         let idx_curve_r = self.ramp_param(&idx_r, &self.prev_fm_idx_r)?;
5322:         let modulator_l = ph_m_l.sin()?.mul(&idx_curve_l)?;
5323:         let modulator_r = ph_m_r.sin()?.mul(&idx_curve_r)?;
5324: 
5325:         let mut dtheta = theta_prev - theta_prev2;
5326:         dtheta -= TWO_PI * (dtheta / TWO_PI).round();
5327:         let theta_curve = self.ramp.affine(dtheta as f64, theta_prev2 as f64)?;
5328:         let omega_c_l = cur_l.affine(TWO_PI as f64, 0.0)?;
5329:         let omega_c_r = cur_r.affine(TWO_PI as f64, 0.0)?;
5330:         let ph_c_l = self
5331:             .t_steps
5332:             .broadcast_mul(&omega_c_l)?
5333:             .affine(1.0, pc_l as f64)?
5334:             .add(&theta_curve)?
5335:             .add(&modulator_l)?
5336:             .add(
5337:                 &temporal_controls
5338:                     .narrow(0, 2, 1)?
5339:                     .reshape((CHUNK_SIZE,))?
5340:                     .affine(0.35, 0.0)?,
5341:             )?;
5342:         let ph_c_r = self
5343:             .t_steps
5344:             .broadcast_mul(&omega_c_r)?
5345:             .affine(1.0, pc_r as f64)?
5346:             .add(&theta_curve)?
5347:             .add(&modulator_r)?
5348:             .add(
5349:                 &temporal_controls
5350:                     .narrow(0, 3, 1)?
5351:                     .reshape((CHUNK_SIZE,))?
5352:                     .affine(0.35, 0.0)?,
5353:             )?;
5354: 
5355:         let morphs = self.wave_morph_head.forward(&refined_hidden)?;
5356:         let morph_l = morphs.narrow(1, 0, 1)?.reshape(())?;
5357:         let morph_r = morphs.narrow(1, 1, 1)?.reshape(())?;
5358:         let oscillator_gains = self.oscillator_gain_head.forward(&refined_hidden)?;
5359:         let carrier_gain_l = oscillator_gains
5360:             .narrow(1, 0, 1)?
5361:             .reshape(())?
5362:             .affine(1.15, 0.05)?;
5363:         let carrier_gain_r = oscillator_gains
5364:             .narrow(1, 1, 1)?
5365:             .reshape(())?
5366:             .affine(1.15, 0.05)?;
5367: 
5368:         let mut audio_l = morph_wave(&ph_c_l, &morph_l)?.broadcast_mul(&carrier_gain_l)?;
5369:         let mut audio_r = morph_wave(&ph_c_r, &morph_r)?.broadcast_mul(&carrier_gain_r)?;
5370:         let auxiliary_pitch = self.auxiliary_pitch_head.forward(&refined_hidden)?;
5371:         let modal_bases = [2.03f64, 3.01, 5.07];
5372:         let mut aux_pairs = Vec::with_capacity(3);
5373:         for (j, base_ratio) in modal_bases.into_iter().enumerate() {
5374:             let ratio_l = auxiliary_pitch
5375:                 .narrow(1, j, 1)?
5376:                 .reshape(())?
5377:                 .affine(std::f64::consts::LN_2, base_ratio.ln())?
5378:                 .exp()?;
5379:             let ratio_r = auxiliary_pitch
5380:                 .narrow(1, j + 3, 1)?
5381:                 .reshape(())?
5382:                 .affine(std::f64::consts::LN_2, base_ratio.ln())?
5383:                 .exp()?;
5384:             aux_pairs.push((
5385:                 cur_l.mul(&ratio_l)?.clamp(24.0f32, 12000.0f32)?,
5386:                 cur_r.mul(&ratio_r)?.clamp(24.0f32, 12000.0f32)?,
5387:             ));
5388:         }
5389:         let mut aux_freqs_l = Vec::with_capacity(3);
5390:         let mut aux_freqs_r = Vec::with_capacity(3);
5391:         for (j, (f_l, f_r)) in aux_pairs.into_iter().enumerate() {
5392:             let p_l = self
5393:                 .t_steps
5394:                 .broadcast_mul(&f_l.affine(TWO_PI as f64, 0.0)?)?
5395:                 .affine(1.0, self.aux_phase_l[j] as f64)?
5396:                 .add(&theta_curve)?;
5397:             let p_r = self
5398:                 .t_steps
5399:                 .broadcast_mul(&f_r.affine(TWO_PI as f64, 0.0)?)?
5400:                 .affine(1.0, self.aux_phase_r[j] as f64)?
5401:                 .add(&theta_curve)?;
5402:             let aux_gain_l = oscillator_gains
5403:                 .narrow(1, 2 + j, 1)?
5404:                 .reshape(())?
5405:                 .affine(0.43, 0.02)?;
5406:             let aux_gain_r = oscillator_gains
5407:                 .narrow(1, 5 + j, 1)?
5408:                 .reshape(())?
5409:                 .affine(0.43, 0.02)?;
5410:             audio_l = audio_l.add(&morph_wave(&p_l, &morph_l)?.broadcast_mul(&aux_gain_l)?)?;
5411:             audio_r = audio_r.add(&morph_wave(&p_r, &morph_r)?.broadcast_mul(&aux_gain_r)?)?;
5412:             aux_freqs_l.push(f_l.reshape((1,))?);
5413:             aux_freqs_r.push(f_r.reshape((1,))?);
5414:         }
5415:         let aux_l_refs: Vec<&Tensor> = aux_freqs_l.iter().collect();
5416:         let aux_r_refs: Vec<&Tensor> = aux_freqs_r.iter().collect();
5417:         let aux_freqs_l = Tensor::cat(&aux_l_refs, 0)?;
5418:         let aux_freqs_r = Tensor::cat(&aux_r_refs, 0)?;
5419: 
5420:         // --- REGIONAL SPECTRAL FIELD ---
5421:         // The previous row/column projection discarded most 2-D topology.  A
5422:         // 4x8 regional readout now drives thirty-two independently panned partial
5423:         // agents.  The ratios continuously interpolate between harmonic and
5424:         // inharmonic modal sets; local temporal change adds micro-detuning.
5425:         let field_cm = next_micro.mean(1)?; // (1, H, W)
5426:         let region_grid = field_cm
5427:             .reshape((REGION_ROWS, REGION_H, REGION_COLS, REGION_W))?
5428:             .mean(D::Minus1)?
5429:             .mean(1)?; // (4,8)
5430:         let region_activity = region_grid.reshape((REGION_COUNT,))?;
5431:         let delta_cm = next_micro.sub(micro)?.mean(1)?;
5432:         let region_change = delta_cm
5433:             .reshape((REGION_ROWS, REGION_H, REGION_COLS, REGION_W))?
5434:             .abs()?
5435:             .mean(D::Minus1)?
5436:             .mean(1)?
5437:             .reshape((REGION_COUNT,))?;
5438: 
5439:         let amps = region_activity
5440:             .affine(0.5, 0.5)?
5441:             .add(&region_change.affine(0.55, 0.0)?)?
5442:             .relu()?
5443:             .reshape((SCAN_PARTIALS, 1))?;
5444:         let tilt = self
5445:             .scan_brightness
5446:             .affine(control.spectral_tilt as f64, 1.0)?
5447:             .clamp(0.18f32, 1.82f32)?;
5448:         let learned_amps = self
5449:             .partial_amplitude_head
5450:             .forward(&refined_hidden)?
5451:             .reshape((SCAN_PARTIALS, 1))?
5452:             .affine(1.6, 0.20)?;
5453:         let damping = self
5454:             .partial_damping_head
5455:             .forward(&refined_hidden)?
5456:             .reshape((SCAN_PARTIALS, 1))?;
5457:         let amps = amps.broadcast_mul(&tilt)?.broadcast_mul(&learned_amps)?;
5458:         let amp_sum = amps.sum_all()?.affine(1.0, 1e-4)?;
5459:         let amps_n = amps.broadcast_div(&amp_sum)?;
5460: 
5461:         let inh = control.inharmonicity.clamp(0.0, 1.0);
5462:         let learned_ratio = self
5463:             .partial_ratio_head
5464:             .forward(&refined_hidden)?
5465:             .reshape((SCAN_PARTIALS, 1))?
5466:             .affine(0.35, 0.0)?
5467:             .exp()?;
5468:         let ratios = self
5469:             .scan_harmonic
5470:             .affine((1.0 - inh) as f64, 0.0)?
5471:             .add(&self.scan_inharmonic.affine(inh as f64, 0.0)?)?
5472:             .add(
5473:                 &region_change
5474:                     .reshape((REGION_COUNT, 1))?
5475:                     .broadcast_mul(&self.scan_detune)?
5476:                     .affine((0.7 + 0.8 * inh) as f64, 0.0)?,
5477:             )?
5478:             .broadcast_mul(&learned_ratio)?;
5479:         // Damping is modal attenuation rather than an external filter: every
5480:         // audible partial remains an explicit solution of the learned
5481:         // oscillator bank. Higher modes may decay more strongly, closing the
5482:         // bright-comb shortcut while preserving phase continuity.
5483:         let modal_decay = damping.broadcast_mul(&ratios)?.affine(-0.10, 0.0)?.exp()?;
5484:         let amps_n = amps_n.broadcast_mul(&modal_decay)?;
5485: 
5486:         // Keep one global column envelope as a slow, coherent breath while the
5487:         // regional agents retain spatially independent spectra.
5488:         let cols = field_cm.mean(1)?.reshape((GRID_W, 1))?;
5489:         let offset = self.scan_column_offset % GRID_W;
5490:         let cols = if offset == 0 {
5491:             cols
5492:         } else {
5493:             Tensor::cat(
5494:                 &[
5495:                     &cols.narrow(0, offset, GRID_W - offset)?,
5496:                     &cols.narrow(0, 0, offset)?,
5497:                 ],
5498:                 0,
5499:             )?
5500:         };
5501:         let env = self
5502:             .scan_interp
5503:             .matmul(&cols)?
5504:             .affine(0.30, 0.70)?
5505:             .clamp(0.18f32, 1.25f32)?
5506:             .reshape((1, CHUNK_SIZE))?;
5507:         let scan_freqs_l = ratios.broadcast_mul(&cur_l)?.clamp(24.0f32, 18000.0f32)?;
5508:         let scan_freqs_r = ratios.broadcast_mul(&cur_r)?.clamp(24.0f32, 18000.0f32)?;
5509:         let scan_phase_l = Tensor::from_vec(self.scan_phase_l.to_vec(), (SCAN_PARTIALS, 1), dev)?;
5510:         let scan_phase_r = Tensor::from_vec(self.scan_phase_r.to_vec(), (SCAN_PARTIALS, 1), dev)?;
5511:         let ph_l = scan_freqs_l
5512:             .broadcast_mul(&self.t_steps)?
5513:             .affine(TWO_PI as f64, 0.0)?
5514:             .broadcast_add(&scan_phase_l)?;
5515:         let ph_r = scan_freqs_r
5516:             .broadcast_mul(&self.t_steps)?
5517:             .affine(TWO_PI as f64, 0.0)?
5518:             .broadcast_add(&scan_phase_r)?;
5519:         let partial_mod_l = temporal_controls
5520:             .narrow(0, DECODER_GLOBAL_CONTROLS, SCAN_PARTIALS)?
5521:             .affine(0.65, 1.0)?;
5522:         let partial_mod_r = temporal_controls
5523:             .narrow(0, DECODER_GLOBAL_CONTROLS + SCAN_PARTIALS, SCAN_PARTIALS)?
5524:             .affine(0.65, 1.0)?;
5525:         let partials_l = ph_l
5526:             .sin()?
5527:             .broadcast_mul(&amps_n)?
5528:             .broadcast_mul(&self.scan_pan_l)?
5529:             .mul(&partial_mod_l)?;
5530:         let partials_r = ph_r
5531:             .sin()?
5532:             .broadcast_mul(&amps_n)?
5533:             .broadcast_mul(&self.scan_pan_r)?
5534:             .mul(&partial_mod_r)?;
5535:         let scan_l = partials_l.sum(0)?.reshape((1, CHUNK_SIZE))?.mul(&env)?;
5536:         let scan_r = partials_r.sum(0)?.reshape((1, CHUNK_SIZE))?.mul(&env)?;
5537:         let scan_gain_l = oscillator_gains
5538:             .narrow(1, 8, 1)?
5539:             .reshape(())?
5540:             .affine(0.75, 0.05)?;
5541:         let scan_gain_r = oscillator_gains
5542:             .narrow(1, 9, 1)?
5543:             .reshape(())?
5544:             .affine(0.75, 0.05)?;
5545:         audio_l = audio_l.add(&scan_l.broadcast_mul(&scan_gain_l)?.reshape((CHUNK_SIZE,))?)?;
5546:         audio_r = audio_r.add(&scan_r.broadcast_mul(&scan_gain_r)?.reshape((CHUNK_SIZE,))?)?;
5547: 
5548:         // Learned broad-band and transient excitation is inside the
5549:         // differentiable renderer. The deterministic tables carry no target
5550:         // information; the temporal decoder must learn when and how strongly
5551:         // each mid/side component is audible.
5552:         let excitation_mid = ring_window(&self.excitation_mid, self.excitation_offset, CHUNK_SIZE)?;
5553:         let excitation_side =
5554:             ring_window(&self.excitation_side, self.excitation_offset, CHUNK_SIZE)?;
5555:         let noise_mid = excitation_mid.mul(
5556:             &temporal_controls
5557:                 .narrow(0, 6, 1)?
5558:                 .reshape((CHUNK_SIZE,))?
5559:                 .affine(0.055, 0.0)?,
5560:         )?;
5561:         let noise_side = excitation_side.mul(
5562:             &temporal_controls
5563:                 .narrow(0, 7, 1)?
5564:                 .reshape((CHUNK_SIZE,))?
5565:                 .affine(0.045, 0.0)?,
5566:         )?;
5567:         audio_l = audio_l.add(&noise_mid)?.add(&noise_side)?;
5568:         audio_r = audio_r.add(&noise_mid)?.sub(&noise_side)?;
5569:         audio_l = audio_l.mul(
5570:             &temporal_controls
5571:                 .narrow(0, 0, 1)?
5572:                 .reshape((CHUNK_SIZE,))?
5573:                 .affine(0.35, 1.0)?,
5574:         )?;
5575:         audio_r = audio_r.mul(
5576:             &temporal_controls
5577:                 .narrow(0, 1, 1)?
5578:                 .reshape((CHUNK_SIZE,))?
5579:                 .affine(0.35, 1.0)?,
5580:         )?;
5581:         let fold_drive_l = temporal_controls
5582:             .narrow(0, 10, 1)?
5583:             .reshape((1, CHUNK_SIZE))?
5584:             .affine(0.30, 1.0)?;
5585:         let fold_drive_r = temporal_controls
5586:             .narrow(0, 11, 1)?
5587:             .reshape((1, CHUNK_SIZE))?
5588:             .affine(0.30, 1.0)?;
5589: 
5590:         let audio_l = self
5591:             .wavefolder_l
5592:             .forward(&audio_l.unsqueeze(0)?.mul(&fold_drive_l)?, &refined_hidden)?
5593:             .reshape((1, CHUNK_SIZE))?;
5594:         let audio_r = self
5595:             .wavefolder_r
5596:             .forward(&audio_r.unsqueeze(0)?.mul(&fold_drive_r)?, &refined_hidden)?
5597:             .reshape((1, CHUNK_SIZE))?;
5598: 
5599:         let open_t = refined_hidden
5600:             .abs()?
5601:             .mean_all()?
5602:             .affine(5.0, 0.0)?
5603:             .add(&movement_t)?
5604:             .clamp(0.4f32, 1.0f32)?
5605:             .affine(energy_factor as f64, 0.0)?;
5606:         let open_curve = self.ramp_param(&open_t, &self.prev_openness)?;
5607:         let open_l = open_curve
5608:             .add(
5609:                 &temporal_controls
5610:                     .narrow(0, 8, 1)?
5611:                     .reshape((CHUNK_SIZE,))?
5612:                     .affine(0.25, 0.0)?,
5613:             )?
5614:             .clamp(0.10f32, 1.25f32)?;
5615:         let open_r = open_curve
5616:             .add(
5617:                 &temporal_controls
5618:                     .narrow(0, 9, 1)?
5619:                     .reshape((CHUNK_SIZE,))?
5620:                     .affine(0.25, 0.0)?,
5621:             )?
5622:             .clamp(0.10f32, 1.25f32)?;
5623:         let audio_l = audio_l.broadcast_mul(&open_l.unsqueeze(0)?)?;
5624:         let audio_r = audio_r.broadcast_mul(&open_r.unsqueeze(0)?)?;
5625: 
5626:         let mid = audio_l
5627:             .add(&audio_r)?
5628:             .affine(0.5, 0.0)?
5629:             .mul(&temporal_controls.narrow(0, 4, 1)?.affine(0.30, 1.0)?)?;
5630:         let side = audio_l
5631:             .sub(&audio_r)?
5632:             .affine(0.5, 0.0)?
5633:             .mul(&temporal_controls.narrow(0, 5, 1)?.affine(0.55, 1.0)?)?;
5634:         // These two scalar heads sit after a deep residual stack. Normalize
5635:         // their shared input without adding parameters so a single Adam step
5636:         // cannot turn a large hidden-state norm into a rail-to-rail jump.
5637:         let renderer_control_hidden = refined_hidden.broadcast_div(
5638:             &refined_hidden
5639:                 .sqr()?
5640:                 .mean_keepdim(D::Minus1)?
5641:                 .affine(1.0, 1e-5)?
5642:                 .sqrt()?,
5643:         )?;
5644:         let pan_raw = self
5645:             .spatial_panner
5646:             .forward(&renderer_control_hidden)?
5647:             .reshape(())?;
5648:         let pan_t = soft_global_pan(&pan_raw)?;

## EVIDENCE src/main.rs:7020-7270
SHA256 92881d7f7e18a6f244796900b64f3a86ad904d2cbca28e1a90133275f5cbac45

7020:         max_depth: if freeze_morph {
7021:             model.depth()
7022:         } else {
7023:             max_morph_depth
7024:         },
7025:         frozen: freeze_morph,
7026:     };
7027:     println!(
7028:         "--> Morph policy: {} at/through L{:02}; growth requires a strict development-probe plateau.",
7029:         if morph_policy.frozen { "frozen" } else { "capped" },
7030:         morph_policy.max_depth
7031:     );
7032: 
7033:     let reset_mode = if fresh_model {
7034:         "fresh_model"
7035:     } else if fresh_decoder {
7036:         "fresh_decoder"
7037:     } else if fresh_world {
7038:         "fresh_world"
7039:     } else if loaded_world {
7040:         "resume"
7041:     } else {
7042:         "new_world"
7043:     };
7044:     let run_id = format!("{}-p{}-s{}", run_started_unix_ms, std::process::id(), seed);
7045:     // Keep filename uniqueness outside RuntimeRng so a random filename never
7046:     // perturbs deterministic model/world evolution for a fixed training seed.
7047:     let audio_file_hash = unique_audio_hash(&base_dir)?;
7048:     let start_global_step = global_step;
7049:     let start_depth = model.depth();
7050:     let start_rad_amp = rad_amp;
7051:     println!(
7052:         "--> Telemetry run {} · schema v{} · build {}{} · mode {}",
7053:         run_id,
7054:         TRACE_SCHEMA_VERSION,
7055:         BUILD_COMMIT,
7056:         if BUILD_DIRTY == "true" { "-dirty" } else { "" },
7057:         reset_mode
7058:     );
7059: 
7060:     let total_chunks = ((SAMPLE_RATE as f32 * sim_duration / CHUNK_SIZE as f32) as usize).max(1);
7061:     // Mobile-safe output path: stream DC-blocked f32 samples to a temporary
7062:     // file, then perform one normalization/transcode pass.  A 16-minute run
7063:     // no longer retains ~350 MB of stereo f32 audio in RAM.
7064:     let raw_audio_path = artifact_path(&base_dir, ".titan_audio_f32", "tmp", run_tag.as_deref());
7065:     let topology_path = artifact_path(&base_dir, "ca_topology_rust", "csv", run_tag.as_deref());
7066:     let topology_tmp_path = format!("{}.tmp", topology_path);
7067:     let topology_index_path = artifact_path(
7068:         &base_dir,
7069:         "ca_topology_index_rust",
7070:         "csv",
7071:         run_tag.as_deref(),
7072:     );
7073:     let morph_events_path =
7074:         artifact_path(&base_dir, "morph_events_rust", "csv", run_tag.as_deref());
7075:     let uncertainty_trace_path = artifact_path(
7076:         &base_dir,
7077:         "uncertainty_trace_rust",
7078:         "csv",
7079:         run_tag.as_deref(),
7080:     );
7081:     let trace_spool_path = artifact_path(
7082:         &base_dir,
7083:         ".titan_uncertainty_spool",
7084:         "jsonl.tmp",
7085:         run_tag.as_deref(),
7086:     );
7087:     let topology_index_spool_path = artifact_path(
7088:         &base_dir,
7089:         ".titan_topology_index_spool",
7090:         "jsonl.tmp",
7091:         run_tag.as_deref(),
7092:     );
7093:     let raw_audio_file = File::create(&raw_audio_path)?;
7094:     let mut raw_audio_writer = BufWriter::with_capacity(1 << 20, raw_audio_file);
7095:     let mut topology_writer = csv::WriterBuilder::new()
7096:         .has_headers(false)
7097:         .from_path(&topology_tmp_path)?;
7098:     let mut trace_spool = BufWriter::new(File::create(&trace_spool_path)?);
7099:     let mut topology_index_spool = BufWriter::new(File::create(&topology_index_spool_path)?);
7100:     let mut raw_chunk_bytes = vec![0u8; CHUNK_SIZE * 2 * std::mem::size_of::<f32>()];
7101:     let (mut dc_x1_l, mut dc_y1_l, mut dc_x1_r, mut dc_y1_r) = (
7102:         host_runtime.dc_x1_l,
7103:         host_runtime.dc_y1_l,
7104:         host_runtime.dc_x1_r,
7105:         host_runtime.dc_y1_r,
7106:     );
7107:     let dc_pole = 0.998f32;
7108:     let mut raw_peak = 1e-6f32;
7109:     let mut chunk_scores: Vec<f32> = Vec::with_capacity(total_chunks);
7110:     let mut trace_rows = 0usize;
7111:     let mut topology_rows = 0usize;
7112:     let mut trace_phi_sum = 0.0f64;
7113:     let mut trace_aperture_sum = 0.0f64;
7114:     let mut trace_synergy_sum = 0.0f64;
7115:     let mut trace_temp_sum = 0.0f64;
7116:     let mut trace_sigma_sum = 0.0f64;
7117:     let mut trace_pi_sum = 0.0f64;
7118:     let mut morph_events = Vec::new();
7119:     let mut pending_morph_event: Option<(u64, &'static str)> = None;
7120:     let mut development_plateau_tracker = DevelopmentPlateauTracker::new();
7121:     let mut development_plateau = DevelopmentPlateauStatus::default();
7122: 
7123:     let mut phi = uncertainty.phi;
7124:     let mut tape_loss: Option<Tensor> = None;
7125:     let mut audio_sequence: Vec<Tensor> = Vec::with_capacity(tape_chunks);
7126:     let mut target_sequence: Vec<Tensor> = Vec::with_capacity(tape_chunks);
7127:     let mut prev_audio_tail: Option<Tensor> = None;
7128:     let mut prev_target_tail: Option<Tensor> = None;
7129:     let mut feature_history: VecDeque<DetachedFeatureFrame> =
7130:         VecDeque::with_capacity(FEATURE_HISTORY_CHUNKS);
7131:     let mut steps_in_tape = 0usize;
7132:     let mut train_ecology_tape = true;
7133:     let mut steps_since_update = 0usize;
7134:     let mut accumulated_steps = 0usize;
7135:     let mut accumulated_grads: Option<GradStore> = None;
7136:     let mut tape_lr_gain_sum = 0.0f64;
7137:     let mut accumulated_lr_gain_sum = 0.0f64;
7138:     let mut latest_lr_gain: f64;
7139:     let mut latest_grad_norm = 0.0f32;
7140:     let mut latest_clip_scale = 1.0f32;
7141:     let mut optimizer_update_count = 0usize;
7142:     // Persisted host-side adaptive state is unpacked into local scalars for
7143:     // the hot loop, then packed again only when checkpointing.
7144:     let mut morph_history = std::mem::take(&mut host_runtime.morph_history);
7145:     let mut morph_baseline = host_runtime.morph_baseline;
7146:     let mut warmup_sum = host_runtime.warmup_sum;
7147:     let mut field_entropy_sum = host_runtime.field_entropy_sum;
7148:     let mut field_entropy_n = host_runtime.field_entropy_n;
7149:     let mut total_complexity = host_runtime.total_complexity;
7150:     let mut boost_state = host_runtime.boost_state;
7151:     let mut prev_loss_vec = host_runtime.prev_loss_vec;
7152:     let mut prev_archetype = std::mem::take(&mut host_runtime.prev_archetype);
7153:     let mut stagnation_ticks = host_runtime.stagnation_ticks;
7154:     let mut last_temp = if loaded_world {
7155:         host_runtime.last_temp
7156:     } else {
7157:         potential.temp
7158:     };
7159:     let mut smoothed_control = host_runtime.smoothed_control;
7160:     let mut sigma = criticality.sigma;
7161:     let chunk_dt = CHUNK_SIZE as f32 / SAMPLE_RATE as f32;
7162: 
7163:     let timer_start = std::time::Instant::now();
7164:     let mut profiling_lap = std::time::Instant::now();
7165:     let mut phase_profiler = PhaseProfiler::default();
7166:     let mut completed_chunks = 0usize; // chunks committed to the raw audio stream
7167:     let mut evolved_chunks = 0usize; // includes a NaN-triggered ecological reset
7168: 
7169:     for step in 0..total_chunks {
7170:         if !keep_running.load(AtomicOrdering::SeqCst) {
7171:             println!(
7172:                 "\n--> Stop requested: finalizing {} completed chunks and saving the organism.",
7173:                 completed_chunks
7174:             );
7175:             break;
7176:         }
7177:         let absolute_step = global_step + step as u64;
7178:         if steps_in_tape == 0 {
7179:             let tape_index = absolute_step / tape_chunks.max(1) as u64;
7180:             train_ecology_tape = tape_index.is_multiple_of(core_update_every as u64);
7181:         }
7182:         let aperture = uncertainty.branch_aperture();
7183:         let escape_strength = adaptive_dynamics.escape_strength();
7184:         let curiosity_factor = ecological_curiosity(&adaptive_dynamics, stagnation_ticks);
7185:         let control_slew =
7186:             (0.10 + 0.18 * controller.meta.surprise() + 0.14 * escape_strength).clamp(0.08, 0.38);
7187: 
7188:         // Causally correct self-model: the state/action pair retained from the
7189:         // previous chunk predicts the post-DSP observation that is now known.
7190:         let (self_model_loss, predicted_mean_t, predicted_log_var_t) = if let (
7191:             Some(input),
7192:             Some(actual_obs),
7193:         ) =
7194:             (&pending_predictor_input, &last_observation)
7195:         {
7196:             let (mean, log_var) = monitor_head.forward(input)?;
7197:             let actual = Tensor::from_vec(actual_obs.values.to_vec(), (1, OBS_DIM), &device)?;
7198:             let nll = mean
7199:                 .sub(&actual)?
7200:                 .sqr()?
7201:                 .mul(&log_var.neg()?.exp()?)?
7202:                 .add(&log_var)?
7203:                 .mean_all()?
7204:                 .affine(0.5, 0.0)?;
7205:             (nll, mean, log_var)
7206:         } else {
7207:             (
7208:                 Tensor::new(0.0f32, &device)?,
7209:                 Tensor::zeros((1, OBS_DIM), DType::F32, &device)?,
7210:                 Tensor::zeros((1, OBS_DIM), DType::F32, &device)?,
7211:             )
7212:         };
7213: 
7214:         // Compact model-based planning is amortized: one batched prediction
7215:         // for all ten actions every PLAN_EVERY chunks. Between plans, the
7216:         // model-free values continue adapting each chunk.
7217:         if absolute_step.is_multiple_of(PLAN_EVERY as u64) {
7218:             match plan_action_scores(
7219:                 &monitor_head,
7220:                 &hidden_mem.detach(),
7221:                 &device,
7222:                 &adaptive_dynamics,
7223:                 smoothed_control,
7224:                 control_slew,
7225:             ) {
7226:                 Ok(scores) => controller.cached_model_scores = scores,
7227:                 Err(e) => println!("! planner skipped at step {}: {}", absolute_step, e),
7228:             }
7229:         }
7230:         let motif_available = motifs.has_recallable(absolute_step);
7231:         let current_action =
7232:             controller.choose(last_temp, &mut adaptive_dynamics, motif_available, &mut rng);
7233:         let mut target_control = SynthesisControl::for_action(current_action);
7234:         if current_action == ControlAction::Recall {
7235:             if let Some((remembered, strength)) = motifs.recall(
7236:                 last_observation.as_ref(),
7237:                 absolute_step,
7238:                 &mut motif_diagnostics,
7239:             ) {
7240:                 target_control =
7241:                     target_control.blend(remembered, (0.35 + 0.55 * strength).clamp(0.0, 0.9));
7242:             }
7243:         }
7244:         // A prolonged low-information attractor adds a bounded rescue bias
7245:         // even when the learned controller still proposes an ordering action.
7246:         // This is a residual intervention, not a hard world reset.
7247:         if escape_strength > 0.0 {
7248:             let rescue = SynthesisControl::for_action(ControlAction::Turbulence);
7249:             target_control =
7250:                 target_control.blend(rescue, (0.18 + 0.62 * escape_strength).clamp(0.0, 0.82));
7251:         }
7252:         // Action selection is discrete, but the acoustic intervention is a
7253:         // persistent continuous state.  This avoids chunk-boundary spectral
7254:         // jumps while still allowing rapid changes during self-surprise.
7255:         let current_control = smoothed_control.blend(target_control, control_slew);
7256:         smoothed_control = current_control;
7257:         let current_predictor_input = predictor_input(
7258:             &hidden_mem.detach(),
7259:             current_action,
7260:             current_control,
7261:             &device,
7262:         )?
7263:         .detach();
7264:         let force_probability = (0.18
7265:             + aperture * 0.48
7266:             + 0.12 * (current_control.shear_mult - 1.0).max(0.0)
7267:             + 0.10 * controller.meta.surprise()
7268:             + 0.32 * escape_strength)
7269:             .clamp(0.05, 0.98);
7270:         let force_macro = absolute_step.is_multiple_of(MACRO_UPDATE_EVERY)

## EVIDENCE src/main.rs:7780-7995
SHA256 853bcf18c18e224b452282234f759e8c93a8d8af7a040a176967af6d14146223

7780:         let stereo_correlation_loss_val = metrics[tail_start + 15];
7781:         let stereo_level_loss_val = metrics[tail_start + 16];
7782:         let output_low_band_ratio_val = metrics[tail_start + 17];
7783:         let output_side_ratio_val = metrics[tail_start + 18];
7784:         let target_low_band_ratio_val = metrics[tail_start + 19];
7785:         let target_side_ratio_val = metrics[tail_start + 20];
7786:         let output_stereo_corr_val = metrics[tail_start + 21];
7787:         let target_stereo_corr_val = metrics[tail_start + 22];
7788:         let output_stereo_level_ratio_val = metrics[tail_start + 23];
7789:         let target_stereo_level_ratio_val = metrics[tail_start + 24];
7790:         let development_best_val = metrics[tail_start + 25];
7791:         let development_mean_val = metrics[tail_start + 26];
7792:         let development_chroma_val = metrics[tail_start + 27];
7793:         let validation_best_val = metrics[tail_start + 28];
7794:         let validation_mean_val = metrics[tail_start + 29];
7795:         let validation_chroma_val = metrics[tail_start + 30];
7796:         let pan_center_loss_val = metrics[tail_start + 31];
7797:         let decoder_side_control_val = metrics[tail_start + 32];
7798:         let decoder_width_control_val = metrics[tail_start + 33];
7799:         let decoder_width_raw_val = metrics[tail_start + 34];
7800:         development_plateau =
7801:             development_plateau_tracker.update(development_mean_val, development_chroma_val);
7802:         if let (Some(actual), Some(_)) = (&last_observation, &pending_predictor_input) {
7803:             controller
7804:                 .meta
7805:                 .update(actual, predicted_mean_host, predicted_log_host);
7806:         }
7807: 
7808:         // NaN bio-reset rides the same readback — no dedicated check sync.
7809:         if !movement.is_finite() || !micro_abs.is_finite() {
7810:             println!("! BIO-RESET: Tape corruption detected (NaN). Re-seeding primordial soup.");
7811:             micro_tape = model.project_manifold(&randn_t(
7812:                 &mut rng,
7813:                 &[1, CA_CHANNELS, GRID_H, GRID_W],
7814:                 1.0,
7815:                 &device,
7816:             )?)?;
7817:             macro_tape = model.project_manifold(&randn_t(
7818:                 &mut rng,
7819:                 &[1, CA_CHANNELS, MACRO_H, MACRO_W],
7820:                 1.0,
7821:                 &device,
7822:             )?)?;
7823:             hidden_mem = Tensor::zeros((1, MEMORY_DIM), DType::F32, &device)?;
7824:             tape_loss = None;
7825:             steps_in_tape = 0;
7826:             steps_since_update = 0;
7827:             accumulated_steps = 0;
7828:             accumulated_grads = None;
7829:             tape_lr_gain_sum = 0.0;
7830:             accumulated_lr_gain_sum = 0.0;
7831:             audio_sequence.clear();
7832:             target_sequence.clear();
7833:             prev_audio_tail = None;
7834:             prev_target_tail = None;
7835:             feature_history.clear();
7836:             adaptive_dynamics.low_motion_run = 0;
7837:             adaptive_dynamics.stagnation = 0.0;
7838:             adaptive_dynamics.escape_cooldown = 0;
7839:             controller.action_age = 0;
7840:             evolved_chunks = step + 1;
7841:             continue;
7842:         }
7843:         let mimic_drift_n = mimic_drift / (1.0 + mimic_drift);
7844:         total_complexity += movement;
7845: 
7846:         // Host mirrors for phase advance + metabolism.
7847:         model.current_freq_l = f_l;
7848:         model.current_freq_r = f_r;
7849:         phases[0] = (phases[0] + TWO_PI * f_l * chunk_dt).rem_euclid(TWO_PI);
7850:         phases[1] = (phases[1] + TWO_PI * f_r * chunk_dt).rem_euclid(TWO_PI);
7851:         phases[2] = (phases[2] + TWO_PI * mf_l * chunk_dt).rem_euclid(TWO_PI);
7852:         phases[3] = (phases[3] + TWO_PI * mf_r * chunk_dt).rem_euclid(TWO_PI);
7853: 
7854:         // Mean-centered multi-lag criticality estimate.  Unlike the old
7855:         // through-origin lag-1 regression, this does not mistake a non-zero
7856:         // movement baseline for branching persistence.
7857:         sigma = criticality.update(movement);
7858: 
7859:         let boost_target = if abs_max < 0.25 {
7860:             (0.25 / (abs_max + 1e-6)).clamp(1.0, 4.0)
7861:         } else {
7862:             1.0
7863:         };
7864:         boost_state = boost_state * 0.9 + boost_target * 0.1;
7865:         let audio_normalized = audio_for_loss.affine(boost_state as f64, 0.0)?;
7866: 
7867:         let m_sig = movement_mon.analyze(movement)?;
7868: 
7869:         // Archetype / stagnation from the channel summary (already in the readback).
7870:         let field01: Vec<f32> = field_summary.iter().map(|&x| (x + 1.0) * 0.5).collect();
7871:         let (arch_summary, field_entropy, dom_arch) = SemanticField::archetype_field(&field01);
7872:         semantic.record(mimic_drift_n, dom_arch);
7873:         let trend = semantic.trend();
7874:         field_entropy_sum += field_entropy as f64;
7875:         field_entropy_n += 1;
7876:         let current_arch = semantic.dominant_archetype();
7877:         if prev_archetype == current_arch {
7878:             stagnation_ticks += 1;
7879:         } else {
7880:             stagnation_ticks = 0;
7881:             prev_archetype = current_arch.to_string();
7882:         }
7883: 
7884:         // Metabolism now has a genuine interior fixed point. The former flat
7885:         // recharge term overwhelmed cost and pinned energy at 0.99.
7886:         let metabolic_cost = 0.0020
7887:             + rms_val.clamp(0.0, 1.0) * 0.0060
7888:             + movement.clamp(0.0, 0.20) * 0.045
7889:             + (current_control.kick_mult - 1.0).max(0.0) * 0.0015;
7890:         energy_state = (energy_state - metabolic_cost).max(0.18);
7891:         let recharge_capacity = (1.0 - energy_state).max(0.0);
7892:         let energy_recharge = recharge_capacity * (1.0 - rms_val).clamp(0.0, 1.0) * 0.012;
7893:         energy_state += energy_recharge;
7894:         energy_state += ENERGY_HOMEO_RATE * (POT_ENERGY_SET - energy_state);
7895:         energy_state = energy_state.clamp(0.18, 0.96);
7896: 
7897:         // Morphic growth/pruning. Capacity responds both to relative mimic
7898:         // difficulty and to sustained ecological collapse; structural events
7899:         // are cooldown-gated so a hot episode cannot cascade through layers.
7900:         let mut morph_event: Option<&'static str> = None;
7901:         let manifold_depth_before = model.manifold_depth();
7902:         let spatial_rings_before = model.active_spatial_rings();
7903:         if step < MORPH_WARMUP {
7904:             warmup_sum += mimic_drift_n;
7905:         } else {
7906:             if morph_baseline.is_none() {
7907:                 let b = (warmup_sum / MORPH_WARMUP as f32).max(1e-4);
7908:                 morph_baseline = Some(b);
7909:                 println!(
7910:                     "--> Morph baseline calibrated: mimic≈{:.3}  (grow>{:.3}, prune<{:.3})",
7911:                     b,
7912:                     b * MORPH_GROWTH_REL,
7913:                     b * MORPH_PRUNE_REL
7914:                 );
7915:             }
7916:             morph_history.push(mimic_drift_n);
7917:             let patience = MORPH_PATIENCE_BASE + model.depth() * 2;
7918:             if morph_history.len() >= patience {
7919:                 let avg = morph_history.iter().sum::<f32>() / morph_history.len() as f32;
7920:                 let window_len = morph_history.len();
7921:                 morph_history.clear();
7922:                 let base = morph_baseline.unwrap();
7923:                 match morph_decision(
7924:                     MorphEvidence {
7925:                         mimic_avg: avg,
7926:                         mimic_baseline: base,
7927:                         field_entropy_norm: (field_entropy / 3.0).clamp(0.0, 1.0),
7928:                         predictive_structure: last_observation
7929:                             .as_ref()
7930:                             .map(|obs| obs.values[10])
7931:                             .unwrap_or(0.0),
7932:                         development_plateau: development_is_strict && development_plateau.plateau,
7933:                     },
7934:                     &adaptive_dynamics,
7935:                     model.depth(),
7936:                     absolute_step,
7937:                     window_len,
7938:                     morph_policy,
7939:                 ) {
7940:                     MorphDecision::GrowPressure if model.grow() => {
7941:                         rad_amp = (rad_amp * RAD_COOL).max(RAD_AMP_MIN);
7942:                         morph_event = Some("NEUROGENESIS · capacity pressure");
7943:                     }
7944:                     MorphDecision::GrowDevelopment if model.grow() => {
7945:                         rad_amp = (rad_amp * RAD_COOL).max(RAD_AMP_MIN);
7946:                         morph_event = Some("NEUROGENESIS · structural development");
7947:                     }
7948:                     MorphDecision::Prune if model.prune() => {
7949:                         rad_amp = (rad_amp * RAD_HEAT).min(RAD_AMP_MAX);
7950:                         morph_event = Some("PRUNING");
7951:                     }
7952:                     _ => {}
7953:                 }
7954:                 // Track the organism's current operating scale instead of
7955:                 // comparing forever against its first 48 chunks. A slow EMA
7956:                 // retains a meaningful relative-pressure signal across runs.
7957:                 morph_baseline = Some(base + 0.04 * (avg - base));
7958:             }
7959:         }
7960:         if let Some(ev) = morph_event {
7961:             println!(
7962:                 "  ◄ {} ►  Depth L{:02} | Manifold {}→{} sheets | Rings {}→{} | Rad {:.2}",
7963:                 ev,
7964:                 model.depth(),
7965:                 manifold_depth_before,
7966:                 model.manifold_depth(),
7967:                 spatial_rings_before,
7968:                 model.active_spatial_rings(),
7969:                 rad_amp
7970:             );
7971:             morph_events.push(serde_json::json!({
7972:                 "step": absolute_step,
7973:                 "run_step": step,
7974:                 "event": ev,
7975:                 "morph_depth": model.depth(),
7976:                 "manifold_depth_before": manifold_depth_before,
7977:                 "manifold_depth": model.manifold_depth(),
7978:                 "active_spatial_rings_before": spatial_rings_before,
7979:                 "active_spatial_rings": model.active_spatial_rings(),
7980:                 "far_ring_gain": model.far_ring_gain(),
7981:                 "rad_amp": rad_amp,
7982:             }));
7983:             pending_morph_event = Some((absolute_step, ev));
7984:         }
7985: 
7986:         // --- POTENTIAL CONTROLLER: the entire crash cart, one call ---
7987:         let recursive_curiosity = curiosity_factor
7988:             .max(controller.meta.surprise() * 0.80)
7989:             .max(adaptive_dynamics.stagnation * 0.90);
7990:         let pot = potential.update(PotentialState {
7991:             micro_amp: micro_abs,
7992:             macro_amp: macro_abs,
7993:             coupling: synergy_val,
7994:             movement,
7995:             energy: energy_state,

## EVIDENCE src/main.rs:8048-8150
SHA256 629d1a58fe1a0cbd01f25bc65d8e6dfe52013dfbd7d8ba27bf2250313cf2bc6a

8048:             let target_long_spec = spec_proj_long.log_mag(&target_long)?.detach();
8049:             Some(spec_proj_long.spectral_loss(&out_long_spec, &target_long_spec, 0.03)?)
8050:         } else {
8051:             None
8052:         };
8053: 
8054:         // --- TOTAL LOSS ---
8055:         // Source evidence is an invariant, not a preference the learned
8056:         // arbiter may switch off. The arbiter modulates additional emphasis
8057:         // around a non-zero grounding floor.
8058:         let source_weight = 0.80 + 0.35 * lw[1] * (1.0 - RESONANT_AUTONOMY);
8059:         let mut total_loss = mimic_loss.affine(source_weight as f64, 0.0)?;
8060:         total_loss = total_loss.add(&level_loss.affine((lw[0] * 0.75) as f64, 0.0)?)?;
8061:         total_loss = total_loss.add(&saturation_loss.affine(2.0, 0.0)?)?;
8062:         total_loss = total_loss.add(&movement_loss.affine((lw[2] * 0.20) as f64, 0.0)?)?;
8063:         let anti_weld_weight = 0.06 + 0.20 * adaptive_dynamics.stagnation;
8064:         let regional_weld_weight = 0.04 + 0.12 * adaptive_dynamics.stagnation;
8065:         total_loss = total_loss.add(&movement_floor_loss.affine(anti_weld_weight as f64, 0.0)?)?;
8066:         total_loss =
8067:             total_loss.add(&regional_floor_loss.affine(regional_weld_weight as f64, 0.0)?)?;
8068:         total_loss = total_loss.add(&roughness_loss.affine(lw[3] as f64, 0.0)?)?;
8069:         total_loss = total_loss.add(&boundary_loss.affine(0.50, 0.0)?)?;
8070:         if let Some(long) = trajectory_loss {
8071:             total_loss = total_loss.add(&long.affine(0.35, 0.0)?)?;
8072:         }
8073:         total_loss = total_loss.add(&reg_loss.affine(0.002, 0.0)?)?;
8074:         total_loss = total_loss.add(&rg_loss.affine((0.15 * lw[4].max(0.2)) as f64, 0.0)?)?;
8075:         total_loss =
8076:             total_loss.add(&self_model_loss.affine((0.30 * lw[5].max(0.2)) as f64, 0.0)?)?;
8077:         total_loss = total_loss.add(&empowerment_loss.affine((0.10 * lw[6]) as f64, 0.0)?)?;
8078:         // Coupling band, released by the controller's over-coupling/heat signal.
8079:         let synergy_w = (SYNERGY_BAND_W * (1.0 - 0.7 * pot.couple_release)).max(0.1f32);
8080:         total_loss = total_loss.add(&synergy_loss.affine(synergy_w as f64, 0.0)?)?;
8081:         // Entropy BONUS (v3 minimized it — see AudioArbiter): adding neg_entropy
8082:         // with a positive coefficient maximizes mixing entropy.
8083:         total_loss = total_loss.add(&neg_entropy.affine(0.05, 0.0)?)?;
8084:         total_loss = total_loss.add(&arb_progress_loss)?;
8085:         if let Some(nl) = novelty_loss {
8086:             let novelty_weight = NOVELTY_W * adaptive_dynamics.stagnation as f64;
8087:             total_loss = total_loss.add(&nl.affine(novelty_weight, 0.0)?)?;
8088:         }
8089: 
8090:         tape_loss = Some(match tape_loss.take() {
8091:             None => total_loss,
8092:             Some(w) => w.add(&total_loss)?,
8093:         });
8094:         steps_in_tape += 1;
8095:         steps_since_update += 1;
8096: 
8097:         // `sigma` is movement persistence, not a calibrated branching ratio.
8098:         // Retain it as evidence, but do not shape LR around an unproved sigma=1
8099:         // singularity.
8100:         let crit_gain = 1.0f32;
8101:         let phi_gate = 1.0 / (1.0 + phi);
8102:         let curiosity_lr_gain = 1.0 + curiosity_factor as f64 * LR_CURIOSITY_MAX;
8103:         latest_lr_gain = crit_gain as f64 * pot.lr_heat * phi_gate as f64 * curiosity_lr_gain;
8104:         tape_lr_gain_sum += latest_lr_gain;
8105: 
8106:         if tape_boundary {
8107:             if let Some(w) = tape_loss.take() {
8108:                 let segment_steps = steps_in_tape;
8109:                 let segment_mean = w.affine(1.0 / segment_steps as f64, 0.0)?;
8110:                 match segment_mean.to_scalar::<f32>() {
8111:                     Ok(loss_val) if loss_val.is_finite() && tape_lr_gain_sum.is_finite() => {
8112:                         let backward_started = Instant::now();
8113:                         let backward_result = segment_mean.backward();
8114:                         phase_profiler.backward += backward_started.elapsed();
8115:                         match backward_result {
8116:                             Ok(mut segment_grads) => {
8117:                                 // Store a weighted SUM across bounded tape segments.
8118:                                 // Dividing once at the optimizer boundary makes -w 64
8119:                                 // a 64-chunk gradient average without a 64-chunk graph.
8120:                                 let segment_weight = segment_steps as f64;
8121:                                 if let Some(grads) = accumulated_grads.as_mut() {
8122:                                     for var in varmap.all_vars() {
8123:                                         if let Some(g) = segment_grads.remove(var.as_tensor()) {
8124:                                             let weighted = g.affine(segment_weight, 0.0)?.detach();
8125:                                             let merged = match grads.remove(var.as_tensor()) {
8126:                                                 Some(previous) => previous.add(&weighted)?.detach(),
8127:                                                 None => weighted,
8128:                                             };
8129:                                             grads.insert(var.as_tensor(), merged);
8130:                                         }
8131:                                     }
8132:                                 } else {
8133:                                     for var in varmap.all_vars() {
8134:                                         if let Some(g) = segment_grads.remove(var.as_tensor()) {
8135:                                             segment_grads.insert(
8136:                                                 var.as_tensor(),
8137:                                                 g.affine(segment_weight, 0.0)?.detach(),
8138:                                             );
8139:                                         }
8140:                                     }
8141:                                     accumulated_grads = Some(segment_grads);
8142:                                 }
8143:                                 accumulated_steps += segment_steps;
8144:                                 accumulated_lr_gain_sum += tape_lr_gain_sum;
8145:                             }
8146:                             Err(e) => println!(
8147:                                 "! WARNING: backward failed: {} — dropping tape segment.",
8148:                                 e
8149:                             ),
8150:                         }

## EVIDENCE src/main.rs:8170-8390
SHA256 3e676cea91a2a78ba3affdfef3e7ad90c6b37625cd72167369da91969d4e16b6

8170:                         if let Some(g) = grads.get(var.as_tensor()) {
8171:                             sq = sq.add(&g.affine(mean_scale, 0.0)?.sqr()?.sum_all()?)?;
8172:                         }
8173:                     }
8174:                     let gnorm = sq.to_scalar::<f32>().unwrap_or(f32::INFINITY).sqrt();
8175:                     if gnorm.is_finite() {
8176:                         // True global-norm clipping happens before AdamW sees the
8177:                         // accumulated mean, so a hot segment cannot poison moments.
8178:                         let clip_scale = (GRAD_NORM_MAX / gnorm.max(1e-6)).min(1.0) as f64;
8179:                         let optimizer_scale = mean_scale * clip_scale;
8180:                         latest_grad_norm = gnorm;
8181:                         latest_clip_scale = clip_scale as f32;
8182:                         for var in varmap.all_vars() {
8183:                             if let Some(g) = grads.remove(var.as_tensor()) {
8184:                                 grads.insert(
8185:                                     var.as_tensor(),
8186:                                     g.affine(optimizer_scale, 0.0)?.detach(),
8187:                                 );
8188:                             }
8189:                         }
8190:                         let mean_lr_gain = accumulated_lr_gain_sum / accumulated_steps as f64;
8191:                         let moment_warmup = (0.20
8192:                             + 0.80 * (optimizer.cumulative_updates() + 1) as f64 / 32.0)
8193:                             .min(1.0);
8194:                         optimizer.set_learning_rate(target_lr * mean_lr_gain * moment_warmup);
8195:                         let optimizer_started = Instant::now();
8196:                         let _ = optimizer.step(&grads);
8197:                         phase_profiler.optimizer += optimizer_started.elapsed();
8198:                         optimizer_update_count += 1;
8199:                     } else {
8200:                         println!("! WARNING: non-finite grad norm — skipping horizon.");
8201:                     }
8202:                 }
8203:             }
8204:             steps_since_update = 0;
8205:             accumulated_steps = 0;
8206:             accumulated_lr_gain_sum = 0.0;
8207:         }
8208: 
8209:         if tape_boundary {
8210:             micro_tape = next_micro.detach();
8211:             macro_tape = next_macro.detach();
8212:             hidden_mem = next_hidden.detach();
8213:         } else {
8214:             micro_tape = next_micro;
8215:             macro_tape = next_macro;
8216:             hidden_mem = next_hidden;
8217:         }
8218: 
8219:         // Radiation is an ecological event, not an optimizer event. Convert
8220:         // the old default-window probability to an equivalent per-chunk
8221:         // hazard so changing --bptt no longer changes organism dynamics.
8222:         let radiation_window_probability = (RADIATE_PROB
8223:             + curiosity_factor * 0.04
8224:             + (current_control.kick_mult - 1.0).max(0.0) * 0.03
8225:             + controller.meta.surprise() * 0.02
8226:             + escape_strength * 0.12)
8227:             .clamp(0.0, 0.30);
8228:         let radiation_probability =
8229:             reference_window_probability_to_chunk(radiation_window_probability);
8230:         if rng.gen::<f32>() < radiation_probability {
8231:             micro_tape = levy_radiate(
8232:                 &micro_tape,
8233:                 rad_amp
8234:                     * (1.0
8235:                         + curiosity_factor * 0.15
8236:                         + controller.meta.surprise() * 0.10
8237:                         + escape_strength * 0.20),
8238:                 &mut rng,
8239:             )?
8240:             .detach();
8241:         }
8242: 
8243:         // --- LANGEVIN STEP: drift (-grad V gains) + temperature noise ---
8244:         let controlled_shear = (pot.shear_amp * current_control.shear_mult).clamp(0.0, 0.75);
8245:         if absolute_step.is_multiple_of(MACRO_UPDATE_EVERY) {
8246:             shear_phase += SHEAR_PHASE_VEL
8247:                 * MACRO_UPDATE_EVERY as f32
8248:                 * (0.82 + 0.28 * current_control.shear_mult);
8249:             let shear = shear_gen.generate(controlled_shear, shear_phase, &device)?;
8250:             macro_tape = macro_tape
8251:                 .add(&shear)?
8252:                 .tanh()?
8253:                 .affine(pot.macro_gain as f64, 0.0)?;
8254:         }
8255:         let controlled_kick = (pot.micro_kick * current_control.kick_mult).clamp(0.0, 0.10);
8256:         if controlled_kick > 1e-3 {
8257:             let kick = randn_t(
8258:                 &mut rng,
8259:                 &[1, CA_CHANNELS, GRID_H, GRID_W],
8260:                 controlled_kick,
8261:                 &device,
8262:             )?;
8263:             micro_tape = micro_tape.add(&kick)?;
8264:         }
8265:         micro_tape = micro_tape
8266:             .affine(pot.micro_gain as f64, 0.0)?
8267:             .clamp(-1.0f32, 1.0f32)?;
8268: 
8269:         // Episodic snapshot cadence.
8270:         if absolute_step.is_multiple_of(EPI_SNAP_EVERY as u64) && absolute_step > 0 {
8271:             episodic.snapshot(&refined_hidden);
8272:         }
8273:         // Novelty buffer cadence.
8274:         if absolute_step.is_multiple_of(NOVELTY_EVERY as u64) {
8275:             if novelty_buf.len() >= NOVELTY_SLOTS {
8276:                 novelty_buf.pop_front();
8277:             }
8278:             novelty_buf.push_back(mono_spec.detach());
8279:         }
8280: 
8281:         // --- TRUTHFUL POST PATH ---
8282:         // The prior system let discrete controller actions inject spectral noise, modal
8283:         // ringing, and FDN echo *after* the differentiable source loss. That
8284:         // created an acoustic shortcut: the controller could buy entropy that
8285:         // the organism could neither predict nor learn to synthesize. Those
8286:         // effects and states are gone; the audible path is the learned renderer
8287:         // followed only by bounded saturation and a stateful DC blocker.
8288:         let output_started = Instant::now();
8289:         let audio_normalized_vec = audio_normalized.to_vec2::<f32>()?;
8290:         let mut audio_l = audio_normalized_vec[0].clone();
8291:         let mut audio_r = audio_normalized_vec[1].clone();
8292:         for i in 0..CHUNK_SIZE {
8293:             audio_l[i] = (audio_l[i] * 0.92).tanh();
8294:             audio_r[i] = (audio_r[i] * 0.92).tanh();
8295: 
8296:             // Stateful DC blocker is part of the observed signal path, so the
8297:             // recursive controller hears the same waveform later mastered.
8298:             let xl = audio_l[i];
8299:             let yl = xl - dc_x1_l + dc_pole * dc_y1_l;
8300:             dc_x1_l = xl;
8301:             dc_y1_l = yl;
8302:             audio_l[i] = yl;
8303:             let xr = audio_r[i];
8304:             let yr = xr - dc_x1_r + dc_pole * dc_y1_r;
8305:             dc_x1_r = xr;
8306:             dc_y1_r = yr;
8307:             audio_r[i] = yr;
8308:             raw_peak = raw_peak.max(yl.abs()).max(yr.abs());
8309:         }
8310: 
8311:         let post = spectral_mon.analyze(
8312:             &audio_l,
8313:             &audio_r,
8314:             movement,
8315:             synergy_val,
8316:             field_entropy,
8317:             sigma,
8318:         );
8319:         let s_sig = &post.json;
8320:         uncertainty.update(
8321:             s_sig,
8322:             &m_sig,
8323:             mimic_drift_n,
8324:             synergy_val,
8325:             empowerment_val,
8326:             &controller.meta,
8327:         );
8328:         phi = uncertainty.phi;
8329: 
8330:         let region_change_mean = region_change_host.iter().sum::<f32>() / REGION_COUNT as f32;
8331:         let observation_delta = last_observation
8332:             .as_ref()
8333:             .map(|prev| post.observation.distance(prev))
8334:             .unwrap_or(0.08);
8335:         let structured_complexity = post.observation.structured_complexity();
8336:         adaptive_dynamics.observe(
8337:             movement,
8338:             region_change_mean,
8339:             observation_delta,
8340:             structured_complexity,
8341:             sigma,
8342:             controller.meta.confidence,
8343:         );
8344:         let recurrence = motifs.recurrence(&post.observation, absolute_step);
8345:         let reward = post.observation.reward_against(
8346:             last_observation.as_ref(),
8347:             recurrence,
8348:             &adaptive_dynamics,
8349:             controller.meta.confidence,
8350:             controller.action_age,
8351:         );
8352:         controller.bandit.update(current_action, reward);
8353:         adaptive_dynamics.observe_reward(reward);
8354:         if absolute_step.is_multiple_of(MOTIF_EVERY as u64) {
8355:             motifs.maybe_store(
8356:                 &post.observation,
8357:                 current_control,
8358:                 absolute_step,
8359:                 motif_capacity,
8360:                 &adaptive_dynamics,
8361:                 &mut motif_diagnostics,
8362:             );
8363:         }
8364:         last_observation = Some(post.observation.clone());
8365:         pending_predictor_input = Some(current_predictor_input);
8366: 
8367:         for i in 0..CHUNK_SIZE {
8368:             let byte = i * 8;
8369:             raw_chunk_bytes[byte..byte + 4].copy_from_slice(&audio_l[i].to_le_bytes());
8370:             raw_chunk_bytes[byte + 4..byte + 8].copy_from_slice(&audio_r[i].to_le_bytes());
8371:         }
8372:         raw_audio_writer.write_all(&raw_chunk_bytes)?;
8373:         phase_profiler.output_io += output_started.elapsed();
8374:         let stereo_corr = s_sig["stereo_corr"].as_f64().unwrap_or(0.0) as f32;
8375:         chunk_scores.push(prime_chunk_score(
8376:             field_entropy,
8377:             adaptive_dynamics.activity_health,
8378:             structured_complexity,
8379:             adaptive_dynamics.stagnation,
8380:             post.observation.width(),
8381:             stereo_corr,
8382:             experiments.prime_width_score,
8383:         ));
8384:         completed_chunks += 1;
8385:         evolved_chunks = step + 1;
8386: 
8387:         if step % TRACE_EVERY == 0 {
8388:             let sample_index = trace_rows;
8389:             let (trace_morph_event_step, trace_morph_event) =
8390:                 pending_morph_event.unwrap_or((0, ""));

## EVIDENCE src/main.rs:8448-8542
SHA256 648b38aca4057d67eace47cc5879ea944db2e24198b0b8c43da466a9af69147a

8448:                 "planner_proposal": controller.model_proposal().label(),
8449:                 "bandit_proposal": controller.bandit_proposal().label(),
8450:                 "reward": reward, "recurrence": recurrence,
8451:                 "model_confidence": controller.meta.confidence,
8452:                 "effective_model_weight": adaptive_dynamics.effective_model_weight,
8453:                 "prediction_error": controller.meta.error_ema,
8454:                 "calibration_error": controller.meta.calibration_error,
8455:                 "region_change": region_change_mean,
8456:                 "observation_delta": observation_delta,
8457:                 "activity_health": adaptive_dynamics.activity_health,
8458:                 "stagnation": adaptive_dynamics.stagnation,
8459:                 "low_motion_run": adaptive_dynamics.low_motion_run,
8460:                 "escape_strength": adaptive_dynamics.escape_strength(),
8461:                 "reward_mean": adaptive_dynamics.reward_mean,
8462:                 "reward_std": adaptive_dynamics.reward_std(),
8463:                 "lr_gain": latest_lr_gain, "grad_norm": latest_grad_norm,
8464:                 "clip_scale": latest_clip_scale, "radiation_probability": radiation_probability,
8465:                 "motifs": motifs.entries.len(),
8466:                 "motif_candidates": motif_diagnostics.candidates,
8467:                 "motif_stored_total": motif_diagnostics.stored_total,
8468:                 "motif_rejected_quality": motif_diagnostics.rejected_quality,
8469:                 "motif_rejected_similarity": motif_diagnostics.rejected_similarity,
8470:                 "motif_last_quality": motif_diagnostics.last_quality,
8471:                 "motif_last_distance": motif_diagnostics.last_nearest_distance,
8472:                 "carrier_freq_l": f_l, "carrier_freq_r": f_r,
8473:                 "carrier_beat_hz": (f_l - f_r).abs(),
8474:                 "decoder_pan": decoder_pan_val,
8475:                 "decoder_pan_raw": decoder_pan_raw_val,
8476:                 "pan_center_loss": pan_center_loss_val,
8477:                 "decoder_side_control": decoder_side_control_val,
8478:                 "decoder_width_control": decoder_width_control_val,
8479:                 "decoder_width_raw": decoder_width_raw_val,
8480:                 "mimic_coarse": mimic_drift, "mimic_fine": mimic_fine_val,
8481:                 "band_loss": band_loss_val, "chroma_loss": chroma_loss_val,
8482:                 "onset_loss": onset_loss_val, "recurrence_loss": recurrence_loss_val,
8483:                 "modulation_loss": modulation_loss_val,
8484:                 "low_band_loss": low_band_loss_val,
8485:                 "stereo_balance_loss": stereo_balance_loss_val,
8486:                 "stereo_side_geometry_loss": stereo_side_geometry_loss_val,
8487:                 "stereo_correlation_loss": stereo_correlation_loss_val,
8488:                 "stereo_level_loss": stereo_level_loss_val,
8489:                 "output_low_band_ratio": output_low_band_ratio_val,
8490:                 "output_side_mid_log_ratio": output_side_ratio_val,
8491:                 "target_low_band_ratio": target_low_band_ratio_val,
8492:                 "target_side_mid_log_ratio": target_side_ratio_val,
8493:                 "decoder_stereo_corr": output_stereo_corr_val,
8494:                 "target_stereo_corr": target_stereo_corr_val,
8495:                 "decoder_stereo_level_log_ratio": output_stereo_level_ratio_val,
8496:                 "target_stereo_level_log_ratio": target_stereo_level_ratio_val,
8497:                 "development_best_spectral": development_best_val,
8498:                 "development_mean_spectral": development_mean_val,
8499:                 "development_mean_chroma": development_chroma_val,
8500:                 "development_score": development_plateau.score,
8501:                 "development_plateau_ready": development_plateau.ready,
8502:                 "development_relative_improvement": development_plateau.relative_improvement,
8503:                 "validation_best_spectral": validation_best_val,
8504:                 "validation_mean_spectral": validation_mean_val,
8505:                 "validation_mean_chroma": validation_chroma_val,
8506:                 "boundary_loss": boundary_loss_val, "level_loss": level_loss_val,
8507:                 "target_rms": target_rms_val, "target_file": target_file,
8508:                 "target_frame": target_frame, "target_chunks_left": target_chunks_left,
8509:                 "optimizer_updates": optimizer.cumulative_updates(),
8510:                 "optimizer_updates_run": optimizer_update_count,
8511:                 "optimizer_resumed": optimizer_resumed,
8512:                 "ultrasonic_ratio": s_sig["ultrasonic_ratio"].as_f64().unwrap_or(0.0),
8513:             });
8514:             trace_phi_sum += phi as f64;
8515:             trace_aperture_sum += aperture as f64;
8516:             trace_synergy_sum += synergy_val as f64;
8517:             trace_temp_sum += pot.temp as f64;
8518:             trace_sigma_sum += sigma as f64;
8519:             trace_pi_sum += s_sig["pi_proxy"].as_f64().unwrap_or(0.0);
8520:             serde_json::to_writer(&mut trace_spool, &trace_row)?;
8521:             trace_spool.write_all(b"\n")?;
8522:             trace_rows += 1;
8523:             if trace_rows.is_multiple_of(64) {
8524:                 topology_writer.flush()?;
8525:                 trace_spool.flush()?;
8526:                 topology_index_spool.flush()?;
8527:             }
8528:             pending_morph_event = None;
8529:         }
8530:         if step % 50 == 0 {
8531:             let rolling_sec = profiling_lap.elapsed().as_secs_f32();
8532:             let rolling_sps = if step > 0 && rolling_sec > 1e-4 {
8533:                 50.0 / rolling_sec
8534:             } else {
8535:                 0.0
8536:             };
8537:             profiling_lap = std::time::Instant::now();
8538:             println!("Chunk {}/{} (global {}) [SPS: {:.2}] | Move: {:.3} | Mimic: {:.3} {} | {} | L{:02} rad:{:.2} | σ:{:.3}/{:.2} | V:{:.3} T:{:.2} | LRx:{:.2}",
8539:                 step, total_chunks, absolute_step, rolling_sps, movement, mimic_drift_n, trend,
8540:                 current_action.label(), model.depth(), rad_amp, sigma, criticality.confidence,
8541:                 pot.v, pot.temp, latest_lr_gain);
8542:             println!("  field H:{:.2}b · {} · synergy:{:.2} empower:{:.2} | energy:{:.2} | φ:{:.2} PI:{:.2} | model raw/eff:{:.2}/{:.2} err:{:.3}",

## EVIDENCE src/main.rs:8665-8698
SHA256 4762dec17ff62a5a32a230d4829eb5f520198460eb3597e673474e87254c53af

8665:                 }
8666:             }
8667:             phase_profiler.checkpoints += checkpoint_started.elapsed();
8668:         }
8669:     }
8670: 
8671:     topology_writer.flush()?;
8672:     drop(topology_writer);
8673:     std::fs::rename(&topology_tmp_path, &topology_path)?;
8674:     trace_spool.flush()?;
8675:     topology_index_spool.flush()?;
8676:     drop(trace_spool);
8677:     drop(topology_index_spool);
8678: 
8679:     let total_elapsed = timer_start.elapsed().as_secs_f32();
8680:     let overall_sps = completed_chunks as f32 / total_elapsed.max(1e-6);
8681:     println!("\n=== PERFORMANCE REPORT ===");
8682:     println!("Total simulation elapsed: {:.2}s", total_elapsed);
8683:     println!("Overall performance speed: {:.2} steps/sec", overall_sps);
8684:     phase_profiler.report(completed_chunks);
8685: 
8686:     // ---- STREAMED MASTERING + PRIME EXTRACTION ----
8687:     raw_audio_writer.flush()?;
8688:     drop(raw_audio_writer);
8689:     let norm = 0.891 / raw_peak.max(1e-6);
8690:     println!(
8691:         "Mastering: streamed DC-blocked peak {:.3} normalized to -1 dBFS (gain {:.2}x).",
8692:         raw_peak, norm
8693:     );
8694: 
8695:     let total_frames = completed_chunks * CHUNK_SIZE;
8696:     let rendered_seconds = total_frames as f32 / SAMPLE_RATE as f32;
8697:     let prime_secs = 60.0f32.min(rendered_seconds.max(CHUNK_SIZE as f32 / SAMPLE_RATE as f32));
8698:     let win = ((SAMPLE_RATE as f32 * prime_secs / CHUNK_SIZE as f32) as usize)

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

## EVIDENCE src/main.rs:6090-6520
SHA256 75d6f09240fd1d8e72d8cc836a30598b907d95538ce58198ff5e59ceb5cb65e7

6090: 
6091: // --- MAIN RUNTIME LOGIC ---
6092: #[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
6093: struct ExperimentProfile {
6094:     spectral_deemphasis: bool,
6095:     prime_width_score: bool,
6096: }
6097: 
6098: impl ExperimentProfile {
6099:     fn enabled(self) -> bool {
6100:         self.spectral_deemphasis || self.prime_width_score
6101:     }
6102: }
6103: 
6104: fn prime_chunk_score(
6105:     field_entropy: f32,
6106:     activity_health: f32,
6107:     structured_complexity: f32,
6108:     stagnation: f32,
6109:     width: f32,
6110:     stereo_corr: f32,
6111:     width_enabled: bool,
6112: ) -> f32 {
6113:     let legacy = field_entropy * (0.25 + 0.50 * activity_health) + structured_complexity * 0.75
6114:         - stagnation * 0.25;
6115:     if width_enabled && stereo_corr > 0.0 {
6116:         legacy + width * stereo_corr.clamp(0.0, 1.0) * 0.40
6117:     } else {
6118:         legacy
6119:     }
6120: }
6121: 
6122: fn best_prime_window(scores: &[f32], win: usize) -> (usize, f32) {
6123:     if scores.is_empty() {
6124:         return (0, 0.0);
6125:     }
6126:     let win = win.max(1).min(scores.len());
6127:     let mut run: f32 = scores.iter().take(win).sum();
6128:     let mut best_start = 0usize;
6129:     let mut best_sum = run;
6130:     if scores.len() > win {
6131:         for start in 1..=(scores.len() - win) {
6132:             run += scores[start + win - 1] - scores[start - 1];
6133:             if run > best_sum {
6134:                 best_sum = run;
6135:                 best_start = start;
6136:             }
6137:         }
6138:     }
6139:     (best_start, best_sum)
6140: }
6141: 
6142: fn validate_experiment_run(
6143:     base_dir: &str,
6144:     tag: Option<&str>,
6145:     model_override: Option<&str>,
6146:     state_override: Option<&str>,
6147:     import_model: Option<&str>,
6148:     fresh_model: bool,
6149:     profile: ExperimentProfile,
6150: ) -> Result<String> {
6151:     let metadata_path = artifact_path(base_dir, "titan_run_metadata_v9", "json", tag);
6152:     if profile.enabled() {
6153:         if tag.is_none() {
6154:             anyhow::bail!("experimental flags require an isolated --run-tag");
6155:         }
6156:         if model_override.is_some() || state_override.is_some() {
6157:             anyhow::bail!(
6158:                 "experimental runs use tagged model and world paths; omit --model and --state"
6159:             );
6160:         }
6161:     }
6162:     if std::path::Path::new(&metadata_path).exists() && tag.is_some() {
6163:         let metadata: serde_json::Value = serde_json::from_slice(&std::fs::read(&metadata_path)?)?;
6164:         let saved_profile = match metadata.pointer("/invocation/experiments") {
6165:             Some(value) => serde_json::from_value::<ExperimentProfile>(value.clone())?,
6166:             None => ExperimentProfile::default(),
6167:         };
6168:         if saved_profile != profile {
6169:             anyhow::bail!(
6170:                 "tagged continuation profile mismatch: saved {:?}, requested {:?}",
6171:                 saved_profile,
6172:                 profile
6173:             );
6174:         }
6175:         if profile.enabled() && (import_model.is_some() || fresh_model) {
6176:             anyhow::bail!("experimental continuation must load its tagged model; omit --import-model and --fresh-model");
6177:         }
6178:         if profile.enabled() {
6179:             for (stem, extension) in [
6180:                 ("titan_model_v9", "safetensors"),
6181:                 ("titan_world_v9", "bin"),
6182:                 ("titan_optimizer_v9", "safetensors"),
6183:             ] {
6184:                 let path = artifact_path(base_dir, stem, extension, tag);
6185:                 if !std::path::Path::new(&path).is_file() {
6186:                     anyhow::bail!("experimental continuation is missing tagged checkpoint: {path}");
6187:                 }
6188:             }
6189:         }
6190:     } else if profile.enabled() {
6191:         let tag = tag.expect("checked above");
6192:         let mut occupied = false;
6193:         if std::path::Path::new(base_dir).exists() {
6194:             for entry in std::fs::read_dir(base_dir)? {
6195:                 let name = entry?.file_name().to_string_lossy().into_owned();
6196:                 if name.contains(&format!("_{tag}.")) || name.contains(&format!("_{tag}_")) {
6197:                     occupied = true;
6198:                     break;
6199:                 }
6200:             }
6201:         }
6202:         if occupied {
6203:             anyhow::bail!(
6204:                 "experimental --run-tag {tag} already has artifacts but no matching run metadata"
6205:             );
6206:         }
6207:         if fresh_model == import_model.is_some() {
6208:             anyhow::bail!(
6209:                 "first experimental run requires exactly one of --import-model or --fresh-model"
6210:             );
6211:         }
6212:         if let Some(source) = import_model {
6213:             if !std::path::Path::new(source).is_file() {
6214:                 anyhow::bail!("experimental import model does not exist: {source}");
6215:             }
6216:         }
6217:     }
6218:     Ok(metadata_path)
6219: }
6220: 
6221: fn main() -> Result<()> {
6222:     let args: Vec<String> = std::env::args().collect();
6223:     if args.iter().any(|arg| analysis::is_analysis_flag(arg)) {
6224:         return analysis::run_from_args(&args);
6225:     }
6226:     let mut base_dir = "/sdcard/Download".to_string();
6227:     let mut n_threads = std::thread::available_parallelism()
6228:         .map(|n| n.get())
6229:         .unwrap_or(8)
6230:         .min(6);
6231:     let mut target_lr = BASE_LR;
6232:     let mut sim_duration = DURATION_SECONDS;
6233:     let mut bptt_window = BPTT_WINDOW;
6234:     let mut core_update_every = CORE_UPDATE_EVERY;
6235:     let mut fresh_model = false;
6236:     let mut fresh_decoder = false;
6237:     let mut fresh_world = false;
6238:     let mut freeze_morph = false;
6239:     let mut morph_blocks = MORPH_DEFAULT_BLOCKS;
6240:     let mut morph_width = MORPH_DEFAULT_WIDTH;
6241:     let mut motif_capacity = MOTIF_DEFAULT_CAPACITY;
6242:     let mut morph_depth_override: Option<usize> = None;
6243:     let mut max_morph_depth_override: Option<usize> = None;
6244:     let mut state_override: Option<String> = None;
6245:     let mut model_override: Option<String> = None;
6246:     let mut import_model_override: Option<String> = None;
6247:     let mut corpus_manifest_override: Option<String> = None;
6248:     let mut refresh_corpus_manifest_requested = false;
6249:     let mut rebuild_corpus_manifest_requested = false;
6250:     let mut prune_missing_manifest_entries = false;
6251:     let mut manifest_only = false;
6252:     let mut run_tag: Option<String> = None;
6253:     let mut experiments = ExperimentProfile::default();
6254:     let mut seed: u64 = 42;
6255:     let mut arg_idx = 1;
6256:     while arg_idx < args.len() {
6257:         match args[arg_idx].as_str() {
6258:             "--base-dir" | "-b" => {
6259:                 if arg_idx + 1 < args.len() {
6260:                     base_dir = args[arg_idx + 1].clone();
6261:                     arg_idx += 2;
6262:                 } else {
6263:                     anyhow::bail!("Missing value for --base-dir");
6264:                 }
6265:             }
6266:             "--threads" | "-t" => {
6267:                 if arg_idx + 1 < args.len() {
6268:                     n_threads = args[arg_idx + 1].parse::<usize>()?;
6269:                     arg_idx += 2;
6270:                 } else {
6271:                     anyhow::bail!("Missing value for --threads");
6272:                 }
6273:             }
6274:             "--lr" | "-l" => {
6275:                 if arg_idx + 1 < args.len() {
6276:                     target_lr = args[arg_idx + 1].parse::<f64>()?;
6277:                     arg_idx += 2;
6278:                 } else {
6279:                     anyhow::bail!("Missing value for --lr");
6280:                 }
6281:             }
6282:             "--duration" | "-d" => {
6283:                 if arg_idx + 1 < args.len() {
6284:                     sim_duration = args[arg_idx + 1].parse::<f32>()?;
6285:                     arg_idx += 2;
6286:                 } else {
6287:                     anyhow::bail!("Missing value for --duration");
6288:                 }
6289:             }
6290:             "--bptt" | "-w" => {
6291:                 if arg_idx + 1 < args.len() {
6292:                     bptt_window = args[arg_idx + 1].parse::<usize>()?;
6293:                     arg_idx += 2;
6294:                 } else {
6295:                     anyhow::bail!("Missing value for --bptt");
6296:                 }
6297:             }
6298:             "--core-update-every" => {
6299:                 if arg_idx + 1 < args.len() {
6300:                     core_update_every = args[arg_idx + 1].parse::<usize>()?;
6301:                     arg_idx += 2;
6302:                 } else {
6303:                     anyhow::bail!("Missing value for --core-update-every");
6304:                 }
6305:             }
6306:             "--seed" | "-s" => {
6307:                 if arg_idx + 1 < args.len() {
6308:                     seed = args[arg_idx + 1].parse::<u64>()?;
6309:                     arg_idx += 2;
6310:                 } else {
6311:                     anyhow::bail!("Missing value for --seed");
6312:                 }
6313:             }
6314:             "--state" => {
6315:                 if arg_idx + 1 < args.len() {
6316:                     state_override = Some(args[arg_idx + 1].clone());
6317:                     arg_idx += 2;
6318:                 } else {
6319:                     anyhow::bail!("Missing value for --state");
6320:                 }
6321:             }
6322:             "--model" => {
6323:                 if arg_idx + 1 < args.len() {
6324:                     model_override = Some(args[arg_idx + 1].clone());
6325:                     arg_idx += 2;
6326:                 } else {
6327:                     anyhow::bail!("Missing value for --model");
6328:                 }
6329:             }
6330:             "--import-model" => {
6331:                 if arg_idx + 1 < args.len() {
6332:                     import_model_override = Some(args[arg_idx + 1].clone());
6333:                     arg_idx += 2;
6334:                 } else {
6335:                     anyhow::bail!("Missing value for --import-model");
6336:                 }
6337:             }
6338:             "--corpus-manifest" => {
6339:                 if arg_idx + 1 < args.len() {
6340:                     corpus_manifest_override = Some(args[arg_idx + 1].clone());
6341:                     arg_idx += 2;
6342:                 } else {
6343:                     anyhow::bail!("Missing value for --corpus-manifest");
6344:                 }
6345:             }
6346:             "--refresh-corpus-manifest" => {
6347:                 refresh_corpus_manifest_requested = true;
6348:                 arg_idx += 1;
6349:             }
6350:             "--rebuild-corpus-manifest" => {
6351:                 rebuild_corpus_manifest_requested = true;
6352:                 arg_idx += 1;
6353:             }
6354:             "--prune-missing-manifest-entries" | "--prune-missing" => {
6355:                 prune_missing_manifest_entries = true;
6356:                 arg_idx += 1;
6357:             }
6358:             "--manifest-only" => {
6359:                 manifest_only = true;
6360:                 arg_idx += 1;
6361:             }
6362:             "--run-tag" => {
6363:                 if arg_idx + 1 < args.len() {
6364:                     validate_run_tag(&args[arg_idx + 1])?;
6365:                     run_tag = Some(args[arg_idx + 1].clone());
6366:                     arg_idx += 2;
6367:                 } else {
6368:                     anyhow::bail!("Missing value for --run-tag");
6369:                 }
6370:             }
6371:             "--spectral-deemphasis" => {
6372:                 experiments.spectral_deemphasis = true;
6373:                 arg_idx += 1;
6374:             }
6375:             "--prime-width-score" => {
6376:                 experiments.prime_width_score = true;
6377:                 arg_idx += 1;
6378:             }
6379:             "--freeze-morph" => {
6380:                 freeze_morph = true;
6381:                 arg_idx += 1;
6382:             }
6383:             "--max-morph-depth" => {
6384:                 if arg_idx + 1 < args.len() {
6385:                     max_morph_depth_override = Some(args[arg_idx + 1].parse::<usize>()?);
6386:                     arg_idx += 2;
6387:                 } else {
6388:                     anyhow::bail!("Missing value for --max-morph-depth");
6389:                 }
6390:             }
6391:             "--morph-layers" | "--morph-blocks" => {
6392:                 if arg_idx + 1 < args.len() {
6393:                     morph_blocks = args[arg_idx + 1].parse::<usize>()?;
6394:                     arg_idx += 2;
6395:                 } else {
6396:                     anyhow::bail!("Missing value for --morph-layers");
6397:                 }
6398:             }
6399:             "--morph-width" => {
6400:                 if arg_idx + 1 < args.len() {
6401:                     morph_width = args[arg_idx + 1].parse::<usize>()?;
6402:                     arg_idx += 2;
6403:                 } else {
6404:                     anyhow::bail!("Missing value for --morph-width");
6405:                 }
6406:             }
6407:             "--motif-capacity" => {
6408:                 if arg_idx + 1 < args.len() {
6409:                     motif_capacity = args[arg_idx + 1].parse::<usize>()?;
6410:                     arg_idx += 2;
6411:                 } else {
6412:                     anyhow::bail!("Missing value for --motif-capacity");
6413:                 }
6414:             }
6415:             "--morph-depth" => {
6416:                 if arg_idx + 1 < args.len() {
6417:                     morph_depth_override = Some(args[arg_idx + 1].parse::<usize>()?);
6418:                     arg_idx += 2;
6419:                 } else {
6420:                     anyhow::bail!("Missing value for --morph-depth");
6421:                 }
6422:             }
6423:             "--fresh-world" => {
6424:                 fresh_world = true;
6425:                 arg_idx += 1;
6426:             }
6427:             "--fresh-decoder" => {
6428:                 fresh_decoder = true;
6429:                 fresh_world = true;
6430:                 arg_idx += 1;
6431:             }
6432:             "--fresh-model" | "--fresh" | "-f" => {
6433:                 fresh_model = true;
6434:                 fresh_world = true;
6435:                 arg_idx += 1;
6436:             }
6437:             "--help" | "-h" => {
6438:                 println!(
6439:                     "TITAN v9 Morphogenic Manifold\n\n\
6440: Usage: titan [BASE_DIR] [options]\n\n\
6441:   -b, --base-dir DIR   Output/training root (default /sdcard/Download)\n\
6442:   -d, --duration SEC   Render duration (default 240)\n\
6443:   -t, --threads N      Rayon/Candle CPU threads (default min(device cores, 6))\n\
6444:   -w, --bptt N         Gradient horizon 1..64; tape is memory-capped at 8 (default 16)\n\
6445:       --core-update-every N  Full CA/GRU backward every N tapes (default 4)\n\
6446:   -l, --lr VALUE       Base AdamW learning rate\n\
6447:   -s, --seed N         Seed for a fresh deterministic organism\n\
6448:       --state PATH     World-checkpoint path\n\
6449:       --model PATH     v9 model output/resume path\n\
6450:       --import-model P Import compatible experimental tensors without overwriting source\n\
6451:       --corpus-manifest PATH  Explicit train/development/validation/exclude manifest\n\
6452:       --refresh-corpus-manifest  Add new WAVs while preserving existing family roles\n\
6453:       --rebuild-corpus-manifest  Replace the manifest from current WAVs (creates backup)\n\
6454:       --prune-missing  Remove manifest entries whose WAV files are absent (refresh only)\n\
6455:       --manifest-only  Create, repair, refresh, or rebuild the manifest, then exit\n\
6456:       --run-tag NAME   Isolate output, telemetry, model, and world artifacts\n\
6457:       --spectral-deemphasis  Experimental 2-6 kHz spectral loss weighting\n\
6458:       --prime-width-score   Experimental width-aware prime selection\n\
6459:       --freeze-morph   Hold the checkpoint's current morphic depth for this run\n\
6460:       --morph-layers N Physically construct N append-preserving morph blocks (default 12)\n\
6461:       --morph-width N  Internal morph-block width, 64..4096 (default 512)\n\
6462:       --motif-capacity N  Runtime motif-memory slots, 1..4096 (default 64)\n\
6463:       --morph-depth N  Override the resumed/initially active morph depth\n\
6464:       --max-morph-depth N  Allow adaptive growth only through depth N\n\
6465:       --fresh-world    Reset CA/DSP/memory while retaining compatible weights\n\
6466:       --fresh-decoder  Retain CA/memory weights; reset only audible decoder tensors\n\
6467:   -f, --fresh-model    Reset both learned weights and the world\n\
6468: \nScientific instrumentation (read-only; see --analysis-only --analysis-help):\n\
6469:       --analysis-only  Load frozen v9 checkpoints and write sidecar reports only\n\
6470:       Frozen analysis target feedback uses the legacy unweighted distance.\n\
6471: \nCtrl-C finishes the active chunk, finalizes audio, and saves the organism.\n"
6472:                 );
6473:                 return Ok(());
6474:             }
6475:             _ => {
6476:                 if arg_idx == 1 && !args[arg_idx].starts_with('-') {
6477:                     base_dir = args[arg_idx].clone();
6478:                     arg_idx += 1;
6479:                 } else {
6480:                     println!("Unknown parameter: {}", args[arg_idx]);
6481:                     arg_idx += 1;
6482:                 }
6483:             }
6484:         }
6485:     }
6486:     if refresh_corpus_manifest_requested && rebuild_corpus_manifest_requested {
6487:         anyhow::bail!(
6488:             "--refresh-corpus-manifest and --rebuild-corpus-manifest are mutually exclusive"
6489:         );
6490:     }
6491:     if prune_missing_manifest_entries && !refresh_corpus_manifest_requested {
6492:         anyhow::bail!("--prune-missing requires --refresh-corpus-manifest");
6493:     }
6494:     if experiments.enabled() && manifest_only {
6495:         anyhow::bail!("experimental flags require a training/render run, not --manifest-only");
6496:     }
6497:     let run_metadata_path = if manifest_only {
6498:         artifact_path(
6499:             &base_dir,
6500:             "titan_run_metadata_v9",
6501:             "json",
6502:             run_tag.as_deref(),
6503:         )
6504:     } else {
6505:         validate_experiment_run(
6506:             &base_dir,
6507:             run_tag.as_deref(),
6508:             model_override.as_deref(),
6509:             state_override.as_deref(),
6510:             import_model_override.as_deref(),
6511:             fresh_model,
6512:             experiments,
6513:         )?
6514:     };
6515:     std::fs::create_dir_all(&base_dir)?;
6516:     let wav_dir = format!("{}/OLD_WAVS", base_dir);
6517:     let corpus_manifest_path = corpus_manifest_override
6518:         .unwrap_or_else(|| format!("{}/titan_corpus_manifest_v7.json", base_dir));
6519:     if rebuild_corpus_manifest_requested {
6520:         let (manifest, backup) = rebuild_corpus_manifest(&wav_dir, &corpus_manifest_path)?;

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
