# Audio research context contract

This is the shared core for every Audio research-team role. Read the selected
source excerpts as evidence. Report factual reconstruction and supported
conclusions, not hidden reasoning. A larger context does not certify an answer.

1. **Decision:** Diagnose harshness and center bias in the latest `v9-long-01`
   exported audio; choose a reversible experiment before changing learned
   behavior. Useful sound and subsequent human use are the practical goals.
2. **Modality:** This is Titan AUDIO, a CPU Rust NCA/GRU/DDSP ecosystem. It is
   neither Titan Text nor an audio-conditioned DiT. No Suno/API experiment is
   authorized or available in this investigation.
3. **Forward inputs:** Fields, recurrent/episodic memory, oscillator phase,
   prior controls, timing, energy, and host synthesis controls. `forward` has
   NO text prompt, target audio, prime, or external reference argument.
4. **Causal order:** Current world -> forward audio -> target sampling/loss ->
   optimizer and later host feedback. Current target cannot change that same
   precomputed chunk; prior targets can affect weights and later ecology.
5. **Prompt direction:** `suno_priming_prompt*.txt` is written AFTER rendering
   and prime extraction. Its descriptive words cannot cause the run. It may
   influence a later external music system, which is outside this experiment.
6. **Clocks:** 4096 samples at 48 kHz per chunk; macro eligibility every four
   chunks; bounded eight-chunk tapes; latest BPTT 64, full core every four
   tapes. Render duration is not wall time or optimizer update count.
7. **Anatomy:** Folded micro/macro NCA, GRU, residual MorphicStack, episodic
   and host motif memories, learned low-rate controls, oscillators, separate
   wavefolders, mid/side stage, bounded pan. Latest active depth is 16.
8. **Optimizer:** AdamW moments are persisted. The three latest WAVs form one
   sequential continuation lineage, not independent seeds or matched arms.
9. **Initialization/resume:** The untagged parent remains separate from the
   tagged run. Exact checkpoint/model/world/moment hashes identify inputs;
   hashes alone establish neither sound quality nor mathematical consistency.
10. **Corpus:** Family roles and generated-name exclusions matter. Two local
    manifest entries have quarantine provenance but non-exclude roles;
    production excludes their filenames. Report declared and candidate roles
    separately; do not infer they entered gradients from manifest roles alone.
11. **Targets:** Streaming family/variant episodes, sampled after forward.
    No inference-time reconstruction target exists. Target mix and online
    learning confound comparisons between elapsed portions of a run.
12. **Losses:** Multiscale magnitude, envelope, band, chroma, pitch, onset,
    recurrence, modulation, low-band, stereo geometry/correlation/balance,
    plus ecology. Read actual weights in selected code. The low-band term
    measures the first log band; it is not our whole 20-200 Hz measurement.
13. **Stereo controls:** Side derives from L-R before a delayed scaled side
    stage. Learned width is bounded 0.05..0.50; global pan +/-0.10. Pan-center
    loss penalizes bounded pan; it does not directly reward correlation=1.
    Increasing side gain cannot synthesize missing side information.
14. **DSP/mastering:** Loss-visible tanh, later gain/saturation/DC blocker,
    and whole-file peak normalization. A sample peak below 0 dBFS does not
    exclude nonlinear saturation or perceptual harshness.
15. **Prime selection:** Approximately 60 seconds selected using field entropy,
    activity health, complexity, stagnation. No listener rating or direct
    stereo/harshness ranking. Selection can amplify a within-run outlier.
16. **Metrics:** The supplied first survey uses MID-channel sampled FFT power
    fractions. Changes in fraction can reflect denominator or stereo changes.
    Measure absolute and total L+R band power before claiming bass recovery.
    Correlation and side/mid energy describe stereo, not musical usefulness.
17. **Observations:** Read the measurement packet. The listener reports
    harsh/centered sound. Neither the PI nor agents have performed controlled
    blinded listening yet; no perceptual improvement is established.
18. **Historical boundaries:** Image smoothing reduced some state high-frequency
    energy without establishing better output/recovery. Renderer proposal
    documents contain unimplemented ideas; they are not available modules.
    Gemini's original Audio E2/E3 diagnostics are superseded: their telemetry
    describes a later 9.56-second continuation, not the 599.98-second WAV.
19. **Inference limits:** Offline waveform filtering tests signal processing.
    It is not a proxy experiment for latent-field diffusion, and neither is
    a trained diffusion/flow decoder. A zero-output identity branch proves
    bypass behavior only. Latent field entropy cannot be recovered from a WAV.
20. **Preservation/budget:** Canonical checkpoints, corpus, existing WAVs, normal
    renderer and training stay intact. An offline test may create copied WAVs
    and sidecar receipts. Use paired intervals and record level-matching gains.
21. **Missing evidence:** Perceptual ratings, independent training replicas,
    pre-side-scale renderer taps, frequency-specific loss gradients, detailed
    target mix attribution, and downstream priming outcomes. Flag missing
    information explicitly; do not invent controls, inputs, or measurements.

The working tree may contain an opt-in spectral-loss and prime-score candidate.
Check its current flags, metadata profile, tests, and actual binary before
interpreting a new run. The eight independent team reviews and synthesis
challenger in `analysis/audio_center_harshness_20260926/team_v2/` used the
earlier immutable packet; they are advisory and do not validate later edits.

Each review must include an object `context_receipt` with these exact factual
fields: `project: "titan_audio_ecosystem"`, `direct_reference_input: false`,
`prompt_is_post_render_output: true`, `waveform_filter_tests_latent_diffusion:
false`, `canonical_mutation_allowed: false`. Give `summary`, `evidence`
(citing source and line numbers), `hypotheses`, `missing_evidence`,
`discriminating_experiment`, and `recommendation`. Label speculation. Keep
different causal mechanisms separable. These receipt facts are a minimum
comprehension gate; the PI independently checks the substantive answer.
