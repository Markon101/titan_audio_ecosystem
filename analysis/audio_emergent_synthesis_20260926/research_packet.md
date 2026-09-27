# Titan Audio synthesis and downstream transfer review packet

Decision: What do the present frozen and saved-output measurements support
about learned audio synthesis and any stronger claim of emergent temporal
organization? What is the smallest manual downstream experiment that can
separate timbre influence from timing/order transfer through a music model?
Do not treat specialist opinion as empirical evidence. The user has no
retained one-to-one TITAN-upload/Suno-output pairs for this analysis.

Source: `titan_audio_ecosystem` commit `6b9f680`, dirty opt-in research branch.
The legacy normal output was byte-matched against the earlier binary on an
isolated synthetic one-chunk test. Current frozen checkpoint: tagged v9 model,
world step 81,483, optimizer update 3,117, active depth 16/16, model SHA-256
`0778fc0ab7b27f3d144e666de7a582de5e25d408913ceb1c31453670de99849a`.
Optimizer step matches world. Analysis uses weights frozen, zero backward and
optimizer steps, one analysis seed 424242, fixed morphology, deterministic
common exogenous forcing. It branches before normal writers and recorded
canonical checkpoint byte nonmutation. Fast provenance skipped corpus WAV
byte hashes, so corpus identity is incomplete. These are single-checkpoint
diagnostics, not seed replications.

Audio v9 model.forward takes fields, recurrent/episodic state, phases,
controls and time. It does not take target audio or text. Target audio is
sampled after current forward during training. The generated prompt is
written after output. Frozen analysis may apply later coarse target-error
feedback unless explicitly disabled. Source filenames with quarantine
provenance were not enough to infer gradient use; the production name filter
excluded two mismarked local manifest entries.

Frozen 256-chunk full rollout from the saved world (21.85 s): at horizon 64
the sampled audio interval had mid RMS 0.06684, stereo correlation 0.508,
onset strength 0.00939; at horizon 256 mid RMS 0.06735, correlation 0.433,
onset strength 0.00926. Combined state distance from origin rose 0.382 to
0.688. Near-recurrence distances 0.102 at 64 and 0.064 at 256 were labeled
approximate candidates; neither proves a cycle or stable attractor. The
rollout was target-error-feedback conditioned. The output summary is
`analysis/audio_emergent_synthesis_20260926/frozen_panel/analysis_report.json`.

Four 64-chunk matched controls used the same active depth and deterministic
analysis seed: trained/saved world, trained/fresh world, initialized/fresh
world at depth 16, initialized/fresh world at native depth 1. At the last
sampled window, trained saved/fresh mid RMS were 0.06684/0.06714; initialized
depth-matched/native were 0.01206/0.01159. Trained/fresh versus initialized
depth-matched began with equal constructed state and differed from the first
chunk (audio waveform relative L2 0.971 at offset 1, 1.007 at offset 64;
last-sample spectral distance 4.507). This establishes a large learned
weight effect on a frozen output path, but the initialized arm is much
quieter, so amplitude can dominate waveform metrics. It does not establish
coherent music or useful temporal organization. Carried-world comparison
also differed (state origin distance 2.061); its causal interpretation is
limited by different starting state. The separability label was
`differentiated_or_drifting_trajectories`, not concepts or attractors.

A separate 64-chunk clone ablation with baseline plus four interventions
reported: target-error-feedback disabled, zero audio/state distance at
sampled offset 64; GRU state commit held, also zero measured audio/state
distance; micro NCA next-state hold, waveform relative L2 1.191 and aligned
correlation 0.289 with combined state distance 0.222; episodic readout zero,
waveform relative L2 0.312 and aligned correlation 0.951 with state distance
0.046. All started with identical state. The full baseline did sample
nonempty target errors. The zero target-feedback result is a 64-chunk output
null, not proof targets never matter during training or at longer horizons.
The GRU null needs validation that its proposed update and committed update
actually differ under the intervention; an analysis-only diagnostic has been
added, with optimized rerun pending. `gru_hold` is applied after the forward
chunk, so any sound effect can first appear later. A zero output difference
cannot by itself prove the recurrent path is unused. See
`analysis/audio_emergent_synthesis_20260926/ablation_64/analysis_report.json`.

The user subjectively hears interesting TITAN effects after uploading to
Suno, but reports no currently mappable input/output pairs. A preferred
local input has been selected by one listener: a −3 dB 3.5 kHz bell EQ of a
59.989-second v9 prime (SHA-256
`f8be8ddba43d6a4bdf9b1991d4466c4421c1fba5fbdb27192289e7419f341f01`).
That preference is about audibility, not transfer. A 600-second online-trained
WAV from the prior continuation contains more and less stereo-separated
intervals; its retained metadata/trace were overwritten by a later 9.56 s
run, so no causal width-head/target attribution should be made from them.

For a future manual downstream test, P=text prompt only, B=selected upload,
O=same B audio with identical seam windows in original block order, and
S=the same fourteen four-second blocks shuffled while retaining a final
partial block, identical seam positions and matched stereo RMS. O/S have
59.989 s duration, correlation 0.9907, side/mid −23.25 dB, RMS spread
0.000021 dB; sampled 2–6 kHz L+R fraction 35.11%/35.46%. The seam/control
pack is under `analysis/audio_emergent_synthesis_20260926/temporal_input_controls/`.
This preserves short-term timbre and within-block timing while disrupting
multi-second order, but changed joins remain a confound. Future outputs must
record exact model/mode/prompt/upload settings, all attempts, seed if
available, source and output hashes. Compare repeated outputs and blind human
ratings, plus level-normalized timbre, modulation/onset, and recurrence
features against wrong-source or shuffled nulls. See `DOWNSTREAM_PROTOCOL.md`.

Please separate MEASUREMENT, INFERENCE, HYPOTHESIS, and SPECULATION. Name a
cheap falsifier and a stronger confirmatory experiment. Do not claim
autonomous emergence, causal temporal transfer, mutual information, or a
downstream benefit from a single WAV or proxy score.
