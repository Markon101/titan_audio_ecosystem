# Frozen Audio v9 synthesis assessment, 2026-09-27

The current evidence establishes **learned and state-dependent audio
generation**. It does not yet establish coherent emergent composition,
reference-independent autonomy over long horizons, or transfer of TITAN's
timing into a downstream music model. Those are separate, testable claims.

## Protected object and analysis scope

The analyzed checkpoint is tagged `v9-long-01`, world step 81,483, optimizer
update 3,117, active morphology 16/16, model SHA-256
`0778fc0ab7b27f3d144e666de7a582de5e25d408913ceb1c31453670de99849a`.
The frozen panel (`frozen_panel/`), 64-chunk ablation (`ablation_64/`),
and one-step GRU reach check (`gru_fresh_compare_release/`) all report
unchanged canonical checkpoint bytes. They use one analysis seed 424242,
fixed morphology, frozen weights, and zero optimizer steps. Fast corpus
provenance omitted per-WAV byte hashes. Audio v9 has no direct prompt or
target-audio input in the current forward pass; the frozen full condition
did include later coarse target-error host feedback.

## Measurements

At the 64-chunk endpoint, trained/saved and trained/fresh worlds had mid RMS
0.06684 and 0.06714. The initialized/fresh arm at *matched active depth 16*
had mid RMS 0.01206. Their sampled stereo correlation was approximately
0.508, 0.489, and 0.956 respectively; spectral flatness was about 0.656,
0.613, and 0.161. The trained-fresh and initialized-depth-matched arms began
with equal constructed state, but their first-chunk waveform relative L2 was
0.971. This directly shows a learned-weight effect on the frozen output path.
Large loudness and timbre differences remain confounds for any quality or
temporal-organization interpretation. The initialized native-depth arm is a
different-capacity descriptive control, not the primary comparison.

The 256-chunk trained/saved rollout stayed finite. Its combined state
distance from origin was 0.382 at 64 and 0.688 at 256; mid RMS stayed near
0.067. Approximate nearest prior-state distances 0.102 and 0.064 are
descriptive returns, not a proven cycle or attractor. The report labels its
separability as `differentiated_or_drifting_trajectories`.

The matched 64-chunk clone ablation began with identical state per condition:

| Intervention | Last-sampled waveform relative L2 | Aligned waveform correlation | State relative L2 |
| --- | ---: | ---: | ---: |
| Disable target-error feedback | 0.000 | 1.000 | 0.000 |
| Hold committed GRU state | 0.000 | 1.000 | 0.000 |
| Hold micro NCA next state | 1.191 | 0.289 | 0.222 |
| Zero episodic readout | 0.312 | 0.951 | 0.046 |

The micro NCA is causally audible under this cloned protocol, and the
episodic readout has a smaller audible effect. The target-feedback null
describes 64 frozen chunks only; it says nothing about target influence on
prior training. State distance here includes micro, macro, and recurrent
state, not every host variable.

The GRU hold needed an intervention-reach check. New analysis-only JSON
step fields separately report its proposed and committed update RMS. The
optimized 16-chunk saved-world probe measured **0.0 for both on every
chunk**; the hold had no proposed update to suppress. A separate one-step
control from a fresh world with the same trained weights measured proposed
and committed GRU change **0.026855**. Initialized, depth-matched/fresh
weights gave **0.002107**. Thus the GRU implementation can update, while
the saved carried state is locally fixed for the measured window. The gate
statistics and longer/multiple-start behavior remain unmeasured. This is
not proof of a global attractor or that the GRU was irrelevant during
training. The aborted unoptimized debug attempt has no scientific result;
the release probes completed and preserved canonical bytes.

## Downstream claim boundary

The user hears interesting effects when TITAN audio is used with Suno, and
selected the local B EQ derivative as the most pleasant prime. No retained
TITAN input, exact downstream prompt/mode/settings, and Suno output are
paired for direct measurement. The present evidence cannot identify how
much timbre, motif, or timing reaches a generated song. The input-only
`temporal_input_controls/` packet and `DOWNSTREAM_PROTOCOL.md` are ready
for a future manual test; they are not downstream results.

Two bounded DeepSeek reviews (`deepseek_dynamics.json`,
`deepseek_transfer.json`) returned nonempty final answers with
`finish_reason="stop"`. They independently challenge temporal-organization
and transfer claims. Their suggested rotation and wrong-source arms are
useful controls, but remain unimplemented. One review describes three
no-effect generations as killing the hypothesis; that is too strong at
such a small sample size, so the protocol treats a null pilot as
inconclusive unless its uncertainty is small enough to rule out a useful
effect. Agent agreement is advisory, not evidence.

## Next discriminating work

1. Level-match trained and initialized frozen audio, then compare
   temporal descriptors and blinded listening across multiple analysis
   seeds and at least one earlier checkpoint. Include simple oscillator,
   noise, and persistence baselines before calling structure emergent.
2. Probe the saved-world GRU update gate/candidate geometry, then repeat the
   hold from fresh and saved worlds over longer horizons. Check proposed
   versus committed updates and output distance per chunk.
3. For downstream transfer, first retain several interleaved P (prompt-only)
   and B (selected upload) outputs with exact metadata. If a repeatable
   effect appears, add seam-matched O/S ordering controls and an unrelated
   audio control. Use generation attempts as independent units and blind
   texture/timing ratings; do not infer transfer from one attractive song.
