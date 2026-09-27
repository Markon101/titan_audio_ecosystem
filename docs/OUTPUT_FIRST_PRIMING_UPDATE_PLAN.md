# Output-first update plan for TITAN Audio v9

Status: basic local implementation begun 2026-09-25. Source reviewed at
`6b9f680` on `v7-truthful-resonant-ecology`; the remote branch tip matched the
local commit. No checkpoint, training behavior, renderer, corpus role, or
canonical output was changed.

## Goal and decision rule

Make TITAN's exported audio more useful as an audio reference for downstream
music generation, especially Suno v6 and compatible audio-conditioned models.
The practical endpoint is a better downstream result heard by blinded listeners,
with measurable influence from the TITAN clip and no unacceptable loss of audio
quality or diversity. Internal dynamical scores are diagnostic variables.

"Priming" here means supplying a TITAN WAV as an audio input or reference. A
separate fine-tuning use would need its own dataset, permissions, and evaluation
protocol. Each downstream model must declare its actual audio-input operation
(for example, extend, cover, remix, or reference conditioning); a generic
"audio DiT" label does not establish a shared interface. Suno's September 2026
v6 release describes audio and multiple inputs as supported creation inputs:
<https://suno.com/release-notes> and
<https://help.suno.com/en/articles/13924481>.

## What the source and local artifacts establish

- The v9 forward pass has no target-audio argument. Target audio is sampled
  after forward, contributes training losses, and can affect later host feedback.
  Frozen synthetic proximity or validation loss therefore does not measure
  reconstruction or direct reference-conditioned synthesis.
- The renderer emits 48 kHz stereo audio in 4,096-sample chunks. Its ordinary
  output receives bounded saturation, a DC blocker, whole-file peak
  normalization, and 16-bit WAV encoding. A 60-second prime is cut from that
  output with short fades.
- The prime window is selected from `field_entropy * (0.25 + 0.50 *
  activity_health) + 0.75 * structured_complexity - 0.25 * stagnation`.
  That is an internal-state heuristic; it does not score musical coherence,
  audible defects, conditioning influence, or downstream quality. The chosen
  window and mean score are printed but not placed in a durable candidate
  manifest.
- `suno_priming_prompt*.txt` describes the whole run, not necessarily the
  selected prime. Its style words derive from proxy thresholds; its bracketed
  `Phi`, `Sigma`, `PI-proxy`, and related numbers are research telemetry, not
  reliable musical instructions. Prompt files use stable names and may be
  overwritten by later runs.
- The local manifest has family-disjoint roles, but two entries declare
  `titan_generated_quarantine` while retaining `train` or `validation` roles.
  The production target loader rejects both by filename, so this is not
  evidence they reached gradients or validation. The analysis corpus report
  counts declared roles and places listed files in `scheduler_order` without
  applying that exclusion, so its apparent corpus roles differ from the
  effective scheduler. `validation_is_strict` must be checked before citing
  held-out results. Past human use of validation to choose changes requires a
  newly locked final set for confirmatory claims.
- The local `/sdcard/Download` v9 set includes model, optimizer, world,
  metadata, full WAV, and prime WAV from a 2026-08-20 run ending at step 60,278.
  Its metadata reports 2,786 cumulative optimizer updates and compatible moment
  resume. A direct PCM check of that prime found 59.989 seconds, 48 kHz stereo
  16-bit, peak -1.0 dBFS, stereo correlation 0.603, and zero clipped frames.
  These are historical artifacts and basic signal checks, not a listening or
  downstream quality verdict.
- One frozen analysis sidecar exists for that checkpoint. It recorded 256
  chunks of frozen rollout and canonical non-mutation, but no synthetic
  benchmark or downstream comparison. Its report warns that the binary build
  commit `202872a` differs from the current source commit `6b9f680`. Rebuild
  and identify the binary before relying on new measurements.
- Audio already has a read-only analysis harness, provenance sidecars, typed
  ablations, perturbation controls, and output metrics. Use that infrastructure
  first. Its implementation and numerical tests do not prove sound quality.

## Ordered update

### 0. Freeze an output baseline and repair the evidence gap

1. Inventory and hash the exact v9 model/world/optimizer/morph/metadata,
   corpus manifest, full WAV, prime WAV, and prompt. Record source commit,
   executable build commit, run tag, step, active depth, seed, sample rate,
   channel layout, and output processing in one immutable receipt. Keep legacy
   files byte-for-byte and use a new analysis/output directory.
   Reconcile declared corpus roles with the production loader's effective
   exclusions and repair the read-only analysis provenance view. Preserve the
   original manifest as evidence; make any later role correction separately
   reviewable and apply the same effective-role rule to every corpus summary.
2. Build the current source in a reproducible environment, run locked tests,
   and compare its read-only analysis against the historical receipt. Stop any
   new training decision if checkpoint consistency or binary identity is
   ambiguous. At the initial review `cargo test --locked --offline` could not
   start because `bincode v1.3.3` was absent from the local Cargo cache. The
   locked dependency was subsequently fetched; the full Rust suite passed 77
   tests, and the release build and strict Clippy passed.
3. Audit exported WAVs directly: nonfinite samples, clipping, true peak,
   loudness, silence, DC, ultrasonic/foldback energy, seams, stereo correlation,
   mono compatibility, onset continuity, repetition, and ending quality.
   Compare the final WAV and its extracted prime; include blind listening.
4. Capture the prior "interesting downstream effects" as a small historical
   case table: exact TITAN input hash, prompt, downstream model/mode/settings,
   output hash, date, and a short listening note. Missing settings remain
   explicitly unknown.

Gate: a reproducible baseline packet and a list of audible problems or strengths.
No conclusion about downstream benefit follows from TITAN metrics alone.

### 1. Build a versioned priming package as a sidecar

Implement an opt-in exporter outside the normal training path. It must never
rewrite the canonical v9 checkpoint or replace the existing prime. From a
fixed full WAV or frozen rollout, emit several candidate clips with exact
start/end sample offsets, independent WAV hashes, processing details, and a
manifest tying them to the source receipt. Include the current 60-second
heuristic selection as a control and at least one random, length-matched window.

Score candidate *audio* for hard defects first. Rank remaining windows with
transparent measures of continuity, temporal development, nontrivial but
nonrepeating structure, tonal/noise balance, stereo compatibility, and clip
boundaries. Predeclare weights on a development set. Keep multiple candidates
when their audible roles differ; a single scalar should not erase useful
variation. Do not select on downstream test outputs.

Write three prompt variants per candidate: (a) audio alone with a neutral
instruction, (b) short audible descriptors checked by a listener, and (c) the
legacy telemetry prompt as a control. Bind prompt text to the candidate hash
and prompt version. Put research telemetry in the manifest, not in the default
creative prompt. Record any loudness matching or resampling as a new derived
artifact, retaining the original PCM.

Gate: deterministic re-export from the same receipt yields the same candidate
bytes and manifest; normal training/output/checkpoint hashes remain unchanged.

### 2. Run a small randomized downstream pilot

Use one documented task per downstream system, such as Suno v6 audio upload
followed by Extend or Cover. Freeze each system's model version, mode, text,
settings, upload length, and generation date. Do not assume different systems
consume the reference in the same way. Run randomized blocks of at least three
outputs per condition; increase repeats if variation overwhelms the effect.
If a provider exposes no seed, record that fact and use paired prompt blocks.

| Arm | Audio input | Text input | Purpose |
| --- | --- | --- | --- |
| A | TITAN candidate | neutral | Audio contribution |
| B | none | same neutral text | No-audio baseline |
| C | TITAN candidate | audible-descriptor prompt | Combined practical workflow |
| D | none | same descriptor prompt | Text-only contribution |
| E | legacy prime | same prompt as C | Current exporter control |
| F | length/loudness-matched simple audio control | same prompt as C | Generic-audio control |

Keep the candidate selection set separate from the final evaluation set.
Randomize presentation order and blind listeners to arm and model. Rate
downstream musical quality, usefulness, whether the intended TITAN texture or
motif survived, unwanted copying/repetition, and preference. Also retain
source-to-output audio similarity and diversity measurements as diagnostics;
similarity alone is not success. Report per-output results, variation, and
failures, not just a best example. The primary comparison is C versus D for
practical combined use, with A versus B isolating audio input and C versus E
testing the new exporter.

Gate: adopt the package only if blinded preference and recognizable influence
improve across repeated outputs without worse defects. If the result is mixed,
keep the legacy path and use the pilot to identify the limiting factor.

### 3. Change synthesis or learning only for an observed output deficit

Use frozen trained/fresh-world/initialized controls and typed ablations to
check whether the deficit comes from weights, ecology, renderer, or mastering.
If the problem is aliasing, test one oversampled/low-pass wavefold stage in an
opt-in decoder fork. If the problem is missing transient or texture detail,
prototype the zero-initialized causal spectral residual already queued in
`DECODER_RESEARCH.md`. If the problem is clip selection or prompt wording,
fix the exporter instead of changing the model.

For each fork, record parent hashes and exact tensor migration, keep a matched
legacy control, start from the same world or a declared fresh world, and hold
optimizer/update budgets equal. Verify zero-initialized paths are output
preserving at step zero. Test numerical health and renderer metrics, then
repeat the blinded downstream pilot on a new locked evaluation set. A lower
spectral loss, a smoother state, or a richer manifold is insufficient if the
exported audio and downstream result do not improve.

## Acceptance and handoff

The first implementation slice is the baseline receipt plus sidecar exporter
and blinded evaluation packet. Do not start a long v9 continuation or change
MorphicStack depth to answer this question. Hand back the code diff, exact
commands, artifact hashes, audio examples, individual listening scores,
downstream settings, failed controls, and a decision to keep or reject each
change. Only promote a renderer/training fork after the output test earns it.

## Basic implementation record, 2026-09-25

The opt-in `scripts/prime_package.py` now creates a local sidecar from an
existing full WAV. It exports opening, middle, ending, and seeded windows,
copies an optional legacy prime, hashes inputs and outputs, records basic PCM
checks, writes a neutral prompt and listening notes, and audits declared versus
production-name/role corpus candidates. It deliberately does not rank music or
invoke Suno. `src/analysis/load.rs` adds the same name/role candidate view to
read-only analysis reports while preserving legacy fields. The historical
manifest remains untouched. A later six-clip, level-matched local listening
panel led one listener to select the original latest-run prime for impact and
its −3 dB EQ derivative for softer, preferred audibility. See
`analysis/audio_center_harshness_20260926/OFFLINE_RESULTS.md`. No matched
TITAN-input/Suno-output comparison has yet been retained; downstream
transfer and model-quality claims remain pending.

The local package test checks byte-identical receipts across output directories,
unchanged source bytes, refusal to overwrite an existing package, corpus role
conflict reporting, and rejection of metadata pointing to a different WAV. A
real-data smoke on the historical v9 WAV produced five clips, including the
byte-preserved legacy prime. Those clips have not received blinded listening
or downstream evaluation.
