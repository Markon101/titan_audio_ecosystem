# Manual downstream transfer test

Status: prospective protocol. The user reports interesting subjective effects
when TITAN audio is uploaded to a music model. At present there is no retained
one-to-one mapping from a TITAN WAV, exact prompt/mode/settings, and generated
Suno WAV to evaluate. No current artifact supports a numerical downstream
transfer claim.

## Question and competing explanations

Can a downstream model use information in a specific TITAN clip beyond the
model's own prior and the text prompt? Separate at least three possibilities:

1. **Timbre transfer:** broad spectrum, texture, pitch color, or noise floor
   influences the output, with no transfer of longer timing.
2. **Timing transfer:** onset patterns, phrase order, or recurring motifs in
   the input measurably influence the generated output.
3. **Generic audio effect:** almost any upload, or the text prompt alone,
   produces a similar result.

The selected practical source is
`/sdcard/Download/titan_audio_prime_choices_2026-09-26/titan_prime_selected_B_soft.wav`
(SHA-256 `f8be8ddba43d6a4bdf9b1991d4466c4421c1fba5fbdb27192289e7419f341f01`).
It is the local −3 dB 3.5 kHz EQ derivative preferred by one listener, not
an independently verified downstream input.

## Small manual comparison

Use **one** downstream model version and **one** audio-input mode throughout
the pilot. Keep the text prompt, uploaded duration, settings, and any available
seed fixed. If generation seeds are unavailable, randomize the order of arms
within blocks and save multiple outputs per arm. Predeclare which output is
the first accepted result; retain failures and all rerolls.

| Arm | Input | What it tests |
| --- | --- | --- |
| P | Text prompt only | Model and prompt baseline |
| B | Selected TITAN B WAV | Practical effect of this audio upload |
| O | Same B material, original block order with seam processing | Fair control for editing seams |
| S | Same blocks shuffled in time, same seam processing | Sensitivity to multi-second order beyond local timbre |

Start with a low-cost P/B pilot of several interleaved generations per arm.
Its purpose is to estimate variability and catch a large audio-input effect;
three null-looking outputs do not falsify transfer. If a repeatable signal
appears, run the full P/B/O/S comparison and add two stronger controls:

| Additional arm | Input | What it checks |
| --- | --- | --- |
| W | Unrelated audio of similar duration/level, with its provenance recorded | Generic upload effect versus this TITAN clip |
| R | Cyclic rotation of B's four-second blocks with matching seams | Preserves almost all block adjacencies while changing where the sequence begins |

W and R are protocol proposals; their WAVs have not been made or validated.

O and S are experimental controls. Their spectrum, duration, sample rate,
stereo balance, and RMS need validation before uploading. If S has audible
clicks or obviously broken transitions that O does not, the timing comparison
is invalid. Matching seam positions does not match the *content* of new
adjacencies, so an O/S difference can still reflect transition artifacts.
R helps separate multi-second order from join-local effects. A second
independent TITAN clip can test whether any identified feature transfers
specifically rather than merely signaling "unusual audio."

Record for every attempt: input SHA-256 and file, model version, exact mode,
prompt text, upload trim/start/end, settings, generation date, any seed,
output file and SHA-256, attempt order, and whether the output was retained or
discarded. Preserve all attempts. Use a new output directory per arm and run.

## Measurements and decision rule

Blindly rate generated outputs for overall usefulness, TITAN-like texture,
recognizable rhythmic/motif carryover, and unwanted copying. Pair these with
level-normalized audio descriptors: band-power shape, pitch or chroma
salience, onset-rate distribution, envelope modulation, and recurrence over
multiple lags. Compare B against P for audio-input effect and O against S
for temporal-order sensitivity. The generated song may change tempo,
instrumentation, length, and phase, so exact waveform correlation is not a
primary outcome. Compare each score with a null distribution obtained by
pairing an output with the wrong source clip or shuffled arm.

The independent unit is one downstream generation attempt, not every frame
or beat in a single song. Report every output, median and spread within each
arm, and uncertainty across attempts. A single attractive track is a case
study. A positive result needs repeated, blinded differences that survive
the prompt-only and edited-audio controls. Do not claim mutual information,
causal transfer of timing, or emergent synthesis from descriptor similarity
alone.
Predeclare a primary blind rating for recognizable rhythmic/motif
carryover before generating outputs. Treat spectral similarity and other
automated descriptors as secondary. At least three outputs per arm are an
exploratory start; a null pilot is inconclusive unless uncertainty excludes
a useful effect. For stronger inference, use more attempts, multiple blind
raters, and a second independent TITAN source.

## Current stopping point

The matched input/output mapping is absent. The next useful work is to prepare
validated O/S control WAVs and a blank attempt ledger, then let the user run a
small manually randomized pilot. No direct Suno/API access is required.
