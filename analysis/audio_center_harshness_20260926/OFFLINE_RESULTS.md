# Audio output investigation, 2026-09-26

Status: local measurements and copied-audio listening material. The listener
reports that the latest v9 continuation sounds harsh and centered. No blind
ratings or downstream music-model results have been collected yet. The
canonical v9 checkpoint and existing `/sdcard/Download` WAVs were read only.

## Lineage and corrections

The 599.9787-second WAV `rust_ecosystem_out_v9-long-01_9bcf582d4b13.wav`
has SHA-256 `22091e4f6fceddd7b61e304bf6b9ddb0d03b98596365346f688cfd3591082185`.
Its 60-second prime has SHA-256
`4f8e6a604f5a3f37ec39818f71af6955a0488d4d937491cf1e051162c05ee48d`.
Exact PCM matching places the prime at 354.816–414.805 seconds. The current
tagged model/world/optimizer set is internally consistent at step 81,483 and
optimizer update 3,117, but its retained metadata and 12-row trace describe a
later 9.557-second run. They cannot explain the long WAV's width controls,
target selection, or prime-scoring decisions. See
`diagnostics/CORRECTION.md` and
`multi_arm_sidecar_package/PROVENANCE_CORRECTION.md`.

The long WAV contains other 60-second intervals with more side energy:
60–120 seconds measures correlation 0.783 and side/mid −8.41 dB;
180–240 seconds measures 0.740 and −7.79 dB. The original prime measures
0.991 and −23.31 dB. These intervals are from the same finished WAV, so this
is a selection and within-run comparison, not evidence from independent
training runs.

## Paired copied-audio probes

The `listening_latest/` sidecar applies five waveform heat steps at ν=0.1,
a 2× gain on existing mid/side side signal, and their combination to the
original prime. It holds the model fixed. Measurements below are from raw
floating transforms before the common listening gain. FFT powers are sampled
Hann-window estimates in full-scale-squared units, so they are diagnostics.

| Condition | L/R corr | Side/mid dB | 20–200 Hz power | 2–6 kHz power | 6–12 kHz power |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 0.9908 | −23.31 | 0.002433 | 0.050015 | 0.003488 |
| Waveform heat | 0.9914 | −23.62 | 0.002433 | 0.043841 | 0.001077 |
| Existing side ×2 | 0.9637 | −17.29 | 0.002534 | 0.050648 | 0.003587 |
| Both | 0.9659 | −17.60 | 0.002534 | 0.044390 | 0.001105 |

Waveform heat is an output low-pass operation. In this recording it mostly
reduced energy above 6 kHz and did not open the centered image. Side gain
changed width metrics without adding new side information. Neither result
establishes better musical output.

A separate −3 dB, 3.5 kHz bell EQ was applied to the original prime and the
two wider intervals. The copies in `eq_level_matched/` attempt the original
stereo RMS before a −1 dBFS peak guard. The original prime and 60–120-second
pair matched RMS within 0.001 dB; the 180–240-second EQ was limited to 0.19 dB
below raw by the peak guard. The `blind_eq_panel/` then attenuated each pair to
one common RMS and shuffled the six labels. Its source and output hashes,
gains, and assignment are in `key.json`; open that file only after rating.

At matched RMS on the original prime, the EQ reduced sampled absolute
2–6 kHz L+R power from 0.050015 to 0.037684, while correlation stayed
0.9908→0.9907. The same raw/EQ comparison on 60–120 seconds gave
0.039709→0.029290 and 0.7832→0.7769; on 180–240 seconds it gave
0.044444→0.031528 and 0.7399→0.7354. Bass-band power rose slightly because
of broad level compensation. This is an EQ/gain effect, not newly generated
bass. Perceptual harshness, transients, and usefulness as a priming clip await
blind listening.

## Research-team interpretation and next gate

Eight independent DeepSeek roles and one synthesis challenger completed with
nonempty final answers and a valid causal-context receipt. They received a
102 KB selected-source packet with 16,384 allowed output tokens per role.
`team_v2/` keeps their request/answer provenance. The team is advisory: it
does not establish causation or sound quality. An early small-context review
incorrectly treated the generated prompt as a causal input, so its prompt
explanation was rejected. The larger default context in `research/` now
states the input/output direction and the corrected lineage explicitly.
The synthesis answer says "six" reviews in one sentence although its saved
request and audit include eight independent roles; use the provenance, not
that prose count. The current default team packet is 144 KB of selected
sources with the same 16,384-token answer allowance. Its `team_v4_preflight/`
directory is an offline dry run, not another completed review.

Listen first to `blind_eq_panel_global/LISTEN.txt` with `listen_A.wav` through
`listen_F.wav`, recording harshness, stereo placement, transient detail, and
reference usefulness before reading `key.json`. Separately compare
`listening_latest/listen_A.wav` through `listen_D.wav` for waveform heat and
existing-side effects. Keep source interval and audio processing as distinct
factors. A listener preference across repeated trials can justify a small
tagged training experiment; current metrics alone cannot.
The first `blind_eq_panel/` matched each raw/EQ pair but left about 1 dB RMS
between source intervals. It is superseded for cross-interval preference by
`blind_eq_panel_global/`, which matches all six WAVs to one stereo RMS target
(measured spread below 0.01 dB) and keeps the −1 dBFS sample-peak guard.
The six anonymous WAVs and `LISTEN.txt` were copied with matching SHA-256
hashes to `/sdcard/Download/titan_audio_blind_eq_2026-09-26/`; `key.json`
remains only in the workspace until ratings are complete.

The WAV-only archives `blind_eq_panel_global/BLIND_EQ_GLOBAL_LISTENING.zip` and
`listening_latest/WAVEFORM_PROBE_LISTENING.zip` contain the instructions and
anonymous clips without an answer key.

The rebuilt default passed a separate, byte-identical one-chunk comparison
against the saved pre-Gemini release binary. The opt-in flags also passed
isolated first-run, tagged-resume, mismatch-rejection, and parent-preservation
checks. See `EXPERIMENT_GATE.md` and `legacy_oracle_report.json` for their
bounded evidence. These checks validate implementation behavior, not an
improvement in sound.

## First listener preference

On 2026-09-26 the user reported leaning toward blinded **E** as the strongest
of the six globally RMS-matched clips. Only after that choice was recorded,
`blind_eq_panel_global/key.json` identified E as the original, unfiltered
prime from 354.816–414.805 seconds. Its direct paired EQ control is B, made
from that same source interval with a −3 dB bell at 3.5 kHz. The user then
called B a very close second, if anything slightly softer to listen to.
This is one listener's preliminary strength/comfort comparison, not proof
that either clip is best for downstream priming.

This feedback lowers the priority of promoting the −3 dB EQ, a wider interval,
or waveform smoothing as a replacement default. Preserve the original prime
as the practical control. A new −1.5 dB bell version of the *same interval*
is in `prime_choices/M_eq_minus1p5_untested.wav`, beside byte-preserved E and
the −3 dB B version. All three stereo RMS values match to better than 0.001 dB.
The files and provenance receipt were copied to
`/sdcard/Download/titan_audio_prime_choices_2026-09-26/`. M remains untested;
its perceived effect and all downstream usefulness await listening. The
opt-in Rust flags remain untrained hypotheses.

The user subsequently selected **B** as the softest and nicest audibly.
`prime_choices/titan_prime_selected_B_soft.wav` is a byte-identical copy of
the rated B file, with `SELECTED.txt` and `selected.receipt.json` recording its
processing and single-listener selection. All three were copied with verified
hashes to `/sdcard/Download/titan_audio_prime_choices_2026-09-26/`. This
selects a practical local audio derivative. It does not establish a downstream
priming benefit or justify changing the renderer's default, the training loss,
or the prime-window selector.
