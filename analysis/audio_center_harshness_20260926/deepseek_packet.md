# Bounded Audio v9 output review packet, 2026-09-26

Question: A listener reports the latest v9 continuation as harsh and biased
toward the center. Consider only small, opt-in interventions that preserve the
existing trained checkpoint and legacy renderer. The user is interested in
diffusion, but no diffusion mechanism has been selected. No downstream Suno/API
access is planned. Treat metrics as diagnostics; audible quality requires
listening.

Source: local `titan_audio_ecosystem` at commit `6b9f680`; production model,
training, and renderer code are unchanged in the working tree. Current tagged
lineage `v9-long-01` finished at global step 81,371 / AdamW update 3,116,
active morphic depth 16; last 600-second invocation used 7 threads, BPTT 64,
core update every 4 tapes. The untagged parent ended at step 60,278 / update
2,786. All WAVs here are finalized 48 kHz stereo PCM16. Latest run is stopped.

Measurements from existing 60-second prime WAVs (no resynthesis). Correlation
is centered Pearson L/R; side/mid is 10log10(mean(((L-R)/2)^2) /
mean(((L+R)/2)^2)); balance is 10log10(E_L/E_R). Spectral fractions estimate
mean mid-channel Hann-window FFT power from 64 evenly spaced 8192-sample
windows, normalized over 20-20,000 Hz. They are approximate and not
perceptual scores.

| Prime | L/R corr | side/mid dB | L/R balance dB | 20-200 Hz | 2-6 kHz | 6-20 kHz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| untagged parent | 0.603 | -6.01 | +1.00 | 20.8% | 33.6% | 3.0% |
| tagged first continuation | 0.314 | -2.81 | +0.79 | 21.9% | 32.4% | 3.2% |
| tagged second continuation | 0.900 | -12.42 | +1.21 | 3.0% | 46.4% | 8.3% |
| tagged latest continuation | 0.991 | -23.31 | -0.13 | 2.1% | 47.3% | 6.2% |

The latest full 600-second WAV is not uniformly centered: non-overlapping
60-second intervals have L/R correlation 0.740-0.992; minutes 5-6 have
0.990-0.992 and side/mid about -23 dB. The current 60-second prime is from a
very center-heavy interval. Its selection score uses field entropy, activity
health, structured complexity, and stagnation, not audible width or harshness.

Latest training trace (704 sampled rows, steps 74,340-81,370): median output
stereo correlation 0.963 versus target correlation 0.862; median truthful
width 0.027; decoder width control 0.120; width raw -0.946; global pan 0.0999
at its +0.10 limit; decoder side control 1.0. The trace's `ultrasonic_ratio`
median is 0.0965, but this is a pre-master guardrail, not a harshness score.
The latest generated prompt says `focused center image` and `bright glassy
upper spectrum`. The target mix and loss history may confound attribution.

Important boundaries: Audio v9 does not take target audio into the current
forward pass; target losses affect learning and later host feedback. The
checkpoint set and existing audio files must remain byte-for-byte intact.
Prior Image experiments found fixed diffusion could suppress high-band state
energy without establishing better reconstruction; do not equate smoothing
with quality. Distinguish signal-domain smoothing, latent-field diffusion, and
a learned spectral/diffusion decoder. Prefer a discriminating frozen or sidecar
test before any training fork.
