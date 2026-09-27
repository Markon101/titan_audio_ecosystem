# Timing-order input controls

Generated from the selected B prime (SHA-256
`f8be8ddba43d6a4bdf9b1991d4466c4421c1fba5fbdb27192289e7419f341f01`)
with `scripts/audio_temporal_control.py` SHA-256
`b3c67628657f4ebab161abe4ca8f9577877cfef6a78480af81b22f685cc26a0e`.
The source was unchanged before and after export. All files are 48 kHz stereo
PCM16 with the original frame count.

The script shuffled fourteen complete four-second blocks using seed 4242,
retained the final partial block, and applied identical 10 ms fade windows at
the same seam positions in both edited conditions. It selected the exact
permutation and processing gains in `receipt.json`. O and S stereo RMS differ
by 0.000021 dB and stay below the −1 dBFS sample-peak guard.

| Input | L/R correlation | Side/mid dB | 20–200 Hz share | 2–6 kHz share |
| --- | ---: | ---: | ---: | ---: |
| B original exact | 0.9907 | −23.25 | 2.81% | 35.11% |
| O ordered, seam matched | 0.9907 | −23.25 | 2.81% | 35.11% |
| S shuffled four-second blocks | 0.9907 | −23.25 | 2.64% | 35.46% |

Band shares are sampled FFT diagnostics. O versus S tests sensitivity to
multi-second ordering if their seams are not objectionable when listened to.
Within-block timing remains. Any downstream difference could also be caused
by the different transition content at the joins; the seam-matched O control
reduces but does not eliminate that confound. B is the practical original
upload, while O and S are experimental controls. No downstream output exists
for any of these arms yet.
