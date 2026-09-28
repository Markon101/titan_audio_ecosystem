# Small-corpus v9 continuation, 2026-09-28

## Run and integrity

The user selected the newer small-corpus lineage. We imported
`v9-newcorpus-retain-01` into a distinct `v9-newcorpus-retain-02` tag and ran
the normal v9 training path for 600 requested audio seconds. The exact command,
parent checkpoint and corpus hashes, executable hash, and child artifact hashes
are in `run_receipt.json`. Neither experimental loss weighting nor experimental
prime selection was enabled. The executable reported build
`6b9f680f1241-dirty`; its SHA-256 is
`e7b96f94c608ce6e00c8e777d89dfce241e6ce93b389ac9e6f96708a675023d2`.
The command requested seven threads, but the runtime had four available and
used four. This run should not be treated as bitwise comparable to an earlier
seven-thread run.

All 200 model tensors and 200 AdamW moment pairs loaded exactly. The world
resumed at step 144,968 with 4,108 cumulative optimizer updates. The process
exited successfully after 7,031 chunks, 599.979 rendered seconds, and 110 new
optimizer updates. Final step is 151,999 with 4,218 cumulative updates. A
read-only postflight reported a consistent model/world/optimizer set at that
step. The four parent checkpoint hashes still matched the pre-run receipt
during training. The postflight warned that its fast corpus provenance omits
per-WAV hashes and that the binary's embedded build commit predates the current
source commit. Its full report remains in the local Termux tmp directory;
`run_receipt.json` retains the relevant result without enumerating unrelated
files in Downloads.

An earlier read-only preflight against `v9-long-cont-02` loaded a consistent
world/optimizer at step 131,072, but its final broad input-stability check
failed after three unrelated files in Downloads reported changed modification
times. This did not implicate that checkpoint. The selected small-corpus
postflight completed successfully.

## Audio and listening

The child full WAV is 48 kHz stereo PCM16, 599.979 seconds, with zero clipped
frames and a -1.00 dBFS peak. Its stereo correlation is 0.917 and its side to
mid energy ratio is -12.54 dB. These are signal descriptions, not perceptual
scores. See `full_audio_checks.json`.

| Selected prime | Source interval | Stereo correlation | Side/mid dB | 2–6 kHz energy |
| --- | ---: | ---: | ---: | ---: |
| Parent | 584.53–644.52 s | 0.942 | -14.19 | 13.38% |
| Child | 54.87–114.86 s | 0.915 | -12.40 | 13.81% |

The source intervals were found by matching exact PCM anchors from each prime
to its full WAV. Two copies were matched to effectively identical RMS and put
in `/sdcard/Download/TITAN_v9_newcorpus_AB_20260928/`. The user listened
blind and selected **A** as less harsh and more useful. The key shows A was the
parent; B was the child. No stereo or transient notes were supplied. See
`audio_metrics.json` and `blind_ab_key.json` for identities, gains, and hashes.

The child prime is modestly wider by these simple measures, but it was not the
listener's preferred prime. Because the exporter selected very different
intervals from runs with different world states, this one comparison cannot
assign the preference to learned weights, interval selection, or trajectory.
Keep the parent prime as the practical reference for now. The child checkpoint
is preserved for a fixed-interval or same-world frozen comparison. No matched
TITAN-upload/Suno-output pairs are available here, so downstream transfer and
emergent composition remain unmeasured.
