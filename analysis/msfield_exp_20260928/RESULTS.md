# First v10-msfield-exp experiment, 2026-09-28

## Provenance and controls

This experiment lives on `experiment/v10-msfield-exp`; the canonical v9 branch
and mature v9 checkpoint sets were not modified. `plan.json` records the exact
pilot executable hash, small-corpus manifest hash, all 41 WAV hashes, seed,
and shared training settings. The user confirmed that the short experiment
should read those WAVs from `/sdcard/Download/OLD_WAVS`. The declared
`/sdcard/Download/Titan_Audio_Corpus_SML` directory was empty when checked.
Some WAVs were reportedly transcoded from 320 kbps Opus to PCM; no source
codec was inferred from WAV bytes.

The legacy regression oracle compared the pre-change release binary with the
experimental-branch release binary on the same synthetic one-chunk fixture.
`legacy_oracle.json` reports byte-identical full and prime audio and world
payloads, the same steps, and model/optimizer tensors within the existing
floating-point tolerance. It is a one-chunk gate, not a proof of all possible
long-run equivalence. The final release build passed the oracle again after
the telemetry and prompt cleanups. Final gates: 89 Rust tests, 31 Python
tests, and offline Clippy passed without warnings.

The v10 synthetic smoke trained one chunk, wrote a `TITANM10` world, and
resumed one more chunk with 229 exact model tensors and 229 exact optimizer
moment pairs. Its metadata advanced from step 1 to 2 with AdamW resumed.
Ordinary v9 rejected explicit v10 model and world paths before changing the
source files. Unit tests establish finite forward/backward, deterministic
first-step behavior, coarse-to-meso-to-fine reach, and a coarse intervention
reaching rendered audio. These are implementation and reachability results,
not learned musical-quality claims.

## Thirty-second small-corpus comparison

Both runs used seed 42, the same 41-file manifest and WAV directory, four
threads, 16 physical Morphic blocks × 512 width, active depth 10 frozen,
`--lr 0.0005`, `--bptt 8`, `--core-update-every 4`, and 351 chunks. The v10
fresh start copied 180 exactly matching non-substrate tensors from the same
deterministic v9 initialization. Neither arm imported a mature v9 world.
The outputs are **training trajectories** with 44 optimizer updates each.
`short_run_metrics.json` retains the run metadata, sampled target schedules,
and descriptive signal metrics.

| Observation | Fresh v9 legacy | Fresh v10 msfield |
| --- | ---: | ---: |
| Total parameters | 16,988,839 | 16,826,171 |
| Substrate parameters | 35,712 CA + 295,488 field bridge | 168,532 msfield |
| Rendered audio | 29.952 s | 29.952 s |
| Wall time | 120.9 s | 109.0 s |
| Steps per second | 2.90 | 3.22 |
| Cumulative optimizer updates in run | 44 | 44 |
| Median sampled micro movement | 0.00035 | 0.01544 |
| Stereo correlation of full WAV | 0.977 | 0.985 |
| Side/mid energy | −15.91 dB | −19.25 dB |
| 2–6 kHz energy fraction | 1.9% | 24.8% |
| Clipped frames | 0 | 0 |

The v10 forward was slower per chunk (105.7 versus 89.0 ms), but its measured
backward phase was shorter (164.5 versus 204.4 ms); total time was about 10%
lower in this sequential phone run. This is a local performance observation,
not a general speed ranking. The two sampled target schedules match through
step 250 and first differ at sampled step 260. Thus later target-based losses
and feedback are not a controlled substrate-only comparison.

The v10-specific trace has 36 rows. Fine, meso, and coarse RMS ended at
0.548, 0.320, and 0.060, respectively. Their median per-step displacement
was 0.0186, 0.0077, and 0.00072. Coarse RMS grew from 0.0012 at the first
sample; it did not remain zero. Fine/meso spatial cosine rose from about
−0.007 to 0.291, but the shared renormalization loss already encourages
fine/meso agreement. Different displacement rates are partly built into the
fixed `dt` values. These observations establish active bounded scales, not
metastability, emergent timing, or useful downstream information.

## Output-first listening

The raw v10 render has more upper-mid energy and a narrower measured stereo
image than the fresh v9 control. A separate −3 dB, 3.5 kHz, Q=1 EQ sidecar
reduced v10's 2–6 kHz energy fraction from 24.8% to 18.5% without changing
its timing. `audio_candidate_panel.py` made a three-way anonymous package of
the fresh v9 render, raw v10 render, and EQ v10 render, matched to effectively
the same RMS. Its public listening files are in
`/sdcard/Download/TITAN_msfield_short_panel_20260928/`; the private key and
manual Suno attempt template are in `candidate_panel_private/`.

The user blind-preferred **C**, which the key identifies as the EQ v10
render. The user said this softer version kept the timing and texture of the
raw v10 clip. This is one listener and one seed. It supports using that EQ
derivative as a candidate upload, not changing the learned substrate or
claiming downstream transfer.

For the earlier v9 parent/child confound, `matched_eval_package.py` rendered
64 frozen chunks from the `retain-01` world with each v9 model. Both arms had
zero optimizer steps, identical 64 target file/frame positions, the same
5.46-second render length, and 41 verified corpus WAV hashes. `matched_v9_key.json`
records the exact identities and level matching. Under that common start,
`retain-01` had stereo correlation 0.823 and side/mid −8.93 dB; `retain-02`
had 0.937 and −13.83 dB. Both had about 13.9% 2–6 kHz energy. The user
first heard the blind pair as **about equal**, then on another listen
tentatively leaned **A**. The private key identifies A as `retain-02`. This
short conditional rollout has a measured narrowing and a tentative listener
preference for the later weights; neither establishes a general quality gain.
Different model weights can still make the worlds diverge after the common
start. The earlier selected-prime preference for `retain-01` came from a
different, much later source interval, so the two listening observations need
not agree.

## Decision and next experiment

The first msfield is numerically viable, differentiable, checkpoint-safe,
parameter-matched, and practical enough for short CPU experiments. The
30-second output is bright and center-heavy; the EQ sidecar improved one
listener's preference without removing their perceived timing/texture.
There is not enough evidence for an hours-long msfield run. One seed,
training-time audio, partial target-schedule divergence, and no retained
TITAN-upload/Suno-output pairs cannot establish downstream usefulness or
emergent composition.

The highest-information next practical test is a manual, repeated downstream
pilot using the anonymous A/B/C WAVs already in the sdcard panel. Hold Suno
model/version, custom-model mode, prompt, style, weirdness, and upload trim
fixed; interleave source order; record every accepted and rejected generation
in `candidate_panel_private/downstream_receipt_template.json`. In particular,
compare raw v10 with EQ v10 to test whether the timing/texture survives when
the harsh band is reduced. No direct Suno API is required.

For a second compute check, run the fresh seed-43 command in
`docs/MSFIELD_EXP.md` under a new tag, with a matching fresh v9 control and
full target-schedule comparison. A v10 frozen common-state and coarse-hold
evaluation path remains unimplemented; the unit intervention only shows
reachability. Future discrete/hybrid and delay/ray substrates remain separate
branches.

The exact run artifacts and executable were copied byte-for-byte to
`/sdcard/Download/TITAN_v10_msfield_exp_20260928/`; `copy_receipt.json`
records every verified copy. `panel_copy_receipt.json` verifies the public
blind-panel copies. The original metadata paths still point to Termux tmp,
so use the copy receipt when interpreting those relocated checkpoints.
The separate `msfield_next` copy was then resumed by the final release binary
for one chunk, step 351→352, with 229 exact model tensors and 229 exact
optimizer moment pairs; the original `msfield/` hashes remained unchanged.
`next_fork_receipt.json` records that verification. The exact bounded
continuation command is in `docs/MSFIELD_EXP.md`.
