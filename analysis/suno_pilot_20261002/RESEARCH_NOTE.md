# First selected Suno pilot, 2026-10-02

Generation date: October 2. Analysis and handoff completed October 3, 2026.

## What the blind preference says

The user's completed qualitative note named S03 and S01 as the preferred
pair, without ranking them or filling numerical rating dimensions. The key
was released after freezing the corrected input/output mapping, notes, and
anonymous descriptors. Uploaded input hashes match the original private key.

| ID | Reference condition | Saved output | Named in preferred pair? |
| --- | --- | --- | --- |
| S01 | Shared-phase spectral surrogate | Machinery Ethereal S03.wav | Yes |
| S02 | Quiet-cut temporal permutation | Máquina Ética - S02.wav | Not named |
| S03 | Intact frozen Titan | Máquina Ética S03.wav | Yes |

The first output filename was mislabeled. The user explicitly corrected it
to S01 before identity release; the original filename and bytes were retained.
S02 is also a selected favorite from its own batch, so "not named" does not
mean it was rejected or disliked. No preference scores were fabricated.

There is a favored intact-reference case and a favored phase-surrogate case.
This pilot does not demonstrate an intact-reference advantage. It is
compatible with usefulness of retained spectral/harmonic/periodic content,
but cannot identify that contribution separately from Suno's model and prompt
prior. The phase control preserves global channel spectra and cross-spectrum
before common fades/encoding; it does not remove every form of organization.
The quiet-cut condition changes segment order while preserving interiors,
with measured residual spectral and join differences from the earlier control
validation. A preference contrast is therefore not a pure test of long-range
order even with more samples.

## Declared settings and selection

- Suno V6 plain, cover mode, audio/reference strength 25%, style strength 50%,
  weirdness 50%, no prompt rewrite; seeds were not exposed.
- Exact prompt: "Rich idm electronica mixed with beautiful synths, ethereal
  vibes, grandiose cinematics, large soundstage".
- Instrumental generation was requested; the MD says the lyrics field was
  empty. No independent vocal-presence classification was performed.
- The entire 21.845-second reference was supplied according to the note.
  These settings and origins are user-declared, not independently verified
  Suno request logs.
- One two-track generation batch per input, only one favorite retained from
  each: three provided tracks out of six declared candidate outputs. The
  other three, actual generation order, output IDs, and prompt-only baseline
  are unavailable. No additional generation credits were spent by the agent.

Condition, generation batch, hidden random variation, and favorite selection
are confounded. There is no within-condition variability estimate and no
independent replication. Do not manufacture a condition p-value from seconds,
beats, descriptor windows, or shuffle repetitions.

## Acoustic description

All originals were preserved; descriptors use the same 24-kHz decode and
4096-frame Hann FFT. Whole-track values are in `RESULTS.json`. The table uses
the prespecified common 20–140-second interval, avoiding unequal song lengths
and endings. Bass share is normalized mono spectral power below 250 Hz;
stereo correlation and side/mid ratio are geometry descriptors, not direct
ratings of stereo motion or quality.

| ID | Full duration | Core RMS | Core power below 250 Hz | Core centroid | Core stereo correlation | Core side/mid RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S01 | 234.0 s | 0.1491 | 59.3% | 663 Hz | 0.679 | 0.438 |
| S02 | 217.6 s | 0.1505 | 62.8% | 597 Hz | 0.699 | 0.421 |
| S03 | 212.6 s | 0.1339 | 66.3% | 571 Hz | 0.711 | 0.412 |

S01 is brighter and wider by these descriptors. S03 is more bass-weighted
and has less one-second RMS variation (CV 0.150 versus 0.170/0.172 for
S01/S02). Core S03 is about 1 dB quieter than S02; the original listening was
not documented as loudness matched. These differences do not explain the
user's preference and do not score fullness, cleanliness, structured
weirdness, transitions, aliveness, or vocal interest. Original 48-kHz PCM16
peaks are 0.721/0.618/0.599, with no exact full-scale samples or runs.
That rules out sample railing in these files, not audible artifacts generally.

All tracks exhibit local spectral-feature continuity versus reordering their
one-second feature windows. At one-second lag, observed mean squared change
is 0.375/0.396/0.292 of the shuffle baseline for S01/S02/S03. At 30-second
lag the ratios are 0.938/0.779/0.863, which do not rank the preferred pair
higher. This describes the generated tracks and cannot establish that their
organization came from Titan, or that one reference produced better structure.

Every output's global band/chroma descriptor is closest to S01's input under
the measured distances, not uniquely to its own uploaded input. The inputs
are intentionally similar in aggregate statistics: input-pair band JS
distances range from 0.000039 to 0.001188. The source/output distances are
far larger and have no unique timing alignment. They cannot identify
input-specific transfer, and exact waveform matching was not the objective.

## Overlap and implementation checks

The two Máquina Ética spectrograms look similar, so an anonymous overlap
screen was run before releasing identities. Four fixed four-second anchors
were searched over the other tracks at 3-kHz mono resolution. Cross-track
maximum absolute correlations range from 0.146 to 0.277, at inconsistent
offsets; self controls equal 1.0. There is no strong literal audio-overlap
evidence in those anchors. This screen does not rule out shared arrangement,
musical ideas, or tempo/pitch-modified copies, and it does not independently
verify the user's generation history.

Three deterministic tests verify descriptor non-mutation, JS/correlation
behavior, a known shift/gain match, and smooth-order versus shuffle behavior.
Readonly-helper corrections leave the actual float32-decoded core metrics
unchanged. Input/key hashes, corrected mapping, frozen notes, and anonymous
results are linked in `intake_receipt.json` and `UNBLINDING_RECEIPT.json`.
All WAVs and full user notes stay local in ignored `runs/`; originals in
Downloads are unchanged. Same-scale core spectrograms are at
`runs/structure_v1/S01_core_spectrogram.png` and the corresponding S02/S03 paths;
their time axis starts at the beginning of the 20-second-trimmed core.

## What this resolves and what remains

The selected cases support keeping both the intact source and the phase
variant as artistically promising references according to this listener.
They do not establish a repeatable conditioning advantage, superiority to
prompt-only generation, transfer of the original long-range order, or
beneficial learned parameter maturation. All three inputs derive from the
same mature frozen Titan; this pilot does not compare older/newer weights.

The cheapest useful additional data would be the already-generated,
unselected siblings if retained, with their source IDs recorded. They would
show how much output variability the favorite selection hides, while still
being siblings within one batch per condition. No new generations are needed
to preserve and analyze the current evidence. Larger randomized replication
and a prompt-only comparison would be required for stronger causal claims.

Reproduce from the repository root using a fresh study/output directory;
completed snapshots and reports deliberately refuse overwrite:

```sh
python3 analysis/suno_pilot_20261002/freeze_intake.py
python3 analysis/weight_world_20261002/ingest_suno_outputs.py \
  --intake analysis/suno_pilot_20261002/runs/intake_snapshot \
  --out analysis/suno_pilot_20261002/runs/anonymous_ingestion
python3 analysis/suno_pilot_20261002/describe_pilot.py
python3 analysis/suno_pilot_20261002/check_overlap.py
python3 analysis/suno_pilot_20261002/unblind_pilot.py
python3 analysis/suno_pilot_20261002/test_descriptors.py
```

No training, architecture change, checkpoint mutation, Suno API, or automatic
generation was performed.
