# Titan / Suno pilot: first readout

Analysis completed October 3, 2026, from your October 2 generations.

Your label correction is applied: **Machinery Ethereal S03.wav is S01**.
The original WAV filename, audio, and your notes are preserved.

## Revealed conditions

| ID | Uploaded reference | Your qualitative preference |
| --- | --- | --- |
| S01 | Shared-phase spectral surrogate | Named in preferred pair |
| S02 | Quiet-cut temporal permutation | Not named in preferred pair |
| S03 | Intact frozen Titan | Named in preferred pair |

You favored both intact Titan and the phase surrogate. This pilot does not
establish an advantage for Titan's original temporal ordering. The phase
surrogate retains spectral, harmonic, and periodic information; it is not a
control that removes every kind of organization. S02 was also your selected
favorite from its own batch, so its omission from the preferred pair does
not mean you disliked it.

## What the files show

Measurements use the same 20–140-second interval in every output:

| ID | Power below 250 Hz | Spectral centroid | Stereo side/mid RMS |
| --- | ---: | ---: | ---: |
| S01 | 59.3% | 663 Hz | 0.438 |
| S02 | 62.8% | 597 Hz | 0.421 |
| S03 | 66.3% | 571 Hz | 0.412 |

S01 is brighter and wider by these descriptors. S03 is more bass-weighted
and about 1 dB quieter than S02 in this interval. These are acoustic
descriptions, not explanations of your preference or ratings of quality.
No exact full-scale samples were found; that does not rule out audible
artifacts. Fixed-anchor checks found no strong literal overlap between the
provided outputs, but cannot rule out shared musical arrangements.

All tracks have local feature continuity. Those measurements describe the
Suno outputs; they cannot establish transfer of Titan's organization.
Aggregate source/output spectral and chroma distances could not identify
each output's own reference reliably.

The spectrograms share frequency and color scales. Their displayed 0–120
seconds correspond to seconds 20–140 of the original outputs:

- [S01 spectrogram](analysis_figures/S01_core_spectrogram.png)
- [S02 spectrogram](analysis_figures/S02_core_spectrogram.png)
- [S03 spectrogram](analysis_figures/S03_core_spectrogram.png)

## Limits and next useful data

There is one generation batch per condition and one selected favorite from
each two-track batch: three provided outputs from six declared candidates.
Selection and random generation variation remain confounded with condition.
There is no prompt-only baseline, repeatability estimate, or numerical rating
panel. This is a promising set of selected cases, not evidence of a systematic
intact-Titan advantage or useful learned parameter maturation.

If the three already-generated unselected siblings are still available,
adding them with their S01/S02/S03 IDs is the cheapest useful follow-up.
It requires no additional generation credits and would expose variability
hidden by favorite selection.

Declared settings: Suno V6 plain, cover mode, audio strength 25%, style 50%,
weirdness 50%, instrumental requested, empty lyrics, no prompt rewrite,
full 21.845-second references. Prompt:

> Rich idm electronica mixed with beautiful synths, ethereal vibes, grandiose cinematics, large soundstage

Three deterministic descriptor tests passed; core metrics reproduced exactly,
and original/snapshot hashes were verified unchanged. No training or new
generation was performed. Full methods, measurements, receipts, and commands
are in the repository's `analysis/suno_pilot_20261002/RESEARCH_NOTE.md`.
