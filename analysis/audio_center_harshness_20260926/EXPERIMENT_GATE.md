# Legacy and experimental path gate, 2026-09-26

This gate uses isolated synthetic 48 kHz stereo WAV training roots under
`/data/data/com.termux/files/usr/tmp`. No canonical `/sdcard/Download` model,
optimizer, world, corpus, or audio file was written by these checks.

## Default path

The pre-Gemini release binary was saved with SHA-256
`7114d3cf37fe49cf4cfd8ab10f026e04c4802a04a3c6901dec03bbd0c7d9e80b`.
The rebuilt opt-in binary's hash and exact file comparisons are in
`legacy_oracle_report.json`. Each binary ran one fresh 4096-sample chunk with
seed 4242, one CPU thread, BPTT 1, full-core update every tape, two physical
morph layers, and the same synthetic source WAV. The output roots were
separate.

The final rebuilt default produced **byte-identical** full WAV, prime WAV,
model tensors, optimizer tensors, and serialized world payload. Both ended at
world step 1 and optimizer update 1. The first opt-in build had shown 20
model elements beyond the numerical tolerance despite identical WAV bytes;
the difference traced to a changed floating-point grouping in spectral
projector construction. Restoring the original expression and constructing
the weighting tensor only for the explicit experiment removed the difference.
The `compare_audio_oracle.py` script records exact hashes, tensor differences,
and the check definition. This one-chunk synthetic gate establishes local
default-path equivalence for that setup, not long-run training quality.

## Opt-in behavior

`--spectral-deemphasis` and `--prime-width-score` are explicit experiment
flags. They require a new `--run-tag` and exactly one of `--import-model` or
`--fresh-model` on the first run. Subsequent tagged continuations require the
same profile and matching tagged checkpoint files. The profile is recorded in
run metadata. Frozen scientific analysis retains its legacy target-feedback
distance and must not be used to infer the experimental objective's feedback.

An isolated first run with both flags and `--fresh-model` saved tagged step 1.
A request omitting `--prime-width-score` failed with a profile-mismatch error
before modifying the tagged checkpoint, metadata, or corpus hashes. The
matching continuation restored AdamW moments and advanced from step 1 to 2.
Another isolated run imported a synthetic parent model/world/optimizer,
enabled spectral de-emphasis, and saved a new tagged step-2 checkpoint; all
three parent file hashes remained unchanged. A request for experimental
training without `--run-tag` failed before creating its base directory.

No real-corpus experimental training has been run, and no audible benefit from
either flag is established. The spectral weight is symmetric in prediction
error: lowering the 2–6 kHz penalty does not imply less 2–6 kHz output energy.
The prime selector can only select among sound already rendered.
