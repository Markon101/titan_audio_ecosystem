# Guaranteed-exposure follow-up (separate from additive A/B/C)

The preregistered additive B arm completed 2,812 chunks without ever selecting
one of its six new families as a training target. That arm is an exposure
failure, so a new, separately labeled test is needed. This follow-up was
specified before inspecting the additive C outcome or any acute-run metric.

Three fresh forks of the exact frozen step-58,107 checkpoint each receive a
single 60-second block (703 chunks) with the same binary, seed 44, 4 threads,
LR 0.00045, BPTT 64/tape 8, core update every tape, active L16, and
measurement-only regime capture at stride 4. All use the same 6 development
and 5 validation files with unchanged roles, families, and bytes. Execution
is sequential in A, C, B order; only one TITAN process runs at once. The sole
changed input is the six-family training ecology:

| Arm | Six training files and families | Added-content interpretation |
| --- | --- | --- |
| acute A | Six existing original train files | Familiar composition control |
| acute C | Matched mild-EQ versions of those six files | Familiar composition with changed timbre |
| acute B | Six genuinely new user-audio families | New composition/family exposure |

The A/C files have exactly matching durations (786.176 seconds in total);
the B files total 818.155 seconds, 4.1% longer. All are 48 kHz stereo 16-bit
PCM. Source, generated, manifest, and held-out hashes are in
`acute_exposure_forks_receipt.json`. The six B files have distinct names,
families, and bytes, with no overlap against the original small corpus. C
variants have no byte overlap with A or held-out files. This is an acute
ecology replacement, not a continuation on the original full 23-family
corpus, and must not be pooled with the additive contrast.

All analysis uses the fixed `regime_calibration.json` from the original A
observer. The exposure gate is at least two distinct sampled training target
families per acute arm in its 703 chunks. Primary descriptive outcomes are
field/GRU/Morphic/audio view trajectory dimensions, persistent regime
confirmation and dwell, full field-state distance from the source checkpoint,
motif additions, health, gradient clipping, and strict fixed-probe changes.
Audio output is retained for listening. New-family effects limited to
decoder/audio views will not be called msfield ecological reorganization.

One seed, three target episodes or so, and different source audio are not
enough for a general capacity or downstream usefulness claim. This test asks
whether genuinely new families have an immediate *measurable* effect after
the additive arm failed to expose the organism to them.
