# TITAN Audio v10-msfield-exp

This is an experimental sibling of the v9 organism. The default
`--substrate legacy` constructs the original folded micro/macro CA and keeps
its original field update and checkpoint format. `--substrate msfield` replaces
only field evolution. The GRU, MorphicStack, temporal decoder, direct spatial
scan synthesizer, losses, corpus policy, ecological forcing, motifs, audio
mastering, and prime export remain shared. No v9 checkpoint is imported into
the incompatible field state.

## State and update

The new persistent state has 64-channel fine 64×64, meso 32×32, and coarse
16×16 fields. The fine and meso tensors retain the shapes consumed by the
existing decoder and direct scan path. The coarse field reaches audio through
learned coarse-to-meso and meso-to-fine exchange. All three scales update each
chunk, with fixed residual time factors `1`, `0.5`, and `0.25`.

At each scale, a bounded local rate combines:

- signed, upwind-like transport from the four adjacent cells, with learned
  velocity bounded to ±0.10 and the existing Klein horizontal seam;
- a four-neighbor diffusion term with a learned coefficient in `[0, 0.08]`;
- a bounded pointwise reaction and recurrent-memory injection;
- gated, learned exchange in both directions between adjacent scales,
  bounded to 0.025 per connection;
- a 0.02 state leak.

The next state is `clamp(x + dt × rate, -1, 1)`. This is differentiable almost
everywhere and numerically bounded; it is not a literal PDE solver or proof of
contraction. Unlike v9, it does not invoke `NeuralCAFolded3D`. Existing host
radiation, shear, kicks, and homeostatic clamps still act after the field
forward. They are shared infrastructure and can influence observed motion.

The v10-specific trace records RMS energy and step displacement by scale,
mean learned-rule velocity and diffusion, hidden-state displacement, and
two cross-scale spatial cosines. Cosine is a descriptive coherence measure,
not information flow. Different movement rates are partly imposed by the
fixed `dt` factors and do not establish learned multiscale organization.

## Isolation and parameter budget

First runs require `--fresh-model --substrate msfield` and an unused
`--run-tag v10-msfield-NAME`. A continuation must reuse the exact tag without
`--fresh-model`. The model, world, optimizer, morph sidecar, and metadata use
distinct `v10_msfield` stems. The v10 world has `TITANM10` magic, a versioned
wrapper around the shared host state, and an additional coarse field. The
model contains `model.msfield.schema_marker`. Ordinary v9 rejects either v10
model or world before loading or overwriting it. V10 requires every saved model
tensor and AdamW moment to reload exactly; it does not accept a v9 import.

For a fresh seed, v10 deterministically constructs a temporary v9 model and
copies 180 same-named, same-shaped non-substrate tensors into the v10 model.
This matches the initial GRU, MorphicStack, decoder, renderer, and auxiliary
weights exactly. A common seed alone would not do this because the normal
initializer draws over sorted tensor names. Fine and meso initial states and
runtime seed are the same; subsequent worlds and target schedules can diverge.

At 16 physical Morphic blocks × 512 width, mature v9 has 16,988,839
parameters and v10 has 16,826,171, **0.96% fewer**. V10 groups are:

| Group | Parameters |
| --- | ---: |
| Msfield substrate | 168,532 |
| Temporal decoder | 7,035,084 |
| Renderer and synthesis heads | 70,812 |
| GRU | 986,112 |
| Morphic recurrent stack | 8,413,184 |
| Host auxiliary | 152,447 |

The v9 spatial CA rules have 35,712 parameters and their recurrent-to-field
bridge has 295,488. Those v9-specific tensors are absent from v10. Parameter
matching is close overall, though effective active-field capacity and update
mechanics still differ.

## Corpus and first command

`--corpus-dir PATH` is available in ordinary training and frozen v9 analysis.
It defaults to `BASE_DIR/OLD_WAVS`, preserving existing commands. On this
device the declared `/sdcard/Download/Titan_Audio_Corpus_SML` directory was
empty when checked. The small manifest's 41 WAVs were available under
`/sdcard/Download/OLD_WAVS`, which the user selected for this pilot.

An isolated fresh 30-second run can be reproduced with a **new, unused** tag
and base directory:

```sh
./target/release/titan \
  --base-dir /sdcard/Download/TITAN_v10_msfield_seed43 \
  --corpus-dir /sdcard/Download/OLD_WAVS \
  --corpus-manifest /sdcard/Download/titan_corpus_manifest_v7_sml.json \
  --substrate msfield --run-tag v10-msfield-seed43 --fresh-model \
  --duration 30 --threads 4 --seed 43 --lr 0.0005 \
  --bptt 8 --core-update-every 4 \
  --morph-layers 16 --morph-width 512 --morph-depth 10 \
  --max-morph-depth 10 --freeze-morph --motif-capacity 512
```

Do not use `--import-model` to seed v10 from mature v9. To continue a v10 tag,
use the same base directory and settings and omit `--fresh-model` and
`--morph-depth`. The exact completed seed-42 pilot and its executable are
preserved in `/sdcard/Download/TITAN_v10_msfield_exp_20260928/`. Its copied
metadata still names the original Termux tmp paths; `analysis/msfield_exp_20260928/copy_receipt.json`
records byte-identical relocation.

A separate writable copy, `msfield_next`, preserves the completed short-run
baseline in `msfield/`. The final release binary verified one exact continuation
chunk on that copy, advancing it from step 351 to 352 with all 229 model
tensors and AdamW moment pairs resumed. To continue the **new-field** lineage
for one bounded 120-second block, run:

```sh
./target/release/titan \
  --base-dir /sdcard/Download/TITAN_v10_msfield_exp_20260928/msfield_next \
  --corpus-dir /sdcard/Download/OLD_WAVS \
  --corpus-manifest /sdcard/Download/titan_corpus_manifest_v7_sml.json \
  --substrate msfield --run-tag v10-msfield-real01 \
  --duration 120 --threads 4 --seed 42 --lr 0.0005 \
  --bptt 8 --core-update-every 4 \
  --morph-layers 16 --morph-width 512 --max-morph-depth 10 \
  --freeze-morph --motif-capacity 512
```

This continues the short v10 pilot, **not** the mature v9 world or learned
v9 checkpoint. Keep this tag and omit `--fresh-model`, `--fresh-world`,
`--morph-depth`, and `--import-model` on subsequent continuations. Assess the
new audio and traces before increasing the duration. The first run's executable
hash and the final verified continuation hash are in the research receipts;
bitwise reproduction requires the same binary and thread count.

## Evidence boundary

The first seed-42 pilot compiled, passed finite forward/backward and seed
repeat tests, saved and exactly resumed model/world/AdamW state, rendered
29.952 seconds, and had no clipped frames. A one-chunk before/after oracle
found byte-identical legacy audio and world payloads with model/optimizer
values within the established floating-point tolerance. The detailed
measurements and listening result are in
`analysis/msfield_exp_20260928/RESULTS.md`.

The standalone audio comparison is an unmatched training trajectory. The
sampled target schedule diverged after step 250. It cannot isolate substrate
quality, and no paired TITAN-upload/Suno-output results exist yet. The
frozen v9 same-world harness in `docs/MATCHED_AUDIO_EVAL.md` is the closer
control for two v9 checkpoints; it does not claim to compare v9 and v10 worlds.

Later branches, outside this experiment's scope: a discrete or hybrid field;
a propagation/delay/ray substrate; and a continuous-discrete hybrid. None is
implemented here.
