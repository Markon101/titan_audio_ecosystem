# Fixed-schedule mature v10 corpus exposure, 2026-10-01

## Decision question and controls

The short six-family test delivered genuinely new audio and raised update
pressure, but did not increase persistent regime discovery. This follow-up
asked whether **longer, exactly matched exogenous target episodes** would
change that result. It did not test an attractor-escape controller.

Six independent **run forks** were copied from the same read-only step-58,107
v10 L16 model, world, AdamW, and Morphic checkpoint. They use one mature
model/world, two fixed **schedule seeds** (`20261002`, `20261003`), and three
audio arms per schedule: familiar originals A, mild EQ variants C, and new
families B. Schedule seeds are not independent model seeds. Each arm ran
1,024 chunks, requested BPTT 64, used a four-chunk autograd tape for device
memory, updated the core every tape, requested four threads, used LR
0.00045, and accumulated 16 AdamW updates. The schedule and alias-corpus
builder, SHA-256 receipts, and preregistered endpoints are in [PLAN.md](PLAN.md).

Each seed's arms used **identical schedule JSON bytes**. A source slot is a
single train-only `slot_NN.wav` alias; a schedule fixes its absolute step,
source frame, and 256-chunk duration. At 103 sampled target rows per run,
alias name, target frame, and remaining chunks matched both the schedule and
the other arms exactly. The scheduled target path never used development or
validation files. All six manifests retained byte-identical strict
development and validation probes. The fixed calibration from the earlier
observer was reused without tuning.

## Primary observations

| Schedule seed | Arm | Confirmed regions | Median trajectory PR, second half | Optimizer trace rows clipped | Median health | Strict validation spectral / chroma loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 20261002 | A familiar | 3 | 9.31 | 5.8% | 0.782 | 0.790 / 0.889 |
| 20261002 | C EQ | 1 | 8.49 | 0.0% | 0.782 | 0.781 / 0.875 |
| 20261002 | B new | 2 | 7.45 | 81.6% | 0.773 | 0.800 / 0.941 |
| 20261003 | A familiar | 3 | 9.34 | 12.6% | 0.781 | 0.787 / 0.871 |
| 20261003 | C EQ | 2 | 8.62 | 12.6% | 0.777 | 0.794 / 0.902 |
| 20261003 | B new | 2 | 8.65 | 68.0% | 0.781 | 0.795 / 0.922 |

B produced **fewer** confirmed long-lived regions than A under both
schedules. None of the six runs revisited an archived region or met the
calibrated over-residence detector. Per-scale field first-difference
participation stayed near 2.25–2.48 in all arms. B's GRU and L16 Morphic
first-difference dimensions were below A in both schedules. These metrics do
not show broader organized use of the field or remaining depth under new
material.

B nonetheless exerted more learning pressure. Its median gradient norm was
8.30 and 5.62 versus A's 2.39 and 2.31. The fraction of sampled optimizer
rows with clip scale below one rose to 81.6% and 68.0%; controls were at
0–12.6%. Net msfield weight L2 change was 0.0212/0.0226 in B versus
0.0165/0.0110 in A; temporal-decoder change was 0.0605/0.0668 versus
0.0424/0.0440. Larger net updates without more field trajectory dimensions
or regimes are **not** evidence of unused capacity being successfully used.
The high clipping makes a null learning response ambiguous: credit/update
limits may be masking adaptation.

Health remained around 0.77–0.78 with no numerical collapse. B's strict
validation chroma loss was worse than A under both schedules; spectral loss
was slightly worse. These are descriptive single-checkpoint results, not a
statistical regression claim. All six final prime WAVs had zero clipped
samples and similar stereo correlation near +0.87. Their subjective quality
and downstream Suno-conditioning value remain untested. Two local blind
panels, one per schedule, are in ignored `runs/listening_20261002/` and
`runs/listening_20261003/`; each answer key remains private there.

Four repeats each of independent-channel time shuffle, four-sample block
shuffle, and matched Gaussian white noise produced zero confirmed regimes
for representative seed-20261002 A and B captures, versus 3 and 2 observed.
This supports that their long residence is not created by those simple
nulls. It does not turn a two-schedule corpus comparison into proof of
general emergence or a fractal topology. Source-to-final field and
weight-group details, per-view covariance proxies, fixed-probe summaries,
and audio descriptors are in [matched_exposure_summary.json](matched_exposure_summary.json).

## Resource and compatibility gates

The phone initially had roughly 3.1–4.2 GiB available RAM and sometimes
almost no free swap. An eight-chunk tape previously peaked near 4.5 GiB RSS.
The new v10-only `--max-autograd-tape 4` kept a 64-chunk optimizer horizon
while bounding the graph. A 17-chunk smoke and a full 64-chunk-horizon smoke
passed; the latter peaked at 2.42 GiB RSS with 1.87 GiB minimum available
RAM. All six full runs used the same setting, peaked near 2.49–2.52 GiB RSS,
and completed without a resource-guard stop. Their memory receipts are
packaged beside the run traces. This tape change limits absolute comparisons
with the earlier eight-chunk experiments.

The schedule test rejects gaps, invalid/held-out slots, frame overflow, and
non-strict probes. The new binary's **disabled** one-chunk path matched the
old world and audio byte-for-byte; model/Adam maximum absolute differences
were 7.45e-9/1.49e-8. All 91 Rust tests, two matched-schedule Python
contract tests, Rust formatting, Python compilation, and provenance checks
passed. [provenance_gate.json](provenance_gate.json) rechecks the canonical
source, frozen copy, alias/held-out WAV bytes, schedules, final checkpoints,
binary, and packaged traces. No canonical checkpoint was overwritten.

## Exact run commands and interpretation boundary

From the repository root, with the local source WAVs present:

```sh
python3 analysis/matched_exposure_20261001/prepare_matched_exposure.py
cargo build --release --locked --offline -j 1
python3 -u analysis/matched_exposure_20261001/run_campaign.py
python3 analysis/matched_exposure_20261001/package_results.py
python3 analysis/matched_exposure_20261001/analyze_results.py
python3 analysis/matched_exposure_20261001/verify_provenance.py
```

The exact six TITAN command arrays, schedule and corpus SHA-256 values,
resource minima, run IDs, checkpoint hashes, and audio hashes are in
[matched_run_receipts.json](matched_run_receipts.json). The raw captured
views, scalar/topology traces, archive states, and transition graphs are
packaged per run; large model/world/optimizer files remain in ignored local
run directories. Re-running the preparation script refuses an existing
corpus directory, and the campaign refuses to overwrite an existing fork.

**Conclusion:** With target episodes genuinely matched, new families caused
greater optimizer pressure and clipping but did not raise persistent regime
discovery or field dimensionality in either schedule seed. This is a bounded
negative result for *this* horizon and optimizer setting. It does not prove
global capacity saturation, because two schedules share one mature model and
the new-family arm spent most of its updates near the clipping ceiling. A
useful next discriminating experiment is a matched optimization-pressure
control—measure and reduce B's clipping without changing only B's effective
compute or target exposure—before implementing structured escapes or a
layer-diversity penalty. No observer-only run has produced a supported deep
trap event, so the anti-attractor controller remains disabled.
