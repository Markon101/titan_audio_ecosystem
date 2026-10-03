# Fixed predictive-probe transfer

This follow-up uses existing frozen audio/state captures. It runs no Titan
forward pass, optimizer, build, training or generation selection. The saved
probe bank freezes scalers, projections, bins, affine-period parameters,
linear weights, residual variances and comparator choices before test scoring.
Old research artifacts and canonical checkpoints are read-only.

From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 python3 analysis/predictive_transfer_20261003/test_transfer.py
OPENBLAS_NUM_THREADS=1 python3 analysis/predictive_density_20261003/test_predictive.py
# New output files only; these refuse to overwrite completed reports:
OPENBLAS_NUM_THREADS=1 python3 analysis/predictive_transfer_20261003/run_transfer.py
OPENBLAS_NUM_THREADS=1 python3 analysis/predictive_transfer_20261003/summarize.py
OPENBLAS_NUM_THREADS=1 python3 analysis/predictive_transfer_20261003/variance_controls.py
python3 analysis/predictive_transfer_20261003/verify.py
```

The completed full report is stored as deterministic `RESULTS.json.gz`.
The local readable `RESULTS.json` is ignored. To read the committed archive:

```sh
python3 - <<'PY'
import gzip, json
from pathlib import Path
p = Path('analysis/predictive_transfer_20261003')
r = json.loads(gzip.decompress((p/'RESULTS.json.gz').read_bytes()))
e = r['episodes']['L16_WP_SP']['strict']['state_to_output']['horizons']['64']
print(e['probes']['field_coarse'])
PY
```

`SUMMARY.json`, `curves.csv`, `DRIFT_DIAGNOSTIC.json` and
`VARIANCE_CONTROLS.json` provide compact readouts. `archive_receipt.json`
binds the full raw/compressed report. Local learned probes are at
`runs/audio_probe.json`, `runs/state_probe.json` and three null files; each
is saved once and reloaded before evaluation. Their hashes remain unchanged
through scoring. The runner may adopt a partial probe only if its complete
serialized state exactly equals the freshly derived source-only state;
incompatible artifacts fail rather than overwrite.

Strict transfer uses source normalization and noise scales. The separate
sigma-adapted diagnostic recalibrates only residual scales using a test's
first 256 chunks, then scores later targets. Its matched strict comparator
uses exactly the same origins. Prediction means/MSE do not change. Future
labels are purged from the calibration boundary; unsupported horizons are
explicitly absent. Neither diagnostic affects Titan or chooses an intervention.

The extra decomposition compares history means with a probe's fixed sigma.
It separates static uncertainty benefits from actual mean-prediction gains.
Proper code reduction is not automatically information uniquely carried by
an internal mechanism. Raw descriptor MSE mixes units and is not audio quality.

All episodes share one model lineage. They are not independent model seeds;
the warmup-128 baseline is a known overlap control. The primary different-RNG
test is held out from fitting, but was previously inspected in other studies.
No validation probes or target IDs enter any probe or comparator selection.
See `PLAN.md` and `RESEARCH_NOTE.md` for interpretation and limitations.
