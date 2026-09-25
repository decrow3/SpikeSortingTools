# motionqc

`motionqc` is a small, estimator-independent toolkit for validating saved
Neuropixels motion fields. It never sorts spikes, changes voltage, or fits a
motion estimator.

All fields use seconds from AP frame zero and the sign convention:

```text
corrected_depth = observed_depth - displacement
```

## Python API

```python
from motionqc import load_field
from motionqc.reference import shift_test, matched_null, canonical_mask

field = load_field("field.npz")
warnings = field.validate(recording_length_s=10473.55)
measurement = shift_test(peaks, (997.0, 1004.5), (993.0, 996.0))
```

`MotionField.save()` writes a deterministic NPZ plus a JSON provenance
manifest. Loaders accept pilot/Q rigid fields, DARTsort `fields.npz` and
`motion.pkl`, raw MEDiCINe directories, and NPX DREDGE directories with an
explicit origin offset.

Runtime analysis dependencies are NumPy, pandas, SciPy, and Matplotlib. They
are already present in the repository's research environment; the package does
not install or run MEDiCINe, DREDGE, DARTsort, or a spike sorter.

## CLI

```bash
motionqc report \
  --field deployed.npz --field slow.npz \
  --peaks population.npz \
  --windows episodes.csv --mask censor_mask.csv \
  --out motionqc_report
```

The report uses an identical shared grid, episode set, quiet samples, and 5 s
increment pairs for every field. It includes episode error/ratio, per-block
error when block references exist, quiet metrics, false-motion fraction, bias
diagnostics and binned leakage profiles, Okabe–Ito/line-style raster and density
overlays, warnings, and the optional upper/lower unit common-mode spectrum.

## Validation

Run the focused suite with:

```bash
pytest -q testing/test_motionqc_field.py \
  testing/test_motionqc_reference.py testing/test_motionqc_report.py
```

The frozen phase-1 reproduction driver is `python -m motionqc.reproduce`.
Results and one known supplied-artifact discrepancy are recorded in
`PHASE1_RESULTS.md`.

## Method provenance

- Label-free maps and matched nulls: `testing/luke_imec1_medicine_m_reference_v1.py`, M/R.
- Per-depth blocks: `testing/luke_imec1_medicine_stage2_o_v1.py`, O.1.
- Slow reference: `testing/luke_imec1_slow_layer_ab_v1.py`, AB.3.
- Matched scoring: `testing/luke_imec1_medicine_stage3_q_v1.py`, Y.
- Hub behaviour references: `t8_episode_shift.py`, `t8b_null.py`,
  `t9_sweep_checks.py`, `t13_quiet_fast_motion.py`, and
  `censor_mask_v1.py` in `incoming/hub_analysis_v1`.

Phase 1 intentionally excludes fitting, sliding-window deployment, catalogue
generation, sweep orchestration, and sorting metrics.
