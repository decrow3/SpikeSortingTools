import numpy as np
import pytest

from testing.ladder_synthetic_donors import (
    SyntheticSpec,
    build_synthetic_cohort,
    default_grid,
    synthetic_template,
)

FS = 30_000.0


def test_synthetic_template_hits_target_amplitude_polarity_and_is_valid():
    neg = synthetic_template(SyntheticSpec(300.0, 0.30), fs=FS)
    assert neg.shape == (61, 33)
    assert np.allclose(neg[:3], 0.0) and np.allclose(neg[-3:], 0.0)  # tapered edges
    assert np.abs(neg).max() == pytest.approx(300.0, rel=0.05)
    peak_col = int(np.argmax(np.max(np.abs(neg), axis=0)))
    trough = neg[:, peak_col]
    assert trough[np.argmax(np.abs(trough))] < 0  # negative-dominant

    pos = synthetic_template(SyntheticSpec(300.0, 0.30, polarity="pos"), fs=FS)
    ptrough = pos[:, int(np.argmax(np.max(np.abs(pos), axis=0)))]
    assert ptrough[np.argmax(np.abs(ptrough))] > 0


def test_trough_width_controls_sharpness():
    sharp = synthetic_template(SyntheticSpec(300.0, 0.20), fs=FS)
    broad = synthetic_template(SyntheticSpec(300.0, 0.55), fs=FS)
    pc = int(np.argmax(np.max(np.abs(sharp), axis=0)))

    def fwhm_samples(w):
        w = -np.abs(w[:, pc]) if True else w
        half = w.min() * 0.5
        below = np.flatnonzero(w <= half)
        return below.max() - below.min() + 1 if below.size else 0

    assert fwhm_samples(sharp) < fwhm_samples(broad)


def test_spatial_lambda_controls_footprint_width():
    compact = synthetic_template(SyntheticSpec(300.0, 0.3, spatial_lambda_um=15.0), fs=FS)
    wide = synthetic_template(SyntheticSpec(300.0, 0.3, spatial_lambda_um=45.0), fs=FS)

    def energy_frac_pm3(t):
        e = np.sqrt((t.astype(np.float64) ** 2).sum(axis=0))
        pk = int(np.argmax(e))
        return e[max(0, pk - 3): pk + 4].sum() / e.sum()

    assert energy_frac_pm3(compact) > energy_frac_pm3(wide)


def test_build_synthetic_cohort_writes_both_polarities_and_a_sharpness_range(tmp_path):
    result = build_synthetic_cohort(tmp_path, fs=FS)
    assert result["n_donors"] == len(default_grid())
    assert set(result["polarity_mix"]) == {"neg", "pos"}
    assert len(result["trough_width_ms_values"]) >= 3
    npz = np.load(tmp_path / "donor_templates.npz")
    assert all(np.isfinite(npz[k]).all() for k in npz.files)


def test_build_synthetic_cohort_refuses_mnt():
    with pytest.raises(ValueError, match="/mnt"):
        build_synthetic_cohort("/mnt/x")
