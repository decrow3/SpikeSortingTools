import numpy as np
import pandas as pd
import pytest

from testing.ladder_donors import (
    DonorConfig,
    _compactness,
    _dewhitened_shape,
    _select,
    _taper,
    build_donor_cohort,
)


def test_compactness_separates_a_footprint_from_a_plateau():
    n_samp, n_chan = 61, 33
    prof = np.array([-0.1, -0.5, -1.0, -0.5, -0.1]) * 120.0
    compact = np.zeros((n_samp, n_chan))
    for dc, w in ((-2, 0.15), (-1, 0.5), (0, 1.0), (1, 0.5), (2, 0.15)):
        compact[28:33, 16 + dc] = prof * w
    c = _compactness(compact)
    assert c["energy_frac_pm3"] > 0.9 and c["half_energy_width_ch"] <= 5

    plateau = np.zeros((n_samp, n_chan))
    plateau[28:33, :] = prof[:, None]  # every channel identical — the pilot-donor pathology
    p = _compactness(plateau)
    assert p["energy_frac_pm3"] < 0.3 and p["half_energy_width_ch"] == n_chan


def test_taper_zeros_edges_and_preserves_the_interior():
    cfg = DonorConfig(edge_guard_samples=3, ramp_samples=5)
    rng = np.random.default_rng(0)
    sta = rng.normal(0, 20, (61, 33)).astype(np.float32)
    sta[25:35] += -150.0
    out = _taper(sta, cfg)
    assert np.allclose(out[:3], 0.0) and np.allclose(out[-3:], 0.0)
    assert np.abs(out[28:32]).max() > 50.0


def _fake_sort(n_units=4, n_chan=48):
    """A compact spike per unit + an identity whitening matrix."""
    templates = np.zeros((n_units, 61, n_chan))
    prof = np.array([-0.2, -0.6, -1.0, -0.6, -0.2])
    for u in range(n_units):
        pk = 10 + u * 8
        for dc, w in ((-1, 0.4), (0, 1.0), (1, 0.4)):
            templates[u, 28:33, pk + dc] = prof * w
    return {
        "st": np.arange(3000 * n_units, dtype=np.int64),
        "cl": np.concatenate([np.full(3000, u) for u in range(n_units)]).astype(np.int64),
        "templates": templates,
        "winv": np.eye(n_chan),
        "positions": np.column_stack([np.zeros(n_chan), np.arange(n_chan) * 10.0]),
        "good": set(range(n_units)),
        "contam_pct": {u: 1.0 for u in range(n_units)},
        "amplitude": {u: 30.0 + 20.0 * u for u in range(n_units)},
    }


def test_dewhitened_shape_is_compact_and_peak_normalised():
    sort = _fake_sort()
    shape, peak_c, polarity = _dewhitened_shape(sort, 1, DonorConfig(radius_ch=16))
    assert polarity == "neg"
    assert np.abs(shape).max() == pytest.approx(1.0)  # unit-peak normalised (µV scale applied later)
    assert _compactness(shape)["energy_frac_pm3"] > 0.8


def test_dewhitened_shape_centres_a_long_template_on_peak_time():
    sort = _fake_sort(n_units=1)
    long = np.zeros((1, 101, 48), dtype=float)
    long[0, 80, 10] = -1.0
    sort["templates"] = long
    shape, _, _ = _dewhitened_shape(sort, 0, DonorConfig(n_samples=61))
    assert shape.shape[0] == 61
    assert np.unravel_index(np.argmax(np.abs(shape)), shape.shape)[0] == 40


def test_select_spreads_across_amplitude_bands_and_polarity():
    rows = []
    for i in range(12):
        rows.append({
            "cluster_id": i, "peak_uv": 20.0 + i * 10,
            "polarity": "neg" if i % 2 else "pos",
            "energy_frac_pm3": 0.9 - i * 0.01,
        })
    picked = _select(
        rows, DonorConfig(per_cell_cap=1, amp_bands_uv=(50.0, 90.0)), n_donors=12
    )
    cells = {(r["amplitude_band"], r["polarity"]) for r in picked}
    assert len(picked) == len(cells)  # per_cell_cap=1 => no duplicate cells
    assert {"low", "mid", "high"} <= {r["amplitude_band"] for r in picked}


def test_build_donor_cohort_refuses_mnt():
    with pytest.raises(ValueError, match="/mnt"):
        build_donor_cohort("x", "y", "/mnt/out")
