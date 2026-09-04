import numpy as np
import pytest

from testing.luke_c2_operator_calibration import (
    CALIB,
    OUTPUT,
    assert_um_channel_inverse,
    fidelity,
    lattice_exactness,
    ramp_expectation,
    um_trajectory,
)


def strip_geometry(n: int = 112) -> np.ndarray:
    """Neuropixels-1.0-like staggered strip: 20 µm rows, two sites per row."""
    rows = n // 2
    x_pattern = [(16.0, 48.0), (0.0, 32.0)]
    positions = []
    for row in range(rows):
        for x in x_pattern[row % 2]:
            positions.append([x, 20.0 * row])
    return np.asarray(positions, dtype=float)


def test_um_and_channel_units_are_exact_inverses():
    geometry = strip_geometry()
    assert assert_um_channel_inverse(geometry) == pytest.approx(1.0, abs=1e-12)


def test_um_channel_inverse_rejects_a_degenerate_geometry():
    flat = np.column_stack([np.zeros(8), np.zeros(8)])  # zero y span
    with pytest.raises(RuntimeError, match="not invertible"):
        assert_um_channel_inverse(flat)


def test_um_trajectory_is_constant_and_scales_with_geometry():
    geometry = strip_geometry()
    traj = um_trajectory(11.0, geometry)
    values = traj(np.linspace(0.0, 120.0, 7))
    assert np.allclose(values, values[0])
    # 11 µm on a 1100 µm / 111-step strip is 1.11 channel indices
    assert values[0] == pytest.approx(11.0 * (len(geometry) - 1) / 1100.0)


def test_fidelity_is_exact_on_an_identity():
    rng = np.random.default_rng(0)
    field = rng.standard_normal((64, 16))
    metrics = fidelity(field, field)
    assert metrics["peak_retention"] == pytest.approx(1.0)
    assert metrics["cosine"] == pytest.approx(1.0)
    assert metrics["rel_rms"] == pytest.approx(0.0, abs=1e-12)


def test_fidelity_peak_retention_is_translation_invariant():
    """The forward metric must not penalise a footprint merely for moving."""
    field = np.zeros((64, 16))
    field[32, 8] = -100.0
    moved = np.roll(field, 4, axis=1)
    assert fidelity(field, moved)["peak_retention"] == pytest.approx(1.0)
    # ... while cosine/rel_rms do fall, which is why they are displacement-only
    assert fidelity(field, moved)["cosine"] == pytest.approx(0.0, abs=1e-12)


def test_lattice_exactness_matches_a_known_roll():
    rng = np.random.default_rng(1)
    field = rng.standard_normal((32, 12))
    assert lattice_exactness(field, np.roll(field, 4, axis=1), 4) == pytest.approx(0.0, abs=1e-12)
    assert lattice_exactness(field, np.roll(field, 4, axis=1), -4) > 0.5


def test_commensurate_offsets_are_two_row_pitches_apart():
    """40 µm is two 20 µm rows; on two sites per row that is four channel indices."""
    geometry = strip_geometry()
    row_pitch = float(np.unique(np.diff(np.unique(geometry[:, 1])))[0])
    for offset_um, channel_shift in CALIB["lattice_commensurate_um"].items():
        assert float(offset_um) % (2 * row_pitch) == 0
        assert channel_shift == int(float(offset_um) / row_pitch) * 2


def test_ramp_expectation_only_averages_offsets_the_ramp_visits():
    curve = [
        {"offset_um": o, "peak_retention": 1.0 - 0.02 * o, "cosine": 1.0,
         "rel_rms": 0.01 * o, "roundtrip_rel_rms": 0.02 * o,
         "roundtrip_peak_retention": 1.0 - 0.03 * o,
         "exact_inverse_amplitude_cost": -0.01 * o, "snr_after": 10.0 - o}
        for o in (0.0, 2.5, 5.0, 11.0, 22.0)
    ]
    expectation = ramp_expectation(curve, 5.0)
    assert expectation["n_offsets"] == 3  # 0, 2.5, 5 only
    assert expectation["min_peak_retention"] == pytest.approx(0.9)
    assert expectation["mean_exact_inverse_amplitude_cost"] < 0.0
    assert ramp_expectation(curve, 0.0) == {}  # a single offset is not a ramp


def test_calibration_never_writes_under_mnt():
    assert not str(OUTPUT).startswith("/mnt/")
