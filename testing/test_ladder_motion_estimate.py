import numpy as np
import pytest

from testing.ladder_motion_estimate import (
    FieldGate,
    _field_diagnostics,
    materialize_qualified_correction,
    qualify_field,
)


class _FakeMotion:
    def __init__(self, disp, t):
        self.displacement = [np.asarray(disp, dtype=float)]
        self.temporal_bins_s = [np.asarray(t, dtype=float)]
        self.spatial_bins_um = [np.array([1000.0])]


def test_field_diagnostics_centres_arbitrary_gauge_and_reads_bandwidth():
    t = np.arange(0, 120, 0.5)
    # a 40 s-period, 15 µm oscillation plus a 6 µm bias
    disp = (7.5 * np.sin(2 * np.pi * t / 40.0) + 6.0)[:, None]
    d = _field_diagnostics(_FakeMotion(disp, t))
    assert d["max_abs_displacement_um"] == pytest.approx(7.5, abs=0.5)
    assert d["temporal_bandwidth_hz"] >= 0.02  # 1/40 s ≈ 0.025 Hz


def test_qualify_field_passes_a_good_field_and_fails_an_over_smoothed_one():
    good = {
        "max_abs_displacement_um": 20.0,
        "estimated_gain_error_fraction": 0.1,
        "support_fraction": 0.99,
        "split_half_correlation": 0.9,
    }
    assert qualify_field(good)["passes"]

    unsupported = {**good, "support_fraction": 0.5, "split_half_correlation": 0.2}
    q = qualify_field(unsupported)
    assert not q["passes"]
    assert "support_measured_and_sufficient" in q["failed"]
    assert "split_half_reproducible" in q["failed"]


def test_qualify_field_fails_out_of_range_displacement_and_error():
    q = qualify_field({
        "max_abs_displacement_um": 120.0,
        "estimated_gain_error_fraction": 0.5,
        "support_fraction": 0.99,
        "split_half_correlation": 0.9,
    })
    assert set(q["failed"]) == {
        "displacement_in_calibrated_range", "estimation_error_measured_and_tolerable"
    }


def test_qualify_field_fails_closed_when_independent_evidence_is_missing():
    q = qualify_field({"max_abs_displacement_um": 20.0})
    assert not q["passes"]
    assert set(q["failed"]) == {
        "estimation_error_measured_and_tolerable",
        "support_measured_and_sufficient",
        "split_half_reproducible",
    }


def test_field_gate_digest_is_config_sensitive():
    assert FieldGate().digest != FieldGate(min_support_fraction=0.9).digest


def test_materialize_applies_motion_to_original_accepted_recording(tmp_path):
    from pipeline.preprocess import validate_accepted_recording
    from spikeinterface.core.motion import Motion
    from spikeinterface.preprocessing.motion import save_motion_info
    from testing.ladder_inject import write_injected_recording

    source = tmp_path / "source"
    geom = np.column_stack([np.zeros(8), np.arange(8) * 20.0])
    write_injected_recording(
        source,
        np.zeros((1000, 8), dtype=np.float32),
        channel_positions=geom,
        fs=30_000.0,
        gain_uv_per_count=1.0,
        n_jobs=1,
    )
    motion = Motion(
        [np.zeros((2, 1), dtype=float)],
        [np.array([0.0, 0.03])],
        np.array([70.0]),
        direction="y",
    )
    motion_dir = tmp_path / "motion_info"
    save_motion_info(
        {
            "parameters": {}, "run_times": {},
            "peaks": np.array([], dtype=[]),
            "peak_locations": np.array([], dtype=[]),
            "motion": motion,
        },
        motion_dir,
    )
    q = qualify_field({
        "max_abs_displacement_um": 0.0,
        "estimated_gain_error_fraction": 0.0,
        "support_fraction": 1.0,
        "split_half_correlation": 1.0,
    })
    manifest = materialize_qualified_correction(
        source, motion_dir, tmp_path / "corrected", qualification=q, n_jobs=1
    )
    assert manifest["kind"] == "qualified_motion_corrected_recording"
    validate_accepted_recording(tmp_path / "corrected")
