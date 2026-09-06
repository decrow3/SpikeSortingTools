from __future__ import annotations

import numpy as np
import pytest

from testing.luke_external_warp_pipeline import (
    _apply_external_warp,
    _assert_no_gain_error,
    _assert_support_policy,
    evaluate_endpoint,
)


def test_gain_disagreement_cannot_be_recorded_as_error_fraction():
    with pytest.raises(ValueError, match="error_fraction"):
        _assert_no_gain_error({"field": {"error_fraction": 0.8}})


def test_zero_motion_uses_the_real_interpolate_motion_recording():
    pytest.importorskip("spikeinterface.sortingcomponents.motion")
    from probeinterface import Probe
    from spikeinterface.core import NumpyRecording

    positions = np.c_[np.zeros(4), np.arange(4, dtype=float) * 20.0]
    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})
    probe.set_device_channel_indices(np.arange(4))
    recording = NumpyRecording(
        [np.zeros((30000, 4), dtype="float32")], sampling_frequency=30000
    ).set_probe(probe)
    field = {
        "displacement_um": np.zeros((2, 4)),
        "recording_time_s": np.array([0.0, 1.0]),
        "depth_um": positions[:, 1],
    }
    warped = _apply_external_warp(
        recording,
        field,
        {
            "application_policy": {
                "border_mode": "remove_channels",
                "spatial_interpolation_method": "kriging",
                "sigma_um": 20.0,
            }
        },
    )
    assert warped.get_num_channels() == recording.get_num_channels()
    assert warped.get_traces(start_frame=0, end_frame=1).shape == (1, 4)


def _endpoint_fixture():
    common = {
        "good_ids": [1],
        "times_by_cluster": {1: np.arange(1000, dtype=np.int64)},
        "depth_by_cluster": {1: 200.0},
        "waveforms": {1: np.ones((4, 3), dtype=float)},
        "contamination": {1: 0.05},
        "samples_by_cluster": {1: np.arange(1000, dtype=np.int64)},
        "amplitudes_by_cluster": {1: np.ones(1000, dtype=float)},
        "fs_hz": 1000.0,
    }
    return dict(common), dict(common)


def _amendment(**overrides):
    result = {
        "identity_correspondence": {
            "event_tolerance_samples": 15,
            "minimum_exclusive_coincidence_fraction": 0.8,
            "maximum_depth_distance_um": 100.0,
            "minimum_waveform_cosine": 0.8,
        },
        "completeness": {
            "spikes_per_window": 1000,
            "max_isi_s": 10.0,
            "window_indexing": "historical",
            "min_finite_interior_windows": 2,
            "minimum_coverage_fraction": 0.5,
            "minimum_improvement_pp": 5.0,
        },
        "guardrails": {
            "maximum_contamination_increase": 0.01,
            "minimum_waveform_cosine_p10": 0.9,
            "minimum_waveform_peak_retention_p10": 0.8,
        },
    }
    for section, values in overrides.items():
        result[section].update(values)
    return result


@pytest.mark.parametrize(
    ("improvement", "coverage", "expected"),
    [(8.0, 1.0, "pass"), (2.0, 1.0, "fail"), (8.0, 0.0, "inconclusive")],
)
def test_endpoint_verdicts_measure_pass_fail_and_coverage(
    monkeypatch, improvement, coverage, expected
):
    baseline, candidate = _endpoint_fixture()
    if coverage == 0.0:
        candidate["good_ids"] = []

    def fake_fit(samples, amplitudes, *, fs_hz, interval_s, config, label):
        missing = 10.0 if label.startswith("baseline") else 10.0 - improvement
        return {"status": "measured", "missing_pct_median": missing}

    monkeypatch.setattr(
        "testing.luke_external_warp_pipeline.train_completeness", fake_fit
    )
    result = evaluate_endpoint(
        baseline, candidate, _amendment(), interval_s=(0.0, 1.0)
    )
    assert result["status"] == expected
    assert result["eligible_units"] == 1
    assert result["coverage_fraction"] == coverage


def test_completeness_improvement_is_baseline_minus_candidate(monkeypatch):
    baseline, candidate = _endpoint_fixture()

    def fake_fit(samples, amplitudes, *, fs_hz, interval_s, config, label):
        return {"status": "measured", "missing_pct_median": 20.0 if label.startswith("baseline") else 10.0}

    monkeypatch.setattr("testing.luke_external_warp_pipeline.train_completeness", fake_fit)
    result = evaluate_endpoint(baseline, candidate, _amendment(), interval_s=(0.0, 1.0))
    assert result["paired_completeness_improvement_pp"] == 10.0


def test_support_artifact_refuses_an_unsupported_interpolation_neighborhood(tmp_path):
    peaks = np.zeros(12, dtype=[("sample_index", "i8"), ("channel_index", "i8"), ("amplitude", "f8"), ("segment_index", "i8")])
    peaks["sample_index"] = np.array([100] * 5 + [1100] * 5 + [100, 1100])
    locations = np.zeros(12, dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"), ("alpha", "f8")])
    locations["y"] = np.array([20.0] * 10 + [120.0, 120.0])
    peaks_path = tmp_path / "peaks.npy"
    locations_path = tmp_path / "peak_locations.npy"
    np.save(peaks_path, peaks)
    np.save(locations_path, locations)
    amendment = {
        "support_artifact": {
            "peaks_npy": str(peaks_path),
            "peak_locations_npy": str(locations_path),
            "minimum_peaks_per_cell": 5,
            "time_bin_s": 1.0,
            "depth_bin_s": 100.0,
            "depth_grid_first_center_um": 20.0,
            "support_fraction_gate": 0.95,
        }
    }
    contract = {
        "domain": {"interval_s": [0.0, 2.0], "depth_band_um": [20.0, 20.0], "n_channels_in_band": 1},
        "application_policy": {"sigma_um": 1.0},
    }
    field = {
        "recording_time_s": np.array([0.0, 1.0, 2.0]),
        "depth_um": np.array([0.0, 20.0, 40.0]),
        "displacement_um": np.zeros((2, 3)),
    }

    class Recording:
        def get_sampling_frequency(self):
            return 1000.0

        def get_channel_locations(self):
            return np.array([[0.0, 20.0]])

    assert _assert_support_policy(Recording(), field, contract, amendment)["support_artifact"]["failed_neighborhoods"] == 0
    locations["y"] = 1000.0
    np.save(locations_path, locations)
    with pytest.raises(ValueError, match="support policy refused"):
        _assert_support_policy(Recording(), field, contract, amendment)