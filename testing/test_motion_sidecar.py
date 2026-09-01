import json
from dataclasses import replace

import numpy as np
import pytest

from pipeline.motion_sidecar import (
    DredgeRigidConfig,
    JobConfig,
    MotionBackend,
    MotionQCConfig,
    MotionSidecarConfig,
    RigidMotionEstimate,
    evaluate_motion_qc,
    run_motion_sidecar,
    run_motion_sidecar_safely,
)


class FakeRecording:
    dtype = np.dtype("int16")

    def __init__(
        self,
        *,
        channel_ids=("AP0", "AP1", "AP2"),
        shift_um=0.0,
        content_id="recording-a",
    ):
        self._channel_ids = np.asarray(channel_ids)
        self.content_id = content_id
        self._locations = np.column_stack(
            (
                np.zeros(len(channel_ids)),
                shift_um + np.arange(len(channel_ids), dtype=float) * 100.0,
            )
        )

    def get_channel_ids(self):
        return self._channel_ids

    def get_channel_locations(self):
        return self._locations

    def get_num_channels(self):
        return self._channel_ids.size

    def get_num_samples(self):
        return 20

    def get_sampling_frequency(self):
        return 10.0


def accepted_manifest(recording):
    return {
        "schema_version": "rescue-recording-manifest-v2",
        "complete": True,
        "request_digest": f"request-{recording.content_id}",
        "recording_content_sha256": f"content-{recording.content_id}",
    }


def run_test_sidecar(estimator_recording, *, recording_for_sorting, **kwargs):
    kwargs.setdefault(
        "accepted_recording_manifest", accepted_manifest(recording_for_sorting)
    )
    return run_motion_sidecar(
        estimator_recording,
        recording_for_sorting=recording_for_sorting,
        **kwargs,
    )


class FakeMotion:
    def __init__(self):
        self.displacement = [np.array([[5.0], [9.0]])]
        self.temporal_bins_s = [np.array([0.5, 1.5])]
        self.spatial_bins_um = np.array([100.0])


def fake_peaks():
    values = np.zeros(
        4,
        dtype=[("sample_index", "i8"), ("channel_index", "i8"), ("amplitude", "f8")],
    )
    values["sample_index"] = [2, 7, 12, 17]
    values["channel_index"] = [0, 1, 1, 2]
    values["amplitude"] = [-10.0, -12.0, -11.0, -13.0]
    return values


def fake_locations():
    values = np.zeros(4, dtype=[("x", "f8"), ("y", "f8")])
    values["y"] = [0.0, 100.0, 100.0, 200.0]
    return values


def successful_backend(calls=None):
    if calls is None:
        calls = {"detect": 0, "localize": 0, "estimate": 0}

    def detect(recording, **kwargs):
        calls["detect"] += 1
        assert kwargs["method"] == "locally_exclusive"
        assert kwargs["detect_threshold"] == 5.0
        return fake_peaks()

    def localize(recording, peaks, **kwargs):
        calls["localize"] += 1
        assert kwargs["method"] == "monopolar_triangulation"
        np.testing.assert_array_equal(peaks, fake_peaks())
        return fake_locations()

    def estimate(**kwargs):
        calls["estimate"] += 1
        assert kwargs["method"] == "dredge_ap"
        assert kwargs["rigid"] is True
        assert kwargs["win_shape"] == "rect"
        assert kwargs["extra_outputs"] is True
        return FakeMotion(), {"pairwise_displacement": np.eye(2)}

    return MotionBackend(detect, localize, estimate, {"test_backend": "1"}), calls


def test_production_config_forbids_nonrigid_and_voltage_paths():
    with pytest.raises(ValueError, match="nonrigid"):
        MotionSidecarConfig(save_nonrigid_for_diagnostics=True)
    with pytest.raises(ValueError, match="not authorized"):
        MotionSidecarConfig(voltage_correction_enabled=True)
    with pytest.raises(ValueError, match="forbidden"):
        MotionSidecarConfig(legacy_correction_cache_export=True)
    with pytest.raises(ValueError, match="not implemented"):
        MotionSidecarConfig(legacy_analysis_export=True)
    with pytest.raises(ValueError, match="direct rigid"):
        DredgeRigidConfig(rigid=False)


def test_unvalidated_qc_cannot_smuggle_thresholds():
    with pytest.raises(ValueError, match="Unvalidated QC"):
        MotionQCConfig(min_peak_count_per_time=10)


def test_success_preserves_identity_and_writes_inspectable_artifacts(tmp_path):
    estimator = FakeRecording()
    sorter = FakeRecording()
    backend, calls = successful_backend()
    result = run_test_sidecar(
        estimator,
        recording_for_sorting=sorter,
        cache_dir=tmp_path / "motion",
        backend=backend,
        job_config=JobConfig(n_jobs=2),
    )

    assert result.recording_for_sorting is sorter
    assert result.status == "ESTIMATE_COMPLETED"
    assert result.qc.status == "NOT_EVALUATED"
    assert calls == {"detect": 1, "localize": 1, "estimate": 1}
    np.testing.assert_array_equal(result.estimate.displacement_native_um[:, 0], [5, 9])
    np.testing.assert_array_equal(
        result.estimate.displacement_reference_centered_um[:, 0], [-2, 2]
    )
    assert result.estimate.reference_value_um == 7.0
    assert result.estimate.peak_count_by_time.tolist() == [2, 2]
    assert result.estimate.support_by_time.tolist() == [2, 2]

    root = tmp_path / "motion"
    method = root / "dredge-rigid-sidecar"
    assert (root / "peaks.npy").exists()
    assert (root / "peak_locations.npy").exists()
    assert (method / "motion_native.npy").exists()
    assert (method / "motion_reference_centered.npy").exists()
    assert (root / "figures/rigid_trace.png").exists()
    assert not (root / "dredge-motion/motion.npy").exists()
    qc = json.loads((root / "motion_qc.json").read_text())
    assert qc["correction_policy_validated"] is False
    assert qc["correction_eligible_epochs"] == "NOT_EVALUATED"
    assert qc["voltage_correction_applied"] is False


def test_exact_cache_reuse_does_not_call_backend(tmp_path):
    recording = FakeRecording()
    backend, calls = successful_backend()
    first = run_test_sidecar(
        recording,
        recording_for_sorting=recording,
        cache_dir=tmp_path / "motion",
        backend=backend,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("backend must not run on exact cache reuse")

    reuse_backend = MotionBackend(forbidden, forbidden, forbidden, {"test_backend": "1"})
    second = run_test_sidecar(
        recording,
        recording_for_sorting=recording,
        cache_dir=tmp_path / "motion",
        backend=reuse_backend,
    )
    assert first.request_digest == second.request_digest
    assert second.cache_lineage["status"] == "reused_exact_match"
    assert calls == {"detect": 1, "localize": 1, "estimate": 1}


def test_cache_request_mismatch_is_refused(tmp_path):
    recording = FakeRecording()
    backend, _ = successful_backend()
    run_test_sidecar(
        recording,
        recording_for_sorting=recording,
        cache_dir=tmp_path / "motion",
        backend=backend,
    )
    changed = replace(MotionSidecarConfig(), support_depth_bin_um=50.0)
    with pytest.raises(RuntimeError, match="another request"):
        run_test_sidecar(
            recording,
            recording_for_sorting=recording,
            cache_dir=tmp_path / "motion",
            config=changed,
            backend=backend,
        )


def test_same_shape_different_recording_content_cannot_reuse_cache(tmp_path):
    first_recording = FakeRecording(content_id="first")
    second_recording = FakeRecording(content_id="second")
    backend, _ = successful_backend()
    run_test_sidecar(
        first_recording,
        recording_for_sorting=first_recording,
        cache_dir=tmp_path / "motion",
        backend=backend,
    )
    with pytest.raises(RuntimeError, match="another request"):
        run_test_sidecar(
            second_recording,
            recording_for_sorting=second_recording,
            cache_dir=tmp_path / "motion",
            backend=backend,
        )


def test_safe_wrapper_turns_cache_mismatch_into_identity_fallback(tmp_path):
    recording = FakeRecording()
    backend, _ = successful_backend()
    run_test_sidecar(
        recording,
        recording_for_sorting=recording,
        cache_dir=tmp_path / "motion",
        backend=backend,
    )
    changed = replace(MotionSidecarConfig(), support_depth_bin_um=50.0)
    result = run_motion_sidecar_safely(
        recording,
        recording_for_sorting=recording,
        cache_dir=tmp_path / "motion",
        config=changed,
        backend=backend,
        accepted_recording_manifest=accepted_manifest(recording),
    )
    assert result.recording_for_sorting is recording
    assert result.status == "ESTIMATE_FAILED_IDENTITY_SORT_CONTINUED"
    failure = json.loads((tmp_path / "motion/estimation_failure.json").read_text())
    assert failure["failure_stage"] == "sidecar_preflight_or_cache_validation"


def test_explicit_recompute_archives_prior_artifact(tmp_path):
    recording = FakeRecording()
    backend, calls = successful_backend()
    first = run_test_sidecar(
        recording,
        recording_for_sorting=recording,
        cache_dir=tmp_path / "motion",
        backend=backend,
    )
    second = run_test_sidecar(
        recording,
        recording_for_sorting=recording,
        cache_dir=tmp_path / "motion",
        backend=backend,
        recompute=True,
    )
    assert first.request_digest == second.request_digest
    assert calls["estimate"] == 2
    assert list((tmp_path / "motion").glob("dredge-rigid-sidecar.superseded-*"))


def test_tampered_cache_is_refused(tmp_path):
    recording = FakeRecording()
    backend, _ = successful_backend()
    run_test_sidecar(
        recording,
        recording_for_sorting=recording,
        cache_dir=tmp_path / "motion",
        backend=backend,
    )
    path = tmp_path / "motion/dredge-rigid-sidecar/motion_native.npy"
    values = np.load(path)
    values[0, 0] += 1
    np.save(path, values)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        run_test_sidecar(
            recording,
            recording_for_sorting=recording,
            cache_dir=tmp_path / "motion",
            backend=backend,
        )


def test_missing_required_figure_invalidates_cache(tmp_path):
    recording = FakeRecording()
    backend, _ = successful_backend()
    run_test_sidecar(
        recording,
        recording_for_sorting=recording,
        cache_dir=tmp_path / "motion",
        backend=backend,
    )
    (tmp_path / "motion/dredge-rigid-sidecar/figures/rigid_trace.png").unlink()
    with pytest.raises(RuntimeError, match="Incomplete motion sidecar"):
        run_test_sidecar(
            recording,
            recording_for_sorting=recording,
            cache_dir=tmp_path / "motion",
            backend=backend,
        )


def test_estimator_failure_writes_receipt_and_preserves_identity(tmp_path):
    recording = FakeRecording()

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic DREDGE failure")

    backend = MotionBackend(fail, fail, fail, {"test_backend": "failure"})
    result = run_test_sidecar(
        recording,
        recording_for_sorting=recording,
        cache_dir=tmp_path / "motion",
        backend=backend,
    )
    assert result.recording_for_sorting is recording
    assert result.estimate is None
    assert result.status == "ESTIMATE_FAILED_IDENTITY_SORT_CONTINUED"
    failure = json.loads((tmp_path / "motion/estimation_failure.json").read_text())
    assert failure["safe_fallback"] == "identity"
    assert failure["voltage_modified"] is False
    assert failure["message"] == "synthetic DREDGE failure"


def test_strict_failure_is_raised_after_receipt_is_written(tmp_path):
    recording = FakeRecording()

    def fail(*args, **kwargs):
        raise RuntimeError("strict failure")

    with pytest.raises(RuntimeError, match="strict failure"):
        run_test_sidecar(
            recording,
            recording_for_sorting=recording,
            cache_dir=tmp_path / "motion",
            backend=MotionBackend(fail, fail, fail),
            strict=True,
        )
    assert (tmp_path / "motion/estimation_failure.json").exists()


def test_success_archives_prior_failure_receipt(tmp_path):
    recording = FakeRecording()

    def fail(*args, **kwargs):
        raise RuntimeError("transient failure")

    run_test_sidecar(
        recording,
        recording_for_sorting=recording,
        cache_dir=tmp_path / "motion",
        backend=MotionBackend(fail, fail, fail, {"test_backend": "1"}),
    )
    backend, _ = successful_backend()
    result = run_test_sidecar(
        recording,
        recording_for_sorting=recording,
        cache_dir=tmp_path / "motion",
        backend=backend,
    )
    assert result.status == "ESTIMATE_COMPLETED"
    assert not (tmp_path / "motion/estimation_failure.json").exists()
    assert list((tmp_path / "motion").glob("estimation_failure.superseded-*.json"))


def test_geometry_mismatch_is_a_hard_failure(tmp_path):
    backend, _ = successful_backend()
    with pytest.raises(ValueError, match="probe geometry"):
        run_test_sidecar(
            FakeRecording(),
            recording_for_sorting=FakeRecording(shift_um=5.0),
            cache_dir=tmp_path / "motion",
            backend=backend,
        )


def test_legacy_correction_ready_path_is_ignored_not_consumed(tmp_path):
    legacy = tmp_path / "motion/dredge-motion"
    legacy.mkdir(parents=True)
    np.save(legacy / "motion.npy", np.zeros((2, 1)))
    backend, _ = successful_backend()
    result = run_test_sidecar(
        FakeRecording(),
        recording_for_sorting=FakeRecording(),
        cache_dir=tmp_path / "motion",
        backend=backend,
    )
    assert result.status == "ESTIMATE_COMPLETED"
    request = json.loads((tmp_path / "motion/request.json").read_text())
    assert request["legacy_correction_cache_detected_and_ignored"] is True
    np.testing.assert_array_equal(np.load(legacy / "motion.npy"), np.zeros((2, 1)))


def test_validated_qc_is_epoch_resolved():
    estimate = RigidMotionEstimate(
        displacement_native_um=np.array([[0.0], [2.0], [20.0]]),
        displacement_reference_centered_um=np.array([[0.0], [2.0], [20.0]]),
        time_s=np.array([0.5, 1.5, 2.5]),
        depth_um=np.array([100.0]),
        peak_count_by_time=np.array([20, 2, 20]),
        peak_count_by_time_depth=np.ones((3, 2), dtype=int),
        depth_bin_centers_um=np.array([50.0, 150.0]),
        support_by_time=np.array([2, 1, 2]),
        reference_method="median_all_finite",
        reference_value_um=0.0,
        provenance={},
        cache_lineage={},
    )
    qc = evaluate_motion_qc(
        estimate,
        MotionQCConfig(
            policy_version="test-qc-v1",
            thresholds_validated=True,
            min_peak_count_per_time=10,
            min_occupied_depth_bins=2,
            max_step_um=10.0,
            max_speed_um_s=10.0,
        ),
    )
    assert qc.status == "PARTIALLY_VALID"
    assert qc.valid_by_time.tolist() == [True, False, False]
    assert "LOW_PEAK_COUNT" in qc.reason_codes_by_time[1]
    assert "IMPLAUSIBLE_STEP" in qc.reason_codes_by_time[2]


def test_qc_json_uses_null_instead_of_nonstandard_nan(tmp_path):
    recording = FakeRecording()
    backend, _ = successful_backend()
    run_test_sidecar(
        recording,
        recording_for_sorting=recording,
        cache_dir=tmp_path / "motion",
        backend=backend,
    )
    text = (tmp_path / "motion/motion_qc.json").read_text()
    assert "NaN" not in text
    parsed = json.loads(
        text,
        parse_constant=lambda value: (_ for _ in ()).throw(
            AssertionError(f"nonstandard JSON constant {value}")
        ),
    )
    assert parsed["uncertainty_by_time_um"] == [None, None]


def test_success_and_failure_timestamps_are_labeled_correctly(tmp_path):
    recording = FakeRecording()
    backend, _ = successful_backend()
    run_test_sidecar(
        recording,
        recording_for_sorting=recording,
        cache_dir=tmp_path / "success",
        backend=backend,
    )
    manifest = json.loads(
        (tmp_path / "success/dredge-rigid-sidecar/manifest.json").read_text()
    )
    assert "estimated_at_utc" in manifest["provenance"]
    assert "failed_at_utc" not in manifest["provenance"]

    def fail(*args, **kwargs):
        raise RuntimeError("timestamp failure")

    run_test_sidecar(
        recording,
        recording_for_sorting=recording,
        cache_dir=tmp_path / "failure",
        backend=MotionBackend(fail, fail, fail),
    )
    failure = json.loads((tmp_path / "failure/estimation_failure.json").read_text())
    assert "failed_at_utc" in failure


def test_split_half_saves_each_halfs_support_map(tmp_path):
    recording = FakeRecording()
    backend, calls = successful_backend()
    result = run_test_sidecar(
        recording,
        recording_for_sorting=recording,
        cache_dir=tmp_path / "motion",
        config=replace(MotionSidecarConfig(), split_half=True),
        backend=backend,
    )
    assert result.status == "ESTIMATE_COMPLETED"
    assert calls["estimate"] == 3
    audit = tmp_path / "motion/dredge-rigid-sidecar/audits/split_half"
    assert (audit / "half_a_peak_count_by_time_depth.npy").exists()
    assert (audit / "half_b_peak_count_by_time_depth.npy").exists()
    metrics = json.loads((audit / "split_half_metrics.json").read_text())
    assert metrics["complete"] is True
    assert metrics["thresholds_validated"] is False
    assert metrics["authorizes_voltage_correction"] is False
    summary = (tmp_path / "motion/motion_summary.md").read_text()
    assert "Split-half audit: completed (diagnostic only)" in summary

    (audit / "half_a_peak_count_by_time_depth.npy").unlink()
    with pytest.raises(RuntimeError, match="Incomplete motion sidecar"):
        run_test_sidecar(
            recording,
            recording_for_sorting=recording,
            cache_dir=tmp_path / "motion",
            config=replace(MotionSidecarConfig(), split_half=True),
            backend=backend,
        )
