import hashlib
import json
import subprocess

import numpy as np
import pytest

from pipeline.bakeoff import (
    BAKEOFF_SCHEMA,
    KS4_PEELER_SCHEMA,
    accept_ks4_reference,
    build_bakeoff_plan,
    validate_dartsort_output,
    resolve_kiasort_installation,
    normalize_dartsort_output,
    normalize_kiasort_curated_output,
    load_ks4_rigid_motion,
    stabilize_ks4_rigid_motion,
    summarize_ks4_seeded_peeler_arm,
    _normalize_peeler_spikes,
    _adapt_kiasort_wrapper_channel_map,
    _validate_kiasort_native_geometry,
    _recover_kiasort_postcurate_rollback,
    resolve_bakeoff_window,
)


def test_kiasort_channel_map_adapter_writes_matlab_column_vectors(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")

    class Wrapper:
        savemat = staticmethod(scipy_io.savemat)

    _adapt_kiasort_wrapper_channel_map(Wrapper)
    path = tmp_path / "channel_map.mat"
    Wrapper.savemat(
        path,
        {
            "chanMap": np.arange(1, 4),
            "connected": np.ones(3, dtype=bool),
            "xcoords": np.array([0.0, 32.0, 16.0]),
            "ycoords": np.array([100.0, 100.0, 120.0]),
        },
    )
    saved = scipy_io.loadmat(path)
    assert saved["xcoords"].shape == (3, 1)
    assert saved["ycoords"].shape == (3, 1)


def test_kiasort_geometry_validator_rejects_flattened_coordinates(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    samples = tmp_path / "RES_Samples"
    samples.mkdir()
    scipy_io.savemat(
        samples / "channel_info.mat",
        {
            "channel_locations": np.arange(9.0).reshape(1, 9),
            "channel_mapping": np.arange(1, 4),
        },
    )
    with pytest.raises(RuntimeError, match="malformed"):
        _validate_kiasort_native_geometry(tmp_path, 3)


def test_kiasort_geometry_validator_reads_matlab_v73_dimension_order(tmp_path):
    h5py = pytest.importorskip("h5py")
    samples = tmp_path / "RES_Samples"
    samples.mkdir()
    with h5py.File(samples / "channel_info.mat", "w") as handle:
        handle.create_dataset("channel_locations", data=np.zeros((3, 4)))
        handle.create_dataset("channel_mapping", data=np.arange(1, 5).reshape(4, 1))
    report = _validate_kiasort_native_geometry(tmp_path, 4)
    assert report["channel_locations_shape"] == [4, 3]
    assert report["channel_info_mat_format"] == "v7.3_hdf5"


def test_kiasort_postcurate_recovery_requires_consistent_atomic_backup(tmp_path):
    h5py = pytest.importorskip("h5py")
    results = tmp_path / "RES_Sorted"
    backup = tmp_path / "Backup/postcurate"
    samples = tmp_path / "Sorted_Samples"
    results.mkdir(parents=True)
    backup.mkdir(parents=True)
    samples.mkdir()
    for path, key, values in (
        (results / "spike_idx.h5", "spike_idx", [10, 20]),
        (results / "unifiedLabels.h5", "unifiedLabels", [9, 9]),
        (backup / "spike_idx.h5", "spike_idx", [10, 20]),
        (backup / "unifiedLabels.h5", "unifiedLabels", [1, 2]),
    ):
        with h5py.File(path, "w") as handle:
            handle.create_dataset(key, data=values)
    (samples / "sorted_samples.mat").write_bytes(b"same")
    (backup / "sorted_samples.mat").write_bytes(b"same")
    (tmp_path / "KIASort_log.txt").write_text("Post-hoc processing finished\n")
    report = _recover_kiasort_postcurate_rollback(tmp_path)
    assert report["recovered"] is True
    with h5py.File(results / "unifiedLabels.h5") as handle:
        np.testing.assert_array_equal(handle["unifiedLabels"][:], [1, 2])


def test_kiasort_installation_diff_hash_includes_staged_changes(tmp_path):
    root = tmp_path / "kiasort"
    wrapper = root / "SpikeInterface_wrapper/kiasort_spikeinterface.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("def run_kiasort():\n    pass\n")
    (root / "kiaSort.m").write_text("% gui\n")
    (root / "run_kiasort_nogui.m").write_text("% nogui\n")
    source = root / "kiaSort_main.m"
    source.write_text("before\n")
    subprocess.run(["git", "init", "-q", root], check=True)
    subprocess.run(["git", "-C", root, "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", root, "config", "user.email", "test@example.com"], check=True
    )
    subprocess.run(["git", "-C", root, "add", "."], check=True)
    subprocess.run(["git", "-C", root, "commit", "-qm", "initial"], check=True)
    source.write_text("after\n")
    subprocess.run(["git", "-C", root, "add", source.name], check=True)
    expected_diff = subprocess.run(
        ["git", "-C", root, "diff", "HEAD", "--binary", "--no-ext-diff"],
        check=True,
        capture_output=True,
    ).stdout
    installation = resolve_kiasort_installation(root)
    assert installation["tracked_diff_sha256"] == hashlib.sha256(expected_diff).hexdigest()
    assert expected_diff


def test_kiasort_postcurate_recovery_rejects_mixed_current_and_backup_files(tmp_path):
    h5py = pytest.importorskip("h5py")
    results = tmp_path / "RES_Sorted"
    backup = tmp_path / "Backup/postcurate"
    samples = tmp_path / "Sorted_Samples"
    results.mkdir(parents=True)
    backup.mkdir(parents=True)
    samples.mkdir()
    for path, key, values in (
        (results / "spike_idx.h5", "spike_idx", [10, 30]),
        (results / "unifiedLabels.h5", "unifiedLabels", [1, 2]),
        (backup / "spike_idx.h5", "spike_idx", [10, 20]),
        (backup / "unifiedLabels.h5", "unifiedLabels", [1, 2]),
    ):
        with h5py.File(path, "w") as handle:
            handle.create_dataset(key, data=values)
    (samples / "sorted_samples.mat").write_bytes(b"same")
    (backup / "sorted_samples.mat").write_bytes(b"same")
    report = _recover_kiasort_postcurate_rollback(tmp_path)
    assert report == {
        "needed": True,
        "recovered": False,
        "reason": "spike backup differs",
    }


def write_recording_manifest(folder):
    folder.mkdir()
    (folder / "rescue_recording_manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "request_digest": "recording-digest",
                "selected_start_frame": 100,
                "selected_end_frame": 1100,
                "sampling_frequency_hz": 30_000.0,
            }
        )
    )


def test_plan_compares_architectures_without_allowing_voltage_warp(tmp_path):
    recording = tmp_path / "recording"
    write_recording_manifest(recording)
    plan = build_bakeoff_plan(
        recording,
        [
            "ks4_no_motion",
            "dartsort_native",
            "kiasort",
            "kiasort_auto_curated",
            "ks4_seeded_static_peeler",
            "ks4_seeded_motion_native_peeler",
            "ks4_seeded_motion_stabilized_peeler",
        ],
    )
    assert plan["schema_version"] == BAKEOFF_SCHEMA
    assert plan["selected_frame_count"] == 1000
    assert plan["shared_input_policy"]["voltage_motion_resampling_allowed"] is False
    assert all(
        candidate["raw_voltage_warp"] is False for candidate in plan["candidates"]
    )
    assert plan["candidates"][0]["runnable_now"] is True
    by_name = {candidate["name"]: candidate for candidate in plan["candidates"]}
    assert by_name["kiasort"]["runnable_now"] is False
    assert by_name["kiasort_auto_curated"]["source_output_required"] is True
    assert by_name["ks4_seeded_static_peeler"]["raw_voltage_warp"] is False
    assert by_name["ks4_seeded_motion_native_peeler"]["motion_source_required"] is True
    assert "unit count alone" in plan["advancement_rule"]


def test_ks4_motion_stabilization_rejects_only_prespecified_large_steps():
    native = np.array([-74.0, -73.0, 4.0, 5.5, -68.0, -66.0])
    stabilized, report = stabilize_ks4_rigid_motion(native, max_step_um=20.0)
    np.testing.assert_allclose(stabilized, [-74.0, -73.0, -73.0, -71.5, -71.5, -69.5])
    assert report["rejected_step_indices"] == [2, 4]
    assert report["maximum_stabilized_step_um"] == 2.0


def test_ks4_rigid_motion_loader_requires_rigid_and_median_centers(tmp_path):
    recording = tmp_path / "recording"
    write_recording_manifest(recording)
    manifest = json.loads((recording / "rescue_recording_manifest.json").read_text())
    window = resolve_bakeoff_window(manifest)
    ops_path = tmp_path / "ops.npy"
    np.save(
        ops_path,
        {
            "dshift": np.array([[-70.0], [-69.0], [5.0], [6.0]]),
            "fs": 30_000.0,
            "batch_size": 200,
        },
        allow_pickle=True,
    )
    loaded = load_ks4_rigid_motion(
        ops_path, window=window, max_step_um=20.0
    )
    assert loaded["native_um"].shape == (4,)
    assert np.median(loaded["native_um"]) == 0.0
    assert np.median(loaded["stabilized_um"]) == 0.0
    np.testing.assert_allclose(loaded["native_um"], [38.0, 37.0, -37.0, -38.0])
    assert loaded["spikeinterface_motion_transform"].startswith("negative_")
    assert loaded["stabilization"]["rejected_step_count"] == 1


def test_peeler_normalizer_restores_ks4_unit_ids_and_rejects_invalid(tmp_path):
    dtype = [
        ("sample_index", "int64"),
        ("cluster_index", "int64"),
        ("channel_index", "int64"),
        ("amplitude", "float32"),
    ]
    spikes = np.array(
        [(7, 1, 3, 1.2), (3, 0, 2, 0.9), (20, 0, 2, 1.0), (4, 9, 1, 1.0)],
        dtype=dtype,
    )
    summary = _normalize_peeler_spikes(
        spikes, np.array([101, 205]), tmp_path, num_samples=10
    )
    np.testing.assert_array_equal(np.load(tmp_path / "spike_times.npy"), [3, 7])
    np.testing.assert_array_equal(np.load(tmp_path / "spike_labels.npy"), [101, 205])
    assert summary["invalid_or_out_of_bounds_spike_count"] == 2
    assert KS4_PEELER_SCHEMA.endswith("-v2")


def test_peeler_guardrails_are_label_preserving_and_detect_cross_unit_pairs():
    metrics = summarize_ks4_seeded_peeler_arm(
        reference_times=np.array([100, 500, 900, 1500]),
        reference_labels=np.array([1, 1, 2, 2]),
        spike_times=np.array([101, 501, 502, 905, 1501]),
        spike_labels=np.array([1, 1, 2, 2, 2]),
        sampling_frequency_hz=1000.0,
        duration_s=2.0,
        event_tolerance_ms=5.0,
        refractory_ms=2.0,
        duplicate_ms=2.0,
        presence_bin_s=1.0,
    )
    assert metrics["label_preserving_reference_event_recovery"] == 1.0
    assert metrics["cross_unit_near_coincident_pair_count"] == 1
    assert metrics["units_present_in_first_and_last_20s_fraction"] == 0.5


def test_peeler_guardrails_match_events_once_and_count_all_cross_unit_pairs():
    metrics = summarize_ks4_seeded_peeler_arm(
        reference_times=np.array([100, 102]),
        reference_labels=np.array([1, 1]),
        spike_times=np.array([101, 500, 500, 500]),
        spike_labels=np.array([1, 1, 2, 3]),
        sampling_frequency_hz=1000.0,
        duration_s=1.0,
        event_tolerance_ms=2.0,
        duplicate_ms=1.0,
    )
    assert metrics["label_preserving_reference_event_recovery"] == 0.5
    assert metrics["cross_unit_near_coincident_pair_count"] == 3


def test_plan_refuses_unknown_candidate(tmp_path):
    recording = tmp_path / "recording"
    write_recording_manifest(recording)
    with pytest.raises(ValueError, match="Unknown"):
        build_bakeoff_plan(recording, ["mystery_sorter"])


def test_window_is_relative_to_materialized_recording_and_tracks_source_frames(tmp_path):
    recording = tmp_path / "recording"
    write_recording_manifest(recording)
    manifest = json.loads((recording / "rescue_recording_manifest.json").read_text())
    window = resolve_bakeoff_window(
        manifest, name="middle", start_s=0.01, duration_s=0.02
    )
    assert window.start_frame == 300
    assert window.end_frame == 900
    assert window.source_start_frame == 400
    assert window.source_end_frame == 1000
    assert window.frame_count == 600
    assert window.directory_name.startswith("middle-")


def test_window_rejects_interval_past_recording(tmp_path):
    recording = tmp_path / "recording"
    write_recording_manifest(recording)
    manifest = json.loads((recording / "rescue_recording_manifest.json").read_text())
    with pytest.raises(ValueError, match="outside"):
        resolve_bakeoff_window(manifest, start_s=0.02, duration_s=0.02)


def test_dartsort_output_validator_uses_documented_final_arrays(tmp_path):
    np.savez(
        tmp_path / "dartsort_sorting.npz",
        times_samples=np.array([10, 20, 30]),
        channels=np.array([1, 2, 3]),
        labels=np.array([5, -1, 7]),
    )
    summary = validate_dartsort_output(tmp_path)
    assert summary == {
        "final_spike_count": 3,
        "assigned_spike_count": 2,
        "unit_count": 2,
    }
    normalized = normalize_dartsort_output(tmp_path, tmp_path, num_samples=100)
    np.testing.assert_array_equal(np.load(tmp_path / "spike_times.npy"), [10, 30])
    np.testing.assert_array_equal(np.load(tmp_path / "spike_labels.npy"), [5, 7])
    assert normalized["normalized_spike_count"] == 2


def test_dartsort_output_validator_rejects_inconsistent_arrays(tmp_path):
    np.savez(
        tmp_path / "dartsort_sorting.npz",
        times_samples=np.array([10, 20]),
        channels=np.array([1]),
        labels=np.array([5, 7]),
    )
    with pytest.raises(RuntimeError, match="inconsistent"):
        validate_dartsort_output(tmp_path)


def test_kiasort_curated_normalizer_applies_labels_and_stable_interval(tmp_path):
    h5py = pytest.importorskip("h5py")
    scipy_io = pytest.importorskip("scipy.io")
    native = tmp_path / "native"
    results = native / "RES_Sorted"
    results.mkdir(parents=True)
    arrays = {
        "spike_idx_curated": np.array([30, 10, 20, 40]),
        "unifiedLabels_curated": np.array([3, 1, -1, 4]),
        "channelNum_curated": np.array([2, 1, 1, 2]),
        "inclusion_curated": np.array([1, 1, 1, 0]),
    }
    for name, values in arrays.items():
        with h5py.File(results / f"{name}.h5", "w") as handle:
            handle.create_dataset(name, data=values)
    scipy_io.savemat(native / "channel_map.mat", {"ycoords": np.array([100.0, 120.0])})
    summary = normalize_kiasort_curated_output(native, tmp_path, num_samples=100)
    np.testing.assert_array_equal(np.load(tmp_path / "spike_times.npy"), [10, 30])
    np.testing.assert_array_equal(np.load(tmp_path / "spike_labels.npy"), [1, 3])
    np.testing.assert_array_equal(np.load(tmp_path / "spike_depths_um.npy"), [100, 120])
    assert summary["curated_assigned_spike_count_before_stable_interval"] == 3
    assert summary["stable_interval_excluded_spike_count"] == 1


def test_existing_ks4_sort_is_registered_without_rerun(tmp_path):
    recording = tmp_path / "recording"
    write_recording_manifest(recording)
    ks4 = tmp_path / "kilosort4"
    ks4.mkdir()
    (ks4 / "rescue_sort_manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "recording_request_digest": "recording-digest",
                "request_digest": "sort-digest",
                "summary": {"final_spike_count": 42, "unit_count": 3},
            }
        )
    )
    output = tmp_path / "bakeoff" / "ks4_no_motion"
    receipt = accept_ks4_reference(recording, ks4, output)
    assert receipt["candidate"] == "ks4_no_motion"
    assert receipt["summary"]["unit_count"] == 3
    assert receipt["raw_voltage_warp"] is False
    assert (output / "bakeoff_sort_manifest.json").exists()


def test_ks4_window_is_extracted_and_rebased_without_rerun(tmp_path):
    recording = tmp_path / "recording"
    write_recording_manifest(recording)
    ks4 = tmp_path / "kilosort4"
    native = ks4 / "sorter_output"
    native.mkdir(parents=True)
    np.save(native / "spike_times.npy", np.array([100, 300, 600, 899, 900]))
    np.save(native / "spike_clusters.npy", np.array([1, 2, 2, 3, 4]))
    (ks4 / "rescue_sort_manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "recording_request_digest": "recording-digest",
                "request_digest": "sort-digest",
                "summary": {"final_spike_count": 5, "unit_count": 4},
            }
        )
    )
    output = tmp_path / "bakeoff" / "middle" / "ks4_no_motion"
    receipt = accept_ks4_reference(
        recording,
        ks4,
        output,
        window_name="middle",
        start_s=0.01,
        duration_s=0.02,
    )
    np.testing.assert_array_equal(np.load(output / "spike_times.npy"), [0, 300, 599])
    np.testing.assert_array_equal(np.load(output / "spike_labels.npy"), [2, 2, 3])
    assert receipt["reference_method"] == "accepted_full_sort_window_extraction"
    assert receipt["summary"]["final_spike_count"] == 3


def test_kiasort_upstream_wrapper_is_resolved_and_fingerprinted(tmp_path):
    wrapper = tmp_path / "SpikeInterface_wrapper" / "kiasort_spikeinterface.py"
    entrypoint = tmp_path / "run_kiasort_nogui.m"
    mirrored_entrypoint = tmp_path / "No_GUI" / "run_kiasort_nogui.m"
    gui = tmp_path / "kiaSort.m"
    wrapper.parent.mkdir()
    mirrored_entrypoint.parent.mkdir()
    wrapper.write_text("def run_kiasort(): pass\n")
    entrypoint.write_text("function run_kiasort_nogui()\nend\n")
    mirrored_entrypoint.write_text("function run_kiasort_nogui()\nend\n")
    gui.write_text("function kiaSort()\nend\n")
    resolved = resolve_kiasort_installation(tmp_path)
    assert resolved["configured"] is True
    assert resolved["wrapper_sha256"]
    assert resolved["gui_entrypoint_sha256"]
    assert resolved["nogui_dir"] == str(entrypoint.parent)
