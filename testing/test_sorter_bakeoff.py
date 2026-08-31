import json

import numpy as np
import pytest

from pipeline.bakeoff import (
    BAKEOFF_SCHEMA,
    accept_ks4_reference,
    build_bakeoff_plan,
    validate_dartsort_output,
    resolve_kiasort_installation,
    normalize_dartsort_output,
    resolve_bakeoff_window,
)


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
        ["ks4_no_motion", "dartsort_native", "kiasort", "si_motion_aware_peeler"],
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
    assert by_name["si_motion_aware_peeler"]["runnable_now"] is False
    assert "unit count alone" in plan["advancement_rule"]


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
