import json
import math
from pathlib import Path

import numpy as np
import pytest

from pipeline import (
    RescueConfig,
    build_kilosort4_params,
    fingerprint,
    pin_sort_identity,
    validate_applied_settings,
)
from pipeline.artifacts import threshold_points
from pipeline.preprocess import (
    _recover_completed_binary_folder,
    recording_binary_receipt,
    select_bad_channel_ids,
    validate_accepted_recording,
)
from pipeline.staging import (
    SOURCE_STAGE_MANIFEST,
    SOURCE_STAGE_SCHEMA,
    stage_spikeglx_stream,
    validate_staged_spikeglx_stream,
)
from pipeline.kilosort_compat import ORIGINAL_BLOCK, PATCHED_BLOCK, patch_source_text
from pipeline.sorting import _archive_declared_failed_partial
from SpikeGLX_ext_ref_rescue import build_run_plan, physical_channel_ids


def sorter_defaults():
    return {
        "do_correction": True,
        "do_CAR": True,
        "artifact_threshold": 300,
        "save_extra_vars": False,
        "Th_universal": 9,
        "Th_learned": 8,
        "duplicate_spike_ms": 0.25,
        "ccg_threshold": 0.25,
        "nearest_chans": 10,
        "nearest_templates": 100,
        "max_channel_distance": 32,
        "clear_cache": False,
        "cross_peel_claim_ms": 0.25,
        "cross_peel_claim_um": 75.0,
    }


def test_rescue_config_fingerprint_is_stable_and_sensitive():
    first = RescueConfig()
    second = RescueConfig()
    changed = RescueConfig(noise_threshold=0.4)
    assert first.digest == second.digest
    assert first.digest != changed.digest
    assert fingerprint({"b": 2, "a": 1}) == fingerprint({"a": 1, "b": 2})
    assert first.materialize_chunk_duration == "10s"


def test_sorter_params_explicitly_disable_rejected_features():
    params = build_kilosort4_params(sorter_defaults())
    assert params["do_correction"] is False
    assert params["do_CAR"] is True
    assert math.isinf(params["artifact_threshold"])
    assert params["cross_peel_claim_ms"] == 0
    assert params["cross_peel_claim_um"] == 0
    assert params["bad_channels"] is None
    assert params["save_extra_vars"] is True
    assert params["Th_universal"] == 12
    assert params["Th_learned"] == 9


def test_sorter_params_refuse_an_unknown_kilosort_schema():
    defaults = sorter_defaults()
    defaults.pop("artifact_threshold")
    with pytest.raises(RuntimeError, match="artifact_threshold"):
        build_kilosort4_params(defaults)


def test_saved_sorter_settings_are_verified():
    requested = build_kilosort4_params(sorter_defaults())
    receipt = validate_applied_settings(requested, requested)
    assert receipt["artifact_threshold"] == "Infinity"
    changed = dict(requested, do_correction=True)
    with pytest.raises(RuntimeError, match="do_correction"):
        validate_applied_settings(changed, requested)


def test_native_nblocks_proves_wrapper_correction_was_disabled():
    requested = build_kilosort4_params(sorter_defaults())
    applied = dict(requested)
    applied.pop("do_correction")
    applied["nblocks"] = 0
    receipt = validate_applied_settings(applied, requested)
    assert receipt["do_correction"] is False
    assert receipt["effective_nblocks"] == 0


def test_native_nblocks_refuses_effective_motion_correction():
    requested = build_kilosort4_params(sorter_defaults())
    applied = dict(requested)
    applied.pop("do_correction")
    applied["nblocks"] = 1
    with pytest.raises(RuntimeError, match="do_correction"):
        validate_applied_settings(applied, requested)


def test_select_bad_channels_uses_both_frozen_metrics():
    selected = select_bad_channel_ids(
        ["AP0", "AP1", "AP2"],
        np.array([0.0, -0.6, 0.0]),
        np.array([0.1, 0.1, 0.4]),
        similarity_threshold=-0.5,
        noise_threshold=0.3,
    )
    assert selected == ["AP1", "AP2"]


def test_threshold_points_excludes_synthetic_channel_from_claim_samples():
    traces = np.array([[0, 214, 0], [-214, 0, 0], [0, -213, 214]], dtype=np.int16)
    samples, channels, values, claim_samples = threshold_points(
        traces,
        start_frame=100,
        channel_ids=np.array([190, 191, 192]),
        threshold_counts=213.33333333333334,
        excluded_channel_ids=[191],
    )
    assert samples.tolist() == [100, 101, 102]
    assert channels.tolist() == [191, 190, 192]
    assert values.tolist() == [214, -214, 214]
    assert claim_samples.tolist() == [101, 102]


def test_threshold_points_rejects_mismatched_channel_ids():
    with pytest.raises(ValueError, match="incompatible"):
        threshold_points(np.zeros((3, 2)), 0, np.array([1, 2, 3]), 10)


def test_physical_bad_channel_resolves_spikeglx_channel_name():
    class Recording:
        def get_channel_ids(self):
            return np.array(["imec1.ap#AP190", "imec1.ap#AP191"])

    assert physical_channel_ids(Recording(), [191]) == ["imec1.ap#AP191"]


def test_run_sheet_plan_uses_frozen_overrides_without_loading_sorter_defaults():
    payload = build_run_plan()
    assert payload["stream_id"] == "imec0.ap"
    assert payload["sorter_overrides"]["do_correction"] is False
    assert payload["sorter_overrides"]["artifact_threshold"] == "Infinity"
    assert payload["motion_sidecar"]["estimator_mode"] == "rigid"
    assert payload["job_settings"]["motion_chunk_duration"] == "2s"
    assert payload["voltage_motion_correction"] is False


def test_accepted_recording_detects_same_size_content_change(tmp_path):
    recording_dir = tmp_path / "recording"
    recording_dir.mkdir()
    binary = recording_dir / "traces.raw"
    binary.write_bytes(b"abcdefgh")
    receipt = recording_binary_receipt(recording_dir)
    manifest = {
        "schema_version": "rescue-recording-manifest-v2",
        "complete": True,
        "expected_binary_bytes": 8,
        **receipt,
    }
    validate_accepted_recording(recording_dir, manifest)
    binary.write_bytes(b"abcdEfgh")
    with pytest.raises(RuntimeError, match="content digest"):
        validate_accepted_recording(recording_dir, manifest)


def test_complete_interrupted_binary_folder_can_be_recovered(tmp_path):
    si = pytest.importorskip("spikeinterface")
    traces = np.arange(800, dtype=np.int16).reshape(200, 4)
    recording = si.NumpyRecording(traces_list=[traces], sampling_frequency=30_000.0)
    partial = tmp_path / "recording.partial"
    recording.save(folder=partial, n_jobs=1, progress_bar=False)
    (partial / "binary.json").unlink()
    (partial / "si_folder.json").unlink()

    _recover_completed_binary_folder(partial, recording)

    loaded = si.load_extractor(partial)
    np.testing.assert_array_equal(loaded.get_traces(), traces)


def test_incomplete_interrupted_binary_folder_is_not_recovered(tmp_path):
    si = pytest.importorskip("spikeinterface")
    traces = np.arange(800, dtype=np.int16).reshape(200, 4)
    recording = si.NumpyRecording(traces_list=[traces], sampling_frequency=30_000.0)
    partial = tmp_path / "recording.partial"
    recording.save(folder=partial, n_jobs=1, progress_bar=False)
    (partial / "binary.json").unlink()
    (partial / "si_folder.json").unlink()
    binary = partial / "traces_cached_seg0.raw"
    binary.write_bytes(binary.read_bytes()[:-2])

    with pytest.raises(RuntimeError, match="binary is incomplete"):
        _recover_completed_binary_folder(partial, recording)


def test_spikeglx_stream_stage_copies_only_selected_stream_and_reuses_it(tmp_path):
    source = tmp_path / "server" / "run"
    source.mkdir(parents=True)
    (source / "run_t0.imec0.ap.bin").write_bytes(b"selected-ap-data")
    (source / "run_t0.imec0.ap.meta").write_text("nSavedChans=2\n")
    (source / "run_t0.imec0.lf.bin").write_bytes(b"do-not-copy")
    (source / "run_t0.imec0.lf.meta").write_text("nSavedChans=2\n")
    staged = tmp_path / "nvme" / "source"

    first = stage_spikeglx_stream(source, staged, stream_id="imec0.ap")
    second = stage_spikeglx_stream(source, staged, stream_id="imec0.ap")

    assert first == second
    assert (staged / "run_t0.imec0.ap.bin").read_bytes() == b"selected-ap-data"
    assert not (staged / "run_t0.imec0.lf.bin").exists()
    validate_staged_spikeglx_stream(staged)


def test_spikeglx_stream_stage_resumes_partial_binary(tmp_path):
    source = tmp_path / "server"
    source.mkdir()
    data = b"abcdefghijklmnopqrstuvwxyz"
    (source / "run_t0.imec1.ap.bin").write_bytes(data)
    (source / "run_t0.imec1.ap.meta").write_text("nSavedChans=2\n")
    staged = tmp_path / "nvme" / "source"
    partial = staged.with_name("source.partial")
    partial.mkdir(parents=True)
    (partial / "run_t0.imec1.ap.bin").write_bytes(data[:10])
    stat = (source / "run_t0.imec1.ap.bin").stat()
    meta_stat = (source / "run_t0.imec1.ap.meta").stat()
    (partial / SOURCE_STAGE_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": SOURCE_STAGE_SCHEMA,
                "source_folder": str(source.resolve()),
                "stream_id": "imec1.ap",
                "source_files": [
                    {
                        "relative_path": "run_t0.imec1.ap.bin",
                        "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    },
                    {
                        "relative_path": "run_t0.imec1.ap.meta",
                        "size_bytes": meta_stat.st_size,
                        "mtime_ns": meta_stat.st_mtime_ns,
                    },
                ],
                "complete": False,
            }
        )
        + "\n"
    )

    stage_spikeglx_stream(source, staged, stream_id="imec1.ap")

    assert (staged / "run_t0.imec1.ap.bin").read_bytes() == data


def test_kilosort_empty_center_patch_is_exact_and_minimal():
    source = "before\n" + ORIGINAL_BLOCK + "after\n"
    patched = patch_source_text(source)
    assert patched == "before\n" + PATCHED_BLOCK + "after\n"
    assert "all(value is None for value in data_result)" in patched
    with pytest.raises(RuntimeError, match="not unique"):
        patch_source_text(patched)


def test_declared_failed_sort_partial_is_archived(tmp_path):
    partial = tmp_path / "kilosort4.partial"
    partial.mkdir()
    (partial / "spikeinterface_log.json").write_text(
        json.dumps({"error": True, "error_trace": ["synthetic failure"]})
    )
    (partial / "evidence.txt").write_text("preserve me")

    archived = _archive_declared_failed_partial(partial)

    assert not partial.exists()
    assert archived.name.startswith("kilosort4.failed-")
    assert (archived / "evidence.txt").read_text() == "preserve me"


def test_ambiguous_sort_partial_is_not_archived(tmp_path):
    partial = tmp_path / "kilosort4.partial"
    partial.mkdir()
    (partial / "spikeinterface_log.json").write_text(
        json.dumps({"error": False, "error_trace": []})
    )
    with pytest.raises(RuntimeError, match="requires inspection"):
        _archive_declared_failed_partial(partial)
    assert partial.exists()


def test_pinned_sort_identity_reuses_exact_files_and_rejects_replacement(tmp_path):
    kilosort = tmp_path / "kilosort4"
    sorter = kilosort / "sorter_output"
    sorter.mkdir(parents=True)
    manifest = {
        "complete": True,
        "request_digest": "sort-request",
        "recording_request_digest": "recording-request",
        "summary": {"final_spike_count": 3, "unit_count": 2},
    }
    (kilosort / "rescue_sort_manifest.json").write_text(json.dumps(manifest))
    np.save(sorter / "spike_times.npy", np.array([1, 2, 3], dtype=np.int64))
    np.save(sorter / "spike_clusters.npy", np.array([0, 0, 1], dtype=np.int32))
    np.save(sorter / "templates.npy", np.zeros((2, 3, 2), dtype=np.float32))
    np.save(sorter / "ops.npy", {"fs": 30_000}, allow_pickle=True)
    (sorter / "cluster_KSLabel.tsv").write_text(
        "cluster_id\tKSLabel\n0\tgood\n1\tmua\n"
    )
    identity_path = tmp_path / "rescue_sort_identity.json"

    first = pin_sort_identity(kilosort, identity_path)
    second = pin_sort_identity(kilosort, identity_path)
    assert first["identity_digest"] == second["identity_digest"]

    np.save(sorter / "spike_clusters.npy", np.array([0, 1, 1], dtype=np.int32))
    with pytest.raises(RuntimeError, match="pinned sort identity"):
        pin_sort_identity(kilosort, identity_path)
