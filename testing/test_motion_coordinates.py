import json

import numpy as np
import pytest

from pipeline.motion_coordinates import (
    MOTION_FIELD_SCHEMA,
    interpolate_motion_at_spikes,
    build_spikeinterface_motion,
    load_qualified_motion_field,
    write_motion_coordinate_sidecar,
)


def write_field(path, *, qualified=True, time_reference="selected_recording_start"):
    np.savez(
        path,
        schema_version=MOTION_FIELD_SCHEMA,
        qualification_passed=qualified,
        time_reference=time_reference,
        depth_reference="probe_y_um",
        displacement_convention="observed_depth_offset_um",
        displacement_um=np.array([[0.0, 10.0], [20.0, 30.0]]),
        time_s=np.array([0.0, 10.0]),
        depth_um=np.array([100.0, 200.0]),
        support=np.full((2, 2), 5.0),
        confidence=np.full((2, 2), 0.9),
        estimator="coarse-test",
        polarity="both",
        qualification_digest="test-qualification",
    )


def test_bilinear_coordinate_correction_is_support_gated(tmp_path):
    field_path = tmp_path / "field.npz"
    write_field(field_path)
    field = load_qualified_motion_field(field_path)
    sampled = interpolate_motion_at_spikes(
        field,
        spike_time_s=np.array([5.0, 11.0]),
        raw_depth_um=np.array([150.0, 150.0]),
    )
    assert sampled["supported"].tolist() == [True, False]
    assert sampled["displacement_um"][0] == pytest.approx(15.0)
    assert np.isnan(sampled["displacement_um"][1])


def test_low_confidence_corner_invalidates_interpolation(tmp_path):
    field_path = tmp_path / "field.npz"
    write_field(field_path)
    field = load_qualified_motion_field(field_path)
    field["confidence"][1, 1] = 0.1
    sampled = interpolate_motion_at_spikes(
        field, np.array([5.0]), np.array([150.0]), min_confidence=0.5
    )
    assert not sampled["supported"][0]
    assert np.isnan(sampled["displacement_um"][0])


@pytest.mark.parametrize(
    "qualified,time_reference,match",
    [
        (False, "selected_recording_start", "independent qualification"),
        (True, "acquisition_start", "time_reference"),
    ],
)
def test_field_handoff_refuses_unqualified_or_ambiguous_time(
    tmp_path, qualified, time_reference, match
):
    field_path = tmp_path / "field.npz"
    write_field(field_path, qualified=qualified, time_reference=time_reference)
    with pytest.raises(ValueError, match=match):
        load_qualified_motion_field(field_path)


def test_sidecar_preserves_raw_depth_and_records_gain(tmp_path):
    root = tmp_path / "result"
    sort_dir = root / "kilosort4"
    sorter_output = sort_dir / "sorter_output"
    recording_dir = root / "recording"
    sorter_output.mkdir(parents=True)
    recording_dir.mkdir()
    (sort_dir / "rescue_sort_manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "request_digest": "sort-digest",
                "recording_request_digest": "recording-digest",
            }
        )
    )
    (recording_dir / "rescue_recording_manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "request_digest": "recording-digest",
                "sampling_frequency_hz": 10.0,
            }
        )
    )
    np.save(sorter_output / "spike_times.npy", np.array([[50], [110]]))
    np.save(sorter_output / "spike_clusters.npy", np.array([3, 4]))
    np.save(
        sorter_output / "spike_positions.npy", np.array([[0.0, 150.0], [0.0, 150.0]])
    )
    field_path = tmp_path / "field.npz"
    write_field(field_path)

    manifest = write_motion_coordinate_sidecar(
        sort_dir, field_path, root / "motion_coordinates", gain=0.5, chunk_spikes=1
    )
    sidecar = root / "motion_coordinates"
    np.testing.assert_array_equal(np.load(sidecar / "raw_depth_um.npy"), [150.0, 150.0])
    corrected = np.load(sidecar / "motion_corrected_depth_um.npy")
    assert corrected[0] == pytest.approx(142.5)
    assert np.isnan(corrected[1])
    assert np.load(sidecar / "supported.npy").tolist() == [True, False]
    assert manifest["gain"] == 0.5
    assert manifest["voltage_modified"] is False
    assert manifest["supported_spike_fraction"] == 0.5
    assert manifest["chunk_spikes"] == 1


def test_qualified_field_converts_to_spikeinterface_motion(tmp_path):
    pytest.importorskip("spikeinterface.core.motion")
    field_path = tmp_path / "field.npz"
    write_field(field_path)
    motion = build_spikeinterface_motion(field_path, gain=0.5)
    np.testing.assert_array_equal(
        motion.displacement[0], np.array([[0.0, 5.0], [10.0, 15.0]])
    )
    np.testing.assert_array_equal(motion.temporal_bins_s[0], [0.0, 10.0])


def test_spikeinterface_motion_refuses_partial_support(tmp_path):
    field_path = tmp_path / "field.npz"
    write_field(field_path)
    field = dict(np.load(field_path))
    field["confidence"] = np.asarray(field["confidence"]).copy()
    field["confidence"][0, 0] = 0.1
    np.savez(field_path, **field)
    with pytest.raises(RuntimeError, match="fully supported"):
        build_spikeinterface_motion(field_path, min_confidence=0.5)
