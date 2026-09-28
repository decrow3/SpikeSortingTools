import numpy as np
import pytest

from npx_preprocessing.motion.lattice_remap_si import (
    ExactLatticeRemapRecording,
    audit_lattice_mappings,
    audit_spikeinterface_nearest_kernel,
    exact_coordinate_mapping,
    sample_stepwise_shifts,
    write_mapping_audit,
)


def np1_geometry(rows=10):
    sites = []
    for row in range(rows):
        xs = (16.0, 48.0) if row % 2 == 0 else (0.0, 32.0)
        sites.extend((x, row * 20.0) for x in xs)
    return np.asarray(sites, dtype=np.float64)


def expected_remap(source, geometry, sampled_shifts):
    output = np.zeros_like(source)
    for shift in np.unique(sampled_shifts):
        rows = np.flatnonzero(sampled_shifts == shift)
        mapping = exact_coordinate_mapping(geometry, shift)
        valid = np.flatnonzero(mapping >= 0)
        output[np.ix_(rows, valid)] = source[np.ix_(rows, mapping[valid])]
    return output


def make_recording(geometry, *, fs=40.0, n_frames=40, t_start=10.0, dtype=np.int16):
    si = pytest.importorskip("spikeinterface")
    values = (
        np.arange(n_frames, dtype=np.int64)[:, None] * 1000
        + np.arange(len(geometry), dtype=np.int64)[None, :]
        + 1
    ).astype(dtype)
    recording = si.NumpyRecording(values, sampling_frequency=fs, t_starts=[t_start])
    recording.set_channel_locations(geometry)
    return recording, values


def test_exact_mapping_preserves_np1_phase_and_zero_fills_outer_support():
    geometry = np1_geometry(8)
    for shift in (-80, -40, 0, 40, 80):
        mapping = exact_coordinate_mapping(geometry, shift)
        valid = np.flatnonzero(mapping >= 0)
        np.testing.assert_array_equal(geometry[mapping[valid], 0], geometry[valid, 0])
        np.testing.assert_allclose(
            geometry[mapping[valid], 1], geometry[valid, 1] + shift
        )
        assert np.unique(mapping[valid]).size == valid.size
    np.testing.assert_array_equal(
        exact_coordinate_mapping(geometry, 0), np.arange(len(geometry))
    )


def test_mapping_audit_exposes_interior_nearest_substitution():
    full = np1_geometry(8)
    missing_coordinate = np.array([16.0, 80.0])
    keep = ~np.all(full == missing_coordinate, axis=1)
    geometry = full[keep]
    rows = audit_lattice_mappings(geometry, [40])
    interior = [row for row in rows if row["support_class"].startswith("interior_missing")]
    assert interior
    assert all(row["exact_source_index"] == -1 for row in interior)
    assert all(row["nearest_source_index"] >= 0 for row in interior)
    assert all(row["nearest_distance_um"] > 0 for row in interior)


def test_stock_nearest_force_zeros_still_substitutes_at_interior_hole():
    pytest.importorskip("spikeinterface")
    from spikeinterface.core.motion import Motion
    from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording

    full = np1_geometry(10)
    missing_coordinate = np.array([16.0, 80.0])
    geometry = full[~np.all(full == missing_coordinate, axis=1)]
    recording, source = make_recording(geometry)
    centers = 10.0 + np.array([0.125, 0.375, 0.625, 0.875])
    shifts = np.full(centers.size, 40.0)
    exact = ExactLatticeRemapRecording(
        recording, centers, shifts, cell_width_s=0.25
    ).get_traces()
    motion = Motion(
        displacement=shifts[:, None],
        temporal_bins_s=centers,
        spatial_bins_um=np.array([geometry[:, 1].mean()]),
        direction="y",
    )
    stock = InterpolateMotionRecording(
        recording,
        motion,
        border_mode="force_zeros",
        spatial_interpolation_method="nearest",
        dtype="float32",
    ).get_traces()

    target = int(np.flatnonzero(np.all(geometry == [16.0, 40.0], axis=1))[0])
    assert np.all(exact[:, target] == 0)
    assert np.all(stock[:, target] != 0)
    mapping = exact_coordinate_mapping(geometry, 40)
    valid = np.flatnonzero(mapping >= 0)
    np.testing.assert_array_equal(stock[:, valid], source[:, mapping[valid]])

    audit = audit_spikeinterface_nearest_kernel(geometry, [40])
    audited_target = audit[target]
    assert audited_target["exact_source_index"] == -1
    assert audited_target["stock_source_index"] >= 0
    assert not audited_target["stock_matches_exact"]


def test_mapping_audit_writes_compact_inspectable_evidence(tmp_path):
    full = np1_geometry(8)
    geometry = np.delete(full, 8, axis=0)
    summary = write_mapping_audit(tmp_path / "audit", geometry, [-40, 0, 40])
    assert summary["exact_rows_all_match_nearest"]
    assert summary["unsupported_targets_nearest_would_substitute"] > 0
    assert (tmp_path / "audit" / "MAPPING_AUDIT.csv").stat().st_size > 0
    assert (tmp_path / "audit" / "SUMMARY.json").stat().st_size > 0


def test_mapping_audit_accepts_distinct_source_and_target_geometries(tmp_path):
    source = np1_geometry(10)
    source = source[~np.all(source == [16.0, 80.0], axis=1)]
    targets = source[(source[:, 1] >= 40.0) & (source[:, 1] <= 140.0)]
    summary = write_mapping_audit(
        tmp_path / "audit", source, [-40, 0, 40], target_locations=targets
    )
    assert summary["num_source_channels"] == len(source)
    assert summary["num_target_channels"] == len(targets)
    assert summary["unsupported_targets_nearest_would_substitute"] > 0


def test_stepwise_sampling_assigns_midpoint_to_later_cell_and_rejects_tail():
    centers = np.array([10.125, 10.375, 10.625])
    shifts = np.array([0.0, 40.0, -40.0])
    times = np.array([10.249, 10.25, 10.251, 10.499, 10.5, 10.501])
    sampled = sample_stepwise_shifts(centers, shifts, times, cell_width_s=0.25)
    np.testing.assert_array_equal(sampled, [0, 40, 40, 40, -40, -40])
    with pytest.raises(ValueError, match="do not cover"):
        sample_stepwise_shifts(centers, shifts, [10.75], cell_width_s=0.25)


@pytest.mark.parametrize("sign", [-1, 1])
def test_lazy_adapter_matches_complete_oracle_for_signs_holes_and_borders(sign):
    full = np1_geometry(10)
    geometry = np.delete(full, 8, axis=0)  # AP191-like interior support hole.
    recording, source = make_recording(geometry)
    centers = 10.0 + np.array([0.125, 0.375, 0.625, 0.875])
    shifts = sign * np.array([0.0, 40.0, -40.0, 80.0])
    corrected = ExactLatticeRemapRecording(
        recording, centers, shifts, cell_width_s=0.25
    )
    times = 10.0 + np.arange(source.shape[0]) / 40.0
    sampled = sample_stepwise_shifts(centers, shifts, times, cell_width_s=0.25)
    expected = expected_remap(source, geometry, sampled)
    actual = corrected.get_traces()
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == source.dtype
    np.testing.assert_array_equal(actual[:10], source[:10])  # q=0 is byte-exact.


def test_chunked_nonzero_clock_reads_equal_full_read_and_channel_subset():
    geometry = np1_geometry(10)
    recording, _ = make_recording(geometry)
    centers = 10.0 + np.array([0.125, 0.375, 0.625, 0.875])
    shifts = np.array([0.0, 40.0, -40.0, 80.0])
    corrected = ExactLatticeRemapRecording(
        recording, centers, shifts, cell_width_s=0.25
    )
    full = corrected.get_traces()
    chunked = np.concatenate(
        [
            corrected.get_traces(start_frame=0, end_frame=7),
            corrected.get_traces(start_frame=7, end_frame=23),
            corrected.get_traces(start_frame=23, end_frame=40),
        ]
    )
    np.testing.assert_array_equal(chunked, full)
    selected_ids = corrected.channel_ids[[1, 4, 9]]
    np.testing.assert_array_equal(
        corrected.get_traces(channel_ids=selected_ids), full[:, [1, 4, 9]]
    )


def test_single_segment_accepts_plain_lists_and_preserves_acquisition_clock():
    geometry = np1_geometry(6)
    recording, _ = make_recording(geometry)
    corrected = ExactLatticeRemapRecording(
        recording,
        [10.125, 10.375, 10.625, 10.875],
        [0, 40, -40, 0],
        cell_width_s=0.25,
    )
    assert corrected.sample_index_to_time(0) == pytest.approx(10.0)
    assert corrected.get_traces().shape == (40, len(geometry))


def test_multiple_segments_use_their_own_clocks_and_motion_tables():
    si = pytest.importorskip("spikeinterface")
    geometry = np1_geometry(6)
    first = np.tile(np.arange(len(geometry), dtype=np.int16) + 1, (8, 1))
    second = first + 100
    recording = si.NumpyRecording(
        [first, second], sampling_frequency=8.0, t_starts=[10.0, 20.0]
    )
    recording.set_channel_locations(geometry)
    centers = [np.array([10.25, 10.75]), np.array([20.25, 20.75])]
    shifts = [np.array([0.0, 40.0]), np.array([-40.0, 0.0])]
    corrected = ExactLatticeRemapRecording(
        recording, centers, shifts, cell_width_s=0.5
    )
    for segment_index, source in enumerate((first, second)):
        times = [10.0, 20.0][segment_index] + np.arange(8) / 8.0
        sampled = sample_stepwise_shifts(
            centers[segment_index], shifts[segment_index], times, cell_width_s=0.5
        )
        expected = expected_remap(source, geometry, sampled)
        np.testing.assert_array_equal(
            corrected.get_traces(segment_index=segment_index), expected
        )


def test_constructor_rejects_gapped_temporal_table():
    geometry = np1_geometry(6)
    recording, _ = make_recording(geometry)
    with pytest.raises(ValueError, match="contiguous"):
        ExactLatticeRemapRecording(
            recording,
            np.array([10.125, 10.375, 10.700]),
            np.array([0.0, 40.0, 0.0]),
            cell_width_s=0.25,
        )
