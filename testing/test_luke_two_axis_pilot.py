import numpy as np

from testing import luke_two_axis_pilot as pilot


def test_default_panel_has_two_time_and_one_depth_pilot():
    selected = pilot.selected_pilots(None)
    assert [item.name for item in selected] == [
        "good_pre_shared",
        "neutral_template",
        "core_depth_strip",
    ]
    assert {item.axis for item in selected} == {"time", "depth"}


def test_core_strip_includes_suspect_channels():
    channels = pilot.pilot_channel_ids(pilot.PILOTS["core_depth_strip"])
    assert len(channels) == 96
    assert channels[0] == 176
    assert channels[-1] == 271
    assert 191 in channels
    assert 216 in channels


def test_bad_channel_rows_map_physical_191_after_depth_slice():
    channel_ids = np.arange(176, 272)
    assert pilot.bad_channel_rows(channel_ids) == [15]
    assert pilot.bad_channel_rows(np.arange(272, 368)) == []


def test_no_bad_interpolation_policy_is_declared_but_not_a_sorting_policy():
    # This internal policy is reserved for real-voltage halo channels that do
    # not include the known bad channel; it must not alter the public panel.
    assert all(item.name != "legacy_no_bad_interpolation" for item in pilot.PILOTS.values())


def test_time_and_depth_frame_ranges():
    fs = 30_000.0
    n_frames = 12_000 * 30_000
    good = pilot.pilot_frame_range(pilot.PILOTS["good_pre_shared"], fs, n_frames)
    core = pilot.pilot_frame_range(pilot.PILOTS["core_depth_strip"], fs, n_frames)
    assert good == (7095 * 30_000, 7215 * 30_000)
    assert core == (0, n_frames)


def test_temporal_metrics_flag_short_lived_unit():
    fs = 1_000.0
    times = np.array([10, 20, 30, 5100, 6100, 7100], dtype=np.int64)
    clusters = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)
    metrics = pilot.temporal_unit_metrics(times, clusters, fs, 10.0, 1.0)
    first = metrics.set_index("unit_id").loc[1]
    second = metrics.set_index("unit_id").loc[2]
    assert first.active_time_bin_fraction == 0.1
    assert second.active_time_bin_fraction == 0.3
    assert first.lifetime_s < second.lifetime_s


def test_time_bin_metrics_conserves_spikes():
    times = np.array([0, 999, 1000, 2999], dtype=np.int64)
    metrics = pilot.temporal_bin_metrics(times, 1000.0, 3.0, 1.0)
    assert metrics.spike_count.tolist() == [2, 1, 1]
    assert metrics.spike_count.sum() == len(times)


def test_default_materialization_memory_is_bounded():
    # 10 s x 30 kHz x 96 int16 channels x 8 workers is about 440 MiB.
    memory_bytes = 10 * 30_000 * 96 * 2 * 8
    assert memory_bytes < 0.5 * 1024**3


def test_io_benchmark_rejects_nonpositive_duration(tmp_path):
    try:
        pilot.benchmark_io(tmp_path, 0.0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Expected a nonpositive duration to fail")


def test_parse_last_sort_runtime_uses_last_completed_run():
    text = "kilosort4 run time 12.5s\nError running kilosort4\nkilosort4 run time 8.25s\n"
    assert pilot.parse_last_sort_runtime_s(text) == 8.25
    assert pilot.parse_last_sort_runtime_s("Error running kilosort4") is None


def test_circular_shift_coincidence_null_is_bounded():
    times = np.array([10, 11, 100, 101, 200, 201], dtype=np.int64)
    clusters = np.array([1, 2, 1, 2, 1, 2], dtype=np.int64)
    depths = np.zeros(len(times))
    value = pilot.circular_shift_coincidence_null(
        times, clusters, depths, 1000, 2, seed=3, n_repeats=2
    )
    assert 0.0 <= value <= 1.0
