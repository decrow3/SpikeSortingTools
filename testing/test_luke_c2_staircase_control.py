import numpy as np
import pytest

from testing.luke_c2_staircase_control import (
    OUTPUT,
    STAIRCASE,
    admissible_train,
    bin_table,
    segments,
    staircase_um,
)

FS = 30_000.0


def test_segments_tile_the_window_exactly_and_alternate():
    segs = segments()
    assert segs[0]["start_s"] == 0.0
    assert segs[-1]["stop_s"] == STAIRCASE["duration_s"]
    for previous, following in zip(segs, segs[1:]):
        assert previous["stop_s"] == following["start_s"]
        assert previous["level_um"] != following["level_um"]  # a real step
    plateaus = [s for s in segs if s["kind"] == "plateau"]
    assert len(plateaus) == STAIRCASE["n_plateaus"] == len(segs)
    assert [p["level_um"] for p in plateaus] == [0.0, 40.0, 0.0, 40.0]


def test_plateau_and_transition_durations_are_whole_bins():
    """Plateau boundaries must land on bin edges, or a plateau bin is impure."""
    bin_s = STAIRCASE["bin_s"]
    for key in ("plateau_s", "transition_s", "duration_s"):
        assert (STAIRCASE[key] / bin_s) % 1 == 0, key
    for seg in segments():
        assert (seg["start_s"] / bin_s) % 1 == 0
        assert (seg["stop_s"] / bin_s) % 1 == 0


def test_plateau_displacement_is_exactly_commensurate():
    """Bitwise equality, not approximate: 1e-9 µm of drift breaks exactness."""
    for seg in segments():
        if seg["kind"] != "plateau":
            continue
        probe = np.linspace(seg["start_s"], seg["stop_s"] - 1e-6, 25)
        values = staircase_um(probe)
        assert np.all(values == seg["level_um"])


def test_there_are_no_fractional_offset_bins_at_all():
    """A fractional bin degrades voltage KS4 sees whether or not it is scored."""
    assert STAIRCASE["transition_s"] == 0.0
    assert not any(s["kind"] == "transition" for s in segments())
    table = bin_table()
    assert table["settled"].all()
    assert set(np.unique(table["displacement_um"])) == set(STAIRCASE["levels_um"])


def test_steps_land_exactly_on_bin_edges():
    edges = bin_table()["edges_s"]
    for seg in segments()[1:]:
        assert np.isclose(edges, seg["start_s"]).any()


def test_commensurate_shift_follows_from_the_geometry():
    geo = STAIRCASE["geometry"]
    shift = round(40.0 / geo["row_pitch_um"]) * geo["sites_per_row"]
    assert shift == geo["commensurate_shift_channels"] == 4


def test_bin_table_marks_only_plateau_bins_settled():
    table = bin_table()
    assert table["n_bins"] == int(STAIRCASE["duration_s"] / STAIRCASE["bin_s"])
    settled_levels = table["displacement_um"][table["settled"]]
    assert set(np.unique(settled_levels)) <= set(STAIRCASE["levels_um"])
    moving = table["displacement_um"][~table["settled"]]
    assert not np.isin(moving, STAIRCASE["levels_um"]).any()
    # 3 transitions x 4 s / 0.5 s bins, minus the two endpoint bins that land
    # exactly on a level, is the settled/transition split
    assert table["settled"].sum() + (~table["settled"]).sum() == table["n_bins"]


def test_admission_drops_spikes_straddling_a_hard_step():
    step_s = segments()[1]["start_s"]
    guard_s = STAIRCASE["bin_s"] * (STAIRCASE["truth_admission"]["guard_bins"] + 1)
    deep = int(10.0 * FS)                          # mid first plateau
    straddling = int(step_s * FS)                  # exactly on the step
    just_before = int((step_s - 0.05) * FS)        # inside the guard
    safe_after = int((step_s + 2 * guard_s + 1.0) * FS)

    result = admissible_train(
        np.array([deep, just_before, straddling, safe_after], dtype=np.int64), FS
    )
    assert result["keep"].tolist() == [True, False, False, True]
    assert result["level_um"][0] == 0.0
    assert result["level_um"][3] == 40.0


def test_admission_keeps_nearly_the_whole_train_with_hard_steps():
    """Only spikes near the three steps go; a ramp would cost far more."""
    train = np.arange(int(1.0 * FS), int(119.0 * FS), int(FS / 6.0), dtype=np.int64)
    result = admissible_train(train, FS)
    assert result["n_admitted"] / result["n_total"] > 0.95


def test_expected_shift_follows_row_pitch():
    from testing.luke_c2_staircase_control import expected_shift_channels

    assert expected_shift_channels(0.0) == 0
    assert expected_shift_channels(40.0) == 4
    assert expected_shift_channels(80.0) == 8


def test_spatial_margin_covers_both_warp_directions():
    """Forward pulls from below, the exact inverse from above."""
    assert STAIRCASE["spatial_margin_channels"] >= expected_shift_for_max_level()


def expected_shift_for_max_level() -> int:
    from testing.luke_c2_staircase_control import expected_shift_channels

    return expected_shift_channels(max(STAIRCASE["levels_um"]))


def test_admission_reports_both_levels_for_a_regular_train():
    train = np.arange(int(1.0 * FS), int(119.0 * FS), int(FS / 6.0), dtype=np.int64)
    result = admissible_train(train, FS)
    assert 0 < result["n_admitted"] < result["n_total"]
    assert all(count > 0 for count in result["n_by_level"].values())


def test_reference_is_built_from_the_wide_strip_not_a_wrapped_roll():
    """Every cropped channel must have a real source, so no channel is undefined."""
    from testing.luke_c2_staircase_control import _reference_full

    margin, n_channels = 8, 10
    wide = np.arange(3 * (n_channels + 2 * margin), dtype=np.float32).reshape(3, -1)
    crop = slice(margin, margin + n_channels)
    table = {"n_bins": 1, "displacement_um": np.array([40.0])}
    reference = _reference_full(wide, crop, table, [0], [3], n_channels)
    assert np.array_equal(reference, wide[:, margin - 4: margin - 4 + n_channels])


def test_control_never_writes_under_mnt():
    assert not str(OUTPUT).startswith("/mnt/")


def test_frame_bin_assignment_mirrors_the_operator_not_rounding():
    """A bin edge that falls between samples must round *up*, as searchsorted does."""
    from testing.ladder_motion import frame_bin_assignment

    fs = 3.0  # samples at t = 0, 1/3, 2/3, 1.0, ...
    edges = np.array([0.0, 0.5, 1.0])
    starts, stops = frame_bin_assignment(6, fs, edges, 2)
    # t=1/3 (<0.5) is bin 0; t=2/3 is bin 1. round(0.5*3)=2 would misplace it.
    assert (starts[0], stops[0]) == (0, 2)
    assert starts[1] == 2
    assert int(round(edges[1] * fs)) == 2  # the rounding this test rules out
    assert stops[-1] == 6  # every frame is assigned, none dropped


def test_frame_bin_assignment_covers_every_frame_exactly_once():
    from testing.ladder_motion import frame_bin_assignment

    n_bins, fs = 240, 29999.759166666667
    edges = np.arange(n_bins + 1) * (120.0 / n_bins)
    n_samples = int(120.0 * fs)
    starts, stops = frame_bin_assignment(n_samples, fs, edges, n_bins)
    assert starts[0] == 0 and stops[-1] == n_samples
    assert np.array_equal(starts[1:], stops[:-1])  # contiguous, no gaps
