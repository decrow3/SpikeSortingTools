import json

import numpy as np
import pandas as pd
import pytest

from testing.luke_yates_stable_window_overlap import (
    COMMON_ESTIMATORS,
    GATE_MIN_YATES,
    MotionSource,
    SourceValidationError,
    build,
    default_sources,
    enumerate_windows,
    evaluate_gate,
    high_motion_luke_controls,
    mark_overlap,
    overlap_box,
    validate_source_matrix,
    window_signature,
    _yates_unique_quiet_intervals,
)


# --------------------------------------------------------------------------- #
# synthetic source helpers
# --------------------------------------------------------------------------- #
def _write_source(root, dataset, probe, estimator, motion, times, depths):
    d = root / f"{dataset}-{probe}" / estimator
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "motion.npy", np.asarray(motion, float))
    np.save(d / "time_bins.npy", np.asarray(times, float))
    np.save(d / "depth_bins.npy", np.asarray(depths, float))
    return MotionSource(dataset, probe, estimator, d)


def _ramp_source(root, tag, dataset, probe, excursion_um, depth_span_um=1000.0,
                 nonrigid_range_um=0.0, estimator="medicine", t0=0.0, dt=1.0):
    times = np.arange(t0, t0 + 120.0, dt)
    rigid = np.linspace(0.0, excursion_um, times.size)
    # two depth columns separated by depth_span_um, differing by nonrigid_range_um
    motion = np.column_stack([rigid, rigid + nonrigid_range_um])
    depths = np.array([0.0, depth_span_um])
    return _write_source(root / tag, dataset, probe, estimator, motion, times, depths)


def _multiwindow_source(root, tag, dataset, probe, per_window_excursions,
                        depth_span_um=1000.0, nonrigid_range_um=0.0,
                        estimator="medicine", t0=0.0, dt=1.0):
    """A recording spanning len(per_window_excursions) back-to-back 120 s windows."""
    segs_t, segs_m = [], []
    for w, exc in enumerate(per_window_excursions):
        wt = np.arange(0.0, 120.0, dt) + t0 + w * 120.0
        rigid = np.linspace(0.0, exc, wt.size)
        segs_t.append(wt)
        segs_m.append(np.column_stack([rigid, rigid + nonrigid_range_um]))
    times = np.concatenate(segs_t)
    motion = np.vstack(segs_m)
    return _write_source(root / tag, dataset, probe, estimator, motion, times,
                         np.array([0.0, depth_span_um]))


# --------------------------------------------------------------------------- #
# window_signature
# --------------------------------------------------------------------------- #
def test_signature_flat_is_quiet():
    times = np.arange(0, 120, 1.0)
    sig = window_signature(np.zeros((times.size, 4)), times, np.array([0.0, 100, 200, 300]))
    assert sig["rigid_excursion_um"] == 0.0
    assert sig["nonrigid_grad_um_per_mm"] == 0.0
    assert sig["rigid_speed_um_s"] == 0.0
    assert sig["finite_fraction"] == 1.0
    assert sig["max_time_gap_s"] == pytest.approx(1.0)


def test_signature_rigid_excursion_is_p95_minus_p5():
    times = np.arange(0, 120, 1.0)
    ramp = np.linspace(0, 10, times.size)
    sig = window_signature(ramp[:, None] + np.zeros((1, 2)), times, np.array([0.0, 1000.0]))
    assert sig["rigid_excursion_um"] == pytest.approx(9.0, abs=0.2)


def test_signature_nonrigid_gradient_is_depth_span_normalised():
    times = np.arange(0, 120, 1.0)
    rigid = np.zeros(times.size)
    # constant 6 µm across-depth difference
    motion = np.column_stack([rigid, rigid + 6.0])
    over_1mm = window_signature(motion, times, np.array([0.0, 1000.0]))
    over_2mm = window_signature(motion, times, np.array([0.0, 2000.0]))
    assert over_1mm["nonrigid_grad_um_per_mm"] == pytest.approx(6.0)
    assert over_2mm["nonrigid_grad_um_per_mm"] == pytest.approx(3.0)  # same range, wider span


def test_signature_single_depth_bin_has_zero_gradient():
    times = np.arange(0, 120, 1.0)
    sig = window_signature(np.linspace(0, 5, times.size)[:, None], times, np.array([100.0]))
    assert sig["nonrigid_grad_um_per_mm"] == 0.0
    assert sig["n_depth_bins"] == 1


def test_signature_counts_nonfinite_and_reports_gap():
    times = np.arange(0, 120, 1.0)
    motion = np.zeros((times.size, 2))
    motion[:12] = np.nan
    sig = window_signature(motion, times, np.array([0.0, 1000.0]))
    assert sig["finite_fraction"] == pytest.approx(0.9, abs=0.01)  # below the floor
    # the kept samples are still consecutive (leading rows are simply dropped)
    assert sig["max_time_gap_s"] == pytest.approx(1.0)


def test_signature_reports_internal_hole_as_gap():
    times = np.arange(0, 120, 1.0)
    motion = np.zeros((times.size, 2))
    motion[40:50] = np.nan  # 10 s hole in the middle
    sig = window_signature(motion, times, np.array([0.0, 1000.0]))
    assert sig["max_time_gap_s"] == pytest.approx(11.0)


# --------------------------------------------------------------------------- #
# enumerate_windows
# --------------------------------------------------------------------------- #
def test_enumerate_non_overlapping_with_nonzero_origin(tmp_path):
    times = np.arange(500.0, 1100.0, 1.0)  # native origin 500
    motion = np.zeros((times.size, 2))
    src = _write_source(tmp_path, "Luke", "imec0", "medicine", motion, times, np.array([0.0, 1000.0]))
    rows = enumerate_windows(src)
    assert [r["window_start_native_s"] for r in rows] == [500.0, 620.0, 740.0, 860.0, 980.0]
    assert [r["window_start_recording_s"] for r in rows] == [0.0, 120.0, 240.0, 360.0, 480.0]
    assert rows[0]["time_origin_native_s"] == 500.0


def test_enumerate_excludes_window_below_coverage(tmp_path):
    # last window [480,600) only carries 100 s of data -> below 0.9*120
    times = np.concatenate([np.arange(0, 480, 1.0), np.arange(480, 580, 1.0)])
    motion = np.zeros((times.size, 2))
    src = _write_source(tmp_path, "Yates", "shank1", "medicine", motion, times, np.array([0.0, 1000.0]))
    starts = [r["window_start_recording_s"] for r in enumerate_windows(src)]
    assert starts == [0.0, 120.0, 240.0, 360.0]


def test_enumerate_includes_exactly_complete_final_window(tmp_path):
    times = np.arange(0, 480 + 119.0 + 1e-9, 1.0)  # window [480,600) spans 480..599
    motion = np.zeros((times.size, 2))
    src = _write_source(tmp_path, "Yates", "shank1", "medicine", motion, times, np.array([0.0, 1000.0]))
    starts = [r["window_start_recording_s"] for r in enumerate_windows(src)]
    assert starts == [0.0, 120.0, 240.0, 360.0, 480.0]


def test_load_transposes_depth_major_motion(tmp_path):
    times = np.arange(0, 120, 1.0)
    motion_depth_major = np.zeros((2, times.size))  # (n_depth, n_time)
    src = _write_source(tmp_path, "Luke", "imec0", "medicine", motion_depth_major, times, np.array([0.0, 1000.0]))
    m, t, d = src.load()
    assert m.shape == (times.size, 2)


# --------------------------------------------------------------------------- #
# overlap_box
# --------------------------------------------------------------------------- #
def test_overlap_box_is_yates_q75_on_every_axis():
    yates = pd.DataFrame(
        {
            "rigid_excursion_um": np.arange(0.0, 10.0),
            "nonrigid_grad_um_per_mm": np.arange(0.0, 10.0) * 2,
            "rigid_speed_um_s": np.arange(0.0, 10.0) * 0.1,
        }
    )
    box = overlap_box(yates)
    assert box["rigid_excursion_um"] == pytest.approx(np.quantile(np.arange(0.0, 10.0), 0.75))
    assert box["nonrigid_grad_um_per_mm"] == pytest.approx(np.quantile(np.arange(0.0, 10.0) * 2, 0.75))
    assert box["rigid_speed_um_s"] == pytest.approx(np.quantile(np.arange(0.0, 10.0) * 0.1, 0.75))


def test_overlap_box_rejects_nonfinite():
    yates = pd.DataFrame(
        {
            "rigid_excursion_um": [1.0, 2.0, np.nan],
            "nonrigid_grad_um_per_mm": [1.0, 2.0, 3.0],
            "rigid_speed_um_s": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(SourceValidationError):
        overlap_box(yates)


# --------------------------------------------------------------------------- #
# mark_overlap
# --------------------------------------------------------------------------- #
def _mo_frame(**over):
    row = {
        "rigid_excursion_um": 1.0,
        "nonrigid_grad_um_per_mm": 1.0,
        "rigid_speed_um_s": 1.0,
        "finite_fraction": 1.0,
        "max_time_gap_s": 1.0,
        "dt_median_s": 1.0,
    }
    row.update(over)
    return pd.DataFrame([row])


def test_mark_overlap_boundary_is_inclusive():
    box = {"rigid_excursion_um": 1.0, "nonrigid_grad_um_per_mm": 1.0, "rigid_speed_um_s": 1.0}
    assert mark_overlap(_mo_frame(), box).tolist() == [True]


def test_mark_overlap_respects_finite_floor_and_gap():
    box = {"rigid_excursion_um": 2.0, "nonrigid_grad_um_per_mm": 2.0, "rigid_speed_um_s": 2.0}
    assert mark_overlap(_mo_frame(finite_fraction=0.5), box).tolist() == [False]
    assert mark_overlap(_mo_frame(max_time_gap_s=10.0, dt_median_s=1.0), box).tolist() == [False]


# --------------------------------------------------------------------------- #
# evaluate_gate  (hand-built frames, independently varied axes)
# --------------------------------------------------------------------------- #
def _win(dataset, probe, estimator, interval, exc, grad=0.0, speed=0.0):
    return {
        "dataset": dataset,
        "probe": probe,
        "estimator": estimator,
        "time_interval_id": interval,
        "window_start_native_s": float(interval) * 120.0,
        "window_start_recording_s": float(interval) * 120.0,
        "rigid_excursion_um": exc,
        "nonrigid_grad_um_per_mm": grad,
        "rigid_speed_um_s": speed,
        "finite_fraction": 1.0,
        "max_time_gap_s": 1.0,
        "dt_median_s": 1.0,
    }


def _panel_frame(luke_imec0, luke_imec1=(), yates_by_interval=None, estimator="medicine"):
    """yates_by_interval: {interval_id: (shank1_exc, shank2_exc)}"""
    rows = [_win("Luke", "imec0", estimator, i, e) for i, e in enumerate(luke_imec0)]
    rows += [_win("Luke", "imec1", estimator, i, e) for i, e in enumerate(luke_imec1)]
    for interval, (s1, s2) in (yates_by_interval or {}).items():
        rows.append(_win("Yates", "shank1", estimator, interval, s1))
        rows.append(_win("Yates", "shank2", estimator, interval, s2))
    return pd.DataFrame(rows)


def test_gate_passes_at_exactly_six_six():
    yates = {i: (2.0, 2.0) for i in range(8)}  # box q75 ~ 2, so all 8 intervals quiet
    windows = _panel_frame(luke_imec0=[0, 0, 0, 1, 1, 1], yates_by_interval=yates)
    gate = evaluate_gate(windows)
    med = gate["by_estimator"]["medicine"]
    assert med["n_luke_imec0_overlap"] == 6
    assert med["n_yates_unique_quiet_intervals"] >= GATE_MIN_YATES
    assert gate["overall_pass"] is True


def test_gate_fails_at_five_luke_windows():
    yates = {i: (2.0, 2.0) for i in range(8)}
    windows = _panel_frame(luke_imec0=[0, 0, 1, 1, 1, 40, 40], yates_by_interval=yates)
    gate = evaluate_gate(windows)
    assert gate["by_estimator"]["medicine"]["n_luke_imec0_overlap"] == 5
    assert gate["overall_pass"] is False


def test_yates_unique_quiet_intervals_needs_every_shank():
    df = pd.DataFrame(
        [
            {"probe": "shank1", "time_interval_id": 0, "in_overlap": True},
            {"probe": "shank2", "time_interval_id": 0, "in_overlap": True},   # counted
            {"probe": "shank1", "time_interval_id": 1, "in_overlap": True},
            {"probe": "shank2", "time_interval_id": 1, "in_overlap": False},  # one shank loud
            {"probe": "shank1", "time_interval_id": 2, "in_overlap": True},   # only one shank present
        ]
    )
    assert _yates_unique_quiet_intervals(df) == 1


def test_gate_counts_unique_intervals_not_shank_windows():
    # a small loud minority on shank2 does not move Yates Q75; those intervals drop out
    yates = {i: (1.0, 1.0 if i < 9 else 40.0) for i in range(12)}
    windows = _panel_frame(luke_imec0=[0] * 8, yates_by_interval=yates)
    med = evaluate_gate(windows)["by_estimator"]["medicine"]
    assert med["n_yates_unique_quiet_intervals"] == 9
    assert med["n_yates_shank_windows_overlap"] == 21  # 12 shank1 + 9 shank2


def test_gate_ignores_imec1_for_the_verdict():
    yates = {i: (2.0, 2.0) for i in range(8)}
    windows = _panel_frame(
        luke_imec0=[40, 40, 40, 40, 40, 40],  # imec0 never quiet
        luke_imec1=[0, 0, 0, 0, 0, 0],  # imec1 quiet, must not rescue the gate
        yates_by_interval=yates,
    )
    gate = evaluate_gate(windows)
    assert gate["by_estimator"]["medicine"]["n_luke_imec1_overlap"] == 6
    assert gate["overall_pass"] is False


def test_overall_pass_follows_primary_estimator_only():
    frames = []
    # medicine: Luke never reaches the quiet region -> fail
    frames.append(_panel_frame([40] * 8, yates_by_interval={i: (2.0, 2.0) for i in range(8)},
                               estimator="medicine"))
    # ks-motion: Luke fully inside -> would pass on its own
    frames.append(_panel_frame([0] * 8, yates_by_interval={i: (2.0, 2.0) for i in range(8)},
                               estimator="ks-motion"))
    gate = evaluate_gate(pd.concat(frames, ignore_index=True))
    assert gate["by_estimator"]["ks-motion"]["pass"] is True
    assert gate["by_estimator"]["medicine"]["pass"] is False
    assert gate["overall_pass"] is False


def test_gate_json_serialisable():
    windows = _panel_frame([0] * 8, yates_by_interval={i: (2.0, 2.0) for i in range(8)})
    json.dumps(evaluate_gate(windows))


# --------------------------------------------------------------------------- #
# high_motion_luke_controls
# --------------------------------------------------------------------------- #
def test_high_motion_controls_are_exactly_the_top_decile():
    excursions = list(range(0, 40, 4))  # 10 windows, 0..36
    windows = _panel_frame(luke_imec0=excursions,
                           yates_by_interval={0: (1.0, 1.0)})
    controls = high_motion_luke_controls(windows)
    edge = np.quantile(np.array(excursions, float), 0.90)
    assert controls
    assert all(c["rigid_excursion_um"] >= edge for c in controls)
    excluded_max = max(e for e in excursions if e < edge)
    assert all(c["rigid_excursion_um"] > excluded_max for c in controls)


# --------------------------------------------------------------------------- #
# source-matrix validation
# --------------------------------------------------------------------------- #
def test_validate_source_matrix_flags_missing(tmp_path):
    sources = [
        _ramp_source(tmp_path, f"s{i}", ds, pr, 1.0, estimator=est)
        for i, (ds, pr, est) in enumerate(
            [("Luke", "imec0", e) for e in COMMON_ESTIMATORS]
        )
    ]  # Luke imec1 and all Yates missing
    with pytest.raises(SourceValidationError, match="missing required"):
        validate_source_matrix(sources)


def test_validate_source_matrix_flags_shape_mismatch(tmp_path):
    good = []
    combos = (
        [("Luke", p, e) for p in ("imec0", "imec1") for e in COMMON_ESTIMATORS]
        + [("Yates", s, e) for s in ("shank1", "shank2") for e in COMMON_ESTIMATORS]
    )
    for i, (ds, pr, est) in enumerate(combos):
        good.append(_ramp_source(tmp_path, f"s{i}", ds, pr, 1.0, estimator=est))
    # corrupt one: motion cols != depth bins
    bad = good[0]
    np.save(bad.motion_dir / "depth_bins.npy", np.array([0.0, 1.0, 2.0]))
    with pytest.raises(SourceValidationError, match="depth bins"):
        validate_source_matrix(good)


def test_default_sources_returns_required_matrix_paths(tmp_path):
    sources = default_sources(tmp_path / "luke", tmp_path / "yates")
    combos = {(s.dataset, s.probe, s.estimator) for s in sources}
    for probe in ("imec0", "imec1"):
        for est in COMMON_ESTIMATORS:
            assert ("Luke", probe, est) in combos
    for shank in ("shank1", "shank2"):
        for est in COMMON_ESTIMATORS:
            assert ("Yates", shank, est) in combos


# --------------------------------------------------------------------------- #
# end-to-end on synthetic sources
# --------------------------------------------------------------------------- #
def test_build_then_gate_on_synthetic_sources(tmp_path):
    sources = [
        _multiwindow_source(tmp_path, "luke_i0", "Luke", "imec0", [1.0] * 8,
                            depth_span_um=4000.0, nonrigid_range_um=1.0),
    ]
    for shank in ("shank1", "shank2"):
        sources.append(
            _multiwindow_source(tmp_path, f"yates_{shank}", "Yates", shank,
                                [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 40.0, 40.0],  # last 2 loud
                                depth_span_um=1200.0, nonrigid_range_um=2.0)
        )
    windows = build(sources)
    assert set(windows["estimator"]) == {"medicine"}
    gate = evaluate_gate(windows)
    med = gate["by_estimator"]["medicine"]
    assert med["n_luke_imec0_overlap"] == 8
    assert med["n_yates_unique_quiet_intervals"] == 6  # the 6 quiet intervals, both shanks
    assert gate["overall_pass"] is True
