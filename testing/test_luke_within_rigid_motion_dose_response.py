import json

import numpy as np
import pandas as pd
import pytest

from testing.luke_within_rigid_motion_dose_response import (
    DOSE_AXES,
    N_WINDOWS,
    N_WINDOWS_MAX,
    SelectedWindow,
    dose_response,
    frame_relative_start,
    load_frozen_list,
    mann_kendall,
    partial_spearman,
    select_windows,
    spearman_ci,
    write_frozen_list,
    _increment1_luke_imec0,
)


# --------------------------------------------------------------------------- #
# phase 1 -- clock conversion
# --------------------------------------------------------------------------- #
def test_frame_relative_start_matches_repo_convention():
    from testing.luke_motion_regime_windows import relative_times

    native = np.arange(3057.7, 3057.7 + 20.0, 1.0)  # dt = 1.0, origin 3057.7
    # a window that starts at the 5th native bin
    want = float(relative_times(native)[5])
    got = frame_relative_start(native[5], native[0], 1.0)
    assert got == pytest.approx(want)


def test_frame_relative_start_two_second_bins():
    from testing.luke_motion_regime_windows import relative_times

    native = np.arange(1.0, 1.0 + 40.0, 2.0)  # dt = 2.0, origin 1.0
    want = float(relative_times(native)[3])
    got = frame_relative_start(native[3], native[0], 2.0)
    assert got == pytest.approx(want)


# --------------------------------------------------------------------------- #
# phase 1 -- synthetic increment-1 CSV
# --------------------------------------------------------------------------- #
def _fake_csv(tmp_path, n_luke=87, seed=0):
    rng = np.random.default_rng(seed)
    origin, dt = 3057.7, 1.0
    rows = []
    # Luke imec0 medicine: rigid excursion 4..23 um, speed loosely correlated
    exc = np.linspace(4.0, 23.0, n_luke) + rng.normal(0, 0.3, n_luke)
    spd = 0.02 * exc + rng.normal(0, 0.05, n_luke) + 0.2
    for k in range(n_luke):
        rows.append(dict(
            dataset="Luke", probe="imec0", estimator="medicine",
            window_start_native_s=origin + k * 120.0,
            window_start_recording_s=k * 120.0,
            time_origin_native_s=origin, window_duration_s=120.0, time_interval_id=k,
            rigid_excursion_um=float(exc[k]), nonrigid_grad_um_per_mm=1.5,
            p95_nonrigid_grad_um_per_mm=2.0, rigid_speed_um_s=float(max(spd[k], 0.01)),
            finite_fraction=1.0, n_time_bins=120, n_depth_bins=2,
            depth_span_um=4074.0, dt_median_s=dt, max_time_gap_s=1.0,
        ))
    # a couple of QC-failing Luke windows that must be dropped
    rows.append({**rows[0], "window_start_native_s": origin + 999 * 120.0,
                 "finite_fraction": 0.5, "rigid_excursion_um": 5.0})
    rows.append({**rows[0], "window_start_native_s": origin + 998 * 120.0,
                 "max_time_gap_s": 99.0, "rigid_excursion_um": 6.0})
    # noise rows for other datasets/estimators
    for est in ("ks-motion", "decentralized-motion"):
        rows.append({**rows[0], "estimator": est})
    rows.append({**rows[0], "dataset": "Yates", "probe": "shank1"})
    p = tmp_path / "window_signatures.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_increment1_filter_drops_qc_failures_and_sorts(tmp_path):
    df = _increment1_luke_imec0(_fake_csv(tmp_path), "medicine")
    assert (df["dataset"] == "Luke").all() and (df["probe"] == "imec0").all()
    assert (df["estimator"] == "medicine").all()
    assert len(df) == 87  # the two QC failures dropped
    assert df["rigid_excursion_um"].is_monotonic_increasing


def test_select_windows_spans_range_and_is_deterministic(tmp_path):
    csv = _fake_csv(tmp_path)
    a = select_windows(csv)
    b = select_windows(csv)
    assert [w.rank for w in a] == [w.rank for w in b]
    assert N_WINDOWS <= len(a) <= N_WINDOWS_MAX
    excs = [w.rigid_excursion_um for w in a]
    assert min(excs) < 5.0 and max(excs) > 22.0  # both tails covered
    assert a[0].rank == 0  # quietest window always included
    # frame starts are the native starts shifted by (origin - dt/2)
    assert a[0].frame_start_s == pytest.approx(a[0].native_start_s - (3057.7 - 0.5))


def test_select_windows_speed_topup_flag(tmp_path):
    windows = select_windows(_fake_csv(tmp_path, seed=3))
    # base picks are the 24 even ranks; any extra are flagged
    base = [w for w in windows if not w.added_for_speed_coverage]
    extra = [w for w in windows if w.added_for_speed_coverage]
    assert len(base) <= N_WINDOWS
    assert len(base) + len(extra) == len(windows)


def test_frozen_list_roundtrip_and_no_overwrite(tmp_path):
    windows = select_windows(_fake_csv(tmp_path))
    path = tmp_path / "frozen.json"
    write_frozen_list(windows, path)
    loaded = load_frozen_list(path)
    assert [w.rank for w in loaded] == [w.rank for w in windows]
    assert loaded[0] == windows[0]
    with pytest.raises(RuntimeError, match="frozen once"):
        write_frozen_list(windows, path)


def test_select_windows_raises_when_too_few(tmp_path):
    with pytest.raises(RuntimeError, match="need"):
        select_windows(_fake_csv(tmp_path, n_luke=10))


# --------------------------------------------------------------------------- #
# phase 4 -- statistics
# --------------------------------------------------------------------------- #
def test_spearman_ci_monotone():
    x = np.arange(24.0)
    y = x ** 1.3
    r = spearman_ci(x, y, n_boot=200)
    assert r["rho"] == pytest.approx(1.0)
    assert r["ci_lo"] > 0.8 and r["n"] == 24


def test_spearman_ci_degenerate_returns_nan():
    r = spearman_ci(np.ones(10), np.arange(10.0))
    assert np.isnan(r["rho"])


def test_mann_kendall_detects_increasing_trend():
    x = np.arange(20.0)
    y = 0.5 * x + np.sin(x)  # increasing with wobble
    mk = mann_kendall(x, y)
    assert mk["tau"] > 0.5 and mk["p"] < 0.05


def test_partial_spearman_removes_confound():
    rng = np.random.default_rng(1)
    z = np.linspace(0, 1, 60)              # session time
    x = z + rng.normal(0, 0.03, 60)        # dose correlated with session time
    y_conf = 3 * z + rng.normal(0, 0.05, 60)   # driven by z only
    y_real = 3 * x + rng.normal(0, 0.05, 60)   # driven by x
    assert abs(partial_spearman(y_conf, x, z)) < 0.4          # confound removed
    assert partial_spearman(y_real, x, z) > 0.5               # real effect survives


def test_dose_response_structure():
    rng = np.random.default_rng(2)
    n = 24
    exc = np.linspace(4, 23, n)
    df = pd.DataFrame({
        "frame_start_s": np.arange(n) * 120.0,
        "rigid_excursion_um": exc,
        "rigid_speed_um_s": 0.02 * exc + rng.normal(0, 0.02, n),
        "E3_qualified_units_per_mm": 10 - 0.3 * exc + rng.normal(0, 0.2, n),
        "E4_refractory_burden_median": 0.001 + 0.0002 * exc,
    })
    out = dose_response(df)
    assert out["n_windows"] == n
    assert set(out["by_endpoint"]) == {"E3_qualified_units_per_mm", "E4_refractory_burden_median"}
    e3 = out["by_endpoint"]["E3_qualified_units_per_mm"]
    assert set(e3) == set(DOSE_AXES)
    assert e3["rigid_excursion_um"]["spearman"]["rho"] < -0.7  # declines with motion


def test_dose_response_needs_session_time_column():
    with pytest.raises(ValueError, match="frame_start_s"):
        dose_response(pd.DataFrame({"rigid_excursion_um": [1.0, 2.0], "E1_compact_events_per_mm_per_s": [1.0, 2.0]}))
