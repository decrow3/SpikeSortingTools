import json

import pytest

from testing.luke_holdout_validation import (
    DISCOVERY_EXCLUSIONS_S,
    MOTION_STRATA,
    PROBES,
    MotionRow,
    make_mock_motion_summary,
    normalize_motion_rows,
    plan_manifest,
)


DURATIONS = {"imec0": 10_000.0, "imec1": 10_000.0}


def test_planner_selects_complete_balanced_sealed_design():
    rows = make_mock_motion_summary(DURATIONS, points_per_cell=10)
    manifest = plan_manifest(rows, DURATIONS, seed=17)

    windows = manifest["windows"]
    assert len(windows) == 24
    cells = {(w["probe"], w["time_quartile"], w["motion_stratum"]) for w in windows}
    expected = {(probe, q, stratum) for probe in PROBES for q in range(1, 5) for stratum in MOTION_STRATA}
    assert cells == expected
    assert manifest["manifest_only"] is True
    assert manifest["sealed"] is True
    assert manifest["provenance"]["raw_files_opened"] is False
    assert manifest["event_quotas"]["extraction_performed"] is False
    assert manifest["event_quotas"]["target_total"] == 384
    assert manifest["event_quotas"]["target_per_polarity_depth_cell"] == 2
    assert sum(
        manifest["event_quotas"]["marginal_balance_dimension"]["target_counts_per_window"]
    ) == 16


def test_selection_is_seed_deterministic_and_manifest_is_hashable():
    rows = make_mock_motion_summary(DURATIONS, points_per_cell=10)
    first = plan_manifest(rows, DURATIONS, seed=123)
    second = plan_manifest(rows, DURATIONS, seed=123)
    assert first == second
    assert len(first["manifest_content_sha256"]) == 64
    json.dumps(first, allow_nan=False)


def test_windows_respect_edges_exclusions_and_global_spacing():
    manifest = plan_manifest(make_mock_motion_summary(DURATIONS, 12), DURATIONS, seed=9)
    windows = manifest["windows"]
    for window in windows:
        assert window["start_s"] >= 60
        assert window["stop_s"] <= DURATIONS[window["probe"]] - 60
        for lo, hi, _ in DISCOVERY_EXCLUSIONS_S:
            assert not (window["start_s"] < hi and window["stop_s"] > lo)
    centers = sorted(window["center_s"] for window in windows)
    assert all(right - left >= 60 for left, right in zip(centers, centers[1:]))


def test_explicit_strata_and_interval_rows_are_supported():
    rows = []
    row_id = 0
    for probe in PROBES:
        for q in range(4):
            base = q * 2500 + 900
            for stratum_index, stratum in enumerate(MOTION_STRATA):
                for repeat in range(3):
                    center = base + 200 * stratum_index + 40 * repeat
                    rows.append(
                        {
                            "probe": probe,
                            "start_s": center - 5,
                            "stop_s": center + 5,
                            "motion_score": stratum_index,
                            "motion_stratum": stratum,
                            "source_row": row_id,
                        }
                    )
                    row_id += 1
    normalized = normalize_motion_rows(rows)
    manifest = plan_manifest(normalized, DURATIONS, seed=3, min_spacing_s=30)
    assert len(manifest["windows"]) == 24


def test_missing_cell_fails_closed_instead_of_weakening_design():
    rows = [row for row in make_mock_motion_summary(DURATIONS, 5) if not (row.probe == "imec1" and row.time_s < 2500)]
    with pytest.raises(ValueError, match="no eligible candidates for cells"):
        plan_manifest(rows, DURATIONS)


def test_partial_explicit_strata_are_rejected():
    rows = [
        MotionRow("imec0", 200, 1, "quiet", 0),
        MotionRow("imec0", 300, 2, None, 1),
    ]
    with pytest.raises(ValueError, match="partially supplied"):
        plan_manifest(rows, DURATIONS)
