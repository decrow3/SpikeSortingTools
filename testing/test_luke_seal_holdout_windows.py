import pandas as pd

from testing.luke_seal_holdout_windows import overlaps_exclusion, select_windows


def test_discovery_overlap_guard():
    assert overlaps_exclusion(7000, 7120)
    assert not overlaps_exclusion(1000, 1120)


def test_select_windows_takes_quiet_and_high_per_third():
    rows = []
    for third in range(3):
        for index, score in enumerate((0.1, 0.5, 0.9)):
            rows.append(
                {
                    "start_s": float(third * 1000 + index * 120),
                    "stop_s": float(third * 1000 + index * 120 + 120),
                    "time_third": third,
                    "combined_motion_score": score,
                }
            )
    selected = select_windows(pd.DataFrame(rows))
    assert len(selected) == 6
    assert set(selected.motion_stratum) == {"relative_quiet", "high_motion"}
    assert selected.groupby("time_third").size().eq(2).all()
