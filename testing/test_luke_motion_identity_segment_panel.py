import pandas as pd

from testing.luke_motion_identity_segment_panel import (
    SELECTIONS,
    SEGMENT_DURATION_S,
    build_panel,
)


def test_panel_is_nonoverlapping_and_sorter_blind():
    rows = []
    for _, start_s, _ in SELECTIONS:
        for probe, scale in (("imec0", 1.0), ("imec1", 1.5)):
            rows.append(
                {
                    "probe": probe,
                    "start_s": start_s,
                    "duration_s": 120.0,
                    "dredge_excursion_um": scale * (start_s / 1000),
                    "decentralized_excursion_um": scale * (start_s / 900),
                    "dredge_decentralized_r": 0.9,
                    "input_anomaly_score": 0.0,
                    "support_instability_score": 0.0,
                }
            )
    panel, probe_rows = build_panel(pd.DataFrame(rows))

    assert len(panel) == len(SELECTIONS)
    assert len(probe_rows) == 2 * len(SELECTIONS)
    assert panel.segment_duration_s.eq(SEGMENT_DURATION_S).all()
    assert not panel.selection_uses_sorter_outcomes.any()
    assert (
        panel.segment_end_s.iloc[:-1].to_numpy()
        <= panel.segment_start_s.iloc[1:].to_numpy()
    ).all()
