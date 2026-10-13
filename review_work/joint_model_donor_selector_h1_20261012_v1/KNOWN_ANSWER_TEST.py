import numpy as np
import pytest

from testing.joint_model_donor_selector import select_donors


def test_frozen_donor_panel_known_answer_and_fail_closed():
    units = np.arange(100, 118, dtype=np.int64)
    amplitudes = np.arange(1, 19, dtype=np.float64)
    cosines = np.asarray([
        .10, .20, .30, .40, .50, .60,
        .11, .21, .31, .41, .51, .61,
        .12, .22, .32, .42, .52, .62,
    ])
    distances = np.asarray([
        10, 300, 300, 300, 300, 300,
        20, 300, 300, 300, 300, 300,
        30, 40, 300, 300, 300, 300,
    ], dtype=np.float64)
    ledger = select_donors(units, amplitudes, cosines, distances)
    assert len(ledger) == 12
    assert [sum(row["amplitude_tier"] == tier for row in ledger) for tier in ("low", "medium", "high")] == [4, 4, 4]
    assert sum(row["within_risk_radius"] for row in ledger) == 4
    assert [row["unit_id"] for row in ledger] == [100, 101, 102, 103, 106, 107, 108, 109, 112, 113, 114, 115]
    assert all(row["nearest_template_cosine"] < 0.9 for row in ledger)

    with pytest.raises(ValueError, match="fewer than four"):
        select_donors(units, amplitudes, cosines, np.full(18, 300.0))
    with pytest.raises(ValueError, match="aligned"):
        select_donors(units[:-1], amplitudes, cosines, distances)
