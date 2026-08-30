from testing import luke_conditioning_2x2_sort as audit


def test_factorial_cells_cover_all_combinations():
    combinations = {
        (cell.saturation_policy, cell.channel_191_policy)
        for cell in audit.CELLS.values()
    }
    assert len(combinations) == 4


def test_only_two_cells_require_new_sorts(tmp_path):
    current = audit.plan(list(audit.CELLS.values()), tmp_path)
    assert sum(row["new_sort_required"] for row in current["cells"]) == 2
    assert current["motion_correction"] is False
    assert current["positive_polarity_excess"]["status"] == "deferred_not_resolved"


def test_next_gate_stays_motion_free_and_does_not_require_full_sort(tmp_path):
    current = audit.plan(list(audit.CELLS.values()), tmp_path)
    assert current["duration_s"] == 120.0
    assert current["motion_correction"] is False
    assert current["claim_mask"] == "off"
