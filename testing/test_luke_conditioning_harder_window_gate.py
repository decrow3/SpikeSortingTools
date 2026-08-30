from testing import luke_conditioning_harder_window_gate as gate


def test_gate_uses_neutral_and_pathological_only():
    assert set(gate.WINDOWS) == {"neutral_template", "pathological"}


def test_gate_plan_is_short_motion_free_and_defers_polarity(tmp_path):
    plan = gate.make_plan(list(gate.WINDOWS.values()), tmp_path)
    assert plan["motion_correction"] is False
    assert plan["claim_mask"] == "off"
    assert all(row["duration_s"] == 120.0 for row in plan["windows"])
    assert plan["positive_polarity_excess"]["status"] == "deferred_not_resolved"
