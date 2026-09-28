import numpy as np
import pandas as pd

from testing import luke_imec1_medicine_stage2_p_v1 as p


def test_tile_ranking_and_initial_third_cap():
    table = p.tile_ranking()
    chosen = p.initial_tiles(table)
    assert len(chosen) == 4
    rows = table.set_index("tile_index").loc[chosen]
    assert not rows.excluded_margin_overlap.any()
    assert rows.session_third.value_counts().max() <= 2
    assert (rows.field_bins == 480).all()


def test_false_motion_excludes_expanded_candidates():
    times = np.arange(.125, 20, .25)
    values = np.zeros((len(times), 1)); values[(times >= 9) & (times <= 11)] = 100
    field = {"session_time_s": times, "depth_um": np.array([100.]), "displacement_um": values}
    cand = pd.DataFrame({"start_s": [9.], "stop_s": [11.]})
    assert p.false_motion(field, cand, (0., 20.)) == 0


def test_selection_uses_quiet_false_motion_gate_and_simplicity():
    quiet = pd.DataFrame([
        {"config":"amp50","quiet_abs":1.,"quiet_inc":1.,"false_motion_frac":.01},
        {"config":"amp50_d1","quiet_abs":1.,"quiet_inc":1.,"false_motion_frac":.01},
        {"config":"amp25","quiet_abs":1.,"quiet_inc":1.,"false_motion_frac":.10},
    ])
    episode = pd.DataFrame([
        {"config":"amp50","episode_err":10.,"episode_ratio":1.,"block_err":10.},
        {"config":"amp50_d1","episode_err":12.,"episode_ratio":1.,"block_err":11.},
        {"config":"amp25","episode_err":1.,"episode_ratio":1.,"block_err":1.},
    ])
    got = p.select(quiet, episode)
    assert got["config"] == "amp50_d1"


def test_planned_development_reuse_contract():
    assert p.DEV_REUSE == {"amp50", "amp25", "depth2"}
    assert p.GPU_BUDGET_S == 7 * 3600


def test_r_null_rule_uses_mode_and_median_absolute_shift():
    nulls = pd.DataFrame({"window":["e"]*5,"resolved":[True]*5,
                          "best_shift_um":[0,0,40,40,120]})
    got = p.validate_nulls_r(nulls,{"e":(0,120)})
    assert got["windows"]["e"]["mode_um"] == 0
    assert got["windows"]["e"]["median_abs_um"] == 40
    assert got["status"] == "fail"


def test_r_null_rule_allows_one_quantisation_step():
    nulls = pd.DataFrame({"window":["e"]*5,"resolved":[True]*5,
                          "best_shift_um":[0,0,10,10,40]})
    assert p.validate_nulls_r(nulls,{"e":(0,120)})["status"] == "pass"
