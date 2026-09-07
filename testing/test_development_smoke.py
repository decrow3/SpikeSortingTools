import json

import pytest

from testing.development_smoke import build_smoke_contract, pin_smoke_plan


def test_smoke_contract_is_bounded_engineering_only(tmp_path):
    from testing.development_ladder import load_contract
    parent = load_contract("testing/configs/luke0804_hindsight_ladder_v1.json")
    smoke, plan = build_smoke_contract(
        parent,
        candidate_names=["rescue_12_9_motion_off", "rescue_12_9_native_rigid"],
        start_s=0,
        duration_s=120,
    )
    assert smoke.raw["development_status"] == "engineering_only_not_for_scientific_selection"
    assert smoke.raw["recording"]["duration_s"] == 120
    assert smoke.raw["recording"]["full_session"] is False
    assert plan["engineering_only"] is True
    path = tmp_path / "smoke_plan.json"
    assert pin_smoke_plan(plan, path) == plan
    assert pin_smoke_plan(plan, path) == plan


def test_luke_execution_groups_are_frozen_to_current_contract():
    from testing.development_ladder import load_contract
    parent = load_contract("testing/configs/luke0804_hindsight_ladder_v1.json")
    execution = json.loads(
        open("testing/configs/luke0804_hindsight_ladder_execution_groups_v1.json").read()
    )
    assert execution["contract_digest"] == parent.digest
    assert execution["groups"]["group_1_motion_axis"]["arms"] == [
        "rescue_12_9_motion_off",
        "rescue_12_9_native_rigid",
        "rescue_12_9_native_nonrigid",
    ]
    assert execution["groups"]["group_2_threshold_axis"]["authorized_after_prerequisites"] is False


@pytest.mark.parametrize(
    "names,duration,match",
    [([], 120, "nonempty"), (["missing"], 120, "unknown"),
     (["rescue_12_9_motion_off"], 301, "duration")],
)
def test_smoke_contract_refuses_unbounded_or_unknown_requests(names, duration, match):
    from testing.development_ladder import load_contract
    parent = load_contract("testing/configs/luke0804_hindsight_ladder_v1.json")
    with pytest.raises(ValueError, match=match):
        build_smoke_contract(parent, candidate_names=names, duration_s=duration)
