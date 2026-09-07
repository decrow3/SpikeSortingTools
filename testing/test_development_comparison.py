import json

import pytest

from testing.development_ladder import (
    CONTRACT_SCHEMA,
    RESULT_SCHEMA,
    ContractError,
    build_plan,
    evaluate_results,
    load_contract,
    pin_plan,
    validate_contract,
)


def example_raw():
    return json.loads(open("configs/example.development_comparison.v1.json").read())


def contract(tmp_path):
    path = tmp_path / "contract.json"
    raw = example_raw()
    raw["candidates"] = raw["candidates"][:2]
    path.write_text(json.dumps(raw))
    return load_contract(path)


def evaluator(valid=True):
    invariants = {
        name: valid
        for name in (
            "exclusive_event_matching", "chance_aware_coincidence",
            "common_physical_time", "measurement_coverage_reported",
            "content_bound_inputs", "acquisition_and_selected_clocks_verified",
            "physical_probe_geometry_preserved",
        )
    }
    return {
        "invariants": invariants,
        "negative_controls": {
            "pathological_claim_mask": {"outcome": "worse"},
            "harmful_external_warp": {"outcome": "worse"},
            "fake_improvement_by_dropping_difficult_spikes": {"outcome": "rejected"},
        },
    }


def row(name, primary=(0.0, 0.0), guardrail=0.0, measured=80, common=0.8):
    return {
        "name": name,
        "eligible_units": 100,
        "measurable_units": measured,
        "common_time_fraction": common,
        "primary_improvements": {
            "median_common_time_missingness_improvement_pp": primary[0],
            "exclusive_identity_continuity_improvement": primary[1],
        },
        "guardrail_regressions": {
            "sliding_rp_contamination": guardrail,
            "chance_aware_duplicate_burden": guardrail,
            "split_merge_burden": guardrail,
            "waveform_instability": guardrail,
            "boundary_burden": guardrail,
        },
    }


def test_example_contract_is_valid_and_plan_records_halo(tmp_path):
    loaded = contract(tmp_path)
    plan = build_plan(loaded)
    assert plan["schema_version"] == CONTRACT_SCHEMA
    assert plan["reference"] == "rescue_12_9_motion_off"
    assert plan["halo_below_um"] == plan["halo_above_um"] == 300.0
    assert "no composite score" in plan["ranking_policy"]


def test_resolved_luke_contract_is_loadable_without_touching_recording_data():
    loaded = load_contract("testing/configs/luke0804_hindsight_ladder_v1.json")
    plan = build_plan(loaded)
    assert plan["experiment_id"] == "luke0804-imec0-hindsight-ladder-v1"
    assert plan["processing_width_um"] == 980.0
    assert plan["scoring_width_um"] == 580.0


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda x: x["recording"].update(duration_s=120), "full_session"),
        (lambda x: x["spatial_contract"].update(scoring_depth_um=[1100, 2900]), "halo"),
        (lambda x: x["metrics"]["primary"].update(unit_count={"minimum_improvement": 1, "repeatability": 0}), "unit count"),
        (lambda x: x["candidates"][1].update(sorter_config="peeler"), "unknown standard"),
    ],
)
def test_contract_refuses_prescription_violations(mutate, match):
    raw = example_raw()
    mutate(raw)
    with pytest.raises(ContractError, match=match):
        validate_contract(raw)


def test_evaluator_controls_gate_every_candidate(tmp_path):
    loaded = contract(tmp_path)
    raw = {
        "schema_version": RESULT_SCHEMA,
        "contract_digest": loaded.digest,
        "evaluator": evaluator(valid=False),
        "candidates": [
            row("rescue_12_9_motion_off"),
            row("rescue_12_9_native_rigid", primary=(8.0, 0.1)),
        ],
    }
    report = evaluate_results(loaded, raw)
    assert not report["evaluator_valid"]
    assert report["pareto_frontier"] == []
    assert next(d for d in report["decisions"] if d["name"].endswith("rigid"))["status"] == "stop"


def test_coverage_is_feasibility_gate_not_an_efficacy_result(tmp_path):
    loaded = contract(tmp_path)
    raw = {
        "schema_version": RESULT_SCHEMA,
        "contract_digest": loaded.digest,
        "evaluator": evaluator(),
        "candidates": [
            row("rescue_12_9_motion_off"),
            row("rescue_12_9_native_rigid", primary=(20.0, 0.2), measured=2),
        ],
    }
    report = evaluate_results(loaded, raw)
    candidate = next(d for d in report["decisions"] if d["name"].endswith("rigid"))
    assert candidate["status"] == "stop"
    assert "inadequate measurable-unit coverage" in candidate["reasons"]


def test_guardrail_failure_blocks_primary_win(tmp_path):
    loaded = contract(tmp_path)
    raw = {
        "schema_version": RESULT_SCHEMA,
        "contract_digest": loaded.digest,
        "evaluator": evaluator(),
        "candidates": [
            row("rescue_12_9_motion_off"),
            row("rescue_12_9_native_rigid", primary=(8.0, 0.1), guardrail=0.06),
        ],
    }
    report = evaluate_results(loaded, raw)
    assert report["pareto_frontier"] == []
    assert any("guardrail" in reason for reason in report["decisions"][1]["reasons"])


def test_passing_candidate_advances(tmp_path):
    loaded = contract(tmp_path)
    raw = {
        "schema_version": RESULT_SCHEMA,
        "contract_digest": loaded.digest,
        "evaluator": evaluator(),
        "candidates": [
            row("rescue_12_9_motion_off"),
            row("rescue_12_9_native_rigid", primary=(8.0, 0.1)),
        ],
    }
    report = evaluate_results(loaded, raw)
    assert report["pareto_frontier"] == ["rescue_12_9_native_rigid"]
    assert report["decisions"][1]["status"] == "advance"


def test_results_are_content_bound(tmp_path):
    loaded = contract(tmp_path)
    with pytest.raises(ContractError, match="another comparison"):
        evaluate_results(loaded, {"schema_version": RESULT_SCHEMA, "contract_digest": "wrong"})


def test_resolved_plan_is_pinned_before_expensive_work(tmp_path):
    loaded = contract(tmp_path)
    path = tmp_path / "output/comparison_plan.json"
    assert pin_plan(loaded, path)["contract_digest"] == loaded.digest
    assert pin_plan(loaded, path)["contract_digest"] == loaded.digest
    changed = json.loads(path.read_text())
    changed["experiment_id"] = "changed"
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="another contract"):
        pin_plan(loaded, path)
