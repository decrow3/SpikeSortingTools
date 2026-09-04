import json

import numpy as np
import pytest

from testing.luke_rescue_c2_drift_challenge_v4 import (
    PRESPEC,
    SCHEMA,
    V4IsolationError,
    V3_OUTPUT,
    output_root,
    prespec_digest,
    trajectory_for,
    verify,
)


def test_schema_is_v4_and_not_v3():
    assert PRESPEC["schema"] == SCHEMA == "luke-rescue-c2-drift-challenge-v4"
    assert "v3" in PRESPEC["supersedes"] and "void" in PRESPEC["supersedes"]


def test_output_root_refuses_mnt(tmp_path):
    with pytest.raises(V4IsolationError, match="under /mnt"):
        output_root("/mnt/NPX/whatever")
    assert output_root(tmp_path).resolve() == tmp_path.resolve()


def test_output_root_refuses_any_overlap_with_the_v3_tree():
    for candidate in (V3_OUTPUT, V3_OUTPUT / "runs", V3_OUTPUT.parent / "luke_rescue_c2_drift_challenge_v3"):
        with pytest.raises(V4IsolationError, match="v3 tree"):
            output_root(candidate)


def test_ramps_reach_exactly_their_frozen_excursion():
    duration = PRESPEC["background"]["duration_s"]
    for name, spec in PRESPEC["conditions"].items():
        if spec["kind"] != "rigid_ramp":
            continue
        fn = trajectory_for(name)
        assert fn(0.0) == pytest.approx(0.0)
        assert float(fn(np.array([duration]))[0]) == pytest.approx(spec["total_um"])
        assert f"{int(spec['total_um'])}um" in name


def test_luke_calibrated_ramps_are_labelled_forward_model_confounded():
    ramps = [s for s in PRESPEC["conditions"].values() if s["kind"] == "rigid_ramp"]
    assert ramps and all(s["forward_model_confounded"] for s in ramps)
    assert {5.0, 11.0, 22.0} == {s["total_um"] for s in ramps}
    control = PRESPEC["conditions"]["staircase_40um"]
    assert control["forward_model_confounded"] is False
    assert "separately" in control["reported"]


def test_the_staircase_only_ever_visits_commensurate_levels():
    from testing.luke_c2_staircase_control import STAIRCASE

    fn = trajectory_for("staircase_40um")
    values = np.unique(fn(np.linspace(0.0, 119.99, 5000)))
    assert set(values) <= set(STAIRCASE["levels_um"])


def test_the_stationary_rigid_control_is_required():
    control = PRESPEC["required_stationary_control"]
    assert control == {"arm": "static", "sorter": "rescue_rigid"}
    assert "interaction" in PRESPEC["correction_effect"] or "-" in PRESPEC["correction_effect"]
    assert "static_rigid" in PRESPEC["correction_effect"]


def test_qualification_is_contrast_specific_and_forbids_per_arm_exclusion():
    rule = PRESPEC["static_qualification"]
    assert "contrast" in rule["rule"]
    assert "per-arm" in rule["forbidden"]
    assert set(rule["reported_cohorts"]) == {
        "common_primary", "all_donor", "operator_qualified_sensitivity"
    }


def test_rescue_rigid_must_not_move_the_thresholds():
    """verify() fails closed if the isolating contrast stops isolating."""
    import testing.luke_rescue_c2_drift_challenge_v4 as v4
    from testing.ladder_sorter import SorterConfig

    original = dict(v4.NAMED_CONFIGS)
    v4.NAMED_CONFIGS["rescue_rigid"] = SorterConfig(
        "rescue_rigid", {"do_correction": True, "nblocks": 1, "Th_universal": 9}
    )
    try:
        with pytest.raises(V4IsolationError, match="detection thresholds"):
            verify()
    finally:
        v4.NAMED_CONFIGS.clear()
        v4.NAMED_CONFIGS.update(original)


def test_prespec_digest_is_stable_and_content_sensitive():
    first = prespec_digest()
    assert first == prespec_digest() and len(first) == 64


def test_verify_freezes_a_prespec_and_refuses_a_changed_one(tmp_path):
    checks = verify(tmp_path)
    frozen = json.loads((tmp_path / "prespec.json").read_text())
    assert frozen == PRESPEC
    assert checks["donor_cohort"] == "verified"
    (tmp_path / "prespec.json").write_text(json.dumps({**PRESPEC, "frozen": "later"}))
    with pytest.raises(SystemExit, match="run-once"):
        verify(tmp_path)


def test_scoring_constants_match_the_scorer_actually_used():
    from testing.ladder_score import (
        ACCURACY_GATE, CAPTURE_FRAC, CHANCE_MARGIN, DEFAULT_TOL_MS,
        SCORE_SCHEMA, TRUTH_CONTRACT_SCHEMA,
    )

    scoring = PRESPEC["scoring"]
    assert scoring["score_schema"] == SCORE_SCHEMA
    assert scoring["truth_contract_schema"] == TRUTH_CONTRACT_SCHEMA
    assert scoring["accuracy_gate"] == ACCURACY_GATE
    assert scoring["capture_frac"] == CAPTURE_FRAC
    assert scoring["chance_margin"] == CHANCE_MARGIN
    assert scoring["tol_ms"] == DEFAULT_TOL_MS


def test_the_recorded_operator_matches_the_one_the_arms_use():
    from testing.luke_c2_operator_calibration import REFERENCE_OPERATOR

    operator = PRESPEC["motion_operator"]
    for key, value in REFERENCE_OPERATOR.items():
        assert operator[key] == value
