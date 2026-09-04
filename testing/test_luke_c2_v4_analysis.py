import numpy as np
import pandas as pd
import pytest

from testing.luke_c2_v4_analysis import (
    ACCURACY_MIN,
    CONTRASTS,
    check_denominators,
    cohorts,
    interaction,
    penalties,
    qualify,
)


def cell(template, condition, arm, sorter, accuracy, n_truth=708, sha="a"):
    return {"template": template, "condition": condition, "arm": arm,
            "sorter": sorter, "accuracy": accuracy, "n_truth": n_truth,
            "truth_sha256": sha, "n_output_units_capturing": 1}


def matrix(**overrides):
    """Two donors x two conditions, all three arms, all sorters."""
    rows = []
    spec = {
        ("D01", "ramp_11um"): {"static": {"rescue": 0.99, "rescue_rigid": 0.55,
                                          "legacy_style": 0.99},
                               "moved": {"rescue": 0.50, "rescue_rigid": 0.98,
                                         "legacy_style": 0.90}},
        ("D02", "ramp_11um"): {"static": {"rescue": 0.99, "rescue_rigid": 0.99,
                                          "legacy_style": 0.99},
                               "moved": {"rescue": 0.60, "rescue_rigid": 0.95,
                                         "legacy_style": 0.90}},
        ("D01", "staircase_40um"): {"static": {"rescue": 0.99, "rescue_rigid": 0.99,
                                               "legacy_style": 0.99},
                                    "moved": {"rescue": 0.40, "rescue_rigid": 0.98,
                                              "legacy_style": 0.90}},
        ("D02", "staircase_40um"): {"static": {"rescue": 0.99, "rescue_rigid": 0.99,
                                               "legacy_style": 0.99},
                                    "moved": {"rescue": 0.40, "rescue_rigid": 0.98,
                                              "legacy_style": 0.90}},
    }
    for (t, c), arms in spec.items():
        n = 687 if c.startswith("staircase") else 708
        sha = "stair" if c.startswith("staircase") else "ramp"
        for arm, sorters in arms.items():
            for s, a in sorters.items():
                rows.append(cell(t, c, arm, s, a, n, sha))
        rows.append(cell(t, c, "moved_corrected", "rescue",
                         arms["static"]["rescue"], n, sha))
    frame = pd.DataFrame(rows)
    for key, value in overrides.items():
        frame.loc[frame.index[-1], key] = value
    return frame


def test_qualification_is_contrast_specific_and_per_condition():
    q = qualify(matrix())
    primary = q[q.contrast == "rescue_vs_rescue_rigid"]
    # D01 passes under rescue (0.99) but fails under rescue_rigid (0.55) on the
    # ramp, so it is out of THAT contrast in THAT condition only
    d01_ramp = primary[(primary.template == "D01") & (primary.condition == "ramp_11um")]
    assert not d01_ramp.qualified.item()
    d01_stair = primary[(primary.template == "D01")
                        & (primary.condition == "staircase_40um")]
    assert d01_stair.qualified.item()
    # ... and it still qualifies for the legacy contrast on the ramp
    legacy = q[(q.contrast == "rescue_vs_legacy_style") & (q.template == "D01")
               & (q.condition == "ramp_11um")]
    assert legacy.qualified.item()


def test_common_primary_cohort_is_the_intersection_across_conditions():
    """A donor qualifying in only some conditions must not enter the primary."""
    cohort = cohorts(qualify(matrix()), "rescue_vs_rescue_rigid")
    assert cohort["common_primary"] == ["D02"]
    assert "D01" in cohort["excluded_from_common"]
    assert set(cohort["qualified_by_condition"]["staircase_40um"]) == {"D01", "D02"}


def test_per_arm_exclusion_is_not_what_cohorts_does():
    """Cohorts are built from static arms only; a bad moving arm never excludes."""
    frame = matrix()
    frame.loc[(frame.template == "D02") & (frame.arm == "moved"), "accuracy"] = 0.01
    cohort = cohorts(qualify(frame), "rescue_vs_rescue_rigid")
    assert "D02" in cohort["common_primary"]


def test_interaction_is_a_difference_of_differences():
    frame = matrix()
    inter = interaction(penalties(frame), "rescue_vs_rescue_rigid")
    row = inter[(inter.template == "D01") & (inter.condition == "ramp_11um")].iloc[0]
    # rescue: 0.50 - 0.99 = -0.49 ; rigid: 0.98 - 0.55 = +0.43
    assert row["penalty_rescue"] == pytest.approx(-0.49)
    assert row["penalty_rescue_rigid"] == pytest.approx(0.43)
    assert row["interaction"] == pytest.approx(0.92)
    # the naive moving-arm difference would have been 0.98 - 0.50 = 0.48,
    # attributing the stationary rigid collapse to motion recovery
    assert row["interaction"] != pytest.approx(0.48)


def test_registration_delta_is_measured_against_the_same_condition_static():
    penalty = penalties(matrix())
    assert (penalty.registration_delta.abs() < 1e-12).all()


def test_mixed_denominators_within_a_condition_fail_closed():
    frame = matrix()
    frame.loc[(frame.template == "D01") & (frame.condition == "ramp_11um")
              & (frame.arm == "moved"), "n_truth"] = 687
    with pytest.raises(ValueError, match="do not share one truth train"):
        check_denominators(frame)


def test_different_conditions_may_hold_different_trains():
    """687 for the staircase and 708 for the ramps is by design, not an error."""
    info = check_denominators(matrix())
    assert info["n_truth_by_condition"]["staircase_40um"] == [687]
    assert info["n_truth_by_condition"]["ramp_11um"] == [708]


def test_contrast_registry_names_the_primary_and_the_comparator():
    assert CONTRASTS["rescue_vs_rescue_rigid"] == ("rescue", "rescue_rigid")
    assert CONTRASTS["rescue_vs_legacy_style"] == ("rescue", "legacy_style")
    assert ACCURACY_MIN == 0.8
