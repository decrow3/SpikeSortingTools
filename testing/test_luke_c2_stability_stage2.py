import numpy as np
import pandas as pd
import pytest

from testing.ladder_score import truth_digest
from testing.luke_c2_stability_stage2 import (
    CANDIDATES as RUN_CANDIDATES, EXPECTED_DONORS, EXPECTED_EVENTS, STAGE2,
    check_disk, output_root, plan, realisations,
)
from testing.luke_c2_stability_stage2_analysis import (
    BASELINE, DECISION_ENDPOINTS, PER_COMPARISON_ALPHA, STATISTICS, classify,
    contrast, decide, decide_contrast, eligible_donors, exact_mcnemar,
    paired_donor_bootstrap, validate_cells,
)

FS = 29999.759166666667
DONORS = [f"D{i:02d}" for i in range(1, 15)]
REALS = [f"r{i:02d}" for i in range(14)]
CONFIGS = [BASELINE, "th_8_8", "th_9_9"]


# --------------------------------------------------------------------------- #
# design: the confounds that must not leak in
# --------------------------------------------------------------------------- #
def test_every_realisation_holds_exactly_the_prespecified_count():
    """Equal-but-wrong would pass a mere equality check."""
    trains = realisations(FS)
    assert {t.size for t in trains.values()} == {EXPECTED_EVENTS} == {687}


def test_composition_and_phase_are_crossed_not_confounded():
    trains = realisations(FS)
    for seed in range(1, 7):
        p0, ph = trains[f"random_s{seed}_p0"], trains[f"random_s{seed}_phalf"]
        offsets = np.unique(ph - p0)
        assert offsets.size == 1 and offsets[0] > 0        # same events, shifted
    assert len({truth_digest({"inj0": t}) for t in trains.values()}) == 14


def test_runner_requires_the_exact_frozen_cohort():
    from testing.luke_rescue_c2_drift_challenge import _resolve_frozen_cohort

    assert _resolve_frozen_cohort(DONORS, None) == DONORS
    with pytest.raises(ValueError):
        _resolve_frozen_cohort(DONORS, ["D01", "D02"])          # a subset
    assert STAGE2["n_donors"] == EXPECTED_DONORS == 14


def test_disk_plan_and_guard(tmp_path):
    lean, fat = plan(14, 14, False), plan(14, 14, True)
    assert lean["cells"] == 588 and lean["recordings"] == 196
    assert lean["estimated_gb"] == pytest.approx(588 * 0.21 + 0.75, abs=0.1)
    assert fat["estimated_gb"] == pytest.approx(588 * 0.21 + 196 * 0.75, abs=0.1)
    # boundary behaviour, measured against the actual volume rather than an
    # absurd constant: a request that fits passes, one that eats the headroom fails
    import shutil
    free_gb = shutil.disk_usage(tmp_path).free / 1e9
    check_disk(max(free_gb - 60, 1), tmp_path)                     # comfortably fits
    with pytest.raises(RuntimeError, match="headroom"):
        check_disk(free_gb, tmp_path)                              # no headroom left
    assert all(c.overrides["do_correction"] is False for c in RUN_CANDIDATES)


def test_never_writes_under_mnt():
    with pytest.raises(ValueError, match="under /mnt"):
        output_root("/mnt/NPX/nope")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def matrix(fail_map=None, fp_map=None, split_map=None):
    """A complete 588-cell design; fail_map[(donor, config)] = list of 14 bools."""
    fail_map, fp_map, split_map = fail_map or {}, fp_map or {}, split_map or {}
    rows = []
    for d in DONORS:
        for i, r in enumerate(REALS):
            for c in CONFIGS:
                failed = fail_map.get((d, c), [False] * 14)[i]
                rows.append({
                    "template": d, "realisation": r, "candidate": c,
                    "n_events": EXPECTED_EVENTS,
                    "accuracy": 0.45 if failed else 0.99,
                    "fp": fp_map.get((d, c), 500 if failed else 2),
                    "n_output_units_capturing": split_map.get((d, c), 2 if failed else 1),
                    "refractory_violation_median": 0.001,
                })
    return pd.DataFrame(rows)


def fails(n):
    return [True] * n + [False] * (14 - n)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_incomplete_or_duplicated_matrices_are_refused():
    assert validate_cells(matrix())["cells"] == 588
    with pytest.raises(ValueError, match="incomplete design"):
        validate_cells(matrix().iloc[:-1])
    dup = pd.concat([matrix(), matrix().iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        validate_cells(dup)


def test_a_wrong_event_count_is_refused():
    bad = matrix()
    bad.loc[0, "n_events"] = 686
    with pytest.raises(ValueError, match="687 events"):
        validate_cells(bad)


# --------------------------------------------------------------------------- #
# CODEX FIX 1 — the paired comparison must use the common eligible population
# --------------------------------------------------------------------------- #
def test_systematic_donors_are_excluded_from_the_paired_comparison():
    """A 14/14 baseline failure must not drive the sporadic contrast.

    Regression for the reviewed defect: the old code pivoted raw `failed` and
    would have reported b=14, c=0, p=0.0001 from D10 alone.
    """
    tagged = classify(matrix({("D10", BASELINE): fails(14)}))
    assert "D10" not in eligible_donors(tagged, BASELINE, "th_9_9")
    result = contrast(tagged, BASELINE, "th_9_9")
    assert result["excluded_systematic_donors"] == ["D10"]
    assert result["unadjusted_mcnemar"]["baseline_fails_candidate_ok"] == 0
    assert result["differences"]["failure_rate"]["point"] == 0.0


def test_eligibility_is_the_union_so_a_config_cannot_gain_by_failing_more():
    """At 11 failures the donor stays; at 12 it leaves BOTH arms, not just one."""
    tagged11 = classify(matrix({("D05", "th_8_8"): fails(11)}))
    tagged12 = classify(matrix({("D05", "th_8_8"): fails(12)}))
    assert "D05" in eligible_donors(tagged11, BASELINE, "th_8_8")
    assert "D05" not in eligible_donors(tagged12, BASELINE, "th_8_8")
    # and it is removed from the baseline arm too, so both describe one population
    for tagged in (tagged11, tagged12):
        donors = eligible_donors(tagged, BASELINE, "th_8_8")
        result = contrast(tagged, BASELINE, "th_8_8")
        assert result["differences"]["failure_rate"]["n_donors"] == len(donors)


# --------------------------------------------------------------------------- #
# CODEX FIX 2/3 — donor-clustered paired inference, per-endpoint CIs
# --------------------------------------------------------------------------- #
def test_bootstrap_resamples_donors_not_cells():
    """Clustering matters: one donor failing every realisation is 1 unit, not 14."""
    tagged = classify(matrix({("D03", "th_9_9"): fails(14)}))
    donors = [d for d in DONORS if d != "D03"] + ["D03"]
    out = paired_donor_bootstrap(
        tagged, BASELINE, "th_9_9", STATISTICS["failure_rate"], donors)
    assert out["n_donors"] == 14
    # a single deviant donor cannot give a CI that excludes zero at 14 donors
    assert not out["excludes_zero"]


def test_every_decision_endpoint_has_its_own_paired_ci():
    tagged = classify(matrix())
    result = contrast(tagged, BASELINE, "th_8_8")
    for endpoint in DECISION_ENDPOINTS:
        diff = result["differences"][endpoint]
        assert set(diff) >= {"point", "ci", "excludes_zero", "n_donors"}
        assert diff["ci"][0] <= diff["point"] <= diff["ci"][1]
    assert diff["ci_level"] == pytest.approx(1 - PER_COMPARISON_ALPHA)


def test_all_prespecified_secondary_endpoints_are_reported():
    result = contrast(classify(matrix()), BASELINE, "th_9_9")
    for endpoint in ("median_accuracy", "p10_accuracy", "fp_p90", "fp_max",
                     "split_rate", "refractory_violation_median"):
        assert endpoint in result["differences"]
        assert endpoint in result["marginals"][BASELINE]


def test_mcnemar_is_reported_but_flagged_as_not_decisive():
    result = contrast(classify(matrix()), BASELINE, "th_8_8")
    assert "not used for decisions" in result["unadjusted_mcnemar"]["caveat"]
    assert exact_mcnemar(10, 0) == pytest.approx(2 / 2 ** 10)
    assert exact_mcnemar(0, 10) == exact_mcnemar(10, 0)


# --------------------------------------------------------------------------- #
# CODEX FIX 5 — decision coherence
# --------------------------------------------------------------------------- #
def verdict_from(differences, systematic=None):
    result = {"candidate": "th_8_8", "baseline": BASELINE,
              "differences": differences,
              "systematic_by_config": systematic or {BASELINE: [], "th_8_8": []}}
    return decide_contrast(result)


def diff(point, excludes):
    return {"point": point, "excludes_zero": excludes, "ci": [point, point]}


def test_a_split_regression_disqualifies_even_a_better_candidate():
    """The reviewed defect: branch order returned replaces_baseline regardless."""
    v = verdict_from({"failure_rate": diff(-0.05, True), "fp_p90": diff(-1.0, False),
                      "split_rate": diff(0.1, True)})
    assert v["verdict"] == "dropped" and "split_rate" in v["regressions"]


def test_fp_regression_is_judged_on_its_own_ci_not_the_failure_p_value():
    """A large FP regression must drop even when failure rates are tied."""
    v = verdict_from({"failure_rate": diff(0.0, False), "fp_p90": diff(395.0, True),
                      "split_rate": diff(0.0, False)})
    assert v["verdict"] == "dropped" and v["regressions"] == ["fp_p90"]


def test_a_tiny_split_increase_without_a_ci_excluding_zero_does_not_drop():
    v = verdict_from({"failure_rate": diff(0.0, False), "fp_p90": diff(0.0, False),
                      "split_rate": diff(0.0001, False)})
    assert v["verdict"] == "not_separated"


def test_a_new_systematic_failure_disqualifies():
    v = verdict_from({"failure_rate": diff(-0.05, True), "fp_p90": diff(0.0, False),
                      "split_rate": diff(0.0, False)},
                     systematic={BASELINE: [], "th_8_8": ["D07"]})
    assert v["verdict"] == "dropped" and v["new_systematic_donors"] == ["D07"]


def test_qualification_needs_a_ci_excluding_zero():
    assert verdict_from({"failure_rate": diff(-0.05, False), "fp_p90": diff(0.0, False),
                         "split_rate": diff(0.0, False)})["verdict"] == "not_separated"
    assert verdict_from({"failure_rate": diff(-0.05, True), "fp_p90": diff(0.0, False),
                         "split_rate": diff(0.0, False)})["verdict"] == "qualifies"


def test_a_baseline_advantage_is_not_described_as_no_difference():
    v = verdict_from({"failure_rate": diff(0.05, True), "fp_p90": diff(0.0, False),
                      "split_rate": diff(0.0, False)})
    assert v["verdict"] == "dropped"
    assert "no paired difference" not in v["reason"]


# --------------------------------------------------------------------------- #
# outcome across candidates
# --------------------------------------------------------------------------- #
def contrasts_for(verdicts):
    out = {}
    for name, (rate, fp, split) in verdicts.items():
        out[name] = {"candidate": name, "baseline": BASELINE,
                     "differences": {"failure_rate": diff(rate, rate != 0),
                                     "fp_p90": diff(fp, False),
                                     "split_rate": diff(split, False)},
                     "systematic_by_config": {BASELINE: [], name: []}}
    return out


def test_no_qualifying_candidate_gives_no_threshold_change():
    d = decide(contrasts_for({"th_8_8": (0.0, 0.0, 0.0), "th_9_9": (0.0, 0.0, 0.0)}))
    assert d["outcome"] == "no_threshold_change" and d["selected"] is None


def test_two_qualifying_candidates_are_ranked_by_the_prespecified_order():
    d = decide(contrasts_for({"th_8_8": (-0.02, 0.0, 0.0), "th_9_9": (-0.05, 0.0, 0.0)}))
    assert d["qualifying"] == ["th_8_8", "th_9_9"]
    assert d["selected"] == "th_9_9"          # the larger reduction
    assert d["outcome"] == "candidate_replaces_baseline"


def test_an_exact_tie_between_two_winners_is_escalated_not_guessed():
    d = decide(contrasts_for({"th_8_8": (-0.05, 0.0, 0.0), "th_9_9": (-0.05, 0.0, 0.0)}))
    assert d["outcome"] == "multiple_candidates_tied_escalate"
    assert d["selected"] is None


def test_multiplicity_is_handled_not_ignored():
    d = decide(contrasts_for({"th_8_8": (0.0, 0.0, 0.0), "th_9_9": (0.0, 0.0, 0.0)}))
    assert d["familywise_alpha"] == 0.05
    assert d["per_comparison_alpha"] == pytest.approx(0.025)
