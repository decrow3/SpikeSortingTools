import numpy as np
import pandas as pd
import pytest

from testing.ladder_score import truth_digest
from testing.luke_c2_stability_stage2 import (
    CANDIDATES as RUN_CANDIDATES, EXPECTED_DONORS, EXPECTED_EVENTS, STAGE2,
    check_disk, output_root, plan, realisations,
)
from testing.luke_c2_stability_stage2_analysis import (
    BASELINE, DECISION_ENDPOINTS, N_BOOTSTRAP, PER_COMPARISON_ALPHA, STATISTICS,
    classify, contrast, decide, decide_contrast, eligible_donors, exact_mcnemar,
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
                    "truth_sha256": f"sha-{d}-{r}",
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
def test_systematic_status_is_a_guardrail_not_a_population_filter():
    """[rev3] A donor only the *candidate* fails systematically disqualifies it.

    The mirror case is not symmetric and must not be: a donor only the baseline
    fails systematically is real evidence for the candidate, so rev3 keeps it in
    the primary population instead of discarding it as "hard".
    """
    # candidate-side: disqualifying, regardless of a better failure rate
    tagged = classify(matrix({("D10", "th_9_9"): fails(14)}))
    v = decide_contrast(contrast(tagged, BASELINE, "th_9_9"))
    assert v["verdict"] == "dropped" and v["new_systematic_donors"] == ["D10"]

    # baseline-side: counted for the candidate, not thrown away
    tagged = classify(matrix({("D10", BASELINE): fails(14)}))
    result = contrast(tagged, BASELINE, "th_9_9")
    assert result["excluded_systematic_donors"] == ["D10"]      # sensitivity only
    assert result["n_primary_donors"] == 14
    assert result["differences"]["failure_rate"]["point"] == pytest.approx(
        -14 / 196, abs=1e-5)                                   # points are 5dp
    assert decide_contrast(result)["new_systematic_donors"] == []


def test_a_candidate_cannot_improve_its_reported_rate_by_failing_more():
    """[rev3] The rev2 version of this test asserted only set membership.

    It therefore passed while the reported contrast still collapsed from +0.056
    to exactly 0.000 as a donor crossed the systematic line — the candidate was
    rewarded for failing once more. This version compares the numbers, which is
    what the title always claimed.
    """
    reported = {}
    for n in (10, 11, 12, 13, 14):
        tagged = classify(matrix({("D05", "th_8_8"): fails(n)}))
        reported[n] = contrast(tagged, BASELINE, "th_8_8")["differences"]["failure_rate"]["point"]
    # more failures must never look better; strictly worse across the boundary
    assert sorted(reported.values()) == list(reported.values())
    assert reported[12] > reported[11], "the 11->12 crossing still pays"
    # the primary population never shrinks in response to an outcome
    for n in (11, 12):
        tagged = classify(matrix({("D05", "th_8_8"): fails(n)}))
        assert contrast(tagged, BASELINE, "th_8_8")["n_primary_donors"] == 14


def test_the_outcome_dependent_population_survives_only_as_sensitivity():
    """The old estimand is still computed, still collapses, and is not decisive."""
    tagged = classify(matrix({("D05", "th_8_8"): fails(12)}))
    result = contrast(tagged, BASELINE, "th_8_8")
    assert "D05" not in eligible_donors(tagged, BASELINE, "th_8_8")
    sens = result["sporadic_only_sensitivity"]
    assert sens["n_donors"] == 13 and sens["differences"]["failure_rate"]["point"] == 0.0
    assert result["differences"]["failure_rate"]["point"] > 0     # primary disagrees
    # and no decision reads the sensitivity: deleting it changes nothing
    stripped = {k: v for k, v in result.items() if k != "sporadic_only_sensitivity"}
    assert decide_contrast(stripped) == decide_contrast(result)


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
    """Endpoints are driven apart, so reusing one CI for another cannot pass.

    The rev2 version used an all-constant matrix in which every endpoint had the
    identical CI; substituting the failure-rate interval for FP or splits would
    have gone unnoticed. Here each endpoint is moved by a different amount.
    """
    tagged = classify(matrix(
        fail_map={("D01", "th_8_8"): fails(6)},                  # failure rate
        fp_map={("D02", "th_8_8"): 900},                         # fp tail
        split_map={("D03", "th_8_8"): 4},                        # splits
    ))
    result = contrast(tagged, BASELINE, "th_8_8")
    points, cis = [], []
    for endpoint in DECISION_ENDPOINTS:
        d = result["differences"][endpoint]
        assert set(d) >= {"point", "ci", "excludes_zero", "n_donors", "ci_level"}
        assert d["ci"][0] <= d["point"] <= d["ci"][1]
        assert d["ci_level"] == pytest.approx(1 - PER_COMPARISON_ALPHA)   # every one
        assert d["n_donors"] == 14
        points.append(d["point"])
        cis.append(tuple(d["ci"]))
    assert len(set(points)) == len(set(cis)) == len(DECISION_ENDPOINTS)


def test_the_bootstrap_actually_computes_at_the_adjusted_level():
    """Reporting alpha=0.025 is not the same as using it.

    rev2 asserted the two reported constants only; a bootstrap hard-wired to 95%
    would have passed. This compares the widths two levels actually produce.
    """
    tagged = classify(matrix({("D01", "th_8_8"): fails(7),
                              ("D06", "th_8_8"): fails(3)}))
    stat = STATISTICS["failure_rate"]
    narrow = paired_donor_bootstrap(tagged, BASELINE, "th_8_8", stat, DONORS,
                                    alpha=0.05)
    wide = paired_donor_bootstrap(tagged, BASELINE, "th_8_8", stat, DONORS,
                                  alpha=PER_COMPARISON_ALPHA)
    assert PER_COMPARISON_ALPHA < 0.05
    width = lambda d: d["ci"][1] - d["ci"][0]
    assert width(wide) > width(narrow)
    assert wide["ci_level"] == pytest.approx(0.975)


SECONDARY = ("median_accuracy", "p10_accuracy", "fp_p90", "fp_max",
             "split_rate", "refractory_violation_median")


def test_all_prespecified_secondary_endpoints_are_reported():
    result = contrast(classify(matrix()), BASELINE, "th_9_9")
    for endpoint in SECONDARY:
        assert endpoint in result["differences"]
        assert endpoint in result["marginals"][BASELINE]
        assert np.isfinite(result["differences"][endpoint]["point"])


def perturbed(column, value, donors):
    """One column of the candidate arm moved on `donors`, everything else flat."""
    cells = matrix()
    mask = cells.candidate.eq("th_9_9") & cells.template.isin(donors)
    cells.loc[mask, column] = value
    # point estimates are exact and draw-count independent, so this sweep uses a
    # token bootstrap; the CIs themselves are covered by their own tests
    return contrast(classify(cells), BASELINE, "th_9_9", n_bootstrap=25)["differences"]


# each endpoint, its driver, and how much of the 196-cell arm that driver must
# reach: a median needs half the cells, a p90 a tenth, a mean or a max just one
ENDPOINT_DRIVERS = {
    "failure_rate": ("accuracy", 0.45, 1),
    "median_accuracy": ("accuracy", 0.95, 8),
    "p10_accuracy": ("accuracy", 0.45, 3),
    "fp_p90": ("fp", 900, 3),
    "fp_max": ("fp", 900, 1),
    "split_rate": ("n_output_units_capturing", 3, 1),
    "refractory_violation_median": ("refractory_violation_median", 0.02, 8),
}


@pytest.mark.parametrize("endpoint", sorted(ENDPOINT_DRIVERS))
def test_each_endpoint_responds_to_its_own_signal_and_not_to_others(endpoint):
    """Presence proved nothing: a hard-wired constant would have passed rev2."""
    flat = contrast(classify(matrix()), BASELINE, "th_9_9",
                    n_bootstrap=25)["differences"]
    column, value, n_donors = ENDPOINT_DRIVERS[endpoint]
    moved = perturbed(column, value, DONORS[:n_donors])
    assert moved[endpoint]["point"] != flat[endpoint]["point"], endpoint

    # and it must ignore a driver belonging to an unrelated column
    others = {"accuracy": ("fp", 900), "fp": ("accuracy", 0.95),
              "n_output_units_capturing": ("fp", 900),
              "refractory_violation_median": ("fp", 900)}[column]
    unrelated = perturbed(others[0], others[1], DONORS[:8])
    assert unrelated[endpoint]["point"] == flat[endpoint]["point"], endpoint


def test_the_smooth_endpoints_carry_a_prespecified_t_sensitivity():
    result = contrast(classify(matrix({("D01", "th_8_8"): fails(9)})),
                      BASELINE, "th_8_8")
    t = result["paired_difference_t_sensitivity"]
    assert set(t) == {"failure_rate", "split_rate"}      # never the tail statistics
    for name, entry in t.items():
        assert entry["n_donors"] == 14
        assert entry["ci"][0] <= entry["point"] <= entry["ci"][1]
        assert entry["point"] == pytest.approx(
            result["differences"][name]["point"], abs=1e-9)


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
    """A coherent synthetic result: the CI now decides, so it must match.

    [rev3] decide_contrast() reads the interval rather than a boolean, so a
    fixture whose CI contradicted its flag would no longer be meaningful input.
    """
    if excludes:
        assert point != 0, "a CI excluding zero cannot sit on zero"
        other = point / 2
    else:
        other = -point - (0.01 if point >= 0 else -0.01)
    lo, hi = sorted((point, other))
    return {"point": point, "excludes_zero": excludes, "ci": [lo, hi]}


def test_the_decision_reads_the_interval_not_the_boolean():
    """[rev3] point, ci and excludes_zero are three facts that can disagree."""
    incoherent = {"failure_rate": {"point": -0.05, "ci": [0.01, 0.10],
                                   "excludes_zero": True},
                  "fp_p90": diff(0.0, False), "split_rate": diff(0.0, False)}
    with pytest.raises(ValueError, match="outside its own CI"):
        verdict_from(incoherent)
    # a CI wholly above zero is a regression whatever the flag says
    honest = {"failure_rate": {"point": 0.05, "ci": [0.01, 0.10],
                               "excludes_zero": False},
              "fp_p90": diff(0.0, False), "split_rate": diff(0.0, False)}
    assert verdict_from(honest)["verdict"] == "dropped"


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
    """Rule 4 in full: failure rate, then FP p90, then split rate.

    rev2 exercised only the first key, so deleting or swapping the other two
    would have passed.
    """
    # level 1 — failure rate decides
    d = decide(contrasts_for({"th_8_8": (-0.02, 0.0, 0.0), "th_9_9": (-0.05, 0.0, 0.0)}))
    assert d["qualifying"] == ["th_8_8", "th_9_9"]
    assert d["selected"] == "th_9_9" and d["outcome"] == "candidate_replaces_baseline"

    # level 2 — failure rates tie, lower FP p90 decides
    d = decide(contrasts_for({"th_8_8": (-0.05, 3.0, 0.0), "th_9_9": (-0.05, -2.0, 0.0)}))
    assert d["selected"] == "th_9_9"
    d = decide(contrasts_for({"th_8_8": (-0.05, -2.0, 0.0), "th_9_9": (-0.05, 3.0, 0.0)}))
    assert d["selected"] == "th_8_8"          # and it is not order-of-insertion

    # level 3 — failure rate and FP tie, lower split rate decides
    d = decide(contrasts_for({"th_8_8": (-0.05, 1.0, 0.02), "th_9_9": (-0.05, 1.0, -0.01)}))
    assert d["selected"] == "th_9_9"
    d = decide(contrasts_for({"th_8_8": (-0.05, 1.0, -0.01), "th_9_9": (-0.05, 1.0, 0.02)}))
    assert d["selected"] == "th_8_8"

    # the ranking the prespec withdrew must not creep back in
    assert d["tie_break"] == "failure_rate, then fp_p90, then split_rate"


def test_an_exact_tie_between_two_winners_is_escalated_not_guessed():
    d = decide(contrasts_for({"th_8_8": (-0.05, 0.0, 0.0), "th_9_9": (-0.05, 0.0, 0.0)}))
    assert d["outcome"] == "multiple_candidates_tied_escalate"
    assert d["selected"] is None


def test_multiplicity_is_handled_not_ignored():
    d = decide(contrasts_for({"th_8_8": (0.0, 0.0, 0.0), "th_9_9": (0.0, 0.0, 0.0)}))
    assert d["familywise_alpha"] == 0.05
    assert d["per_comparison_alpha"] == pytest.approx(0.025)


# --------------------------------------------------------------------------- #
# [rev3] coverage the reviewer found missing entirely
# --------------------------------------------------------------------------- #
def test_validation_requires_every_column_it_depends_on():
    """A column that was merely optional degraded silently into a NaN endpoint."""
    for column in ("n_events", "refractory_violation_median", "truth_sha256"):
        with pytest.raises(ValueError, match="missing columns"):
            validate_cells(matrix().drop(columns=[column]))


def test_the_right_number_of_configurations_is_not_the_right_configurations():
    renamed = matrix().replace({"candidate": dict(zip(CONFIGS, ("foo", "bar", "baz")))})
    assert renamed.candidate.nunique() == 3            # the old check passed here
    with pytest.raises(ValueError, match="configurations are"):
        validate_cells(renamed)


def test_a_null_endpoint_is_refused_rather_than_counted_as_a_success():
    """`NaN < FAILURE_THRESHOLD` is False, so an unchecked null passes as a win."""
    for column in ("accuracy", "fp", "n_output_units_capturing",
                   "refractory_violation_median"):
        bad = matrix()
        bad.loc[0, column] = np.nan
        with pytest.raises(ValueError, match=f"non-finite {column}"):
            validate_cells(bad)
    infinite = matrix()
    infinite["fp"] = infinite.fp.astype(float)
    infinite.loc[3, "fp"] = np.inf
    with pytest.raises(ValueError, match="non-finite fp"):
        validate_cells(infinite)
    # and the failure it would otherwise have masked is real
    masked = matrix()
    masked.loc[0, "accuracy"] = np.nan
    assert not classify(masked).loc[0, "failed"]


def test_configurations_must_be_proved_to_have_seen_the_identical_train():
    """Matching labels are not evidence; the recorded hashes are."""
    ok = matrix()
    assert validate_cells(ok)["truth_hashes"] == len(REALS) * len(DONORS)
    drifted = matrix()
    row = (drifted.template.eq("D04") & drifted.realisation.eq("r05")
           & drifted.candidate.eq("th_8_8"))
    drifted.loc[row, "truth_sha256"] = "sha-somethingelse"
    with pytest.raises(ValueError, match="not hash-paired"):
        validate_cells(drifted)


def test_thresholds_must_be_proved_applied_not_inferred_from_the_request():
    from testing.luke_c2_stability_stage2 import assert_applied_settings

    requested = {"Th_universal": 8, "Th_learned": 8}
    applied = {"Th_universal": 8, "Th_learned": 8, "effective_nblocks": 0,
               "_sources": {"Th_universal": "applied:applied_Th_universal",
                            "Th_learned": "applied:applied_Th_learned"}}
    assert assert_applied_settings("th_8_8", applied, requested) is applied

    # the tautology: a value echoed back from the request must not satisfy it
    echoed = {**applied, "_sources": {"Th_universal": "requested:Th_universal",
                                      "Th_learned": "applied:applied_Th_learned"}}
    with pytest.raises(RuntimeError, match="not applied-derived"):
        assert_applied_settings("th_8_8", echoed, requested)

    # provenance is necessary, not sufficient
    with pytest.raises(RuntimeError, match="did not resolve as requested"):
        assert_applied_settings("th_8_8", {**applied, "Th_learned": 9}, requested)
    with pytest.raises(RuntimeError, match="did not resolve as requested"):
        assert_applied_settings("th_8_8", {**applied, "effective_nblocks": 1}, requested)


def test_effective_settings_still_reports_when_it_fell_back(tmp_path):
    """The fallback is not removed — it is made visible, so callers can refuse."""
    from testing.ladder_sorter import effective_settings

    eff = effective_settings({
        "summary": {"critical_saved_settings": {"effective_nblocks": 0,
                                                "do_CAR": True}},
        "sorter_params": {"Th_universal": 8, "Th_learned": 8},
    })
    assert eff["Th_universal"] == 8                     # value still resolves
    assert eff["_sources"]["Th_universal"] == "requested:Th_universal"


def test_frozen_outputs_are_written_atomically(tmp_path):
    from testing.luke_c2_stability_stage2 import _atomic_csv, _atomic_write
    from testing.luke_c2_stability_stage2_analysis import _atomic_write as a_write

    for writer, name in ((_atomic_write, "runner.json"), (a_write, "analysis.json")):
        target = tmp_path / name
        writer(target, '{"a": 1}\n')
        assert target.read_text() == '{"a": 1}\n'
        assert not list(tmp_path.glob("*.tmp"))         # no residue left behind
    _atomic_csv(pd.DataFrame([{"a": 1}]), tmp_path / "cells.csv")
    assert (tmp_path / "cells.csv").exists() and not list(tmp_path.glob("*.tmp"))


def test_the_prespec_manifest_is_frozen_by_comparison_not_by_trust(tmp_path):
    """A changed protocol must stop the run, not overwrite the manifest."""
    import json

    from testing.luke_c2_stability_stage2 import STAGE2

    manifest = tmp_path / "prespec.json"
    manifest.write_text(json.dumps({**STAGE2, "plan": {}}, indent=2) + "\n")
    assert json.loads(manifest.read_text())["n_donors"] == EXPECTED_DONORS
    assert json.loads(manifest.read_text()) != {**STAGE2, "plan": {"cells": 588}}
