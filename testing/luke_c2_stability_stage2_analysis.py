"""Stage 2 endpoints and decision rules, written before the data exists.

Prespec: `docs/luke_c2_train_stability_stage2_prespec.md`. Kept separate from the
runner so the endpoints can be reviewed, tested against synthetic frames, and
recomputed without re-sorting.

Inference, and why it is shaped this way
----------------------------------------
Cells are **not** independent: the 14 realisations within a donor share that
donor's waveform, amplitude and placement. Every interval here therefore comes
from a **paired bootstrap that resamples donors**, carrying all of a donor's
realisations together. An exact McNemar test over the 196 donor-realisation
pairs would assume those pairs are independent draws, which they are not; it is
still reported, explicitly labelled unadjusted, but **no decision rests on it**.

Inference is **conditional on the 14 frozen train realisations** and generalises
across donors only. Extending it to other realisations would need the
realisation axis resampled too, and this design cannot support that.

The primary population is **all frozen donors** [rev3]
------------------------------------------------------
Earlier revisions ran the primary contrast on the donors that were not
"systematic" under either arm. The union made both arms describe the same
donors, which fixed an unequal-denominator bug — but membership of that set is
itself a Stage-2 outcome, so the estimand stayed conditioned on the result.
Crossing one donor from 11 to 12 failures drops it from both arms and moves the
reported contrast from +0.056 to exactly 0.000; the fix's own test asserted only
set membership and never caught it.

So the primary contrast now runs on **all 14 frozen donors**. A donor that fails
under both arms contributes equally to both and cancels from the paired
difference, which makes inclusion conservative rather than biased. Systematic
status survives as a *separate disqualifying guardrail* (a candidate is dropped
if it makes a donor systematic that the baseline does not), and the
union-excluded population survives as a labelled **sensitivity** analysis that
no decision reads. Fixing the population also makes the two candidates rankable
against one another, which per-contrast union sets did not.

Every decision endpoint is a **paired difference** on that fixed population,
with its own donor-bootstrap CI. Endpoints are never compared through another
endpoint's p-value.

Interval limitations, stated in advance [rev3]
----------------------------------------------
These are ordinary percentile bootstraps over **14 clusters**. At a 97.5 % level
only ~50 of the 4000 draws populate each nominal tail, and the tail endpoints
(`p10_accuracy`, `fp_p90`, `fp_max`) are order statistics of a heavily tied
pooled sample, so their bootstrap distributions are visibly discrete and can
produce stepwise or zero-width intervals. This is reported, not corrected: BCa's
jackknife acceleration is itself unstable for tied maxima and quantiles, and a
t-interval is unsuitable for non-smooth endpoints. For the two *smooth*
endpoints a donor-level paired t-interval is reported alongside as a
prespecified sensitivity check. More draws would cut Monte-Carlo noise but
cannot manufacture more than 14 independent clusters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

REPO_ROOT = Path(__file__).resolve().parents[1]


def _atomic_write(path: Path, text: str) -> None:
    """Same-directory temp then rename, so a frozen output is never half-written."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
DEFAULT_ROOT = REPO_ROOT / "testing/outputs/luke_c2_stability_stage2"

ANALYSIS_SCHEMA = "luke-c2-stability-stage2-analysis-v3"
FAILURE_THRESHOLD = 0.9
SYSTEMATIC_MIN_FAILURES = 12
# [rev5] Two rules that close the 12/14 cliff. Both can only ever DISQUALIFY a
# candidate, never promote one, so neither can manufacture a false positive.
# The constants are frozen judgement calls, not fitted quantities.
#
# A benchmark that fails more than half its cells cannot serve as a benchmark,
# however much better than production it is: "less bad" is not usable, and that
# situation is already what rule 3 exists to report.
ABSOLUTE_FAILURE_CAP = 0.5
# Of 14 realisations, failing 4 more than production on one donor (29 points) is
# a material per-donor regression. This fires on the candidate-minus-baseline
# difference, so unlike the systematic flag it has no special behaviour at 12 and
# catches a candidate that is much worse on one donor while winning on aggregate.
DONOR_DETERIORATION_CAP = 4
BASELINE = "th_12_9"
CANDIDATES = ("th_8_8", "th_9_9")
EXPECTED_CELLS = 588
EXPECTED_DONORS = 14
EXPECTED_REALISATIONS = 14
EXPECTED_EVENTS = 687
N_BOOTSTRAP = 4000
BOOTSTRAP_SEED = 20260905
FAMILY_ALPHA = 0.05                      # familywise, across both candidates
PER_COMPARISON_ALPHA = FAMILY_ALPHA / len(CANDIDATES)   # Bonferroni


# --------------------------------------------------------------------------- #
# validation — the analysis must refuse an incomplete or duplicated matrix
# --------------------------------------------------------------------------- #
# Decision-critical endpoints. score_sort() returns 0.0, not NaN, when nothing
# captures the train (ladder_score: `best, tp, fp, fn, accuracy = None, 0, 0,
# n_truth, 0.0`), so a null here is corruption, never a legitimate result -- and
# an unchecked one would silently pass, because `NaN < FAILURE_THRESHOLD` is
# False.
FINITE_ENDPOINTS = ("accuracy", "fp", "n_output_units_capturing")
# Guardrails that are *legitimately undefined* for some cells: guardrails() sets
# rv = [nan] when no good unit has two spikes, so a cell where the sorter found
# nothing has no refractory median to report. Those are exactly the cells a
# stability experiment is about, so requiring finiteness here would refuse the
# completed matrix. They are counted and reported instead, and their endpoint
# statistic is nan-aware.
NULLABLE_ENDPOINTS = ("refractory_violation_median",)
NUMERIC_ENDPOINTS = FINITE_ENDPOINTS + NULLABLE_ENDPOINTS
REQUIRED_COLUMNS = ("template", "realisation", "candidate", "n_events",
                    "truth_sha256") + NUMERIC_ENDPOINTS


def validate_cells(cells: pd.DataFrame) -> dict:
    """Refuse anything but the complete, unique, prespecified design.

    Every check here is *required*, never conditional on a column happening to
    be present: an absent column used to sail through and then surface as a
    silent NaN inside an endpoint statistic.
    """
    missing = set(REQUIRED_COLUMNS) - set(cells.columns)
    if missing:
        raise ValueError(f"cell matrix is missing columns {sorted(missing)}")
    duplicated = cells.duplicated(["template", "realisation", "candidate"]).sum()
    if duplicated:
        raise ValueError(f"{duplicated} duplicate donor/realisation/config cells")
    counts = {
        "cells": len(cells),
        "donors": cells.template.nunique(),
        "realisations": cells.realisation.nunique(),
        "configs": cells.candidate.nunique(),
    }
    expected = {"cells": EXPECTED_CELLS, "donors": EXPECTED_DONORS,
                "realisations": EXPECTED_REALISATIONS, "configs": 1 + len(CANDIDATES)}
    if counts != expected:
        raise ValueError(f"incomplete design: got {counts}, expected {expected}")

    # the right *number* of configurations is not the right configurations
    observed_configs = set(cells.candidate.unique())
    if observed_configs != {BASELINE, *CANDIDATES}:
        raise ValueError(
            f"configurations are {sorted(observed_configs)}, expected "
            f"{sorted({BASELINE, *CANDIDATES})}"
        )
    if set(cells.n_events.unique()) != {EXPECTED_EVENTS}:
        raise ValueError(
            f"every cell must hold {EXPECTED_EVENTS} events, got "
            f"{sorted(cells.n_events.unique())}"
        )
    # a NaN accuracy is not a success: `NaN < FAILURE_THRESHOLD` is False, so an
    # unchecked null would silently count as a passing cell in classify().
    for col in FINITE_ENDPOINTS:
        values = pd.to_numeric(cells[col], errors="coerce")
        bad = ~np.isfinite(values.to_numpy(dtype=float))
        if bad.any():
            raise ValueError(
                f"{int(bad.sum())} cells hold a null or non-finite {col}"
            )
    per_pair = cells.groupby(["template", "realisation"]).candidate.nunique()
    if not (per_pair == 1 + len(CANDIDATES)).all():
        raise ValueError("some donor/realisation pairs lack a complete config triplet")
    # identical labels do not prove the three configs saw the identical train.
    # nunique() defaults to dropna=True, so a null hash beside two matching ones
    # counted as agreement; reject nulls and malformed digests before grouping.
    digests = cells.truth_sha256.astype("string")
    blank = digests.isna() | digests.str.strip().eq("")
    if blank.any():
        raise ValueError(f"{int(blank.sum())} cells record no truth hash")
    malformed = ~digests.str.fullmatch(r"[0-9a-f]{12}|[0-9a-f]{64}")
    if malformed.any():
        raise ValueError(
            f"{int(malformed.sum())} cells record a malformed truth hash, e.g. "
            f"{digests[malformed].iloc[0]!r}"
        )
    hashes = cells.groupby(["template", "realisation"]).truth_sha256.nunique(
        dropna=False)
    if not (hashes == 1).all():
        disagree = sorted(hashes[hashes != 1].index)
        raise ValueError(
            f"{len(disagree)} donor/realisation pairs are not hash-paired across "
            f"configurations: {disagree[:5]}"
        )
    counts["truth_hashes"] = int(cells.truth_sha256.nunique())
    # undefined guardrails are permitted, but never silent -- and "undefined"
    # means null, not anything unparseable. The scorer builds this endpoint from
    # finite boolean means and reports NaN only when no good unit has two spikes,
    # so an infinity or a string is corruption wearing missingness as a disguise.
    for col in NULLABLE_ENDPOINTS:
        raw = cells[col]
        null = raw.isna()
        parsed = pd.to_numeric(raw, errors="coerce")
        unparseable = parsed.isna() & ~null
        if unparseable.any():
            raise ValueError(
                f"{int(unparseable.sum())} cells hold a non-numeric {col}, e.g. "
                f"{raw[unparseable].iloc[0]!r}; a null is undefined, this is not"
            )
        values = parsed.to_numpy(dtype=float)
        infinite = np.isinf(values)
        if infinite.any():
            raise ValueError(
                f"{int(infinite.sum())} cells hold an infinite {col}; the scorer "
                "cannot produce one"
            )
        counts[f"undefined_{col}"] = int(np.isnan(values).sum())
    return counts


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def classify(cells: pd.DataFrame, threshold: float = FAILURE_THRESHOLD,
             systematic_min: int = SYSTEMATIC_MIN_FAILURES) -> pd.DataFrame:
    out = cells.copy()
    out["failed"] = out.accuracy < threshold
    flags = out.groupby(["template", "candidate"]).failed.agg(
        n_failures="sum", n_realisations="size").reset_index()
    flags["systematic"] = flags.n_failures >= systematic_min
    return out.merge(flags, on=["template", "candidate"], how="left")


def donor_failure_counts(tagged: pd.DataFrame) -> pd.DataFrame:
    """Failures out of 14 for every donor x configuration.

    [rev5] Reported in full so the 12/14 systematic flag can be audited against
    the distribution it was thresholded from, rather than standing alone.
    """
    return (tagged.groupby(["template", "candidate"]).failed.sum()
            .unstack("candidate").astype(int).sort_index())


def worst_donor_deterioration(tagged: pd.DataFrame, baseline: str,
                              candidate: str) -> dict:
    """The donor on which the candidate fails most often relative to production."""
    counts = donor_failure_counts(tagged)
    delta = (counts[candidate] - counts[baseline]).sort_values(ascending=False)
    donor = delta.index[0]
    return {"donor": str(donor), "extra_failures": int(delta.iloc[0]),
            "candidate_failures": int(counts.loc[donor, candidate]),
            "baseline_failures": int(counts.loc[donor, baseline]),
            "cap": DONOR_DETERIORATION_CAP,
            "exceeds_cap": bool(delta.iloc[0] >= DONOR_DETERIORATION_CAP)}


def eligible_donors(tagged: pd.DataFrame, baseline: str, candidate: str) -> list:
    """Donors systematic under NEITHER arm of this contrast.

    [rev3] The union equalises the two arms' denominators, but it does NOT make
    the exclusion harmless: membership is itself an outcome, so crossing a donor
    from 11 to 12 failures drops it from both arms and moves the reported
    contrast (verified: +0.056 -> 0.000). An earlier docstring here claimed the
    union "stops a configuration improving its own rate by failing a donor one
    more time"; that claim was false and is withdrawn.

    This population therefore drives only the labelled sporadic-only
    *sensitivity* analysis. The primary contrast runs on all frozen donors,
    where a donor that fails under both arms contributes equally to both and so
    cancels from the paired difference.
    """
    pair = tagged[tagged.candidate.isin([baseline, candidate])]
    systematic = set(pair.loc[pair.systematic, "template"])
    return sorted(set(pair.template) - systematic)


# --------------------------------------------------------------------------- #
# paired donor bootstrap
#
# Resampling works on per-donor numpy column blocks rather than DataFrames: the
# bootstrap draws thousands of times per endpoint, and pandas concatenation at
# that rate dominates the runtime.
# --------------------------------------------------------------------------- #
CELL_COLUMNS = ("failed", "accuracy", "fp", "n_output_units_capturing",
                "refractory_violation_median")


def _blocks(frame, config: str, donors: list) -> dict:
    """donor -> {column: np.ndarray} for one configuration, realisation-ordered."""
    out = {}
    subset = frame[frame.candidate == config]
    for donor, group in subset.groupby("template"):
        if donor not in donors:
            continue
        group = group.sort_values("realisation")
        out[donor] = {col: group[col].to_numpy(dtype=float)
                      for col in CELL_COLUMNS if col in group}
    return out


def _pool(blocks: dict, selection: list) -> dict:
    return {col: np.concatenate([blocks[d][col] for d in selection])
            for col in next(iter(blocks.values()))}


def _draws(n: int, n_donors: int, seed: int):
    """One shared resampling plan, so every endpoint uses the same donor draws."""
    return np.random.default_rng(seed).integers(0, n_donors, size=(n, n_donors))



# A resample can legitimately contain no defined cell for a nullable endpoint,
# making that draw's statistic NaN. rev6 dropped those draws and took percentiles
# over the survivors, which fixed the [nan, nan] collapse but bought it with
# something worse: a percentile over surviving draws is conditional on the
# statistic being defined, and with one defined donor out of 14 it reported
# ci=[0.0, 0.0] with undefined_interval=False -- a zero-width interval presented
# as an ordinary CI, from a single independent unit. A loud NaN was replaced by
# quiet false precision.
#
# [rev7] Support is therefore judged on the data, not on how many resamples
# happened to succeed. The draw-survival fraction measures the probability of
# drawing at least one defined donor, which is not donor support: a single
# defined donor survives about 65% of 14-donor resamples and so passed the old
# 0.5 rule. The gate is now the number of DISTINCT donors contributing a defined
# observation. Below it there is no donor-generalising CI at all -- the point
# estimate and the missingness are reported instead.
MIN_DEFINED_DONORS = 10          # of 14; a frozen judgement call, not fitted


def _defined_donors(statistic, donors: list, *blocks) -> int:
    """Donors for which the statistic is defined in every arm supplied."""
    n = 0
    for d in donors:
        values = [statistic(_pool(b, [d])) for b in blocks]
        if all(np.isfinite(v) for v in values):
            n += 1
    return n


def _draw_interval(draws: np.ndarray, alpha: float, n_defined_donors: int,
                   n_donors: int) -> dict:
    """Percentile interval, refused outright when donor support is inadequate."""
    finite = draws[np.isfinite(draws)]
    provenance = {"n_draws": int(draws.size), "n_usable_draws": int(finite.size),
                  "n_defined_donors": int(n_defined_donors),
                  "min_defined_donors": MIN_DEFINED_DONORS}
    if n_defined_donors < MIN_DEFINED_DONORS or finite.size < 2:
        return {"ci": [float("nan")] * 2, "excludes_zero": False,
                "undefined_interval": True,
                "interval_refused_reason": (
                    f"only {n_defined_donors} of {n_donors} donors contribute a "
                    f"defined value; below the prespecified minimum of "
                    f"{MIN_DEFINED_DONORS}, no donor-generalising CI is quoted"),
                **provenance}
    lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"ci": [round(float(lo), 5), round(float(hi), 5)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "undefined_interval": False,
            # honest about what the interval is when any draw was dropped
            "conditional_on_definedness": bool(finite.size < draws.size),
            **provenance}


def paired_donor_bootstrap(tagged, baseline: str, candidate: str, statistic,
                           donors: list, alpha: float = PER_COMPARISON_ALPHA,
                           n: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict:
    """CI for `statistic(candidate) - statistic(baseline)`, resampling donors.

    Donors are the independent unit; all of a resampled donor's realisations
    travel together, so within-donor dependence is preserved. Both arms are
    computed on the *same* resampled donors, which is what makes it paired.
    """
    if not donors:
        return {"point": float("nan"), "ci": [float("nan")] * 2, "n_donors": 0,
                "excludes_zero": False}
    base_blocks = _blocks(tagged, baseline, donors)
    cand_blocks = _blocks(tagged, candidate, donors)
    point = statistic(_pool(cand_blocks, donors)) - statistic(_pool(base_blocks, donors))
    draws = np.empty(n)
    for i, picks in enumerate(_draws(n, len(donors), seed)):
        selection = [donors[j] for j in picks]
        draws[i] = (statistic(_pool(cand_blocks, selection))
                    - statistic(_pool(base_blocks, selection)))
    defined = _defined_donors(statistic, donors, cand_blocks, base_blocks)
    return {"point": round(float(point), 5),
            "ci_level": round(1 - alpha, 4), "n_donors": len(donors),
            **_draw_interval(draws, alpha, defined, len(donors))}


def marginal_bootstrap(tagged, config: str, statistic, donors: list,
                       alpha: float = PER_COMPARISON_ALPHA, n: int = N_BOOTSTRAP,
                       seed: int = BOOTSTRAP_SEED) -> dict:
    if not donors:
        return {"point": float("nan"), "ci": [float("nan")] * 2}
    blocks = _blocks(tagged, config, donors)
    point = statistic(_pool(blocks, donors))
    draws = np.empty(n)
    for i, picks in enumerate(_draws(n, len(donors), seed)):
        draws[i] = statistic(_pool(blocks, [donors[j] for j in picks]))
    interval = _draw_interval(draws, alpha,
                              _defined_donors(statistic, donors, blocks),
                              len(donors))
    return {"point": round(float(point), 5), "ci": interval["ci"],
            "n_usable_draws": interval["n_usable_draws"],
            "n_defined_donors": interval["n_defined_donors"],
            "undefined_interval": interval["undefined_interval"]}


# --------------------------------------------------------------------------- #
# the endpoint statistics, each a function of a cell frame
# --------------------------------------------------------------------------- #
STATISTICS = {
    "failure_rate": lambda a: float(a["failed"].mean()),
    "median_accuracy": lambda a: float(np.median(a["accuracy"])),
    "p10_accuracy": lambda a: float(np.percentile(a["accuracy"], 10)),
    "fp_p90": lambda a: float(np.percentile(a["fp"], 90)),
    "fp_max": lambda a: float(a["fp"].max()),
    "split_rate": lambda a: float((a["n_output_units_capturing"] > 1).mean()),
    # nan-aware: one cell with no scoreable good unit must not poison the
    # pooled endpoint for every donor in the resample
    "refractory_violation_median": lambda a: _nanmedian(
        a["refractory_violation_median"])
    if "refractory_violation_median" in a else float("nan"),
}
def _nanmedian(values) -> float:
    """Median over the defined cells; NaN only if every cell is undefined."""
    values = np.asarray(values, dtype=float)
    defined = values[np.isfinite(values)]
    return float(np.median(defined)) if defined.size else float("nan")


DECISION_ENDPOINTS = ("failure_rate", "fp_p90", "split_rate")

# endpoints that are means, so a donor-level t-interval is meaningful; the tail
# order statistics are deliberately excluded (see the module docstring)
SMOOTH_ENDPOINTS = ("failure_rate", "split_rate")


# [rev6] Every constant a verdict depends on, in one place, so the runner can
# freeze it into prespec.json alongside the protocol. Without this the caps lived
# only here and could be changed after data collection without the frozen
# manifest noticing.
def _analysis_source_digest() -> str:
    """SHA-256 of this module.

    [rev7] The named constants are not the whole protocol: the statistic
    implementations, the comparison semantics and the tie-break are code, and the
    schema string only protects them if someone remembers to bump it. Hashing the
    source binds them too. After the prespec is frozen, changing this file is
    exactly the thing that must stop an analysis and force an explicit re-freeze.
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def decision_protocol() -> dict:
    """The protocol as it is RIGHT NOW, not as it was at import.

    [rev7] This was a module-level dict built once at import. A constant changed
    afterwards -- by a monkeypatch, a REPL, a wrapper script -- diverged from the
    snapshot silently: decide_contrast() read the live global while the frozen
    comparison still saw the old value and passed. Reading the constants at call
    time is what makes the freeze bind the rules actually in force.
    """
    return {
        "analysis_schema": ANALYSIS_SCHEMA,
        "analysis_source_sha256": _analysis_source_digest(),
        "failure_threshold": FAILURE_THRESHOLD,
        "systematic_min_failures": SYSTEMATIC_MIN_FAILURES,
        "absolute_failure_cap": ABSOLUTE_FAILURE_CAP,
        "donor_deterioration_cap": DONOR_DETERIORATION_CAP,
        "baseline": BASELINE,
        "candidates": list(CANDIDATES),
        "decision_endpoints": list(DECISION_ENDPOINTS),
        "smooth_endpoints": list(SMOOTH_ENDPOINTS),
        "family_alpha": FAMILY_ALPHA,
        "per_comparison_alpha": PER_COMPARISON_ALPHA,
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "min_defined_donors": MIN_DEFINED_DONORS,
        "tie_break": ["failure_rate", "fp_p90", "split_rate"],
    }


def donor_paired_t(tagged, baseline: str, candidate: str, statistic,
                   donors: list, alpha: float = PER_COMPARISON_ALPHA) -> dict:
    """Student-t interval on the 14 donor-level paired differences.

    A prespecified sensitivity check on the percentile bootstrap for the smooth
    endpoints only. Reported, never decisive.
    """
    base_blocks = _blocks(tagged, baseline, donors)
    cand_blocks = _blocks(tagged, candidate, donors)
    per_donor = np.array([statistic(_pool(cand_blocks, [d]))
                          - statistic(_pool(base_blocks, [d])) for d in donors])
    n = len(per_donor)
    mean = float(per_donor.mean())
    if n < 2 or not np.isfinite(per_donor).all():
        return {"point": mean, "ci": [float("nan")] * 2, "n_donors": n}
    se = float(per_donor.std(ddof=1) / np.sqrt(n))
    crit = float(student_t.ppf(1 - alpha / 2, df=n - 1))
    return {"point": round(mean, 5),
            "ci": [round(mean - crit * se, 5), round(mean + crit * se, 5)],
            "ci_level": round(1 - alpha, 4), "n_donors": n,
            "excludes_zero": bool(abs(mean) > crit * se)}


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar on discordant pairs. Descriptive only.

    Reported for continuity with the prespec, but it assumes the 196 pairs are
    independent draws, which they are not — they cluster within 14 donors. No
    decision uses it; the donor bootstrap does.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))


def contrast(tagged: pd.DataFrame, baseline: str, candidate: str,
             n_bootstrap: int = N_BOOTSTRAP) -> dict:
    """Every prespecified endpoint for one contrast.

    [rev3] The primary population is *all* frozen donors. Outcome-dependent
    exclusion is confined to a labelled sensitivity analysis; see
    `eligible_donors`. Keeping the primary population fixed also makes the two
    candidates rankable against each other, which the per-contrast union set did
    not — each candidate could otherwise be scored on a different donor set.
    """
    donors = sorted(set(tagged.template))
    pair = tagged
    sporadic_only = eligible_donors(tagged, baseline, candidate)
    excluded = sorted(set(donors) - set(sporadic_only))

    differences = {
        name: paired_donor_bootstrap(pair, baseline, candidate, stat, donors,
                                     n=n_bootstrap)
        for name, stat in STATISTICS.items()
    }
    marginals = {
        config: {name: marginal_bootstrap(pair, config, stat, donors,
                                          n=n_bootstrap)
                 for name, stat in STATISTICS.items()}
        for config in (baseline, candidate)
    }
    sensitivity = {
        name: paired_donor_bootstrap(pair[pair.template.isin(sporadic_only)],
                                     baseline, candidate, stat, sporadic_only,
                                     n=n_bootstrap)
        for name, stat in STATISTICS.items()
    }
    wide = pair.pivot_table(index=["template", "realisation"], columns="candidate",
                            values="failed", aggfunc="first")
    b = int(((wide[baseline] == 1) & (wide[candidate] == 0)).sum())
    c = int(((wide[baseline] == 0) & (wide[candidate] == 1)).sum())
    counts = donor_failure_counts(tagged)
    return {
        "baseline": baseline, "candidate": candidate,
        "primary_donors": donors, "n_primary_donors": len(donors),
        # [rev5] the distribution the 12/14 flag is thresholded from, in full
        "donor_failure_counts": {
            config: {str(d): int(counts.loc[d, config]) for d in counts.index}
            for config in (baseline, candidate)
        },
        "worst_donor_deterioration": worst_donor_deterioration(
            tagged, baseline, candidate),
        # [rev5] the candidate on its own terms: a relative win over a bad
        # baseline is still unusable if the candidate is itself bad
        "absolute_failure_rate": {
            **marginals[candidate]["failure_rate"],
            "cap": ABSOLUTE_FAILURE_CAP,
        },
        "sporadic_only_donors": sporadic_only,
        "excluded_systematic_donors": excluded,
        "systematic_by_config": {
            config: sorted(set(tagged.loc[(tagged.candidate == config)
                                          & tagged.systematic, "template"]))
            for config in (baseline, candidate)
        },
        "differences": differences,
        "marginals": marginals,
        "paired_difference_t_sensitivity": {
            name: donor_paired_t(pair, baseline, candidate, STATISTICS[name], donors)
            for name in SMOOTH_ENDPOINTS
        },
        "sporadic_only_sensitivity": {
            "note": "outcome-dependent population; reported, never decisive",
            "n_donors": len(sporadic_only),
            "differences": sensitivity,
        },
        "unadjusted_mcnemar": {
            "baseline_fails_candidate_ok": b, "candidate_fails_baseline_ok": c,
            "p_value": round(exact_mcnemar(b, c), 6),
            "caveat": "assumes independent pairs; clustered within donors; not used for decisions",
        },
    }


# --------------------------------------------------------------------------- #
# decision rules
# --------------------------------------------------------------------------- #
def _side(entry: dict, name: str) -> tuple[bool, bool]:
    """(improved, regressed) read off the CI itself.

    [rev3] Direction used to come from the sign of `point` gated on an
    `excludes_zero` flag. Those are three facts that can disagree, and a result
    carrying point=-0.05 with ci=[0.01, 0.10] was accepted as an improvement.
    The interval is the authority; a point outside its own CI is incoherent
    input and raises rather than silently picking one of them.
    """
    point, (lo, hi) = entry["point"], entry["ci"]
    if np.isfinite(lo) and np.isfinite(hi):
        if not lo <= point <= hi:
            raise ValueError(
                f"{name}: point {point} lies outside its own CI [{lo}, {hi}]"
            )
        return hi < 0, lo > 0
    return False, False


def decide_contrast(result: dict) -> dict:
    """Rules 1-3, each endpoint judged on its own paired CI."""
    diff = result["differences"]
    better, worse_failure = _side(diff["failure_rate"], "failure_rate")
    _, worse_fp = _side(diff["fp_p90"], "fp_p90")
    _, worse_splits = _side(diff["split_rate"], "split_rate")
    new_systematic = sorted(
        set(result["systematic_by_config"][result["candidate"]])
        - set(result["systematic_by_config"][result["baseline"]]))

    # [rev5] the 12/14 systematic flag is a step, so on its own it both misses a
    # candidate failing 11/14 everywhere and disqualifies one crossing to 12/14
    # on a single donor. These two rules close it from either side. Both can only
    # disqualify, so neither can turn a losing candidate into a winner.
    # [rev6] required-key access. These were read with .get(), so a contrast
    # dict missing them silently passed both guardrails and still reported
    # "qualifies" with absolute_failure_rate_acceptable=True -- fail-open on the
    # public decision function, and a false reason string. A verdict is now
    # refused rather than guessed when the evidence for it is absent.
    for key in ("absolute_failure_rate", "worst_donor_deterioration"):
        if key not in result:
            raise ValueError(
                f"contrast result is missing {key}; the rev5 guardrails cannot "
                "be evaluated and a verdict will not be issued without them"
            )
    absolute = result["absolute_failure_rate"]
    if not np.isfinite([absolute["point"], *absolute["ci"]]).all():
        raise ValueError(
            "absolute failure-rate estimate or CI is undefined; acceptability "
            f"cannot be established (point {absolute['point']}, "
            f"ci {absolute['ci']})"
        )
    upper = absolute["ci"][1]
    too_bad_outright = bool(upper >= ABSOLUTE_FAILURE_CAP)
    worst = result["worst_donor_deterioration"]
    donor_regression = bool(worst["exceeds_cap"])

    regressions = [name for name, flag in
                   (("fp_p90", worse_fp), ("split_rate", worse_splits),
                    ("failure_rate", worse_failure)) if flag]
    if regressions or new_systematic or donor_regression or too_bad_outright:
        causes = list(regressions)
        if new_systematic:
            causes.append(f"new systematic donors {new_systematic}")
        if donor_regression:
            causes.append(
                f"donor {worst['donor']} fails {worst['extra_failures']} more "
                f"times than production (cap {DONOR_DETERIORATION_CAP})")
        if too_bad_outright:
            causes.append(
                f"absolute failure rate is not bounded below "
                f"{ABSOLUTE_FAILURE_CAP} (upper CI {absolute['ci'][1]})")
        verdict, reason = "dropped", "; ".join(causes)
    elif better:
        verdict, reason = "qualifies", (
            "lower paired failure rate over all frozen donors with a CI "
            "excluding zero; no FP, split, per-donor or systematic regression, "
            "and an acceptable absolute failure rate")
    else:
        # a baseline advantage is a failure-rate regression, so it is already
        # handled above; the only way here is a CI that spans zero
        verdict, reason = "not_separated", (
            "no paired difference in failure rate whose CI excludes zero")
    return {
        "verdict": verdict, "reason": reason,
        "failure_rate_difference": diff["failure_rate"],
        "regressions": regressions, "new_systematic_donors": new_systematic,
        "donor_deterioration": worst or None,
        "absolute_failure_rate_acceptable": not too_bad_outright,
    }


def decide(contrasts: dict) -> dict:
    """Rule 1-3 across candidates, with a prespecified tie-break.

    A regression disqualifies a candidate even when its failure rate is better —
    'better' never overrides 'dropped'.
    """
    verdicts = {name: decide_contrast(result) for name, result in contrasts.items()}
    qualifying = [n for n, v in verdicts.items() if v["verdict"] == "qualifies"]
    if not qualifying:
        outcome, selected = "no_threshold_change", None
    elif len(qualifying) == 1:
        outcome, selected = "candidate_replaces_baseline", qualifying[0]
    else:
        # deterministic, prespecified ordering: lower failure rate, then lower
        # FP p90, then lower split rate; a genuine tie is escalated, not guessed
        def key(name):
            d = contrasts[name]["differences"]
            return (d["failure_rate"]["point"], d["fp_p90"]["point"],
                    d["split_rate"]["point"])
        ranked = sorted(qualifying, key=key)
        if key(ranked[0]) == key(ranked[1]):
            outcome, selected = "multiple_candidates_tied_escalate", None
        else:
            outcome, selected = "candidate_replaces_baseline", ranked[0]
    return {
        "per_candidate": verdicts, "qualifying": qualifying,
        "outcome": outcome, "selected": selected,
        "tie_break": "failure_rate, then fp_p90, then split_rate",
        "familywise_alpha": FAMILY_ALPHA,
        "per_comparison_alpha": PER_COMPARISON_ALPHA,
    }


def assert_frozen_protocol(root: Path) -> dict:
    """Refuse to analyse a dataset under rules other than the ones it was run under.

    [rev7] The runner froze decision_protocol() into prespec.json at collection,
    but analyse() read only stage2.csv, so nothing compared them: a completed
    dataset could be analysed under changed constants -- or changed analysis
    code -- without any refusal. Freezing a protocol that is never checked is
    documentation, not a control.
    """
    path = root / "prespec.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; a stage-2 dataset is analysed only under the "
            "protocol frozen with it, and that protocol cannot be recovered"
        )
    stored = json.loads(path.read_text()).get("decision_protocol")
    if stored is None:
        raise ValueError(
            f"{path} records no decision_protocol; it predates the frozen "
            "protocol and cannot be analysed under these rules"
        )
    if stored != decision_protocol():
        differing = sorted(
            k for k in set(stored) | set(decision_protocol())
            if stored.get(k) != decision_protocol().get(k))
        detail = {k: {"frozen": stored.get(k), "current": decision_protocol().get(k)}
                  for k in differing if k != "analysis_source_sha256"}
        raise ValueError(
            f"the decision protocol has changed since collection: {differing}. "
            f"{detail or 'the analysis source itself differs'}. Re-freezing is a "
            "deliberate act, not a side effect of editing."
        )
    return stored


def analyse(root: Path | str = DEFAULT_ROOT) -> dict:
    root = Path(root)
    frozen = assert_frozen_protocol(root)
    cells = pd.read_csv(root / "stage2.csv")
    counts = validate_cells(cells)
    tagged = classify(cells)
    contrasts = {c: contrast(tagged, BASELINE, c) for c in CANDIDATES}
    result = {
        "schema": ANALYSIS_SCHEMA,
        "design": counts,
        "failure_threshold": FAILURE_THRESHOLD,
        "systematic_min_failures": SYSTEMATIC_MIN_FAILURES,
        "baseline": BASELINE,
        "inference": (
            "paired donor bootstrap; conditional on the 14 frozen realisations, "
            "generalising across donors only"
        ),
        "primary_population": (
            "all frozen donors; systematic status is a separate disqualifying "
            "guardrail, and the union-excluded population is reported only as a "
            "labelled sensitivity analysis"
        ),
        "undefined_guardrail_cells": {
            col: counts.get(f"undefined_{col}", 0) for col in NULLABLE_ENDPOINTS
        },
        "interval_caveat": (
            "percentile bootstrap over 14 clusters; tail endpoints are ties-heavy "
            "order statistics with discrete, possibly zero-width intervals. An "
            "endpoint undefined for some cells is quoted only when at least "
            f"{MIN_DEFINED_DONORS} of 14 donors contribute a defined value; where "
            "any resample was dropped the interval is flagged "
            "conditional_on_definedness, meaning it is conditional on the "
            "statistic existing rather than an ordinary unconditional bootstrap"
        ),
        "decision_endpoints": list(DECISION_ENDPOINTS),
        "frozen_protocol_verified": frozen,
        "disqualifying_guardrails": {
            "systematic_min_failures": SYSTEMATIC_MIN_FAILURES,
            "absolute_failure_cap": ABSOLUTE_FAILURE_CAP,
            "donor_deterioration_cap": DONOR_DETERIORATION_CAP,
            "note": ("[rev5] all three can only disqualify a candidate, never "
                     "promote one; the per-donor distribution is reported in "
                     "full under each contrast's donor_failure_counts"),
        },
        "contrasts": contrasts,
        "decision": decide(contrasts),
    }
    _atomic_write(root / "analysis_cells.csv", tagged.to_csv(index=False))
    _atomic_write(root / "analysis.json",
                  json.dumps(result, indent=2, default=str) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()
    print(json.dumps(analyse(args.root), indent=2, default=str))


if __name__ == "__main__":
    main()
