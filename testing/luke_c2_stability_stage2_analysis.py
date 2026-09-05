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

The eligible population is **common to a contrast**
---------------------------------------------------
A donor is excluded from a baseline-versus-candidate contrast if it is
systematic under **either** configuration. Conditioning each configuration on
its own systematic set — the obvious implementation — is biased in both
directions: a configuration that fails a donor 12 times has that donor removed
from its rate entirely, while one that fails 11 times keeps all 14 cells, so a
configuration can improve its reported rate *by failing once more*. The union
rule makes both arms of a contrast describe the same donors.

Every decision endpoint is a **paired difference** on that common population,
with its own donor-bootstrap CI. Endpoints are never compared through another
endpoint's p-value.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "testing/outputs/luke_c2_stability_stage2"

ANALYSIS_SCHEMA = "luke-c2-stability-stage2-analysis-v2"
FAILURE_THRESHOLD = 0.9
SYSTEMATIC_MIN_FAILURES = 12
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
def validate_cells(cells: pd.DataFrame) -> dict:
    """Refuse anything but the complete, unique, prespecified design."""
    required = {"template", "realisation", "candidate", "accuracy", "fp",
                "n_output_units_capturing"}
    missing = required - set(cells.columns)
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
    if "n_events" in cells and set(cells.n_events.unique()) != {EXPECTED_EVENTS}:
        raise ValueError(
            f"every cell must hold {EXPECTED_EVENTS} events, got "
            f"{sorted(cells.n_events.unique())}"
        )
    per_pair = cells.groupby(["template", "realisation"]).candidate.nunique()
    if not (per_pair == 1 + len(CANDIDATES)).all():
        raise ValueError("some donor/realisation pairs lack a complete config triplet")
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


def eligible_donors(tagged: pd.DataFrame, baseline: str, candidate: str) -> list:
    """Donors systematic under NEITHER arm of this contrast.

    The union rule is what stops a configuration improving its own rate by
    failing a donor one more time.
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
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": round(float(point), 5),
            "ci": [round(float(lo), 5), round(float(hi), 5)],
            "ci_level": round(1 - alpha, 4), "n_donors": len(donors),
            "excludes_zero": bool(lo > 0 or hi < 0)}


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
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": round(float(point), 5),
            "ci": [round(float(lo), 5), round(float(hi), 5)]}


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
    "refractory_violation_median": lambda a: float(
        np.median(a["refractory_violation_median"]))
    if "refractory_violation_median" in a else float("nan"),
}
DECISION_ENDPOINTS = ("failure_rate", "fp_p90", "split_rate")


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


def contrast(tagged: pd.DataFrame, baseline: str, candidate: str) -> dict:
    """Every prespecified endpoint for one contrast, on the common population."""
    donors = eligible_donors(tagged, baseline, candidate)
    pair = tagged[tagged.template.isin(donors)]
    excluded = sorted(set(tagged.template) - set(donors))

    differences = {
        name: paired_donor_bootstrap(pair, baseline, candidate, stat, donors)
        for name, stat in STATISTICS.items()
    }
    marginals = {
        config: {name: marginal_bootstrap(pair, config, stat, donors)
                 for name, stat in STATISTICS.items()}
        for config in (baseline, candidate)
    }
    wide = pair.pivot_table(index=["template", "realisation"], columns="candidate",
                            values="failed", aggfunc="first")
    b = int(((wide[baseline] == 1) & (wide[candidate] == 0)).sum())
    c = int(((wide[baseline] == 0) & (wide[candidate] == 1)).sum())
    return {
        "baseline": baseline, "candidate": candidate,
        "eligible_donors": donors, "n_eligible_donors": len(donors),
        "excluded_systematic_donors": excluded,
        "systematic_by_config": {
            config: sorted(set(tagged.loc[(tagged.candidate == config)
                                          & tagged.systematic, "template"]))
            for config in (baseline, candidate)
        },
        "differences": differences,
        "marginals": marginals,
        "unadjusted_mcnemar": {
            "baseline_fails_candidate_ok": b, "candidate_fails_baseline_ok": c,
            "p_value": round(exact_mcnemar(b, c), 6),
            "caveat": "assumes independent pairs; clustered within donors; not used for decisions",
        },
    }


# --------------------------------------------------------------------------- #
# decision rules
# --------------------------------------------------------------------------- #
def decide_contrast(result: dict) -> dict:
    """Rules 1-3, each endpoint judged on its own paired CI."""
    diff = result["differences"]
    better = diff["failure_rate"]["point"] < 0 and diff["failure_rate"]["excludes_zero"]
    worse_failure = diff["failure_rate"]["point"] > 0 and diff["failure_rate"]["excludes_zero"]
    worse_fp = diff["fp_p90"]["point"] > 0 and diff["fp_p90"]["excludes_zero"]
    worse_splits = diff["split_rate"]["point"] > 0 and diff["split_rate"]["excludes_zero"]
    new_systematic = sorted(
        set(result["systematic_by_config"][result["candidate"]])
        - set(result["systematic_by_config"][result["baseline"]]))

    regressions = [name for name, flag in
                   (("fp_p90", worse_fp), ("split_rate", worse_splits),
                    ("failure_rate", worse_failure)) if flag]
    if regressions or new_systematic:
        verdict, reason = "dropped", (
            f"material regression on {regressions or 'systematic failures'}"
            + (f"; new systematic donors {new_systematic}" if new_systematic else ""))
    elif better:
        verdict, reason = "qualifies", (
            "lower paired sporadic failure rate with a CI excluding zero, no FP, "
            "split or systematic regression")
    else:
        verdict, reason = "not_separated", (
            "no paired difference in sporadic failure rate whose CI excludes zero"
            if not worse_failure else "baseline better, but not a candidate regression")
    return {
        "verdict": verdict, "reason": reason,
        "failure_rate_difference": diff["failure_rate"],
        "regressions": regressions, "new_systematic_donors": new_systematic,
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


def analyse(root: Path | str = DEFAULT_ROOT) -> dict:
    root = Path(root)
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
        "eligibility": "donors systematic under either arm are excluded from that contrast",
        "decision_endpoints": list(DECISION_ENDPOINTS),
        "contrasts": contrasts,
        "decision": decide(contrasts),
    }
    tagged.to_csv(root / "analysis_cells.csv", index=False)
    (root / "analysis.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()
    print(json.dumps(analyse(args.root), indent=2, default=str))


if __name__ == "__main__":
    main()
