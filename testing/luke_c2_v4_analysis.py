"""C2 v4 analysis — from the frozen cell matrix to the prespecified endpoints.

`luke_rescue_c2_drift_challenge_v4.py` emits cells; it deliberately computes no
conclusions. This module is the analysis half, kept separate so the endpoints can
be recomputed and audited without re-sorting.

What it computes, all of it prespecified in the v4 PRESPEC
---------------------------------------------------------
* **Static qualification, contrast-specific and per condition.** A donor enters a
  contrast if it clears `accuracy_min` under *both* configs of that contrast, in
  that condition. Per-condition matters because each condition sorts its own
  static baseline: the recordings differ by name, so they get separate cache
  leaves and separate sorts, and KS4 is not bit-deterministic (observed spread
  <= 0.005). Reading qualification off one representative condition would be an
  assumption, not a measurement.
* **Motion penalty**, `moved - static`, within a condition and within a config.
* **The correction effect as an interaction**, never a single difference:
  `(moved_rigid - static_rigid) - (moved_rescue - static_rescue)`. Rigid
  correction changes clustering with no motion present, so the moving-arm
  difference alone confounds motion recovery with a different sort.
* **The registration reference**, `moved_corrected - static`. Not a ceiling: the
  exact inverse minimises positional error, not amplitude or accuracy.
* **Fragmentation**, output units capturing the train.
* **Three cohorts**: common primary (qualified in every condition), all-donor,
  and an operator-qualified sensitivity subset. Per-arm exclusion is forbidden --
  it would let each magnitude run on a different, progressively easier cohort.

Ramp conditions are forward-model confounded and are labelled as such in every
table. The staircase is a machinery positive control and is reported separately.

Run: `python testing/luke_c2_v4_analysis.py [--root <v4 output root>]`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "testing/outputs/luke_rescue_c2_drift_challenge_v4"
CALIBRATION = REPO_ROOT / "testing/outputs/luke_c2_operator_calibration/offset_sweep.csv"

ANALYSIS_SCHEMA = "luke-c2-v4-analysis-v1"
ACCURACY_MIN = 0.8
SNR_FLOOR = 3.0

CONTRASTS = {
    "rescue_vs_rescue_rigid": ("rescue", "rescue_rigid"),   # primary
    "rescue_vs_legacy_style": ("rescue", "legacy_style"),    # operational only
}


def load_cells(root: Path | str = DEFAULT_ROOT) -> pd.DataFrame:
    cells = pd.read_csv(Path(root) / "c2_v4.csv")
    required = {"template", "condition", "arm", "sorter", "accuracy", "n_truth"}
    missing = required - set(cells.columns)
    if missing:
        raise ValueError(f"cell matrix is missing {sorted(missing)}")
    return cells


def check_denominators(cells: pd.DataFrame) -> dict:
    """Every arm of a condition must share one truth train. Fails closed."""
    bad = (
        cells.groupby(["template", "condition"])
        .agg(n_truth=("n_truth", "nunique"), n_hash=("truth_sha256", "nunique"))
        .query("n_truth > 1 or n_hash > 1")
    )
    if not bad.empty:
        raise ValueError(
            "arms within a condition do not share one truth train:\n"
            f"{bad.to_string()}"
        )
    return {
        "condition_groups": int(cells.groupby(["template", "condition"]).ngroups),
        "n_truth_by_condition": {
            str(c): sorted(map(int, g.n_truth.unique()))
            for c, g in cells.groupby("condition")
        },
    }


def qualify(cells: pd.DataFrame, accuracy_min: float = ACCURACY_MIN) -> pd.DataFrame:
    """Contrast-specific static qualification, per condition."""
    static = cells[cells.arm == "static"].pivot_table(
        index=["template", "condition"], columns="sorter", values="accuracy"
    )
    rows = []
    for (template, condition), row in static.iterrows():
        for name, (a, b) in CONTRASTS.items():
            rows.append({
                "template": template, "condition": condition, "contrast": name,
                f"static_{a}": row.get(a), f"static_{b}": row.get(b),
                "qualified": bool(row.get(a, 0) >= accuracy_min
                                  and row.get(b, 0) >= accuracy_min),
            })
    return pd.DataFrame(rows)


def cohorts(qualified: pd.DataFrame, contrast: str) -> dict:
    """Common-primary, all-donor and (if available) operator-qualified subsets."""
    sub = qualified[qualified.contrast == contrast]
    by_condition = {
        str(c): sorted(g.loc[g.qualified, "template"])
        for c, g in sub.groupby("condition")
    }
    common = sorted(set.intersection(*(set(v) for v in by_condition.values()))) \
        if by_condition else []
    return {
        "contrast": contrast,
        "common_primary": common,
        "n_common_primary": len(common),
        "qualified_by_condition": by_condition,
        "all_donor": sorted(sub.template.unique()),
        "excluded_from_common": sorted(set(sub.template.unique()) - set(common)),
    }


def operator_qualified(magnitude_um: float, snr_floor: float = SNR_FLOOR) -> dict:
    """Donors whose ramp-mean retained SNR clears the floor (sensitivity only)."""
    if not CALIBRATION.exists():
        return {"available": False}
    sweep = pd.read_csv(CALIBRATION)
    inside = sweep[sweep.offset_um <= magnitude_um]
    mean_snr = inside.groupby("template").snr_after.mean()
    return {
        "available": True, "magnitude_um": magnitude_um, "rule": "ramp-mean",
        "snr_floor": snr_floor,
        "qualified": sorted(mean_snr.index[mean_snr >= snr_floor]),
        "below_floor": sorted(mean_snr.index[mean_snr < snr_floor]),
    }


def penalties(cells: pd.DataFrame) -> pd.DataFrame:
    """Motion penalty per donor/condition/sorter, plus the registration delta."""
    wide = cells.pivot_table(
        index=["template", "condition"], columns=["arm", "sorter"], values="accuracy"
    )
    out = []
    for (template, condition), row in wide.iterrows():
        record = {"template": template, "condition": condition}
        for sorter in ("rescue", "rescue_rigid", "legacy_style"):
            if ("moved", sorter) in row and ("static", sorter) in row:
                record[f"penalty_{sorter}"] = row[("moved", sorter)] - row[("static", sorter)]
        if ("moved_corrected", "rescue") in row:
            record["registration_delta"] = (
                row[("moved_corrected", "rescue")] - row[("static", "rescue")]
            )
        out.append(record)
    return pd.DataFrame(out)


def interaction(penalty: pd.DataFrame, contrast: str) -> pd.DataFrame:
    """(moved_b - static_b) - (moved_a - static_a): the correction effect."""
    a, b = CONTRASTS[contrast]
    frame = penalty.copy()
    frame["interaction"] = frame[f"penalty_{b}"] - frame[f"penalty_{a}"]
    frame["contrast"] = contrast
    return frame[["template", "condition", "contrast",
                  f"penalty_{a}", f"penalty_{b}", "interaction"]]


def fragmentation(cells: pd.DataFrame) -> pd.DataFrame:
    if "n_output_units_capturing" not in cells:
        return pd.DataFrame()
    rescue = cells[cells.sorter == "rescue"]
    wide = rescue.pivot_table(index=["template", "condition"], columns="arm",
                             values="n_output_units_capturing")
    wide = wide.reset_index()
    wide["split_under_motion"] = wide.get("moved", 0) > 1
    return wide


def summarise(cells: pd.DataFrame, contrast: str = "rescue_vs_rescue_rigid") -> dict:
    qualified = qualify(cells)
    cohort = cohorts(qualified, contrast)
    penalty = penalties(cells)
    inter = interaction(penalty, contrast)
    common = set(cohort["common_primary"])
    a, b = CONTRASTS[contrast]

    by_condition = {}
    for condition, group in inter.groupby("condition"):
        primary = group[group.template.isin(common)]
        every = group
        def stats(frame):
            return {
                "n": int(len(frame)),
                f"median_penalty_{a}": round(float(frame[f"penalty_{a}"].median()), 4),
                f"median_penalty_{b}": round(float(frame[f"penalty_{b}"].median()), 4),
                "median_interaction": round(float(frame["interaction"].median()), 4),
                "interaction_positive": int((frame["interaction"] > 0).sum()),
            }
        reg = penalty[penalty.condition == condition]
        by_condition[str(condition)] = {
            "forward_model_confounded": not str(condition).startswith("staircase"),
            "role": ("machinery positive control"
                     if str(condition).startswith("staircase") else "Luke-calibrated"),
            "common_primary": stats(primary),
            "all_donor": stats(every),
            "registration_reference": {
                "median_delta": round(float(reg.registration_delta.median()), 4),
                "max_abs_delta": round(float(reg.registration_delta.abs().max()), 4),
                "within_0.01": int((reg.registration_delta.abs() <= 0.01).sum()),
                "n": int(len(reg)),
            },
        }
    frag = fragmentation(cells)
    return {
        "schema": ANALYSIS_SCHEMA,
        "accuracy_min": ACCURACY_MIN,
        "contrast": contrast,
        "correction_effect": (
            f"(moved_{b} - static_{b}) - (moved_{a} - static_{a}); not a single "
            "difference, because correction changes clustering with no motion present"
        ),
        "denominators": check_denominators(cells),
        "cohorts": cohort,
        "operator_qualified_sensitivity": {
            str(m): operator_qualified(m) for m in (5.0, 11.0, 22.0)
        },
        "by_condition": by_condition,
        "fragmentation": {
            str(c): int(g.split_under_motion.sum())
            for c, g in frag.groupby("condition")
        } if not frag.empty else {},
        "caveat": (
            "ramp conditions are forward-model confounded: the injection operator "
            "attenuates compact donors by 10-32% across these excursions, so a "
            "ramp penalty mixes motion with resampling loss. The staircase carries "
            "no such artifact but is ~2x Luke's largest displacement and "
            "discontinuous, so it bounds the mechanism, not the Luke-scale effect."
        ),
    }


def analyse(root: Path | str = DEFAULT_ROOT) -> dict:
    root = Path(root)
    cells = load_cells(root)
    result = summarise(cells)
    qualify(cells).to_csv(root / "analysis_qualification.csv", index=False)
    penalty = penalties(cells)
    penalty.to_csv(root / "analysis_penalties.csv", index=False)
    interaction(penalty, result["contrast"]).to_csv(
        root / "analysis_interaction.csv", index=False)
    (root / "analysis.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()
    result = analyse(args.root)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
