"""Unblind and score a completed Luke multichannel event review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


ALLOWED_LABELS = {"neural", "artifact", "uncertain"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=Path("testing/outputs/luke_multichannel_event_validation/imec1"),
    )
    return parser.parse_args()


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half = z * np.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return float(center - half), float(center + half)


def score_review(labels: pd.DataFrame, key: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required_labels = {"review_id", "review_label"}
    required_key = {"review_id", "status"}
    if not required_labels.issubset(labels.columns):
        raise ValueError(f"Labels require columns {sorted(required_labels)}")
    if not required_key.issubset(key.columns):
        raise ValueError(f"Key requires columns {sorted(required_key)}")
    if labels["review_id"].duplicated().any() or key["review_id"].duplicated().any():
        raise ValueError("review_id must be unique in both files")

    clean = labels.copy()
    clean["review_label"] = clean["review_label"].fillna("").str.strip().str.lower()
    missing = clean["review_label"] == ""
    if missing.any():
        ids = clean.loc[missing, "review_id"].head(8).tolist()
        raise ValueError(f"{missing.sum()} events are unlabeled; examples: {ids}")
    invalid = ~clean["review_label"].isin(ALLOWED_LABELS)
    if invalid.any():
        values = sorted(clean.loc[invalid, "review_label"].unique())
        raise ValueError(f"Invalid labels {values}; use {sorted(ALLOWED_LABELS)}")

    merged = key.merge(clean, on="review_id", how="inner", validate="one_to_one")
    if len(merged) != len(key) or len(merged) != len(clean):
        raise ValueError("Label and key review_id sets do not match")

    rows = []
    for status, group in merged.groupby("status", sort=True):
        counts = group["review_label"].value_counts()
        neural = int(counts.get("neural", 0))
        definite = int(counts.get("neural", 0) + counts.get("artifact", 0))
        low, high = wilson_interval(neural, definite)
        rows.append(
            {
                "status": status,
                "n_reviewed": len(group),
                "n_neural": neural,
                "n_artifact": int(counts.get("artifact", 0)),
                "n_uncertain": int(counts.get("uncertain", 0)),
                "neural_fraction_excluding_uncertain": neural / definite
                if definite
                else np.nan,
                "neural_fraction_ci95_low": low,
                "neural_fraction_ci95_high": high,
            }
        )
    summary = pd.DataFrame(rows)
    contingency = pd.crosstab(merged["status"], merged["review_label"])
    for column in ("neural", "artifact"):
        if column not in contingency:
            contingency[column] = 0
    contingency = contingency.reindex(["unmatched", "matched"], fill_value=0)
    odds_ratio, p_value = fisher_exact(
        contingency[["neural", "artifact"]].to_numpy()
    )
    unmatched = summary.loc[summary["status"] == "unmatched"]
    unmatched_fraction = (
        float(unmatched.iloc[0]["neural_fraction_excluding_uncertain"])
        if len(unmatched)
        else float("nan")
    )
    result = {
        "primary_endpoint": "manual neural fraction among unmatched candidates, excluding uncertain labels",
        "unmatched_neural_fraction": unmatched_fraction,
        "artifact_hypothesis_screen": {
            "criterion": "unmatched neural fraction < 0.20",
            "passes": bool(unmatched_fraction < 0.20)
            if np.isfinite(unmatched_fraction)
            else None,
        },
        "matched_vs_unmatched_fisher_exact": {
            "table_order": ["unmatched", "matched"],
            "column_order": ["neural", "artifact"],
            "odds_ratio": float(odds_ratio),
            "two_sided_p_value": float(p_value),
        },
        "caveat": "Coincidence-defined matched events are a positive-control population, not perfect neural ground truth.",
    }
    return summary, result


def main() -> None:
    args = parse_args()
    labels = pd.read_csv(args.review_dir / "review_labels.csv")
    key = pd.read_csv(args.review_dir / "review_key.csv")
    summary, result = score_review(labels, key)
    summary.to_csv(args.review_dir / "manual_review_summary.csv", index=False)
    (args.review_dir / "manual_review_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(summary.to_string(index=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
