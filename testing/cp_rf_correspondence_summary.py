"""Join CP W2 RF scores through RF-independent reciprocal-best correspondences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def unique_best(table, source, target, weight):
    result = {}
    ambiguous = 0
    for unit, rows in table.groupby(source):
        maximum = rows[weight].max()
        winners = sorted(map(int, rows.loc[rows[weight] == maximum, target].unique()))
        if len(winners) == 1:
            result[int(unit)] = (winners[0], int(maximum))
        else:
            ambiguous += 1
    return result, ambiguous


def reciprocal(table, left, right, weight):
    forward, left_ties = unique_best(table, left, right, weight)
    reverse, right_ties = unique_best(table, right, left, weight)
    pairs = {
        (source, target): count
        for source, (target, count) in forward.items()
        if target in reverse and reverse[target][0] == source
    }
    return pairs, {"left_max_ties": left_ties, "right_max_ties": right_ties}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-analysis", type=Path, required=True)
    parser.add_argument("--rf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    cross = pd.read_csv(args.h5_analysis / "STATIC_D2L_TIME_ONLY_PAIRS.csv")
    cross = cross[
        (cross.window == "W2")
        & (cross.static_unit >= 0)
        & (cross.d2l_all_force_unit >= 0)
    ]
    static_force, cross_ties = reciprocal(
        cross, "static_unit", "d2l_all_force_unit", "matched_events"
    )
    lineage = pd.read_csv(args.h5_analysis / "SAME_STATE_EVENT_CONTINGENCY.csv")
    lineage = lineage[
        (lineage.window == "W2")
        & (lineage.all_force_unit >= 0)
        & (lineage.no_force_unit >= 0)
    ]
    force_no_force, lineage_ties = reciprocal(
        lineage, "all_force_unit", "no_force_unit", "event_rows"
    )
    metrics = {
        arm: pd.read_csv(args.rf / f"{arm}_unit_rf_metrics.csv").set_index("unit_id")
        for arm in ("static", "all_force", "no_force")
    }
    no_force_by_force = {
        all_force: (no_force, count)
        for (all_force, no_force), count in force_no_force.items()
    }
    rows = []
    for (static, all_force), cross_count in sorted(static_force.items()):
        if all_force not in no_force_by_force:
            continue
        no_force, lineage_count = no_force_by_force[all_force]
        if not (
            static in metrics["static"].index
            and all_force in metrics["all_force"].index
            and no_force in metrics["no_force"].index
        ):
            continue
        row = {
            "static_unit": static,
            "all_force_unit": all_force,
            "no_force_unit": no_force,
            "static_all_force_matched_events": cross_count,
            "all_force_no_force_lineage_rows": lineage_count,
        }
        for arm, unit in (
            ("static", static),
            ("all_force", all_force),
            ("no_force", no_force),
        ):
            row[f"{arm}_cv_snr"] = float(metrics[arm].loc[unit, "cv_snr"])
        rows.append(row)
    joined = pd.DataFrame(rows)
    joined.to_csv(args.output / "RF_RECIPROCAL_TRIPLETS.csv", index=False)
    summary = {
        "status": "complete_development_only",
        "correspondence": (
            "RF-independent reciprocal unique-best unit links: time-only <=12-sample "
            "static/all-force event candidates, then exact saved-row all-force/no-force lineage"
        ),
        "correspondence_is_identity_truth": False,
        "static_all_force_reciprocal_pairs": len(static_force),
        "all_force_no_force_reciprocal_pairs": len(force_no_force),
        "eligible_three_way_pairs": len(joined),
        "cross_arm_ties_excluded": cross_ties,
        "lineage_ties_excluded": lineage_ties,
        "metrics": {},
        "outer_holdout_opened": False,
    }
    for arm in ("static", "all_force", "no_force"):
        values = joined[f"{arm}_cv_snr"].to_numpy()
        summary["metrics"][arm] = {
            "median_cv_snr": float(np.median(values)),
            "mean_cv_snr": float(np.mean(values)),
        }
    for left, right in (
        ("all_force", "static"),
        ("no_force", "static"),
        ("no_force", "all_force"),
    ):
        delta = (
            joined[f"{left}_cv_snr"].to_numpy()
            - joined[f"{right}_cv_snr"].to_numpy()
        )
        summary["metrics"][f"{left}_minus_{right}"] = {
            "median": float(np.median(delta)),
            "mean": float(np.mean(delta)),
            "wins": int(np.sum(delta > 0)),
            "losses": int(np.sum(delta < 0)),
            "exact_ties": int(np.sum(delta == 0)),
        }
    (args.output / "RF_PAIRED_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
