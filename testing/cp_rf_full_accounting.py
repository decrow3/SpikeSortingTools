"""RF eligibility and exact same-state family accounting for CP W2.

This is descriptive.  It does not change the frozen RF eligibility rule, open
the outer holdout, or promote timing correspondences to biological identities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def unique_best(table, source, target, weight):
    out = {}
    for unit, rows in table.groupby(source):
        maximum = rows[weight].max()
        winners = rows.loc[rows[weight] == maximum, target].unique()
        if len(winners) == 1:
            out[int(unit)] = (int(winners[0]), int(maximum))
    return out


def reciprocal(table, left, right, weight):
    forward = unique_best(table, left, right, weight)
    reverse = unique_best(table, right, left, weight)
    return {
        (a, b): n
        for a, (b, n) in forward.items()
        if b in reverse and reverse[b][0] == a
    }


def reason(row):
    failures = []
    if row.n_spikes < 500:
        failures.append("total<500")
    if row.n_spikes_fold0 < 200:
        failures.append("fold0<200")
    if row.n_spikes_fold1 < 200:
        failures.append("fold1<200")
    return "eligible" if not failures else ";".join(failures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-analysis", type=Path, required=True)
    parser.add_argument("--rf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    eligibility = {}
    metrics = {}
    for arm in ("static", "all_force", "no_force"):
        table = pd.read_csv(args.rf / f"{arm}_rf_eligibility.csv")
        table["ineligibility_reason"] = table.apply(reason, axis=1)
        eligibility[arm] = table.set_index("unit_id")
        metrics[arm] = pd.read_csv(
            args.rf / f"{arm}_unit_rf_metrics.csv"
        ).set_index("unit_id")

    lineage_all = pd.read_csv(args.h5_analysis / "SAME_STATE_EVENT_CONTINGENCY.csv")
    lineage_all = lineage_all[lineage_all.window == "W2"].copy()
    lineage = lineage_all[
        (lineage_all.all_force_unit >= 0)
        & (lineage_all.no_force_unit >= 0)
    ].copy()
    parent_totals = lineage.groupby("all_force_unit").event_rows.sum()
    child_totals = lineage.groupby("no_force_unit").event_rows.sum()
    pair_rows = []
    for row in lineage.itertuples(index=False):
        af, nf = int(row.all_force_unit), int(row.no_force_unit)
        a = eligibility["all_force"].loc[af]
        n = eligibility["no_force"].loc[nf]
        pair_rows.append(
            {
                "all_force_unit": af,
                "no_force_unit": nf,
                "event_rows": int(row.event_rows),
                "fraction_of_parent": float(row.event_rows / parent_totals.loc[af]),
                "fraction_of_child": float(row.event_rows / child_totals.loc[nf]),
                "all_force_eligible": bool(a.eligible),
                "all_force_n_spikes": int(a.n_spikes),
                "all_force_fold0": int(a.n_spikes_fold0),
                "all_force_fold1": int(a.n_spikes_fold1),
                "no_force_eligible": bool(n.eligible),
                "no_force_n_spikes": int(n.n_spikes),
                "no_force_fold0": int(n.n_spikes_fold0),
                "no_force_fold1": int(n.n_spikes_fold1),
            }
        )
    pair_table = pd.DataFrame(pair_rows)
    pair_table.to_csv(args.output / "ALL_FORCE_NO_FORCE_ALL_PAIRS.csv", index=False)

    families_meta = pd.read_csv(args.h5_analysis / "SAME_STATE_FAMILIES.csv")
    families_meta = families_meta[families_meta.window == "W2"].set_index(
        "all_force_unit"
    )
    family_rows = []
    for af, children in pair_table.groupby("all_force_unit"):
        af = int(af)
        children = children.sort_values(
            ["event_rows", "no_force_unit"], ascending=[False, True]
        )
        a = eligibility["all_force"].loc[af]
        child_ids = children.no_force_unit.astype(int).tolist()
        eligible_child_ids = children.loc[
            children.no_force_eligible, "no_force_unit"
        ].astype(int).tolist()
        dominant = int(children.iloc[0].no_force_unit)
        meta = families_meta.loc[af]
        family_rows.append(
            {
                "all_force_unit": af,
                "h5_category": str(meta.category),
                "all_force_eligible": bool(a.eligible),
                "all_force_n_spikes": int(a.n_spikes),
                "all_force_fold0": int(a.n_spikes_fold0),
                "all_force_fold1": int(a.n_spikes_fold1),
                "all_force_cv_snr": (
                    float(metrics["all_force"].loc[af, "cv_snr"])
                    if bool(a.eligible)
                    else np.nan
                ),
                "n_no_force_children": len(child_ids),
                "no_force_children": ";".join(map(str, child_ids)),
                "n_eligible_children": len(eligible_child_ids),
                "eligible_children": ";".join(map(str, eligible_child_ids)),
                "dominant_child": dominant,
                "dominant_child_share": float(children.iloc[0].fraction_of_parent),
                "dominant_child_eligible": bool(children.iloc[0].no_force_eligible),
                "dominant_child_cv_snr": (
                    float(metrics["no_force"].loc[dominant, "cv_snr"])
                    if bool(children.iloc[0].no_force_eligible)
                    else np.nan
                ),
            }
        )
    families = pd.DataFrame(family_rows)
    families.to_csv(args.output / "ALL_FORCE_NO_FORCE_RF_FAMILIES.csv", index=False)

    cross = pd.read_csv(args.h5_analysis / "STATIC_D2L_TIME_ONLY_PAIRS.csv")
    cross = cross[
        (cross.window == "W2")
        & (cross.static_unit >= 0)
        & (cross.d2l_all_force_unit >= 0)
    ]
    static_force = reciprocal(
        cross, "static_unit", "d2l_all_force_unit", "matched_events"
    )
    cross_rows = []
    for (static, af), events in sorted(static_force.items()):
        s, a = eligibility["static"].loc[static], eligibility["all_force"].loc[af]
        cross_rows.append(
            {
                "static_unit": static,
                "all_force_unit": af,
                "matched_events": events,
                "static_eligible": bool(s.eligible),
                "all_force_eligible": bool(a.eligible),
                "static_n_spikes": int(s.n_spikes),
                "all_force_n_spikes": int(a.n_spikes),
                "static_cv_snr": float(metrics["static"].loc[static, "cv_snr"])
                if bool(s.eligible)
                else np.nan,
                "all_force_cv_snr": float(
                    metrics["all_force"].loc[af, "cv_snr"]
                )
                if bool(a.eligible)
                else np.nan,
            }
        )
    cross_table = pd.DataFrame(cross_rows)
    cross_table.to_csv(args.output / "STATIC_ALL_FORCE_RECIPROCAL_PAIRS.csv", index=False)

    arm_summary = {}
    count_bins = [500, 1000, 2000, np.inf]
    count_labels = ["500-999", "1000-1999", ">=2000"]
    count_matched = []
    for arm in ("static", "all_force", "no_force"):
        e = eligibility[arm]
        reasons = e.ineligibility_reason.value_counts().to_dict()
        m = metrics[arm].copy()
        m["count_bin"] = pd.cut(
            m.n_spikes, count_bins, right=False, labels=count_labels
        )
        for label, rows in m.groupby("count_bin", observed=True):
            count_matched.append(
                {
                    "arm": arm,
                    "spike_count_bin": str(label),
                    "n_units": len(rows),
                    "median_cv_snr": float(rows.cv_snr.median()),
                    "mean_cv_snr": float(rows.cv_snr.mean()),
                }
            )
        arm_summary[arm] = {
            "n_units": int(len(e)),
            "n_eligible": int(e.eligible.sum()),
            "eligibility_fraction": float(e.eligible.mean()),
            "ineligibility_reason_counts": {
                str(key): int(value) for key, value in reasons.items()
            },
        }
    pd.DataFrame(count_matched).to_csv(
        args.output / "RF_BY_FROZEN_SPIKE_COUNT_BIN.csv", index=False
    )

    af_eligible = families.all_force_eligible
    any_child = families.n_eligible_children > 0
    positive_parent_children = set(lineage.no_force_unit.astype(int))
    no_force_only_from_noise = sorted(
        set(eligibility["no_force"].index.astype(int)) - positive_parent_children
    )
    eligible_only_from_noise = sum(
        bool(eligibility["no_force"].loc[unit, "eligible"])
        for unit in no_force_only_from_noise
    )
    all_force_with_noise_rows = set(
        lineage_all.loc[
            (lineage_all.all_force_unit >= 0) & (lineage_all.no_force_unit < 0),
            "all_force_unit",
        ].astype(int)
    )
    dominant_parent = pair_table.sort_values(
        ["no_force_unit", "event_rows", "all_force_unit"],
        ascending=[True, False, True],
    ).drop_duplicates("no_force_unit")
    eligible_children = dominant_parent[dominant_parent.no_force_eligible]
    comparable_dominant = families[
        families.all_force_eligible & families.dominant_child_eligible
    ].copy()
    dominant_delta = (
        comparable_dominant.dominant_child_cv_snr
        - comparable_dominant.all_force_cv_snr
    )
    summary = {
        "status": "complete_development_only",
        "outer_holdout_opened": False,
        "eligibility_rule": "total>=500 and each frozen inner fold>=200",
        "arms": arm_summary,
        "same_state_family_accounting": {
            "n_all_force_parents": int(len(families)),
            "n_exact_nonnoise_parent_child_pairs": int(len(pair_table)),
            "no_force_units_with_positive_parent": int(len(positive_parent_children)),
            "no_force_units_only_from_all_force_noise": int(len(no_force_only_from_noise)),
            "eligible_no_force_units_only_from_all_force_noise": int(
                eligible_only_from_noise
            ),
            "eligible_no_force_units_with_dominant_eligible_parent": int(
                eligible_children.all_force_eligible.sum()
            ),
            "eligible_no_force_units_with_dominant_ineligible_parent": int(
                (~eligible_children.all_force_eligible).sum()
            ),
            "all_force_units_with_rows_becoming_noise": int(
                len(all_force_with_noise_rows)
            ),
            "eligible_parent_any_eligible_child": int((af_eligible & any_child).sum()),
            "eligible_parent_no_eligible_child": int((af_eligible & ~any_child).sum()),
            "ineligible_parent_any_eligible_child": int((~af_eligible & any_child).sum()),
            "ineligible_parent_no_eligible_child": int((~af_eligible & ~any_child).sum()),
            "eligible_parent_dominant_child_eligible": int(
                (af_eligible & families.dominant_child_eligible).sum()
            ),
            "eligible_parent_vs_eligible_dominant_child_rf": {
                "n": int(len(dominant_delta)),
                "median_child_minus_parent_cv_snr": float(np.median(dominant_delta)),
                "mean_child_minus_parent_cv_snr": float(np.mean(dominant_delta)),
                "child_wins": int((dominant_delta > 0).sum()),
                "child_losses": int((dominant_delta < 0).sum()),
                "exact_ties": int((dominant_delta == 0).sum()),
            },
            "interpretation": (
                "Exact row lineage accounts for every nonnoise all-force/no-force "
                "parent-child pair; child units are not counted as newly recovered neurons."
            ),
        },
        "static_all_force_candidate_correspondence": {
            "reciprocal_unique_best_pairs": int(len(cross_table)),
            "both_eligible": int(
                (cross_table.static_eligible & cross_table.all_force_eligible).sum()
            ),
            "static_eligible_all_force_ineligible": int(
                (cross_table.static_eligible & ~cross_table.all_force_eligible).sum()
            ),
            "static_ineligible_all_force_eligible": int(
                (~cross_table.static_eligible & cross_table.all_force_eligible).sum()
            ),
            "identity_truth": False,
        },
        "count_matched_scope": (
            "Descriptive frozen broad spike-count bins among eligible units; no "
            "threshold optimization and no cross-arm identity assumption."
        ),
        "rf_map_example": (
            "Unavailable from the compact RF endpoint because STAs were not saved; "
            "not reconstructed because it is optional and not decision-critical."
        ),
    }
    (args.output / "RF_FULL_ACCOUNTING.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
