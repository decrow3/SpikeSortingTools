"""Summarize >2,000-spike amplitude completeness and cross-grid unit families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARM_SOURCES = {
    "9/8": ("lower_l_at_u9", "candidate"),
    "9/9": ("lower_l_at_u9", "baseline"),
    "10/8": ("lower_l_at_u10", "candidate"),
    "10/9": ("lower_l_at_u10", "baseline"),
    "10/10": ("lower_u_at_l10", "candidate"),
    "12/10": ("lower_u_at_l10", "baseline"),
    "12/9": ("lower_l_at_u12", "candidate"),
}
ARM_ORDER = ("12/10", "12/9", "10/10", "10/9", "10/8", "9/9", "9/8")


def load_arm(comparisons: Path, arm: str) -> pd.DataFrame:
    edge, side = ARM_SOURCES[arm]
    root = comparisons / edge
    metrics = pd.read_csv(root / f"unit_metrics_{side}.csv")
    windows = pd.read_csv(root / "amplitude_windows.csv")
    windows = windows[(windows.sort_id.astype(str) == arm) & (windows["sort"] == side)]
    rows = []
    for unit in metrics.itertuples(index=False):
        finite = windows[
            (windows.cluster_id == int(unit.cluster_id))
            & (windows.status == "finite_interior")
            & np.isfinite(windows.missing_pct)
        ]
        measurable = len(finite) >= 2
        rows.append(
            {
                "cluster_id": int(unit.cluster_id),
                "spike_count": int(unit.spike_count),
                "eligible_gt2000": int(unit.spike_count) > 2000,
                "finite_window_count": len(finite),
                "measurable": measurable,
                "amplitude_completeness_pct": (
                    100.0 - float(finite.missing_pct.median()) if measurable else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).set_index("cluster_id")


def primary_map(comparisons: Path, edge: str) -> dict[int, int]:
    frame = pd.read_csv(comparisons / edge / "primary_matches.csv")
    return dict(zip(frame.baseline_cluster.astype(int), frame.candidate_cluster.astype(int)))


def cycle_consistent_families(comparisons: Path) -> list[dict[str, int]]:
    u_l9 = primary_map(comparisons, "lower_u_at_l9")
    l_u10 = primary_map(comparisons, "lower_l_at_u10")
    l_u9 = primary_map(comparisons, "lower_l_at_u9")
    u_l8 = primary_map(comparisons, "lower_u_at_l8")
    lower = []
    for unit_10_9, unit_9_9 in u_l9.items():
        unit_10_8 = l_u10.get(unit_10_9)
        via_9_9 = l_u9.get(unit_9_9)
        via_10_8 = u_l8.get(unit_10_8) if unit_10_8 is not None else None
        if via_9_9 is not None and via_9_9 == via_10_8:
            lower.append(
                {"10/9": unit_10_9, "9/9": unit_9_9, "10/8": unit_10_8, "9/8": via_9_9}
            )

    u_l10 = primary_map(comparisons, "lower_u_at_l10")
    l_u12 = primary_map(comparisons, "lower_l_at_u12")
    l_u10_high = primary_map(comparisons, "lower_l_at_u10_high")
    u_l9_high = primary_map(comparisons, "lower_u_at_l9_high")
    upper = []
    for unit_12_10, unit_10_10 in u_l10.items():
        unit_12_9 = l_u12.get(unit_12_10)
        via_10_10 = l_u10_high.get(unit_10_10)
        via_12_9 = u_l9_high.get(unit_12_9) if unit_12_9 is not None else None
        if via_10_10 is not None and via_10_10 == via_12_9:
            upper.append(
                {
                    "12/10": unit_12_10,
                    "10/10": unit_10_10,
                    "12/9": unit_12_9,
                    "10/9": via_10_10,
                }
            )

    lower_by_anchor = {row["10/9"]: row for row in lower}
    return [{**row, **lower_by_anchor[row["10/9"]]} for row in upper if row["10/9"] in lower_by_anchor]


def analyze(comparisons: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    arms = {arm: load_arm(comparisons, arm) for arm in ARM_ORDER}
    arm_rows = []
    for arm in ARM_ORDER:
        frame = arms[arm]
        eligible = frame[frame.eligible_gt2000]
        measured = eligible[eligible.measurable]
        arm_rows.append(
            {
                "arm": arm,
                "eligible_units": len(eligible),
                "eligible_spikes": int(eligible.spike_count.sum()),
                "eligible_spike_fraction": eligible.spike_count.sum() / frame.spike_count.sum(),
                "measured_units": len(measured),
                "measured_fraction": len(measured) / len(eligible),
                "median_completeness_pct": measured.amplitude_completeness_pct.median(),
                "q25_completeness_pct": measured.amplitude_completeness_pct.quantile(0.25),
                "q75_completeness_pct": measured.amplitude_completeness_pct.quantile(0.75),
            }
        )

    raw_families = cycle_consistent_families(comparisons)
    eligible_families = [
        family
        for family in raw_families
        if all(arms[arm].at[cluster, "eligible_gt2000"] for arm, cluster in family.items())
    ]
    fully_measurable = [
        family
        for family in eligible_families
        if all(arms[arm].at[cluster, "measurable"] for arm, cluster in family.items())
    ]
    family_rows = []
    for family_id, family in enumerate(eligible_families, start=1):
        row = {"family_id": family_id}
        for arm in ARM_ORDER:
            unit = arms[arm].loc[family[arm]]
            key = arm.replace("/", "_")
            row[f"cluster_{key}"] = family[arm]
            row[f"spikes_{key}"] = int(unit.spike_count)
            row[f"completeness_{key}_pct"] = unit.amplitude_completeness_pct
        row["fully_amplitude_measurable"] = family in fully_measurable
        family_rows.append(row)

    matched_rows = []
    for arm in ARM_ORDER:
        all_spikes = [arms[arm].at[family[arm], "spike_count"] for family in eligible_families]
        complete = [
            arms[arm].at[family[arm], "amplitude_completeness_pct"] for family in fully_measurable
        ]
        eligible_spikes = next(row["eligible_spikes"] for row in arm_rows if row["arm"] == arm)
        matched_rows.append(
            {
                "arm": arm,
                "matched_eligible_families": len(eligible_families),
                "matched_family_spikes": int(sum(all_spikes)),
                "matched_fraction_of_eligible_spikes": sum(all_spikes) / eligible_spikes,
                "fully_measurable_families": len(fully_measurable),
                "median_completeness_pct_fully_paired": float(np.median(complete)),
                "q25_completeness_pct_fully_paired": float(np.quantile(complete, 0.25)),
                "q75_completeness_pct_fully_paired": float(np.quantile(complete, 0.75)),
            }
        )
    method = {
        "eligibility": "strictly more than 2000 curated spikes in every included arm",
        "amplitude_measurable": "at least two finite_interior 1000-spike windows",
        "matching": "reciprocal-primary matches on each adjacent edge, with agreement along both paths around each 2x2 cell",
        "raw_all_seven_families": len(raw_families),
        "eligible_all_seven_families": len(eligible_families),
        "fully_amplitude_measurable_all_seven_families": len(fully_measurable),
        "warning": "These are cycle-consistent reciprocal-primary families, not isolated degree-one components of the full correspondence graph.",
    }
    return pd.DataFrame(arm_rows), pd.DataFrame(family_rows), pd.DataFrame(matched_rows), method


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    arms, families, matched, method = analyze(args.analysis_root / "comparisons")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arms.to_csv(args.output_dir / "amplitude_completeness_gt2000_all_arms.csv", index=False)
    families.to_csv(args.output_dir / "cycle_consistent_unit_families_gt2000.csv", index=False)
    matched.to_csv(args.output_dir / "cycle_consistent_matched_summary.csv", index=False)
    (args.output_dir / "cycle_consistent_matching_method.json").write_text(
        json.dumps(method, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(method, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
