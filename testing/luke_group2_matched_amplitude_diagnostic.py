"""Bounded matched-cohort diagnostic for the Group 2 9/9 versus 9/8 arms.

This script deliberately reuses the saved correspondence graph.  It does not
recompute matching or tune the overlap threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE_ARM = "rescue_9_9_motion_off"
CANDIDATE_ARM = "rescue_9_8_motion_off"


def _safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--amplitude-units", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    primary = pd.read_csv(args.comparison_dir / "primary_matches.csv")
    edges = pd.read_csv(args.comparison_dir / "correspondence_edges.csv")
    baseline_metrics = pd.read_csv(args.comparison_dir / "unit_metrics_baseline.csv")
    candidate_metrics = pd.read_csv(args.comparison_dir / "unit_metrics_candidate.csv")
    comparison_summary = json.loads((args.comparison_dir / "summary.json").read_text())
    amplitude = pd.read_csv(args.amplitude_units)

    assert primary.primary_match.all()
    assert primary.baseline_cluster.is_unique
    assert primary.candidate_cluster.is_unique
    assert set(amplitude.arm.unique()) >= {BASELINE_ARM, CANDIDATE_ARM}
    assert baseline_metrics.cluster_id.is_unique and candidate_metrics.cluster_id.is_unique

    baseline_degrees = edges.groupby("baseline_cluster").candidate_cluster.nunique()
    candidate_degrees = edges.groupby("candidate_cluster").baseline_cluster.nunique()
    pair = primary.copy()
    profile_columns = [
        "cluster_id",
        "label",
        "mean_rate_hz",
        "refractory_violation_fraction_1_5ms",
        "active_lifetime_fraction",
        "presence_fraction",
        "firing_rate_cv",
        "amplitude_cv",
        "late_early_amplitude_ratio",
        "median_depth_um",
        "depth_excursion_p95_p5_um",
        "edge_spike_fraction",
        "early_middle_waveform_similarity",
        "middle_late_waveform_similarity",
    ]
    pair = pair.merge(
        baseline_metrics[profile_columns].rename(
            columns={
                c: "baseline_cluster" if c == "cluster_id" else f"baseline_{c}"
                for c in profile_columns
            }
        ),
        on="baseline_cluster",
        how="left",
        validate="one_to_one",
    ).merge(
        candidate_metrics[profile_columns].rename(
            columns={
                c: "candidate_cluster" if c == "cluster_id" else f"candidate_{c}"
                for c in profile_columns
            }
        ),
        on="candidate_cluster",
        how="left",
        validate="one_to_one",
    )
    pair["candidate_minus_baseline_rate_hz"] = (
        pair.candidate_mean_rate_hz - pair.baseline_mean_rate_hz
    )
    pair["candidate_over_baseline_rate_ratio"] = (
        pair.candidate_mean_rate_hz / pair.baseline_mean_rate_hz
    )
    pair["rate_change_pct"] = 100.0 * (pair.candidate_over_baseline_rate_ratio - 1.0)
    pair["log2_rate_ratio"] = np.log2(pair.candidate_over_baseline_rate_ratio)
    pair["firing_rate_change_direction"] = np.where(
        pair.candidate_over_baseline_rate_ratio > 1.0,
        "9/8 higher",
        np.where(pair.candidate_over_baseline_rate_ratio < 1.0, "9/8 lower", "equal"),
    )
    for metric in (
        "refractory_violation_fraction_1_5ms",
        "active_lifetime_fraction",
        "presence_fraction",
        "firing_rate_cv",
        "amplitude_cv",
        "late_early_amplitude_ratio",
        "depth_excursion_p95_p5_um",
        "edge_spike_fraction",
    ):
        pair[f"candidate_minus_baseline_{metric}"] = (
            pair[f"candidate_{metric}"] - pair[f"baseline_{metric}"]
        )
    pair["baseline_graph_degree"] = pair.baseline_cluster.map(baseline_degrees).fillna(0).astype(int)
    pair["candidate_graph_degree"] = pair.candidate_cluster.map(candidate_degrees).fillna(0).astype(int)
    pair["strict_degree1_pair"] = (
        (pair.baseline_graph_degree == 1) & (pair.candidate_graph_degree == 1)
    )

    baseline_amp = amplitude[amplitude.arm == BASELINE_ARM].rename(
        columns={c: f"baseline_{c}" for c in amplitude.columns if c != "arm"}
    )
    candidate_amp = amplitude[amplitude.arm == CANDIDATE_ARM].rename(
        columns={c: f"candidate_{c}" for c in amplitude.columns if c != "arm"}
    )
    pair = pair.merge(
        baseline_amp,
        left_on="baseline_cluster",
        right_on="baseline_cluster_id",
        how="left",
        validate="one_to_one",
    ).merge(
        candidate_amp,
        left_on="candidate_cluster",
        right_on="candidate_cluster_id",
        how="left",
        validate="one_to_one",
    )
    pair["baseline_gt2000"] = pair.baseline_spike_count.notna()
    pair["candidate_gt2000"] = pair.candidate_spike_count.notna()

    def status(prefix: str, eligible: pd.Series) -> pd.Series:
        measured = pair[f"{prefix}_measurement_status"].eq("measured")
        return pd.Series(
            np.where(~eligible, "not_eligible_<=2000", np.where(measured, "measured", "unmeasured")),
            index=pair.index,
        )

    pair["baseline_amplitude_status"] = status("baseline", pair.baseline_gt2000)
    pair["candidate_amplitude_status"] = status("candidate", pair.candidate_gt2000)
    pair["transition"] = (
        pair.baseline_amplitude_status + " -> " + pair.candidate_amplitude_status
    )
    both_measured = (
        pair.baseline_amplitude_status.eq("measured")
        & pair.candidate_amplitude_status.eq("measured")
    )
    pair["candidate_minus_baseline_completeness_pp"] = np.where(
        both_measured,
        pair.candidate_amplitude_completeness_pct - pair.baseline_amplitude_completeness_pct,
        np.nan,
    )

    baseline_anchor = pair[pair.baseline_gt2000].copy()
    transitions = (
        baseline_anchor.groupby(
            ["baseline_amplitude_status", "candidate_amplitude_status", "transition"],
            dropna=False,
        )
        .size()
        .rename("pair_count")
        .reset_index()
    )
    transitions["fraction_of_baseline_gt2000_primary_pairs"] = (
        transitions.pair_count / len(baseline_anchor)
    )

    b_total_spikes = int(baseline_metrics.spike_count.sum())
    c_total_spikes = int(candidate_metrics.spike_count.sum())
    b_primary_spikes = int(primary.baseline_events.sum())
    c_primary_spikes = int(primary.candidate_events.sum())
    mass = pd.DataFrame(
        [
            {
                "stratum": "reciprocal_primary_cluster_set",
                "baseline_spikes": b_primary_spikes,
                "candidate_spikes": c_primary_spikes,
            },
            {
                "stratum": "outside_reciprocal_primary_cluster_set",
                "baseline_spikes": b_total_spikes - b_primary_spikes,
                "candidate_spikes": c_total_spikes - c_primary_spikes,
            },
        ]
    )
    mass["candidate_minus_baseline_spikes"] = mass.candidate_spikes - mass.baseline_spikes
    mass["share_of_total_spike_increase"] = mass.candidate_minus_baseline_spikes / (
        c_total_spikes - b_total_spikes
    )

    baseline_gt = baseline_metrics[baseline_metrics.spike_count > 2000]
    candidate_gt = candidate_metrics[candidate_metrics.spike_count > 2000]
    measured_pairs = baseline_anchor[
        baseline_anchor.baseline_amplitude_status.eq("measured")
        & baseline_anchor.candidate_amplitude_status.eq("measured")
    ].copy()
    paired_delta = measured_pairs.candidate_minus_baseline_completeness_pp.dropna()
    measured_rate_ranked = measured_pairs.sort_values(
        "candidate_minus_baseline_rate_hz", ascending=False
    ).copy()
    increasing_measured = measured_rate_ranked[
        measured_rate_ranked.candidate_over_baseline_rate_ratio > 1.0
    ].copy()
    measured_net_spike_delta = int(
        measured_pairs.candidate_events.sum() - measured_pairs.baseline_events.sum()
    )

    def rate_summary(frame: pd.DataFrame) -> dict:
        ratios = frame.candidate_over_baseline_rate_ratio
        equal = np.isclose(ratios, 1.0, rtol=1e-9, atol=1e-12)
        return {
            "pairs": int(len(frame)),
            "median_baseline_rate_hz": float(frame.baseline_mean_rate_hz.median()),
            "median_candidate_rate_hz": float(frame.candidate_mean_rate_hz.median()),
            "median_rate_ratio": float(ratios.median()),
            "median_rate_change_pct": float(100.0 * (ratios.median() - 1.0)),
            "q25_rate_change_pct": float(frame.rate_change_pct.quantile(0.25)),
            "q75_rate_change_pct": float(frame.rate_change_pct.quantile(0.75)),
            "aggregate_rate_ratio": float(
                frame.candidate_mean_rate_hz.sum() / frame.baseline_mean_rate_hz.sum()
            ),
            "rate_higher_pairs": int(((ratios > 1.0) & ~equal).sum()),
            "rate_equal_pairs": int(equal.sum()),
            "rate_lower_pairs": int(((ratios < 1.0) & ~equal).sum()),
        }

    def profile_summary(frame: pd.DataFrame) -> dict:
        return {
            "pairs": int(len(frame)),
            "baseline_labels": {
                str(k): int(v) for k, v in frame.baseline_label.value_counts().items()
            },
            "candidate_labels": {
                str(k): int(v) for k, v in frame.candidate_label.value_counts().items()
            },
            "baseline_spatial_classes": {
                str(k): int(v) for k, v in frame.baseline_spatial_class.value_counts().items()
            },
            "median_baseline_rate_hz": float(frame.baseline_mean_rate_hz.median()),
            "median_rate_change_pct": float(frame.rate_change_pct.median()),
            "median_baseline_retention": float(frame.baseline_retention.median()),
            "median_candidate_retention": float(frame.candidate_retention.median()),
            "median_jaccard": float(frame.jaccard.median()),
            "median_refractory_violation_change_pp": float(
                100.0 * frame.candidate_minus_baseline_refractory_violation_fraction_1_5ms.median()
            ),
            "median_presence_fraction_change_pp": float(
                100.0 * frame.candidate_minus_baseline_presence_fraction.median()
            ),
            "median_active_lifetime_change_pp": float(
                100.0 * frame.candidate_minus_baseline_active_lifetime_fraction.median()
            ),
            "median_amplitude_cv_change": float(
                frame.candidate_minus_baseline_amplitude_cv.median()
            ),
            "median_completeness_change_pp": float(
                frame.candidate_minus_baseline_completeness_pp.median()
            ),
        }

    summary = {
        "comparison": f"{BASELINE_ARM}_vs_{CANDIDATE_ARM}",
        "matching_recomputed": False,
        "strict_amplitude_threshold": "spike_count > 2000",
        "amplitude_measurement_rule": "at least 2 finite_interior 1000-spike windows",
        "all_units": {
            "baseline_units": int(len(baseline_metrics)),
            "candidate_units": int(len(candidate_metrics)),
            "baseline_spikes": b_total_spikes,
            "candidate_spikes": c_total_spikes,
        },
        "gt2000_units": {
            "baseline_units": int(len(baseline_gt)),
            "candidate_units": int(len(candidate_gt)),
            "baseline_spikes": int(baseline_gt.spike_count.sum()),
            "candidate_spikes": int(candidate_gt.spike_count.sum()),
        },
        "reciprocal_primary_representation": {
            "pairs": int(len(primary)),
            "baseline_unit_fraction": _safe_fraction(len(primary), len(baseline_metrics)),
            "candidate_unit_fraction": _safe_fraction(len(primary), len(candidate_metrics)),
            "baseline_spike_fraction": _safe_fraction(b_primary_spikes, b_total_spikes),
            "candidate_spike_fraction": _safe_fraction(c_primary_spikes, c_total_spikes),
            "baseline_gt2000_units_in_pairs": int(len(baseline_anchor)),
            "baseline_gt2000_unit_fraction": _safe_fraction(len(baseline_anchor), len(baseline_gt)),
            "baseline_gt2000_spike_fraction": _safe_fraction(
                baseline_anchor.baseline_events.sum(), baseline_gt.spike_count.sum()
            ),
        },
        "graph_ambiguity": {
            "correspondence_edges": int(len(edges)),
            "strict_degree1_pairs": int(pair.strict_degree1_pair.sum()),
            "primary_pairs_with_baseline_degree_gt1": int((pair.baseline_graph_degree > 1).sum()),
            "primary_pairs_with_candidate_degree_gt1": int((pair.candidate_graph_degree > 1).sum()),
            "interpretation": (
                "The saved overlap graph is too dense to support a strict degree-1 "
                "clean-pair/split-merge classification at its existing threshold."
            ),
        },
        "baseline_gt2000_primary_pairs": {
            "pairs": int(len(baseline_anchor)),
            "both_measured_fraction_of_all_baseline_gt2000_units": _safe_fraction(
                len(measured_pairs), len(baseline_gt)
            ),
            "candidate_also_gt2000": int(baseline_anchor.candidate_gt2000.sum()),
            "both_measured": int(len(measured_pairs)),
            "median_candidate_minus_baseline_completeness_pp": (
                float(paired_delta.median()) if len(paired_delta) else None
            ),
            "q25_candidate_minus_baseline_completeness_pp": (
                float(paired_delta.quantile(0.25)) if len(paired_delta) else None
            ),
            "q75_candidate_minus_baseline_completeness_pp": (
                float(paired_delta.quantile(0.75)) if len(paired_delta) else None
            ),
            "candidate_higher_completeness": int((paired_delta > 0).sum()),
            "equal_completeness": int((paired_delta == 0).sum()),
            "candidate_lower_completeness": int((paired_delta < 0).sum()),
        },
        "spike_increase_decomposition": {
            "total_candidate_minus_baseline_spikes": int(c_total_spikes - b_total_spikes),
            "reciprocal_primary_cluster_set_delta": int(
                mass.loc[mass.stratum == "reciprocal_primary_cluster_set", "candidate_minus_baseline_spikes"].iloc[0]
            ),
            "outside_primary_cluster_set_delta": int(
                mass.loc[mass.stratum == "outside_reciprocal_primary_cluster_set", "candidate_minus_baseline_spikes"].iloc[0]
            ),
        },
        "firing_rate": {
            "both_amplitude_measured_pairs": rate_summary(measured_pairs),
            "baseline_gt2000_primary_pairs": rate_summary(baseline_anchor),
            "all_primary_pairs": rate_summary(pair),
            "definition": (
                "Curated cluster spike count divided by the common full-session duration; "
                "therefore the paired rate ratio equals the paired spike-count ratio."
            ),
            "both_measured_concentration": {
                "net_candidate_minus_baseline_spikes": measured_net_spike_delta,
                "top_1_positive_rate_change_share_of_net": float(
                    measured_rate_ranked.head(1).eval("candidate_events - baseline_events").sum()
                    / measured_net_spike_delta
                ),
                "top_3_positive_rate_change_share_of_net": float(
                    measured_rate_ranked.head(3).eval("candidate_events - baseline_events").sum()
                    / measured_net_spike_delta
                ),
                "top_5_positive_rate_change_share_of_net": float(
                    measured_rate_ranked.head(5).eval("candidate_events - baseline_events").sum()
                    / measured_net_spike_delta
                ),
                "spearman_rate_change_vs_completeness_change": float(
                    measured_pairs[[
                        "rate_change_pct", "candidate_minus_baseline_completeness_pp"
                    ]].corr(method="spearman").iloc[0, 1]
                ),
            },
            "both_measured_unit_profiles": {
                "rate_increasing": profile_summary(increasing_measured),
                "rate_decreasing": profile_summary(
                    measured_pairs[measured_pairs.candidate_over_baseline_rate_ratio < 1.0]
                ),
            },
            "baseline_gt2000_unit_profiles": {
                "rate_increasing": profile_summary(
                    baseline_anchor[baseline_anchor.candidate_over_baseline_rate_ratio > 1.0]
                ),
                "rate_decreasing": profile_summary(
                    baseline_anchor[baseline_anchor.candidate_over_baseline_rate_ratio < 1.0]
                ),
            },
        },
        "existing_comparison_coverage_status": comparison_summary["coverage_status"],
        "coincidence_excess_localization": (
            "Not identifiable from the saved aggregate guardrail and correspondence tables; "
            "localizing it would require a new event-level computation."
        ),
    }

    # QA: denominators, joins, and additive decomposition must reconcile exactly.
    assert b_total_spikes == int(comparison_summary["baseline_spikes"])
    assert c_total_spikes == int(comparison_summary["candidate_spikes"])
    assert len(primary) == int(comparison_summary["primary_matches"])
    assert int(primary.baseline_events.sum()) == int(
        baseline_metrics.set_index("cluster_id").loc[primary.baseline_cluster, "spike_count"].sum()
    )
    assert int(primary.candidate_events.sum()) == int(
        candidate_metrics.set_index("cluster_id").loc[primary.candidate_cluster, "spike_count"].sum()
    )
    assert int(mass.baseline_spikes.sum()) == b_total_spikes
    assert int(mass.candidate_spikes.sum()) == c_total_spikes
    assert int(mass.candidate_minus_baseline_spikes.sum()) == c_total_spikes - b_total_spikes
    assert int(transitions.pair_count.sum()) == len(baseline_anchor)
    assert len(amplitude) == len(amplitude[["arm", "cluster_id"]].drop_duplicates())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair.to_csv(args.output_dir / "primary_pair_units.csv", index=False)
    transitions.to_csv(args.output_dir / "amplitude_status_transitions.csv", index=False)
    mass.to_csv(args.output_dir / "spike_mass_decomposition.csv", index=False)
    measured_pairs.to_csv(args.output_dir / "paired_measured_completeness.csv", index=False)
    measured_pairs.to_csv(args.output_dir / "paired_measured_firing_rates.csv", index=False)
    measured_rate_ranked.to_csv(args.output_dir / "paired_measured_rate_changes_ranked.csv", index=False)
    increasing_measured.to_csv(args.output_dir / "rate_increasing_unit_profiles.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
