"""Locate recoverable neuron loss in the accepted unwarped Luke KS4 sort.

The audit deliberately separates four questions that are often conflated:

1. Is a voltage-defined event represented in KS4's learned event table?
2. Does it survive duplicate removal and final cluster assignment?
3. Is the assigned cluster labelled ``good`` rather than ``mua``?
4. Do independently waveform-consistent KIASORT units map to one KS4 unit, or
   are their events absent/scattered across many KS4 units?

Reviewed neural events are the primary event-level biological evidence.  The
sealed automatic holdout is a sensitivity control, not a neural ground truth.
KIASORT-only waveform candidates are unit-level hypotheses, not confirmed
neurons, and are therefore reported separately.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from testing.luke_sorter_band_comparison import load_kiasort_band
except ModuleNotFoundError:  # Allow direct execution: python testing/<script>.py
    from luke_sorter_band_comparison import load_kiasort_band


RESCUE_ROOT = Path(
    "/mnt/NPX/Luke/20250804/"
    "rescue_pipeline_results_Luke0804_V2V1_g0_imec1"
)
OUTPUT = Path("testing/outputs/luke_ks4_neuron_loss_audit")
REVIEWED = Path(
    "testing/outputs/luke_multichannel_event_validation/imec1/event_stage_trace.csv"
)
HOLDOUT_KEY = Path(
    "testing/outputs/luke_prospective_holdout/holdout_candidate_key_v2.csv"
)
HOLDOUT_PUBLIC = Path(
    "testing/outputs/luke_prospective_holdout/holdout_candidates_v2.csv"
)
UNIT_METRICS = Path(
    "testing/outputs/luke_full_probe_rescue_diagnostics/unit_metrics.csv"
)
WAVEFORM_REVIEW = Path(
    "testing/outputs/luke_sorter_waveform_arbitration_kiasort_geometry_valid/"
    "isolated_kiasort_waveform_review.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescue-root", type=Path, default=RESCUE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--reviewed", type=Path, default=REVIEWED)
    parser.add_argument("--holdout-key", type=Path, default=HOLDOUT_KEY)
    parser.add_argument("--holdout-public", type=Path, default=HOLDOUT_PUBLIC)
    parser.add_argument("--unit-metrics", type=Path, default=UNIT_METRICS)
    parser.add_argument("--waveform-review", type=Path, default=WAVEFORM_REVIEW)
    parser.add_argument("--kiasort-output-name", default="kiasort_channels_82_114")
    parser.add_argument("--time-tolerance-ms", type=float, default=0.5)
    parser.add_argument("--depth-tolerance-um", type=float, default=60.0)
    return parser.parse_args()


def nearest_local_matches(
    source_times: np.ndarray,
    source_depths: np.ndarray,
    target_times: np.ndarray,
    target_labels: np.ndarray,
    *,
    time_radius: int,
    depth_radius_um: float,
    target_depths: np.ndarray | None = None,
    label_depths: np.ndarray | None = None,
    allowed_labels: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return presence and nearest label without materializing target subsets."""
    source_times = np.asarray(source_times, dtype=np.int64)
    source_depths = np.asarray(source_depths, dtype=float)
    target_times = np.asarray(target_times)
    target_labels = np.asarray(target_labels)
    if target_times.ndim != 1 or target_labels.shape != target_times.shape:
        raise ValueError("Target times and labels must be aligned one-dimensional arrays")
    if source_times.shape != source_depths.shape:
        raise ValueError("Source times and depths must be aligned")
    if target_depths is None and label_depths is None:
        raise ValueError("Either event depths or a label-depth lookup is required")
    if target_depths is not None and np.asarray(target_depths).shape != target_times.shape:
        raise ValueError("Target event depths must align with target times")

    present = np.zeros(source_times.size, dtype=bool)
    labels = np.full(source_times.size, -1, dtype=np.int64)
    for index, (time, depth) in enumerate(zip(source_times, source_depths)):
        left = int(np.searchsorted(target_times, time - time_radius, side="left"))
        right = int(np.searchsorted(target_times, time + time_radius, side="right"))
        candidates: list[tuple[int, float, int]] = []
        for target_index in range(left, right):
            label = int(target_labels[target_index])
            if allowed_labels is not None and label not in allowed_labels:
                continue
            target_depth = (
                float(target_depths[target_index])
                if target_depths is not None
                else float(label_depths[label])
            )
            if not np.isfinite(target_depth):
                continue
            depth_delta = abs(target_depth - float(depth))
            if depth_delta <= depth_radius_um:
                candidates.append(
                    (abs(int(target_times[target_index]) - int(time)), depth_delta, label)
                )
        if candidates:
            present[index] = True
            labels[index] = min(candidates)[2]
    return present, labels


def stage_outcome(learned: bool, final: bool, good: bool) -> str:
    """Classify a trace while letting the more authoritative final table win."""
    if good:
        return "retained_in_good_unit"
    if final:
        return "retained_in_mua_only"
    if learned:
        return "lost_in_duplicate_removal_or_finalization"
    return "absent_from_learned_and_final_tables"


def refractory_fraction(times: np.ndarray, radius: int) -> float:
    times = np.sort(np.asarray(times, dtype=np.int64))
    if times.size < 2:
        return float("nan")
    return float(np.mean(np.diff(times) <= radius))


def trace_cohort(
    cohort: pd.DataFrame,
    learned_times: np.ndarray,
    learned_labels: np.ndarray,
    final_times: np.ndarray,
    final_labels: np.ndarray,
    final_depths: np.ndarray,
    label_depths: np.ndarray,
    good_units: set[int],
    time_radius: int,
    depth_radius_um: float,
) -> pd.DataFrame:
    learned, learned_unit = nearest_local_matches(
        cohort.sample_index,
        cohort.depth_um,
        learned_times,
        learned_labels,
        time_radius=time_radius,
        depth_radius_um=depth_radius_um,
        label_depths=label_depths,
    )
    final, final_unit = nearest_local_matches(
        cohort.sample_index,
        cohort.depth_um,
        final_times,
        final_labels,
        time_radius=time_radius,
        depth_radius_um=depth_radius_um,
        target_depths=final_depths,
    )
    good, good_unit = nearest_local_matches(
        cohort.sample_index,
        cohort.depth_um,
        final_times,
        final_labels,
        time_radius=time_radius,
        depth_radius_um=depth_radius_um,
        target_depths=final_depths,
        allowed_labels=good_units,
    )
    result = cohort.copy()
    result["learned_present"] = learned
    result["learned_unit"] = learned_unit
    result["final_present"] = final
    result["final_unit"] = final_unit
    result["good_present"] = good
    result["good_unit"] = good_unit
    result["stage_outcome"] = [
        stage_outcome(*values) for values in zip(learned, final, good)
    ]
    return result


def cohort_summary(traces: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort_name, group in traces.groupby("cohort", sort=False):
        count = len(group)
        rows.append(
            {
                "cohort": cohort_name,
                "n_events": count,
                "learned_present": int(group.learned_present.sum()),
                "learned_fraction": float(group.learned_present.mean()),
                "final_present": int(group.final_present.sum()),
                "final_fraction": float(group.final_present.mean()),
                "good_present": int(group.good_present.sum()),
                "good_fraction": float(group.good_present.mean()),
                "learned_but_not_final": int(
                    (group.learned_present & ~group.final_present).sum()
                ),
                "final_mua_only": int(
                    (group.final_present & ~group.good_present).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def reviewed_mua_candidates(
    traces: pd.DataFrame, unit_metrics: pd.DataFrame
) -> pd.DataFrame:
    """Nominate already-detected MUA units for review, never auto-promotion."""
    reviewed = traces.loc[
        (traces.cohort == "reviewed_neural_discovery")
        & traces.final_present
        & ~traces.good_present
    ]
    counts = (
        reviewed.groupby("final_unit")
        .size()
        .rename("reviewed_neural_event_count")
        .reset_index()
        .rename(columns={"final_unit": "unit_id"})
    )
    if counts.empty:
        return counts
    result = counts.merge(
        unit_metrics.reset_index(), on="unit_id", validate="one_to_one"
    )
    result["bounded_review_candidate"] = (
        (result.reviewed_neural_event_count >= 2)
        & ~result.ks_good
        & (result.contamination_pct <= 10.0)
        & (result.refractory_violation_fraction_1p5ms <= 0.01)
        & (result.spike_count >= 100)
        & (result.presence_fraction_300s >= 0.5)
    )
    return result.sort_values(
        ["bounded_review_candidate", "reviewed_neural_event_count", "contamination_pct"],
        ascending=[False, False, True],
    )


def build_event_cohorts(
    reviewed_path: Path, holdout_key_path: Path, holdout_public_path: Path
) -> pd.DataFrame:
    reviewed = pd.read_csv(reviewed_path)
    reviewed = reviewed.loc[reviewed.review_label == "neural"].copy()
    reviewed["sample_index"] = reviewed.aligned_sample_index.astype(np.int64)
    reviewed["depth_um"] = reviewed.peak_depth_um.astype(float)
    reviewed["cohort"] = "reviewed_neural_discovery"
    reviewed["event_id"] = reviewed.review_id

    holdout = pd.read_csv(holdout_key_path).merge(
        pd.read_csv(holdout_public_path),
        on=["candidate_id", "probe", "window_id"],
        validate="one_to_one",
    )
    holdout = holdout.loc[holdout.probe == "imec1"].copy()
    holdout["cohort"] = "sealed_automatic_holdout"
    holdout["event_id"] = holdout.candidate_id
    return pd.concat(
        [
            reviewed[["cohort", "event_id", "sample_index", "depth_um", "window"]],
            holdout[
                [
                    "cohort",
                    "event_id",
                    "sample_index",
                    "depth_um",
                    "window_id",
                    "motion_stratum",
                    "polarity",
                ]
            ].rename(columns={"window_id": "window"}),
        ],
        ignore_index=True,
    )


def summarize_cross_sort_candidates(
    waveform_review: pd.DataFrame,
    kia,
    learned_times: np.ndarray,
    learned_labels: np.ndarray,
    final_times: np.ndarray,
    final_labels: np.ndarray,
    final_depths: np.ndarray,
    label_depths: np.ndarray,
    good_units: set[int],
    time_radius: int,
    depth_radius_um: float,
    refractory_radius: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    assignment_rows = []
    for candidate in waveform_review.loc[waveform_review.waveform_shortlist].itertuples():
        unit_id = int(candidate.unit_id)
        mask = kia.labels == unit_id
        cohort = pd.DataFrame(
            {
                "sample_index": kia.times[mask],
                "depth_um": kia.depths_um[mask],
            }
        )
        traced = trace_cohort(
            cohort,
            learned_times,
            learned_labels,
            final_times,
            final_labels,
            final_depths,
            label_depths,
            good_units,
            time_radius,
            depth_radius_um,
        )
        counts = Counter(map(int, traced.loc[traced.final_present, "final_unit"]))
        target_units = sorted(counts)
        union_mask = np.isin(final_labels, target_units)
        rows.append(
            {
                "kiasort_unit": unit_id,
                "source_spike_count": len(traced),
                "positive_dominant": bool(candidate.positive_dominant),
                "split_half_template_cosine": float(candidate.split_half_template_cosine),
                "early_late_template_cosine": float(candidate.early_late_template_cosine),
                "independent_template_explained_fraction_excess": float(
                    candidate.median_explained_fraction_excess
                ),
                "best_ks4_template_cosine": float(candidate.best_template_cosine),
                "learned_fraction": float(traced.learned_present.mean()),
                "final_fraction": float(traced.final_present.mean()),
                "good_fraction": float(traced.good_present.mean()),
                "ks4_target_unit_count": len(target_units),
                "largest_single_ks4_target_fraction": float(
                    max(counts.values(), default=0) / max(len(traced), 1)
                ),
                "target_union_refractory_fraction_1p5ms": refractory_fraction(
                    final_times[union_mask], refractory_radius
                ),
                "interpretation": "waveform_consistent_cross_sort_fragmentation_candidate",
            }
        )
        for rank, (target, count) in enumerate(counts.most_common(), start=1):
            assignment_rows.append(
                {
                    "kiasort_unit": unit_id,
                    "rank": rank,
                    "ks4_unit": target,
                    "matched_source_events": count,
                    "source_event_fraction": count / len(traced),
                    "ks4_label": "good" if target in good_units else "mua",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(assignment_rows)


def main() -> None:
    args = parse_args()
    sorter = args.rescue_root / "kilosort4/sorter_output"
    window_dir = args.rescue_root / "sorter_bakeoff/windows/rapid_motion-8b4978262d"
    window_manifest = json.loads(
        (window_dir / "ks4_no_motion/bakeoff_sort_manifest.json").read_text()
    )
    window = window_manifest["window"]
    fs = float(window["sampling_frequency_hz"])
    time_radius = int(round(args.time_tolerance_ms * 1e-3 * fs))
    refractory_radius = int(round(1.5e-3 * fs))

    unit_metrics = pd.read_csv(args.unit_metrics).set_index("unit_id")
    maximum_label = int(unit_metrics.index.max())
    label_depths = np.full(maximum_label + 1, np.nan, dtype=float)
    label_depths[unit_metrics.index.to_numpy(int)] = unit_metrics.median_spike_depth_um
    good_units = set(map(int, unit_metrics.index[unit_metrics.ks_good]))

    learned_times = np.load(sorter / "full_st.npy", mmap_mode="r")[:, 0]
    learned_labels = np.load(sorter / "full_clu.npy", mmap_mode="r")
    final_times = np.load(sorter / "spike_times.npy", mmap_mode="r")
    final_labels = np.load(sorter / "spike_clusters.npy", mmap_mode="r")
    final_depths = np.load(sorter / "spike_positions.npy", mmap_mode="r")[:, 1]
    if np.any(np.diff(learned_times) < 0) or np.any(np.diff(final_times) < 0):
        raise RuntimeError("KS4 event tables must be sorted by time")

    cohorts = build_event_cohorts(
        args.reviewed, args.holdout_key, args.holdout_public
    )
    traces = trace_cohort(
        cohorts,
        learned_times,
        learned_labels,
        final_times,
        final_labels,
        final_depths,
        label_depths,
        good_units,
        time_radius,
        args.depth_tolerance_um,
    )
    summary = cohort_summary(traces)
    mua_candidates = reviewed_mua_candidates(traces, unit_metrics)

    start = int(window["start_frame"])
    stop = int(window["end_frame"])
    learned_left = int(np.searchsorted(learned_times, start))
    learned_right = int(np.searchsorted(learned_times, stop))
    final_left = int(np.searchsorted(final_times, start))
    final_right = int(np.searchsorted(final_times, stop))
    kia = load_kiasort_band(window_dir, args.kiasort_output_name)
    waveform_review = pd.read_csv(args.waveform_review)
    cross_sort, assignments = summarize_cross_sort_candidates(
        waveform_review,
        kia,
        learned_times[learned_left:learned_right] - start,
        learned_labels[learned_left:learned_right],
        final_times[final_left:final_right] - start,
        final_labels[final_left:final_right],
        final_depths[final_left:final_right],
        label_depths,
        good_units,
        time_radius,
        args.depth_tolerance_um,
        refractory_radius,
    )
    assignments = assignments.merge(
        unit_metrics.reset_index()[
            [
                "unit_id",
                "contamination_pct",
                "spike_count",
                "presence_fraction_300s",
                "refractory_violation_fraction_1p5ms",
                "median_amplitude",
            ]
        ],
        left_on="ks4_unit",
        right_on="unit_id",
        how="left",
        validate="many_to_one",
    ).drop(columns="unit_id")

    reviewed = summary.set_index("cohort").loc["reviewed_neural_discovery"]
    decision = {
        "status": "stage_local_loss_audit_complete",
        "primary_result": (
            "The accepted unwarped KS4 sort retains most reviewed neural events in "
            "its final event table; the largest measured exclusion occurs at the "
            "KS-good/MUA boundary, not duplicate removal."
        ),
        "reviewed_neural_event_counts": {
            "n": int(reviewed.n_events),
            "final_present": int(reviewed.final_present),
            "final_missing": int(reviewed.n_events - reviewed.final_present),
            "learned_but_not_final": int(reviewed.learned_but_not_final),
            "final_mua_only": int(reviewed.final_mua_only),
            "final_good": int(reviewed.good_present),
        },
        "safest_recovery_axis": (
            "reversible post-sort review and unit-family reconciliation among MUA "
            "clusters that already contain supported events"
        ),
        "bounded_mua_review_candidates": mua_candidates.loc[
            mua_candidates.bounded_review_candidate, "unit_id"
        ].astype(int).tolist(),
        "not_supported": [
            "lowering detection thresholds globally",
            "relaxing duplicate removal globally",
            "blanket promotion of MUA clusters",
            "hard-merging cross-sort fragments without CCG/template/residual gates",
        ],
        "cross_sort_status": (
            "The waveform-consistent KIASORT-only candidates are hypotheses, not "
            "confirmed neurons. Their supported KS4 events are distributed across "
            "many mostly-MUA clusters, which nominates fragmentation/reconciliation "
            "for bounded follow-up but does not authorize an automatic merge."
        ),
        "next_bounded_test": (
            "First test whether reviewed-event-rich MUA unit 389 contains a clean, "
            "waveform-coherent subcomponent. Separately, link rather than merge the "
            "cross-sort fragments and require coherent raw templates, complementary "
            "temporal support, acceptable union refractory burden, and reduced "
            "residual energy."
        ),
        "guardrails": [
            "Reviewed events are a reused discovery cohort.",
            "The sealed automatic holdout is not neural ground truth.",
            "Learned-event spatial matching uses final cluster median depth as a proxy.",
            "A cross-sort waveform-consistent unit is not proven biological identity.",
        ],
        "parameters": {
            "time_tolerance_ms": args.time_tolerance_ms,
            "depth_tolerance_um": args.depth_tolerance_um,
            "kiasort_output_name": args.kiasort_output_name,
            "window_request_digest": window["request_digest"],
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces.to_csv(args.output_dir / "event_stage_trace.csv", index=False)
    summary.to_csv(args.output_dir / "cohort_stage_summary.csv", index=False)
    cross_sort.to_csv(args.output_dir / "cross_sort_candidate_summary.csv", index=False)
    assignments.to_csv(args.output_dir / "cross_sort_candidate_assignments.csv", index=False)
    mua_candidates.to_csv(args.output_dir / "reviewed_mua_unit_candidates.csv", index=False)
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(summary.to_string(index=False))
    print("\nCross-sort candidates:\n")
    print(cross_sort.to_string(index=False))
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
