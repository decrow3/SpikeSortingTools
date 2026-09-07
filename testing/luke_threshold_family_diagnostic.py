"""Decompose learned-threshold spike additions in cycle-consistent unit families."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

from testing.luke_threshold_grid_analysis import ARM_PATHS, _input_paths


TRANSITIONS = (
    ("12/10->12/9", "lower_l_at_u12", "12/10", "12/9"),
    ("10/10->10/9", "lower_l_at_u10_high", "10/10", "10/9"),
    ("10/9->10/8", "lower_l_at_u10", "10/9", "10/8"),
    ("9/9->9/8", "lower_l_at_u9", "9/9", "9/8"),
)
CORRESPONDENCE_TOLERANCE_MS = 0.5
COINCIDENCE_TOLERANCE_MS = 0.5
COINCIDENCE_DEPTH_UM = 75.0
COINCIDENCE_SEED = 20250804
LONGITUDINAL_BIN_S = 300.0
TIME_DECILES = 10

# Deliberately prospective, interpretable signature thresholds. The continuous
# metrics and correlations remain primary; these labels are descriptive aids.
HIGH_BASELINE_RETENTION = 0.80
MATERIAL_RV_DELTA = 0.01
MATERIAL_COINCIDENCE_DELTA = 0.01
MATERIAL_AMP_CV_DELTA = 0.05
MATERIAL_LONGITUDINAL_GAIN = 0.05
LOW_AMPLITUDE_MEDIAN_PERCENTILE = 0.25
TEMPLATE_CORE_SEEN_FRACTION = 0.90
TEMPLATE_JS_DIVERGENCE = 0.10


@dataclass
class SortArrays:
    times: np.ndarray
    clusters: np.ndarray
    amplitudes: np.ndarray
    detection_templates: np.ndarray
    depths: np.ndarray


@njit(cache=True)
def exclusive_candidate_core_mask(
    baseline_times: np.ndarray, candidate_times: np.ndarray, tolerance: int
) -> np.ndarray:
    """Mark candidate events used by chronological one-to-one matching."""
    core = np.zeros(candidate_times.size, dtype=np.bool_)
    i = 0
    j = 0
    while i < baseline_times.size and j < candidate_times.size:
        if baseline_times[i] < candidate_times[j] - tolerance:
            i += 1
        elif candidate_times[j] < baseline_times[i] - tolerance:
            j += 1
        else:
            core[j] = True
            i += 1
            j += 1
    return core


@njit(cache=True)
def near_coincident_marks(
    times: np.ndarray,
    clusters: np.ndarray,
    depths: np.ndarray,
    tolerance: int,
    depth_tolerance: float,
) -> np.ndarray:
    """Return marked spikes using the production coincidence definition."""
    marked = np.zeros(times.size, dtype=np.bool_)
    for left in range(times.size):
        right = left + 1
        while right < times.size and times[right] - times[left] <= tolerance:
            if clusters[right] != clusters[left] and abs(depths[right] - depths[left]) <= depth_tolerance:
                marked[left] = True
                marked[right] = True
            right += 1
    return marked


@njit(cache=True)
def queries_near_other_baseline(
    query_times: np.ndarray,
    query_depths: np.ndarray,
    baseline_times: np.ndarray,
    baseline_clusters: np.ndarray,
    baseline_depths: np.ndarray,
    own_baseline_cluster: int,
    tolerance: int,
    depth_tolerance: float,
) -> np.ndarray:
    """Mark query events near a spatially plausible other baseline unit."""
    marked = np.zeros(query_times.size, dtype=np.bool_)
    left = 0
    for i in range(query_times.size):
        while left < baseline_times.size and baseline_times[left] < query_times[i] - tolerance:
            left += 1
        j = left
        while j < baseline_times.size and baseline_times[j] <= query_times[i] + tolerance:
            if (
                baseline_clusters[j] != own_baseline_cluster
                and abs(baseline_depths[j] - query_depths[i]) <= depth_tolerance
            ):
                marked[i] = True
                break
            j += 1
    return marked


def shifted_null_marks(
    sort: SortArrays, *, duration_frames: int, tolerance: int, seed: int
) -> np.ndarray:
    """Reproduce the deterministic cluster-wise circular-shift null per spike."""
    units = np.unique(sort.clusters)
    offsets = np.zeros(int(units.max()) + 1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    offsets[units] = rng.integers(tolerance + 1, duration_frames, size=units.size)
    shifted = (np.asarray(sort.times, dtype=np.int64) + offsets[sort.clusters]) % duration_frames
    order = np.argsort(shifted, kind="stable")
    marked_sorted = near_coincident_marks(
        shifted[order], sort.clusters[order], sort.depths[order], tolerance, COINCIDENCE_DEPTH_UM
    )
    marked = np.empty(marked_sorted.size, dtype=np.bool_)
    marked[order] = marked_sorted
    return marked


def load_sort(arm: str) -> SortArrays:
    curated, _ = _input_paths(arm)
    times = np.load(curated / "spike_times.npy", mmap_mode="r")
    clusters = np.load(curated / "spike_clusters.npy", mmap_mode="r")
    full_st = np.load(curated / "full_st.npy", mmap_mode="r")
    kept = np.load(curated / "kept_spikes.npy", mmap_mode="r")
    positions = np.load(curated / "spike_positions.npy", mmap_mode="r")
    if (
        kept.dtype != np.bool_
        or kept.shape[0] != full_st.shape[0]
        or not bool(np.all(kept))
        or full_st.shape[0] != times.size
        or clusters.size != times.size
        or positions.shape[0] != times.size
    ):
        raise RuntimeError(f"{arm}: expected aligned, all-retained curated arrays")
    if np.any(np.diff(times) < 0) or not np.array_equal(times, full_st[:, 0].astype(np.int64)):
        raise RuntimeError(f"{arm}: curated time alignment/order failed")
    return SortArrays(
        times=times,
        clusters=clusters,
        amplitudes=full_st[:, 2],
        detection_templates=full_st[:, 1].astype(np.int64),
        depths=positions[:, 1],
    )


def cluster_indices(clusters: np.ndarray, wanted: set[int]) -> dict[int, np.ndarray]:
    return {cluster: np.flatnonzero(clusters == cluster) for cluster in sorted(wanted)}


def amplitude_features(baseline_amp: np.ndarray, added_amp: np.ndarray) -> dict[str, float]:
    if baseline_amp.size == 0 or added_amp.size == 0:
        return {
            "added_amplitude_median_baseline_percentile": np.nan,
            "added_amplitude_below_baseline_q10_fraction": np.nan,
            "added_amplitude_below_baseline_q25_fraction": np.nan,
            "added_amplitude_below_baseline_median_fraction": np.nan,
        }
    ordered = np.sort(np.asarray(baseline_amp, dtype=float))
    percentiles = np.searchsorted(ordered, np.asarray(added_amp, dtype=float), side="right") / ordered.size
    return {
        "added_amplitude_median_baseline_percentile": float(np.median(percentiles)),
        "added_amplitude_below_baseline_q10_fraction": float(np.mean(percentiles <= 0.10)),
        "added_amplitude_below_baseline_q25_fraction": float(np.mean(percentiles <= 0.25)),
        "added_amplitude_below_baseline_median_fraction": float(np.mean(percentiles <= 0.50)),
    }


def template_features(core_templates: np.ndarray, added_templates: np.ndarray) -> dict[str, float]:
    if core_templates.size == 0 or added_templates.size == 0:
        return {
            "added_template_seen_in_core_fraction": np.nan,
            "added_vs_core_template_js_divergence": np.nan,
            "core_detection_template_count": int(np.unique(core_templates).size),
            "added_detection_template_count": int(np.unique(added_templates).size),
        }
    ids = np.union1d(core_templates, added_templates)
    core_count = np.zeros(ids.size, dtype=float)
    added_count = np.zeros(ids.size, dtype=float)
    core_unique, core_n = np.unique(core_templates, return_counts=True)
    added_unique, added_n = np.unique(added_templates, return_counts=True)
    core_count[np.searchsorted(ids, core_unique)] = core_n
    added_count[np.searchsorted(ids, added_unique)] = added_n
    return {
        "added_template_seen_in_core_fraction": float(np.mean(np.isin(added_templates, core_unique))),
        "added_vs_core_template_js_divergence": float(
            jensenshannon(core_count / core_count.sum(), added_count / added_count.sum(), base=2.0) ** 2
        ),
        "core_detection_template_count": int(core_unique.size),
        "added_detection_template_count": int(added_unique.size),
    }


def distribution_js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    """Base-2 Jensen-Shannon divergence for two aligned count vectors."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.sum() <= 0 or second.sum() <= 0:
        return np.nan
    return float(jensenshannon(first / first.sum(), second / second.sum(), base=2.0) ** 2)


def classify_signature(row: dict[str, float]) -> str:
    if row["baseline_retention"] < HIGH_BASELINE_RETENTION:
        return "unstable_correspondence"
    degradations = sum(
        (
            row["refractory_change"] > MATERIAL_RV_DELTA,
            row["unit_coincidence_excess_change"] > MATERIAL_COINCIDENCE_DELTA,
            row["amplitude_cv_change"] > MATERIAL_AMP_CV_DELTA,
        )
    )
    longitudinal = (
        row["presence_change"] >= MATERIAL_LONGITUDINAL_GAIN
        or row["active_lifetime_change"] >= MATERIAL_LONGITUDINAL_GAIN
        or row["added_in_baseline_empty_bins_fraction"] >= MATERIAL_LONGITUDINAL_GAIN
    )
    low_amplitude = (
        row["added_amplitude_median_baseline_percentile"] <= LOW_AMPLITUDE_MEDIAN_PERCENTILE
    )
    template_consistent = (
        row["added_template_seen_in_core_fraction"] >= TEMPLATE_CORE_SEEN_FRACTION
        and row["added_vs_core_template_js_divergence"] <= TEMPLATE_JS_DIVERGENCE
    )
    if longitudinal and low_amplitude and template_consistent and degradations == 0:
        return "recovery_compatible"
    if not longitudinal and not low_amplitude and degradations >= 1:
        return "broadening_compatible"
    return "mixed_or_indeterminate"


def safe_fraction(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if mask.size else np.nan


def analyze_transition(
    *,
    analysis_root: Path,
    families: pd.DataFrame,
    transition: str,
    edge: str,
    baseline_arm: str,
    candidate_arm: str,
    duration_s: float,
    sampling_frequency_hz: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    baseline = load_sort(baseline_arm)
    candidate = load_sort(candidate_arm)
    tolerance = int(round(CORRESPONDENCE_TOLERANCE_MS * 1e-3 * sampling_frequency_hz))
    duration_frames = int(round(duration_s * sampling_frequency_hz))
    observed_baseline = near_coincident_marks(
        baseline.times, baseline.clusters, baseline.depths, tolerance, COINCIDENCE_DEPTH_UM
    )
    observed_candidate = near_coincident_marks(
        candidate.times, candidate.clusters, candidate.depths, tolerance, COINCIDENCE_DEPTH_UM
    )
    null_baseline = shifted_null_marks(
        baseline, duration_frames=duration_frames, tolerance=tolerance, seed=COINCIDENCE_SEED
    )
    null_candidate = shifted_null_marks(
        candidate, duration_frames=duration_frames, tolerance=tolerance, seed=COINCIDENCE_SEED + 1
    )

    comparison = analysis_root / "comparisons" / edge
    primary = pd.read_csv(comparison / "primary_matches.csv").set_index(
        ["baseline_cluster", "candidate_cluster"]
    )
    baseline_metrics = pd.read_csv(comparison / "unit_metrics_baseline.csv").set_index("cluster_id")
    candidate_metrics = pd.read_csv(comparison / "unit_metrics_candidate.csv").set_index("cluster_id")
    baseline_col = f"cluster_{baseline_arm.replace('/', '_')}"
    candidate_col = f"cluster_{candidate_arm.replace('/', '_')}"
    wanted_baseline = set(families[baseline_col].astype(int))
    wanted_candidate = set(families[candidate_col].astype(int))
    baseline_indices = cluster_indices(baseline.clusters, wanted_baseline)
    candidate_indices = cluster_indices(candidate.clusters, wanted_candidate)

    rows = []
    time_rows = []
    longitudinal_edges = np.arange(0.0, duration_s + LONGITUDINAL_BIN_S, LONGITUDINAL_BIN_S)
    if longitudinal_edges[-1] < duration_s:
        longitudinal_edges = np.r_[longitudinal_edges, duration_s]
    decile_edges = np.linspace(0.0, duration_s, TIME_DECILES + 1)
    for family in families.itertuples(index=False):
        family_id = int(family.family_id)
        baseline_cluster = int(getattr(family, baseline_col))
        candidate_cluster = int(getattr(family, candidate_col))
        bi = baseline_indices[baseline_cluster]
        ci = candidate_indices[candidate_cluster]
        baseline_times = np.asarray(baseline.times[bi], dtype=np.int64)
        candidate_times = np.asarray(candidate.times[ci], dtype=np.int64)
        core_local = exclusive_candidate_core_mask(baseline_times, candidate_times, tolerance)
        added_local = ~core_local
        added_indices = ci[added_local]
        pair = primary.loc[(baseline_cluster, candidate_cluster)]
        if int(core_local.sum()) != int(pair.matched_events):
            raise RuntimeError(f"{transition} family {family_id}: exclusive match count mismatch")

        other_baseline = queries_near_other_baseline(
            np.asarray(candidate.times[added_indices], dtype=np.int64),
            np.asarray(candidate.depths[added_indices], dtype=float),
            baseline.times,
            baseline.clusters,
            baseline.depths,
            baseline_cluster,
            tolerance,
            COINCIDENCE_DEPTH_UM,
        )
        b_obs = safe_fraction(observed_baseline[bi])
        b_null = safe_fraction(null_baseline[bi])
        c_obs = safe_fraction(observed_candidate[ci])
        c_null = safe_fraction(null_candidate[ci])
        added_obs = safe_fraction(observed_candidate[added_indices])
        added_null = safe_fraction(null_candidate[added_indices])

        baseline_longitudinal = np.histogram(baseline_times / sampling_frequency_hz, longitudinal_edges)[0]
        core_longitudinal = np.histogram(
            candidate_times[core_local] / sampling_frequency_hz, longitudinal_edges
        )[0]
        added_longitudinal = np.histogram(
            np.asarray(candidate.times[added_indices]) / sampling_frequency_hz, longitudinal_edges
        )[0]
        added_total = int(added_local.sum())
        added_empty = int(added_longitudinal[baseline_longitudinal == 0].sum())
        added_deciles = np.histogram(
            np.asarray(candidate.times[added_indices]) / sampling_frequency_hz, decile_edges
        )[0]
        baseline_deciles = np.histogram(baseline_times / sampling_frequency_hz, decile_edges)[0]
        core_deciles = np.histogram(candidate_times[core_local] / sampling_frequency_hz, decile_edges)[0]
        for decile, count in enumerate(added_deciles, start=1):
            time_rows.append(
                {
                    "transition": transition,
                    "family_id": family_id,
                    "time_decile": decile,
                    "baseline_spikes": int(baseline_deciles[decile - 1]),
                    "core_spikes": int(core_deciles[decile - 1]),
                    "added_spikes": int(count),
                    "added_spike_fraction": count / added_total if added_total else np.nan,
                }
            )

        bm = baseline_metrics.loc[baseline_cluster]
        cm = candidate_metrics.loc[candidate_cluster]
        row = {
            "transition": transition,
            "family_id": family_id,
            "baseline_arm": baseline_arm,
            "candidate_arm": candidate_arm,
            "baseline_cluster": baseline_cluster,
            "candidate_cluster": candidate_cluster,
            "baseline_spikes": len(bi),
            "candidate_spikes": len(ci),
            "core_spikes": int(core_local.sum()),
            "candidate_only_spikes": added_total,
            "baseline_retention": float(core_local.sum() / len(bi)),
            "candidate_retention": float(core_local.sum() / len(ci)),
            "added_spike_fraction": float(added_total / len(ci)),
            "added_near_other_baseline_unit_fraction": safe_fraction(other_baseline),
            "added_candidate_coincidence_observed_fraction": added_obs,
            "added_candidate_coincidence_null_fraction": added_null,
            "added_candidate_coincidence_excess": added_obs - added_null,
            "baseline_unit_coincidence_observed_fraction": b_obs,
            "baseline_unit_coincidence_null_fraction": b_null,
            "baseline_unit_coincidence_excess": b_obs - b_null,
            "candidate_unit_coincidence_observed_fraction": c_obs,
            "candidate_unit_coincidence_null_fraction": c_null,
            "candidate_unit_coincidence_excess": c_obs - c_null,
            "unit_coincidence_excess_change": (c_obs - c_null) - (b_obs - b_null),
            "refractory_change": float(
                cm.refractory_violation_fraction_1_5ms - bm.refractory_violation_fraction_1_5ms
            ),
            "amplitude_cv_change": float(cm.amplitude_cv - bm.amplitude_cv),
            "presence_change": float(cm.presence_fraction - bm.presence_fraction),
            "active_lifetime_change": float(cm.active_lifetime_fraction - bm.active_lifetime_fraction),
            "completeness_change_pp": float(
                getattr(family, f"completeness_{candidate_arm.replace('/', '_')}_pct")
                - getattr(family, f"completeness_{baseline_arm.replace('/', '_')}_pct")
            ),
            "added_in_baseline_empty_bins_fraction": added_empty / added_total if added_total else np.nan,
            "added_time_bin_cv": (
                float(np.std(added_longitudinal) / np.mean(added_longitudinal))
                if added_total and np.mean(added_longitudinal)
                else np.nan
            ),
            "added_vs_baseline_time_js_divergence": distribution_js_divergence(
                baseline_longitudinal, added_longitudinal
            ),
            "added_vs_core_time_js_divergence": distribution_js_divergence(
                core_longitudinal, added_longitudinal
            ),
        }
        row.update(amplitude_features(baseline.amplitudes[bi], candidate.amplitudes[added_indices]))
        row.update(
            template_features(
                candidate.detection_templates[ci[core_local]], candidate.detection_templates[added_indices]
            )
        )
        row["signature"] = classify_signature(row)
        rows.append(row)

    aggregate_validation = {
        "transition": transition,
        "baseline_observed": safe_fraction(observed_baseline),
        "baseline_null": safe_fraction(null_baseline),
        "candidate_observed": safe_fraction(observed_candidate),
        "candidate_null": safe_fraction(null_candidate),
    }
    expected = pd.read_csv(comparison / "guardrail_summary.csv").set_index("metric").loc[
        "chance_aware_near_coincident_excess"
    ]
    aggregate_validation["baseline_excess_difference_from_saved"] = (
        aggregate_validation["baseline_observed"]
        - aggregate_validation["baseline_null"]
        - float(expected.baseline)
    )
    aggregate_validation["candidate_excess_difference_from_saved"] = (
        aggregate_validation["candidate_observed"]
        - aggregate_validation["candidate_null"]
        - float(expected.candidate)
    )
    if (
        abs(aggregate_validation["baseline_excess_difference_from_saved"]) > 1e-12
        or abs(aggregate_validation["candidate_excess_difference_from_saved"]) > 1e-12
    ):
        raise RuntimeError(f"{transition}: aggregate chance-aware coincidence failed reconciliation")
    return pd.DataFrame(rows), pd.DataFrame(time_rows), aggregate_validation


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    for transition, frame in metrics.groupby("transition", sort=False):
        added = frame.candidate_only_spikes.sum()
        summary_rows.append(
            {
                "transition": transition,
                "families": len(frame),
                "baseline_spikes": int(frame.baseline_spikes.sum()),
                "candidate_spikes": int(frame.candidate_spikes.sum()),
                "core_spikes": int(frame.core_spikes.sum()),
                "candidate_only_spikes": int(added),
                "aggregate_added_spike_fraction": added / frame.candidate_spikes.sum(),
                "median_family_added_spike_fraction": frame.added_spike_fraction.median(),
                "added_near_other_baseline_unit_fraction": (
                    np.average(
                        frame.added_near_other_baseline_unit_fraction,
                        weights=frame.candidate_only_spikes,
                    )
                ),
                "added_candidate_coincidence_observed_fraction": np.average(
                    frame.added_candidate_coincidence_observed_fraction,
                    weights=frame.candidate_only_spikes,
                ),
                "added_candidate_coincidence_null_fraction": np.average(
                    frame.added_candidate_coincidence_null_fraction,
                    weights=frame.candidate_only_spikes,
                ),
                "added_amplitude_below_baseline_q25_fraction": np.average(
                    frame.added_amplitude_below_baseline_q25_fraction,
                    weights=frame.candidate_only_spikes,
                ),
                "added_in_baseline_empty_bins_fraction": np.average(
                    frame.added_in_baseline_empty_bins_fraction,
                    weights=frame.candidate_only_spikes,
                ),
                "median_added_amplitude_baseline_percentile": frame.added_amplitude_median_baseline_percentile.median(),
                "median_template_core_seen_fraction": frame.added_template_seen_in_core_fraction.median(),
                "median_template_js_divergence": frame.added_vs_core_template_js_divergence.median(),
                "median_refractory_change": frame.refractory_change.median(),
                "median_unit_coincidence_excess_change": frame.unit_coincidence_excess_change.median(),
                "median_amplitude_cv_change": frame.amplitude_cv_change.median(),
                "median_presence_change": frame.presence_change.median(),
                "median_active_lifetime_change": frame.active_lifetime_change.median(),
                "median_added_vs_baseline_time_js_divergence": frame.added_vs_baseline_time_js_divergence.median(),
                "median_added_vs_core_time_js_divergence": frame.added_vs_core_time_js_divergence.median(),
            }
        )
    outcomes = (
        "refractory_change",
        "unit_coincidence_excess_change",
        "amplitude_cv_change",
        "presence_change",
        "active_lifetime_change",
        "completeness_change_pp",
    )
    correlation_rows = []
    for transition, frame in metrics.groupby("transition", sort=False):
        for outcome in outcomes:
            valid = frame[["added_spike_fraction", outcome]].dropna()
            rho = spearmanr(valid.added_spike_fraction, valid[outcome]).statistic if len(valid) >= 3 else np.nan
            correlation_rows.append(
                {"transition": transition, "outcome": outcome, "n": len(valid), "spearman_rho": rho}
            )
    centered = metrics.copy()
    for column in ("added_spike_fraction", *outcomes):
        centered[column] = centered[column] - centered.groupby("transition")[column].transform("median")
    for outcome in outcomes:
        valid = centered[["added_spike_fraction", outcome]].dropna()
        correlation_rows.append(
            {
                "transition": "pooled_within_transition_centered",
                "outcome": outcome,
                "n": len(valid),
                "spearman_rho": spearmanr(valid.added_spike_fraction, valid[outcome]).statistic,
            }
        )
    classification = (
        metrics.groupby(["transition", "signature"], sort=False)
        .size()
        .rename("families")
        .reset_index()
    )
    return pd.DataFrame(summary_rows), pd.DataFrame(correlation_rows), classification


def plot_results(metrics: pd.DataFrame, classification: pd.DataFrame, output_dir: Path) -> None:
    colors = {
        "12/10->12/9": "#355C7D",
        "10/10->10/9": "#C06C84",
        "10/9->10/8": "#F67280",
        "9/9->9/8": "#F8B195",
    }
    outcomes = (
        ("refractory_change", "RV fraction change"),
        ("unit_coincidence_excess_change", "Chance-aware coincidence change"),
        ("amplitude_cv_change", "Amplitude CV change"),
        ("presence_change", "Presence fraction change"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (field, label) in zip(axes.flat, outcomes):
        for transition, frame in metrics.groupby("transition", sort=False):
            ax.scatter(
                frame.added_spike_fraction,
                frame[field],
                s=24,
                alpha=0.72,
                color=colors[transition],
                label=transition,
                edgecolors="none",
            )
        ax.axhline(0, color="#333333", lw=0.8)
        ax.set_xlabel("Added candidate spike fraction")
        ax.set_ylabel(label)
        ax.grid(color="#DDDDDD", lw=0.6, alpha=0.7)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Added-spike fraction versus independent unit-quality changes")
    fig.savefig(output_dir / "added_fraction_relationships.png", dpi=180)
    plt.close(fig)

    pivot = classification.pivot(index="transition", columns="signature", values="families").fillna(0)
    order = [name for name, *_ in TRANSITIONS]
    pivot = pivot.reindex(order)
    palette = {
        "recovery_compatible": "#355C7D",
        "mixed_or_indeterminate": "#D6B35A",
        "broadening_compatible": "#C06C84",
        "unstable_correspondence": "#AAAAAA",
    }
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    bottom = np.zeros(len(pivot))
    for column in (
        "recovery_compatible",
        "mixed_or_indeterminate",
        "broadening_compatible",
        "unstable_correspondence",
    ):
        values = pivot[column].to_numpy() if column in pivot else np.zeros(len(pivot))
        ax.bar(pivot.index, values, bottom=bottom, label=column.replace("_", " "), color=palette[column])
        bottom += values
    ax.set_ylabel("Cycle-consistent families")
    ax.set_title("Prospective signature classification")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(axis="y", color="#DDDDDD", lw=0.6, alpha=0.7)
    fig.savefig(output_dir / "signature_classification.png", dpi=180)
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--recording-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    families = pd.read_csv(args.families)
    if len(families) != 50 or families.family_id.nunique() != 50:
        raise RuntimeError("expected exactly 50 unique cycle-consistent eligible families")
    recording = json.loads(args.recording_manifest.read_text())
    sampling_frequency_hz = float(recording["sampling_frequency_hz"])
    duration_s = float(recording.get("duration_s") or recording["num_samples"] / sampling_frequency_hz)

    frames = []
    time_frames = []
    validations = []
    for transition in TRANSITIONS:
        metrics, time_rows, validation = analyze_transition(
            analysis_root=args.analysis_root,
            families=families,
            transition=transition[0],
            edge=transition[1],
            baseline_arm=transition[2],
            candidate_arm=transition[3],
            duration_s=duration_s,
            sampling_frequency_hz=sampling_frequency_hz,
        )
        frames.append(metrics)
        time_frames.append(time_rows)
        validations.append(validation)
    metrics = pd.concat(frames, ignore_index=True)
    time_rows = pd.concat(time_frames, ignore_index=True)
    transition_summary, correlations, classification = summarize(metrics)
    plot_results(metrics, classification, output_dir)

    metrics.to_csv(output_dir / "family_transition_metrics.csv", index=False)
    transition_summary.to_csv(output_dir / "transition_summary.csv", index=False)
    correlations.to_csv(output_dir / "added_fraction_correlations.csv", index=False)
    classification.to_csv(output_dir / "signature_classification.csv", index=False)
    time_rows.to_csv(output_dir / "added_spike_time_deciles.csv", index=False)
    source_files = {
        "families": {"path": str(args.families.resolve()), "sha256": sha256(args.families)},
        "recording_manifest": {
            "path": str(args.recording_manifest.resolve()),
            "sha256": sha256(args.recording_manifest),
        },
        "threshold_grid_summary": {
            "path": str((args.analysis_root / "summary.json").resolve()),
            "sha256": sha256(args.analysis_root / "summary.json"),
        },
    }
    payload = {
        "schema_version": "luke-threshold-family-diagnostic-v1",
        "status": "complete",
        "families": len(families),
        "transitions": [row[0] for row in TRANSITIONS],
        "rows": len(metrics),
        "definitions": {
            "added_spike_fraction": "1 - reciprocal-primary candidate retention",
            "core_spikes": "candidate spikes used by exclusive chronological matching to the family baseline within 0.5 ms",
            "candidate_only_spikes": "candidate family spikes not used by that exclusive core match",
            "chance_aware_coincidence": "marked-spike fraction within 0.5 ms and 75 um of another unit minus a deterministic cluster-wise circular-shift null",
            "amplitude": "sorter-native production QC amplitude from full_st[kept_spikes][:, 2]",
            "template_composition": "within-candidate detection-template ID composition; this is not waveform correlation",
        },
        "signature_thresholds": {
            "high_baseline_retention": HIGH_BASELINE_RETENTION,
            "material_rv_delta": MATERIAL_RV_DELTA,
            "material_coincidence_delta": MATERIAL_COINCIDENCE_DELTA,
            "material_amplitude_cv_delta": MATERIAL_AMP_CV_DELTA,
            "material_longitudinal_gain": MATERIAL_LONGITUDINAL_GAIN,
            "low_amplitude_median_percentile": LOW_AMPLITUDE_MEDIAN_PERCENTILE,
            "template_core_seen_fraction": TEMPLATE_CORE_SEEN_FRACTION,
            "template_js_divergence": TEMPLATE_JS_DIVERGENCE,
        },
        "aggregate_coincidence_validation": validations,
        "source_files": source_files,
        "limitations": [
            "Cycle-consistent reciprocal-primary families are not isolated degree-one graph components.",
            "The same biological neurons are not known; correspondence is spike-train identity evidence only.",
            "Template composition uses detection-template IDs and does not substitute for raw-waveform similarity.",
            "Completeness changes exist only where both family arms are amplitude-measurable.",
            "Signature classes are prospective descriptive rules; continuous metrics and correlations are primary.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
