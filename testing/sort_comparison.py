"""Generic, exclusive comparison of long spike-sort outputs.

The calculations are extracted from ``luke_full_session_compare`` and operate on
normalized arrays/data frames.  Paths, cluster IDs, and labels are not Luke
specific.  Correspondence means spike-train correspondence, not biological
identity.
"""

from __future__ import annotations

import json
import io
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from numba import njit

from pipeline.config import fingerprint
from testing.development_strip import classify_unit_depths


COMPARISON_SCHEMA = "long-sort-comparison-v2"
EDGE_COLUMNS = [
    "baseline_cluster", "candidate_cluster", "matched_events", "baseline_events",
    "candidate_events", "unmatched_baseline_events", "unmatched_candidate_events",
    "baseline_retention", "candidate_retention", "jaccard", "primary_match",
]


def load_comparison_inputs(
    name: str,
    curated_output: Path | str,
    qc_dir: Path | str,
    *,
    sampling_frequency_hz: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load identity-bound curated arrays and corrected amplitude windows."""
    from testing.luke_amplitude_dropout_audit import (
        build_windows_table,
        curated_arrays_from_raw,
        read_cached_truncation_qc,
        read_curated_arrays,
    )

    curated = Path(curated_output)
    raw, hashes = read_curated_arrays(curated)
    arrays = curated_arrays_from_raw(name, raw)
    label_bytes = (curated / "cluster_KSLabel.tsv").read_bytes()
    labels_frame = pd.read_csv(io.BytesIO(label_bytes), sep="\t")
    label_column = next(column for column in labels_frame if column != "cluster_id")
    labels = dict(
        zip(labels_frame.cluster_id.astype(int), labels_frame[label_column].astype(str).str.lower())
    )
    cached, qc_hash = read_cached_truncation_qc(name, Path(qc_dir))
    windows = build_windows_table(arrays, cached, sampling_frequency_hz)
    sort = {
        "st": arrays.times,
        "cl": arrays.clusters,
        "amp": arrays.amplitudes,
        "labels": labels,
        "identity_digest": fingerprint({
            "curated": str(curated.resolve()), "files": hashes,
            "labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
        }),
    }
    positions_path = curated / "spike_positions.npy"
    if positions_path.is_file():
        positions = np.load(positions_path, mmap_mode="r")
        if positions.ndim != 2 or positions.shape[0] != len(arrays.times) or positions.shape[1] < 2:
            raise ValueError(f"invalid spike_positions.npy in {curated}")
        sort["depth"] = np.asarray(positions[:, 1])
    qc = {"amplitude_windows": windows, "request_digest": qc_hash}
    return sort, qc


@njit(cache=True)
def coincidence_counts(at, ac, bt, bc, na, nb, tolerance):
    """Enumerate temporally possible cluster-pair edges only."""
    counts = np.zeros((na, nb), dtype=np.int64)
    left = 0
    for i in range(len(at)):
        while left < len(bt) and bt[left] < at[i] - tolerance:
            left += 1
        j = left
        while j < len(bt) and bt[j] <= at[i] + tolerance:
            counts[ac[i], bc[j]] += 1
            j += 1
    return counts


@njit(cache=True)
def exclusive_count(a, b, tolerance):
    """Greedy chronological one-to-one matching for one cluster pair."""
    i = j = count = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j] - tolerance:
            i += 1
        elif b[j] < a[i] - tolerance:
            j += 1
        else:
            count += 1
            i += 1
            j += 1
    return count


def correspondence(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray], tolerance: int,
                   minimum_overlap: float = 0.1, primary_retention: float = 0.5) -> pd.DataFrame:
    """Return full exclusive correspondence graph with reciprocal primaries."""
    at_raw, ac_raw = np.asarray(a["st"]).reshape(-1), np.asarray(a["cl"]).reshape(-1)
    bt_raw, bc_raw = np.asarray(b["st"]).reshape(-1), np.asarray(b["cl"]).reshape(-1)
    if len(at_raw) != len(ac_raw) or len(bt_raw) != len(bc_raw):
        raise ValueError("each sort must have one cluster label per spike time")
    if np.any(np.diff(at_raw) < 0) or np.any(np.diff(bt_raw) < 0):
        raise ValueError("spike times must be globally sorted")
    aid, ac = np.unique(ac_raw, return_inverse=True)
    bid, bc = np.unique(bc_raw, return_inverse=True)
    an, bn = np.bincount(ac), np.bincount(bc)
    possible = coincidence_counts(at_raw, ac, bt_raw, bc, len(aid), len(bid), tolerance)
    ats = np.split(at_raw[np.argsort(ac, kind="stable")], np.cumsum(an)[:-1])
    bts = np.split(bt_raw[np.argsort(bc, kind="stable")], np.cumsum(bn)[:-1])
    rows = []
    for i, j in np.argwhere(possible >= minimum_overlap * np.minimum(an[:, None], bn[None, :])):
        matched = exclusive_count(ats[i], bts[j], tolerance)
        if matched < minimum_overlap * min(an[i], bn[j]):
            continue
        rows.append({
            "baseline_cluster": int(aid[i]),
            "candidate_cluster": int(bid[j]),
            "matched_events": int(matched),
            "baseline_events": int(an[i]),
            "candidate_events": int(bn[j]),
            "unmatched_baseline_events": int(an[i] - matched),
            "unmatched_candidate_events": int(bn[j] - matched),
            "baseline_retention": matched / an[i],
            "candidate_retention": matched / bn[j],
            "jaccard": matched / (an[i] + bn[j] - matched),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    abest = frame.groupby("baseline_cluster").jaccard.transform("max")
    bbest = frame.groupby("candidate_cluster").jaccard.transform("max")
    am = frame.jaccard == abest
    bm = frame.jaccard == bbest
    au = am.groupby(frame.baseline_cluster).transform("sum") == 1
    bu = bm.groupby(frame.candidate_cluster).transform("sum") == 1
    frame["primary_match"] = (
        am & bm & au & bu
        & (frame.baseline_retention >= primary_retention)
        & (frame.candidate_retention >= primary_retention)
    )
    return frame[EDGE_COLUMNS]


def common_time(a, b):
    """Intersect non-overlapping [start, end, value] rows without filling gaps."""
    a, b = np.asarray(a, dtype=float).reshape(-1, 3), np.asarray(b, dtype=float).reshape(-1, 3)
    for values in (a, b):
        if len(values) and (
            not np.isfinite(values).all()
            or np.any(values[:, 1] < values[:, 0])
            or np.any(values[1:, 0] < values[:-1, 1])
        ):
            raise ValueError("invalid or overlapping fit intervals")
    i = j = 0
    seconds = weighted = 0.0
    pieces = []
    while i < len(a) and j < len(b):
        lo, hi = max(a[i, 0], b[j, 0]), min(a[i, 1], b[j, 1])
        if hi > lo:
            difference = a[i, 2] - b[j, 2]
            seconds += hi - lo
            weighted += (hi - lo) * difference
            pieces.append((lo, hi, a[i, 2], b[j, 2]))
        if a[i, 1] <= b[j, 1]:
            i += 1
        else:
            j += 1
    return seconds, weighted / seconds if seconds else np.nan, pieces


@njit(cache=True)
def _near_coincident_fraction(times, clusters, depths, tolerance, depth_tolerance):
    marked = np.zeros(len(times), dtype=np.uint8)
    for left in range(len(times)):
        right = left + 1
        while right < len(times) and times[right] - times[left] <= tolerance:
            if clusters[right] != clusters[left] and abs(depths[right] - depths[left]) <= depth_tolerance:
                marked[left] = 1
                marked[right] = 1
            right += 1
    return float(marked.sum() / len(times)) if len(times) else np.nan


def chance_aware_coincidence(sort: Mapping[str, np.ndarray], *, duration_frames: int,
                             tolerance_frames: int, depth_tolerance_um: float,
                             seed: int) -> dict[str, float | bool]:
    """Observed marked-spike burden and deterministic cluster-shift null."""
    required = {"st", "cl", "depth"}
    if not required <= set(sort):
        return {"available": False, "observed": np.nan, "shift_null": np.nan, "excess": np.nan}
    times = np.asarray(sort["st"], dtype=np.int64)
    clusters = np.asarray(sort["cl"], dtype=np.int64)
    depths = np.asarray(sort["depth"], dtype=float)
    if not (len(times) == len(clusters) == len(depths)) or np.any(np.diff(times) < 0):
        raise ValueError("coincidence arrays must be aligned and time sorted")
    observed = _near_coincident_fraction(times, clusters, depths, tolerance_frames, depth_tolerance_um)
    shifted = times.copy()
    rng = np.random.default_rng(seed)
    if duration_frames <= tolerance_frames + 1:
        raise ValueError("duration is too short for the coincidence null")
    for unit in np.unique(clusters):
        keep = clusters == unit
        offset = int(rng.integers(tolerance_frames + 1, duration_frames))
        shifted[keep] = (shifted[keep] + offset) % duration_frames
    order = np.argsort(shifted, kind="stable")
    null = _near_coincident_fraction(
        shifted[order], clusters[order], depths[order], tolerance_frames, depth_tolerance_um
    )
    return {"available": True, "observed": observed, "shift_null": null, "excess": observed - null}


def _unit_metrics(sort: Mapping[str, Any], *, fs: float, duration_s: float,
                  longitudinal_bin_s: float,
                  spatial_region: Mapping[str, Any] | None) -> pd.DataFrame:
    times = np.asarray(sort["st"]).reshape(-1)
    clusters = np.asarray(sort["cl"]).reshape(-1)
    labels = sort.get("labels", {})
    rows = []
    for unit in np.unique(clusters):
        keep = clusters == unit
        unit_times = times[keep]
        row = {
            "cluster_id": int(unit),
            "label": str(labels.get(int(unit), "unknown")),
            "spike_count": int(keep.sum()),
            "mean_rate_hz": float(keep.sum() / duration_s),
            "refractory_violation_fraction_1_5ms": (
                float(np.mean(np.diff(unit_times) < round(0.0015 * fs))) if len(unit_times) > 1 else np.nan
            ),
            "active_lifetime_fraction": (
                float((unit_times[-1] - unit_times[0]) / (duration_s * fs)) if len(unit_times) > 1 else 0.0
            ),
        }
        edges = np.arange(0.0, duration_s + longitudinal_bin_s, longitudinal_bin_s)
        if len(edges) < 2 or edges[-1] < duration_s:
            edges = np.r_[edges, duration_s]
        counts = np.histogram(unit_times / fs, bins=edges)[0]
        rates = counts / np.diff(edges)
        row["presence_fraction"] = float(np.mean(counts > 0)) if len(counts) else np.nan
        row["firing_rate_cv"] = float(np.std(rates) / np.mean(rates)) if len(rates) and np.mean(rates) else np.nan
        if "amp" in sort:
            amplitudes = np.asarray(sort["amp"])[keep].astype(float)
            row["amplitude_cv"] = float(np.std(amplitudes) / np.mean(amplitudes)) if np.mean(amplitudes) else np.nan
            epoch = np.minimum((unit_times / (duration_s * fs) * 3).astype(int), 2)
            epoch_amp = [float(np.median(amplitudes[epoch == index])) if np.any(epoch == index) else np.nan for index in range(3)]
            row.update(
                early_amplitude=epoch_amp[0], middle_amplitude=epoch_amp[1], late_amplitude=epoch_amp[2],
                late_early_amplitude_ratio=(epoch_amp[2] / epoch_amp[0] if epoch_amp[0] and np.isfinite(epoch_amp[2]) else np.nan),
            )
        else:
            row.update(amplitude_cv=np.nan, early_amplitude=np.nan, middle_amplitude=np.nan,
                       late_amplitude=np.nan, late_early_amplitude_ratio=np.nan)
        if "depth" in sort:
            depths = np.asarray(sort["depth"])[keep].astype(float)
            row["median_depth_um"] = float(np.median(depths))
            row["depth_excursion_p95_p5_um"] = float(np.quantile(depths, 0.95) - np.quantile(depths, 0.05))
            if spatial_region is not None:
                spike_classes = classify_unit_depths(
                    depths,
                    processing_depth_um=spatial_region["processing_depth_um"],
                    scoring_depth_um=spatial_region["scoring_depth_um"],
                    minimum_edge_exclusion_um=spatial_region["minimum_edge_exclusion_um"],
                )
                row["edge_spike_fraction"] = float(np.mean(spike_classes == "edge"))
            else:
                row["edge_spike_fraction"] = np.nan
        else:
            row.update(median_depth_um=np.nan, depth_excursion_p95_p5_um=np.nan, edge_spike_fraction=np.nan)
        # Feature/waveform continuity remains explicit when saved features are unavailable.
        row["early_middle_waveform_similarity"] = np.nan
        row["middle_late_waveform_similarity"] = np.nan
        rows.append(row)
    frame = pd.DataFrame(rows)
    if spatial_region is None or frame.empty or frame.median_depth_um.isna().any():
        frame["spatial_class"] = "unavailable" if spatial_region is not None else "all"
    else:
        frame["spatial_class"] = classify_unit_depths(
            frame.median_depth_um.to_numpy(),
            processing_depth_um=spatial_region["processing_depth_um"],
            scoring_depth_um=spatial_region["scoring_depth_um"],
            minimum_edge_exclusion_um=spatial_region["minimum_edge_exclusion_um"],
        )
    return frame


def _valid_windows(windows: pd.DataFrame, cluster: int) -> np.ndarray:
    required = {"cluster_id", "status", "start_s", "end_s", "missing_pct"}
    if not required <= set(windows.columns):
        raise ValueError(f"amplitude windows missing {sorted(required - set(windows.columns))}")
    return (
        windows[(windows.cluster_id == cluster) & (windows.status == "finite_interior")]
        .sort_values("start_s")[["start_s", "end_s", "missing_pct"]]
        .to_numpy()
    )


def compare_sorts(
    baseline_sort: Mapping[str, Any],
    candidate_sort: Mapping[str, Any],
    baseline_qc: Mapping[str, Any],
    candidate_qc: Mapping[str, Any],
    config: Mapping[str, Any],
    spatial_region: Mapping[str, Any] | None = None,
    *,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Compare normalized long sorts and optionally write the stable artifact set."""
    fs = float(config["sampling_frequency_hz"])
    duration_s = float(config["duration_s"])
    duration_frames = int(round(duration_s * fs))
    for name, sort in (("baseline", baseline_sort), ("candidate", candidate_sort)):
        times = np.asarray(sort["st"]).reshape(-1)
        if not np.issubdtype(times.dtype, np.integer) or (len(times) and (times.min() < 0 or times.max() >= duration_frames)):
            raise ValueError(
                f"{name} spike times are not integer samples on the configured recording clock"
            )
    tolerance = int(round(float(config["correspondence_tolerance_ms"]) * 1e-3 * fs))
    edges = correspondence(
        baseline_sort, candidate_sort, tolerance,
        minimum_overlap=float(config["minimum_correspondence_overlap"]),
        primary_retention=float(config["primary_retention"]),
    )
    baseline_units = _unit_metrics(
        baseline_sort, fs=fs, duration_s=duration_s,
        longitudinal_bin_s=float(config["longitudinal_bin_s"]), spatial_region=spatial_region,
    )
    candidate_units = _unit_metrics(
        candidate_sort, fs=fs, duration_s=duration_s,
        longitudinal_bin_s=float(config["longitudinal_bin_s"]), spatial_region=spatial_region,
    )
    primary = edges[edges.primary_match.astype(bool)].copy()
    if spatial_region is not None:
        bclass = baseline_units.set_index("cluster_id").spatial_class
        cclass = candidate_units.set_index("cluster_id").spatial_class
        primary["baseline_spatial_class"] = primary.baseline_cluster.map(bclass)
        primary["candidate_spatial_class"] = primary.candidate_cluster.map(cclass)
        primary["interior_primary"] = (
            primary.baseline_spatial_class.eq("interior")
            & primary.candidate_spatial_class.eq("interior")
        )
    else:
        primary["interior_primary"] = True

    baseline_windows = baseline_qc["amplitude_windows"].copy()
    candidate_windows = candidate_qc["amplitude_windows"].copy()
    baseline_windows["sort"] = "baseline"
    candidate_windows["sort"] = "candidate"
    amplitude_rows = []
    for edge in primary[primary.interior_primary].itertuples(index=False):
        aw = _valid_windows(baseline_windows, edge.baseline_cluster)
        bw = _valid_windows(candidate_windows, edge.candidate_cluster)
        seconds, difference, _ = common_time(aw, bw)
        measurable = (
            len(aw) >= int(config.get("minimum_valid_amplitude_windows", 2))
            and len(bw) >= int(config.get("minimum_valid_amplitude_windows", 2))
            and seconds / duration_s >= float(config["minimum_common_time_fraction"])
        )
        amplitude_rows.append({
            "baseline_cluster": edge.baseline_cluster,
            "candidate_cluster": edge.candidate_cluster,
            "baseline_valid_windows": len(aw),
            "candidate_valid_windows": len(bw),
            "common_time_s": seconds,
            "common_time_fraction": seconds / duration_s,
            "measurable": measurable,
            "baseline_minus_candidate_missingness_pp": difference if measurable else np.nan,
        })
    amplitude_pairs = pd.DataFrame(amplitude_rows)

    baseline_degree = edges.groupby("baseline_cluster").size() if len(edges) else pd.Series(dtype=int)
    candidate_degree = edges.groupby("candidate_cluster").size() if len(edges) else pd.Series(dtype=int)
    split_merge = {
        "correspondence_edges": int(len(edges)),
        "primary_matches": int(edges.primary_match.sum()) if len(edges) else 0,
        "baseline_units_with_multiple_edges": int((baseline_degree > 1).sum()),
        "candidate_units_with_multiple_edges": int((candidate_degree > 1).sum()),
        "ambiguous_edges": int((~edges.primary_match).sum()) if len(edges) else 0,
        "interpretation": "spike-train correspondence graph; not proof of biological identity",
    }
    coincidence = {
        name: chance_aware_coincidence(
            sort, duration_frames=duration_frames,
            tolerance_frames=int(round(float(config["coincidence_tolerance_ms"]) * 1e-3 * fs)),
            depth_tolerance_um=float(config["coincidence_depth_um"]),
            seed=int(config.get("coincidence_seed", 20250804)) + index,
        )
        for index, (name, sort) in enumerate((("baseline", baseline_sort), ("candidate", candidate_sort)))
    }
    guardrail_rows = [
        {
            "metric": "median_refractory_violation_fraction_1_5ms",
            "baseline": float(baseline_units.refractory_violation_fraction_1_5ms.median()),
            "candidate": float(candidate_units.refractory_violation_fraction_1_5ms.median()),
            "available": True,
        },
        {
            "metric": "chance_aware_near_coincident_excess",
            "baseline": coincidence["baseline"]["excess"],
            "candidate": coincidence["candidate"]["excess"],
            "available": coincidence["baseline"]["available"] and coincidence["candidate"]["available"],
        },
    ]
    if spatial_region is not None:
        for metric, column in (
            ("edge_unit_fraction", "spatial_class"),
            ("edge_spike_fraction", "edge_spike_fraction"),
        ):
            if column == "spatial_class":
                bvalue = float(np.mean(baseline_units[column] == "edge"))
                cvalue = float(np.mean(candidate_units[column] == "edge"))
            else:
                bvalue = float(np.average(baseline_units[column].fillna(0), weights=baseline_units.spike_count))
                cvalue = float(np.average(candidate_units[column].fillna(0), weights=candidate_units.spike_count))
            guardrail_rows.append({"metric": metric, "baseline": bvalue, "candidate": cvalue, "available": True})
    for metric in ("sliding_rp_contamination", "nn_isolation", "nn_miss_rate", "sd_ratio", "noise_cutoff"):
        bvalue = baseline_qc.get("guardrails", {}).get(metric)
        cvalue = candidate_qc.get("guardrails", {}).get(metric)
        guardrail_rows.append({"metric": metric, "baseline": bvalue, "candidate": cvalue,
                               "available": bvalue is not None and cvalue is not None})
    guardrails = pd.DataFrame(guardrail_rows)
    guardrails["candidate_minus_baseline"] = pd.to_numeric(guardrails.candidate, errors="coerce") - pd.to_numeric(guardrails.baseline, errors="coerce")

    interior_pairs = primary[primary.interior_primary]
    measurable = int(amplitude_pairs.measurable.sum()) if len(amplitude_pairs) else 0
    eligible = baseline_units if spatial_region is None else baseline_units[baseline_units.spatial_class == "interior"]
    cohort_rows = []
    for cid in eligible.cluster_id:
        partner = primary[primary.baseline_cluster == cid]
        fitted = amplitude_pairs[amplitude_pairs.baseline_cluster == cid] if len(amplitude_pairs) else pd.DataFrame()
        reason = ("unmatched" if partner.empty else "candidate_not_interior" if not partner.interior_primary.iloc[0]
                  else "insufficient_fit_support" if fitted.empty or not fitted.measurable.iloc[0] else "measurable")
        cohort_rows.append(dict(baseline_cluster=int(cid), candidate_cluster=None if partner.empty else int(partner.candidate_cluster.iloc[0]), status=reason))
    eligibility = pd.DataFrame(cohort_rows, columns=["baseline_cluster", "candidate_cluster", "status"])
    coverage = {
        "baseline_eligible_units": int(len(eligible)),
        "baseline_cohort_status_counts": eligibility.status.value_counts().to_dict(),
        "primary_matches": int(len(primary)),
        "interior_primary_matches": int(len(interior_pairs)),
        "amplitude_measurable_both_common_time": measurable,
        "matched_pair_conditional_measurable_fraction": measurable / len(interior_pairs) if len(interior_pairs) else 0.0,
        "amplitude_measurable_fraction": measurable / len(eligible) if len(eligible) else 0.0,
        "endpoint_status": (
            "measured" if len(eligible) and measurable > 0 and measurable / len(eligible) >= float(config["minimum_measurable_unit_fraction"])
            else "infeasible_insufficient_coverage"
        ),
    }
    summary = {
        "schema_version": COMPARISON_SCHEMA,
        "baseline_units": int(len(baseline_units)),
        "candidate_units": int(len(candidate_units)),
        "baseline_spikes": int(len(baseline_sort["st"])),
        "candidate_spikes": int(len(candidate_sort["st"])),
        "primary_matches": int(len(primary)),
        "interior_primary_matches": int(len(interior_pairs)),
        "median_baseline_retention": float(interior_pairs.baseline_retention.median()) if len(interior_pairs) else None,
        "median_candidate_retention": float(interior_pairs.candidate_retention.median()) if len(interior_pairs) else None,
        "median_jaccard": float(interior_pairs.jaccard.median()) if len(interior_pairs) else None,
        "median_missingness_improvement_pp": (
            float(amplitude_pairs.loc[amplitude_pairs.measurable, "baseline_minus_candidate_missingness_pp"].median())
            if measurable else None
        ),
        "coverage_status": coverage["endpoint_status"],
    }
    evaluation_request = {
        "schema_version": COMPARISON_SCHEMA,
        "config": dict(config),
        "spatial_region": None if spatial_region is None else dict(spatial_region),
        "baseline_identity": baseline_sort.get("identity_digest"),
        "candidate_identity": candidate_sort.get("identity_digest"),
        "baseline_qc_identity": baseline_qc.get("request_digest"),
        "candidate_qc_identity": candidate_qc.get("request_digest"),
    }
    candidate_manifest = {**evaluation_request, "request_digest": fingerprint(evaluation_request)}
    decision = {
        "automatic_rank": None,
        "status": "ready_for_pareto_review" if coverage["endpoint_status"] == "measured" else "endpoint_infeasible",
        "reason": "No composite score is computed; apply the prospective development contract to efficacy and guardrail changes.",
    }
    report = {
        "summary": summary, "candidate_manifest": candidate_manifest, "edges": edges,
        "primary_matches": primary, "split_merge_summary": split_merge,
        "amplitude_windows": pd.concat([baseline_windows, candidate_windows], ignore_index=True),
        "amplitude_completeness_pairs": amplitude_pairs,
        "unit_metrics_baseline": baseline_units, "unit_metrics_candidate": candidate_units,
        "guardrail_summary": guardrails, "coverage_summary": coverage, "decision": decision,
        "baseline_eligibility": eligibility,
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / "candidate_manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous.get("request_digest") != candidate_manifest["request_digest"]:
                raise RuntimeError("existing comparison output belongs to another request")
        for name in ("summary", "candidate_manifest", "split_merge_summary", "coverage_summary", "decision"):
            (output / f"{name}.json").write_text(json.dumps(report[name], indent=2) + "\n")
        for name in (
            "edges", "primary_matches", "amplitude_windows", "amplitude_completeness_pairs", "baseline_eligibility",
            "unit_metrics_baseline", "unit_metrics_candidate", "guardrail_summary",
        ):
            filename = "correspondence_edges.csv" if name == "edges" else f"{name}.csv"
            report[name].to_csv(output / filename, index=False)
    return report
