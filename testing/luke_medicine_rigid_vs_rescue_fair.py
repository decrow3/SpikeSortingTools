"""Fair shared-domain comparison of MEDiCINe rigid and original RESCUE sorts.

This is a spike-array comparison; it never rereads voltage or launches a sort.
The primary cohort is predeclared as the common channel-depth span with a
100-um guard at each boundary. Zero- and 60-um guards are retained as
sensitivity analyses. Edge/halo units are classified and preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import fingerprint
from testing.sort_comparison import correspondence


BASE = Path("/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0")
CAND = Path("/media/huklab/Data/luke_medicine_rigid_sort_20260909_v1")
FS = 30_000.0
MARGINS_UM = (0.0, 60.0, 100.0)
PRIMARY_MARGIN_UM = 100.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def labels(path: Path) -> dict[int, str]:
    frame = pd.read_csv(path, sep="\t")
    value = next(column for column in frame if column != "cluster_id")
    return dict(zip(frame.cluster_id.astype(int), frame[value].astype(str).str.lower()))


def load_sort(root: Path, name: str) -> dict:
    curated = root / "cur/cur_output"
    times = np.load(curated / "spike_times.npy", mmap_mode="r").reshape(-1)
    clusters = np.load(curated / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    positions = np.load(curated / "spike_positions.npy", mmap_mode="r")
    amplitudes = np.load(curated / "amplitudes.npy", mmap_mode="r").reshape(-1)
    if not (len(times) == len(clusters) == len(positions) == len(amplitudes)):
        raise ValueError(f"unaligned curated arrays for {name}")
    if np.any(np.diff(times) < 0):
        raise ValueError(f"unsorted spike times for {name}")
    files = ["spike_times.npy", "spike_clusters.npy", "spike_positions.npy", "amplitudes.npy", "cluster_KSLabel.tsv"]
    return {
        "name": name,
        "root": str(root),
        "st": times,
        "cl": clusters,
        "depth": positions[:, 1],
        "amp": amplitudes,
        "labels": labels(curated / "cluster_KSLabel.tsv"),
        "identity_digest": fingerprint({filename: sha256(curated / filename) for filename in files}),
    }


def unit_metrics(sort: dict, duration_s: float, bin_s: float = 300.0) -> pd.DataFrame:
    times = np.asarray(sort["st"], dtype=np.int64)
    clusters = np.asarray(sort["cl"], dtype=np.int64)
    depths = np.asarray(sort["depth"], dtype=np.float32)
    amplitudes = np.asarray(sort["amp"], dtype=np.float32)
    order = np.argsort(clusters, kind="stable")
    ordered_clusters = clusters[order]
    ids, starts, counts = np.unique(ordered_clusters, return_index=True, return_counts=True)
    edges = np.arange(0.0, duration_s + bin_s, bin_s)
    if edges[-1] < duration_s:
        edges = np.r_[edges, duration_s]
    rows = []
    for cluster, start, count in zip(ids, starts, counts):
        take = order[start:start + count]
        unit_times = times[take]
        unit_depths = depths[take]
        unit_amp = amplitudes[take]
        rates = np.histogram(unit_times / FS, bins=edges)[0] / np.diff(edges)
        rows.append({
            "cluster_id": int(cluster),
            "label": sort["labels"].get(int(cluster), "unknown"),
            "spike_count": int(count),
            "median_depth_um": float(np.median(unit_depths)),
            "depth_p05_um": float(np.quantile(unit_depths, 0.05)),
            "depth_p95_um": float(np.quantile(unit_depths, 0.95)),
            "refractory_violation_fraction_1_5ms": float(np.mean(np.diff(unit_times) < round(0.0015 * FS))) if count > 1 else np.nan,
            "presence_fraction_5min": float(np.mean(rates > 0)),
            "firing_rate_cv_5min": float(np.std(rates) / np.mean(rates)) if np.mean(rates) else np.nan,
            "amplitude_cv": float(np.std(unit_amp) / np.mean(unit_amp)) if np.mean(unit_amp) else np.nan,
        })
    return pd.DataFrame(rows)


def cohort(metrics: pd.DataFrame, lo: float, hi: float) -> pd.Series:
    return metrics.median_depth_um.between(lo, hi, inclusive="both")


def median_or_none(values: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.median()) if len(values) else None


def aggregate_unit_metrics(metrics: pd.DataFrame, lo: float, hi: float) -> list[dict]:
    interior = metrics[cohort(metrics, lo, hi)]
    rows = []
    for label, frame in (("all", interior), ("good", interior[interior.label.eq("good")])):
        rows.append({
            "label_cohort": label,
            "units": int(len(frame)),
            **{f"median_{column}": median_or_none(frame[column]) for column in (
                "spike_count", "refractory_violation_fraction_1_5ms",
                "presence_fraction_5min", "firing_rate_cv_5min", "amplitude_cv",
            )},
        })
    return rows


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    contract_path = CAND / "application_contract.json"
    contract = json.loads(contract_path.read_text())
    duration_s = float(contract["duration_s"])
    common_lo, common_hi = map(float, contract["output_depth_span_um"])
    baseline = load_sort(BASE, "original_rescue")
    candidate = load_sort(CAND, "medicine_rigid")
    bmetrics = unit_metrics(baseline, duration_s)
    cmetrics = unit_metrics(candidate, duration_s)
    bmetrics.to_csv(output / "unit_metrics_original_rescue.csv", index=False)
    cmetrics.to_csv(output / "unit_metrics_medicine_rigid.csv", index=False)

    edges = correspondence(
        baseline, candidate, tolerance=round(0.5e-3 * FS),
        minimum_overlap=0.10, primary_retention=0.50,
    )
    edges.to_csv(output / "correspondence_edges.csv", index=False)
    primary = edges[edges.primary_match.astype(bool)].copy()
    primary = primary.merge(
        bmetrics.add_prefix("baseline_"), left_on="baseline_cluster", right_on="baseline_cluster_id",
    ).merge(
        cmetrics.add_prefix("candidate_"), left_on="candidate_cluster", right_on="candidate_cluster_id",
    )
    for metric in ("refractory_violation_fraction_1_5ms", "presence_fraction_5min", "firing_rate_cv_5min", "amplitude_cv"):
        primary[f"candidate_minus_baseline_{metric}"] = primary[f"candidate_{metric}"] - primary[f"baseline_{metric}"]
    primary.to_csv(output / "reciprocal_primary_matches.csv", index=False)

    summaries = []
    for margin in MARGINS_UM:
        lo, hi = common_lo + margin, common_hi - margin
        bkeep, ckeep = cohort(bmetrics, lo, hi), cohort(cmetrics, lo, hi)
        pairs = primary[
            primary.baseline_median_depth_um.between(lo, hi, inclusive="both")
            & primary.candidate_median_depth_um.between(lo, hi, inclusive="both")
        ]
        belig, celig = bmetrics[bkeep], cmetrics[ckeep]
        bmatched = set(pairs.baseline_cluster.astype(int))
        bgood = belig[belig.label.eq("good")]
        summaries.append({
            "margin_um": margin,
            "depth_lo_um": lo,
            "depth_hi_um": hi,
            "baseline_units": int(len(belig)),
            "candidate_units": int(len(celig)),
            "unit_change_pct": float(100 * (len(celig) / len(belig) - 1)) if len(belig) else None,
            "baseline_good_units": int(belig.label.eq("good").sum()),
            "candidate_good_units": int(celig.label.eq("good").sum()),
            "good_unit_change_pct": float(100 * (celig.label.eq("good").sum() / belig.label.eq("good").sum() - 1)) if belig.label.eq("good").sum() else None,
            "baseline_spikes": int(belig.spike_count.sum()),
            "candidate_spikes": int(celig.spike_count.sum()),
            "spike_change_pct": float(100 * (celig.spike_count.sum() / belig.spike_count.sum() - 1)) if belig.spike_count.sum() else None,
            "reciprocal_primary_matches": int(len(pairs)),
            "baseline_unit_match_fraction": float(len(set(pairs.baseline_cluster)) / len(belig)) if len(belig) else None,
            "candidate_unit_match_fraction": float(len(set(pairs.candidate_cluster)) / len(celig)) if len(celig) else None,
            "baseline_good_matched": int(sum(int(cid) in bmatched for cid in bgood.cluster_id)),
            "baseline_good_match_fraction": float(sum(int(cid) in bmatched for cid in bgood.cluster_id) / len(bgood)) if len(bgood) else None,
            "median_jaccard": median_or_none(pairs.jaccard),
            "median_baseline_retention": median_or_none(pairs.baseline_retention),
            "median_candidate_retention": median_or_none(pairs.candidate_retention),
            "median_candidate_minus_baseline_refractory": median_or_none(pairs.candidate_minus_baseline_refractory_violation_fraction_1_5ms),
            "median_candidate_minus_baseline_presence": median_or_none(pairs.candidate_minus_baseline_presence_fraction_5min),
            "median_candidate_minus_baseline_firing_rate_cv": median_or_none(pairs.candidate_minus_baseline_firing_rate_cv_5min),
            "median_candidate_minus_baseline_amplitude_cv": median_or_none(pairs.candidate_minus_baseline_amplitude_cv),
        })
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output / "shared_domain_sensitivity.csv", index=False)
    primary_summary = next(row for row in summaries if row["margin_um"] == PRIMARY_MARGIN_UM)

    outside_rows = []
    primary_lo, primary_hi = common_lo + PRIMARY_MARGIN_UM, common_hi - PRIMARY_MARGIN_UM
    for name, metrics in (("original_rescue", bmetrics), ("medicine_rigid", cmetrics)):
        depth = metrics.median_depth_um
        classes = np.select(
            [depth < common_lo, depth.between(common_lo, primary_lo, inclusive="left"),
             depth.between(primary_hi, common_hi, inclusive="right"), depth > common_hi],
            ["outside_low", "boundary_low", "boundary_high", "outside_high"], default="primary_interior",
        )
        for spatial_class in np.unique(classes):
            take = metrics[classes == spatial_class]
            outside_rows.append({"sort": name, "spatial_class": spatial_class, "units": int(len(take)),
                                 "good_units": int(take.label.eq("good").sum()), "spikes": int(take.spike_count.sum())})
    pd.DataFrame(outside_rows).to_csv(output / "spatial_inventory.csv", index=False)

    aggregate_rows = []
    for name, metrics in (("original_rescue", bmetrics), ("medicine_rigid", cmetrics)):
        for row in aggregate_unit_metrics(metrics, primary_lo, primary_hi):
            aggregate_rows.append({"sort": name, **row})
    pd.DataFrame(aggregate_rows).to_csv(output / "primary_interior_unit_qc.csv", index=False)

    baseline_manifest = json.loads((BASE / "kilosort4/rescue_sort_manifest.json").read_text())["summary"]
    candidate_manifest = json.loads((CAND / "kilosort4/rescue_sort_manifest.json").read_text())["summary"]
    curated_totals = {
        "baseline": {"units": int(len(bmetrics)), "good_units": int(bmetrics.label.eq("good").sum()), "spikes": int(len(baseline["st"]))},
        "candidate": {"units": int(len(cmetrics)), "good_units": int(cmetrics.label.eq("good").sum()), "spikes": int(len(candidate["st"]))},
    }

    provenance = {
        "schema_version": "luke-medicine-rigid-vs-rescue-fair-v1",
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "baseline_root": str(BASE), "candidate_root": str(CAND),
        "baseline_identity_digest": baseline["identity_digest"],
        "candidate_identity_digest": candidate["identity_digest"],
        "application_contract_sha256": sha256(contract_path),
        "recording_duration_s": duration_s, "sampling_frequency_hz": FS,
        "common_channel_depth_um": [common_lo, common_hi],
        "primary_guard_um": PRIMARY_MARGIN_UM,
        "primary_scoring_depth_um": [primary_lo, primary_hi],
        "sensitivity_guards_um": list(MARGINS_UM),
        "correspondence": {"tolerance_ms": 0.5, "minimum_overlap": 0.10, "reciprocal_primary_retention": 0.50},
        "interpretation": "Spike-train correspondence is descriptive agreement, not biological identity or directional efficacy.",
        "known_limitation": "MEDiCINe input omitted channels 0-19 and 366-383; those channels are absent, not adjudicated bad.",
    }
    result = {
        **provenance,
        "analysis_digest": fingerprint({**provenance, "summaries": summaries}),
        "primary": primary_summary,
        "sensitivity": summaries,
        "correspondence_graph": {
            "edges": int(len(edges)),
            "reciprocal_primary_matches": int(primary.primary_match.sum()) if "primary_match" in primary else int(len(primary)),
            "baseline_units_with_multiple_edges": int((edges.groupby("baseline_cluster").size() > 1).sum()),
            "candidate_units_with_multiple_edges": int((edges.groupby("candidate_cluster").size() > 1).sum()),
        },
        "all_sort_totals": curated_totals,
        "sorter_manifest_totals": {"baseline": baseline_manifest, "candidate": candidate_manifest},
        "primary_interior_unit_qc": aggregate_rows,
        "edge_policy": "Preserve and report boundary/outside inventory; exclude it only from the primary shared-interior endpoint.",
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
