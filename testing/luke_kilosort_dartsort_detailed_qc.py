"""Detailed cached QC comparison for the 930--1030 s KS/DARTsort snippets.

The analysis is identity-aware and does not compare unmatched population
medians as if they represented the same neurons. It builds common per-unit
metrics, strict mutual-best pairs, connected split/merge families, and
shared-versus-exclusive event partitions. No raw voltage is read.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.unit_quality import _isi_metrics
from testing.luke_kilosort_dartsort_unit_comparison import (
    ARMS,
    BASE_DEPTH_UM,
    BASE_MATCH_FRACTION,
    BASE_TOLERANCE_MS,
    DURATION_S,
    FS,
    MIN_PAIR_EVENTS,
    MIN_UNIT_SPIKES,
    corrected_to_observed_depth,
    exclusive_pairs,
    pair_table,
)


SCHEMA = "luke-kilosort-dartsort-detailed-qc-v1"
DEPTH_MIN_UM = 260.0
DEPTH_MAX_UM = 3720.0
BIN_S = 10.0
DRIFT_MIN_SPIKES = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(partial, path)


def _safe(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else np.nan


def load_ks_detailed(arm: str, root: Path, fields) -> dict[str, Any]:
    path = root / "kilosort4/sorter_output"
    times = np.load(path / "spike_times.npy").reshape(-1).astype(np.int64)
    labels = np.load(path / "spike_clusters.npy").reshape(-1).astype(np.int64)
    corrected_depth = np.load(path / "spike_positions.npy")[:, 1].astype(float)
    amplitude = np.load(path / "amplitudes.npy").reshape(-1).astype(float)
    if not (len(times) == len(labels) == len(corrected_depth) == len(amplitude)):
        raise RuntimeError(f"KS arrays differ in length: {arm}")
    order = np.argsort(times, kind="stable")
    times, labels = times[order], labels[order]
    corrected_depth, amplitude = corrected_depth[order], amplitude[order]
    depth = corrected_to_observed_depth(arm, times, corrected_depth, fields)
    labels_table = pd.read_csv(path / "cluster_KSLabel.tsv", sep="\t")
    label_col = next(column for column in labels_table if column != "cluster_id")
    ks_label = dict(zip(labels_table.cluster_id.astype(int), labels_table[label_col].astype(str)))
    contam_table = pd.read_csv(path / "cluster_ContamPct.tsv", sep="\t")
    contam_col = next(column for column in contam_table if column != "cluster_id")
    contamination = dict(zip(contam_table.cluster_id.astype(int), pd.to_numeric(contam_table[contam_col], errors="coerce")))
    templates = np.load(path / "templates.npy", mmap_mode="r")
    similarity = np.load(path / "similar_templates.npy", mmap_mode="r")
    ops = np.load(path / "ops.npy", allow_pickle=True).item()
    geom = np.c_[np.asarray(ops["xc"], dtype=float), np.asarray(ops["yc"], dtype=float)]
    return {
        "sorter": "kilosort", "arm": arm, "times": times, "labels": labels,
        "depth": depth, "amplitude": amplitude, "ks_label": ks_label,
        "contamination_pct": contamination, "templates": templates,
        "similarity": similarity, "geom": geom,
        "source": path,
    }


def load_dartsort_detailed(npz_path: Path, h5_path: Path, receipt_path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "load_dartsort_detailed requires the optional h5py dependency to read "
            f"{h5_path}"
        ) from exc
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "complete" or receipt.get("returncode") != 0:
        raise RuntimeError("DARTsort receipt is not complete")
    z = np.load(npz_path, allow_pickle=False)
    with h5py.File(h5_path, "r") as handle:
        h5_times = np.asarray(handle["times_samples"], dtype=np.int64)
        h5_channels = np.asarray(handle["channels"], dtype=np.int64)
        depth = np.asarray(handle["point_source_localizations"][:, 2], dtype=float)
        amplitude = np.asarray(handle["denoised_ptp_amplitudes"], dtype=float)
        scores = np.asarray(handle["scores"], dtype=float)
    times = np.asarray(z["times_samples"], dtype=np.int64)
    labels = np.asarray(z["labels"], dtype=np.int64)
    channels = np.asarray(z["channels"], dtype=np.int64)
    if not np.array_equal(channels, h5_channels):
        raise RuntimeError("DARTsort NPZ/H5 channels do not align")
    if np.max(np.abs(times - h5_times)) > 1:
        raise RuntimeError("DARTsort final events moved by more than one sample")
    merged_responsibility = np.asarray(z["merged_responsibilities"], dtype=float)
    max_responsibility = np.nanmax(merged_responsibility, axis=1)
    sorted_responsibility = np.sort(merged_responsibility, axis=1)
    responsibility_margin = sorted_responsibility[:, -1] - sorted_responsibility[:, -2]
    train = np.asarray(z["gmm_train"], dtype=bool)
    keep = (
        (labels >= 0) & (times >= 0) & (times < int(round(DURATION_S * FS)))
        & np.isfinite(depth)
    )
    median_depth = {
        int(unit): float(np.median(depth[keep & (labels == unit)]))
        for unit in np.unique(labels[keep])
    }
    eligible = {unit for unit, value in median_depth.items() if DEPTH_MIN_UM <= value <= DEPTH_MAX_UM}
    keep &= np.isin(labels, list(eligible))
    indices = np.flatnonzero(keep)
    order = np.argsort(times[indices], kind="stable")
    indices = indices[order]
    return {
        "sorter": "dartsort", "arm": "dartsort", "times": times[indices],
        "labels": labels[indices], "depth": depth[indices], "amplitude": amplitude[indices],
        "channels": channels[indices], "scores": scores[indices],
        "max_responsibility": max_responsibility[indices],
        "responsibility_margin": responsibility_margin[indices], "gmm_train": train[indices],
        "geom": np.asarray(z["geom"], dtype=float), "source_indices": indices,
        "source": npz_path,
    }


def subset(data: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    result = {"sorter": data["sorter"], "arm": data["arm"]}
    for key in ("times", "labels", "depth", "amplitude", "channels", "scores", "max_responsibility", "responsibility_margin", "gmm_train"):
        if key in data:
            result[key] = data[key][mask]
    return result


def unit_metric_row(data: dict[str, Any], unit: int, mask: np.ndarray | None = None) -> dict[str, Any]:
    if mask is None:
        mask = data["labels"] == unit
    times = np.sort(np.asarray(data["times"][mask], dtype=np.int64))
    # Restore aligned order for non-time arrays; source data are already sorted.
    depth = np.asarray(data["depth"][mask], dtype=float)
    amp = np.asarray(data["amplitude"][mask], dtype=float)
    bin_index = np.floor((data["times"][mask] / FS) / BIN_S).astype(int)
    counts = np.bincount(bin_index, minlength=int(DURATION_S // BIN_S))[: int(DURATION_S // BIN_S)]
    depth_medians, amp_medians = [], []
    for bin_id in range(len(counts)):
        in_bin = bin_index == bin_id
        if np.count_nonzero(in_bin) >= DRIFT_MIN_SPIKES:
            depth_medians.append(float(np.nanmedian(depth[in_bin])))
            amp_medians.append(float(np.nanmedian(amp[in_bin])))
    isi = _isi_metrics(
        times, sampling_frequency=FS, duration_s=DURATION_S,
        isi_threshold_ms=1.5, rp_refractory_ms=1.0, rp_censored_ms=0,
    )
    finite_amp = amp[np.isfinite(amp)]
    amp_mean = np.mean(finite_amp) if finite_amp.size else np.nan
    row = {
        "sorter": data["sorter"], "arm": data["arm"], "unit_id": int(unit),
        "num_spikes": int(times.size), "firing_rate_hz": float(times.size / DURATION_S),
        "first_spike_s": float(times[0] / FS), "last_spike_s": float(times[-1] / FS),
        "unit_lifetime_s": float((times[-1] - times[0]) / FS),
        "presence_ratio_10s": float(np.mean(counts > 0)),
        "firing_rate_10s_cv": float(np.std(counts) / np.mean(counts)) if np.mean(counts) else np.nan,
        "isi_violations_count_1p5ms": isi["isi_violations_count_1p5ms"],
        "isi_violations_fraction_1p5ms": isi["isi_violations_fraction_1p5ms"],
        "si_isi_violations_ratio_1p5ms": isi["si_isi_violations_ratio_1p5ms"],
        "rp_violations_count_1ms": isi["rp_violations_count_1ms"],
        "si_rp_contamination_1ms": isi["si_rp_contamination_1ms"],
        "depth_median_um": _safe(depth, 50), "depth_p05_um": _safe(depth, 5),
        "depth_p95_um": _safe(depth, 95), "depth_excursion_um": _safe(depth, 95) - _safe(depth, 5),
        "edge_spike_fraction_40um": float(np.mean((depth <= DEPTH_MIN_UM + 40) | (depth >= DEPTH_MAX_UM - 40))),
        "valid_10s_depth_bins": len(depth_medians),
        "depth_bin_median_range_um": float(np.ptp(depth_medians)) if len(depth_medians) >= 2 else np.nan,
        "amplitude_scale": "kilosort_feature" if data["sorter"] == "kilosort" else "dartsort_denoised_ptp",
        "amplitude_median": _safe(finite_amp, 50), "amplitude_p05": _safe(finite_amp, 5),
        "amplitude_p95": _safe(finite_amp, 95),
        "amplitude_cv": float(np.std(finite_amp) / amp_mean) if finite_amp.size and amp_mean else np.nan,
        "valid_10s_amplitude_bins": len(amp_medians),
        "amplitude_bin_median_cv": float(np.std(amp_medians) / np.mean(amp_medians)) if len(amp_medians) >= 2 and np.mean(amp_medians) else np.nan,
        "ks_label": data.get("ks_label", {}).get(int(unit), pd.NA),
        "ks_contamination_pct": data.get("contamination_pct", {}).get(int(unit), np.nan),
        "dartsort_max_responsibility_median": _safe(data["max_responsibility"][mask], 50) if "max_responsibility" in data else np.nan,
        "dartsort_responsibility_margin_median": _safe(data["responsibility_margin"][mask], 50) if "responsibility_margin" in data else np.nan,
        "dartsort_gmm_train_fraction": float(np.mean(data["gmm_train"][mask])) if "gmm_train" in data else np.nan,
        "dartsort_score_median": _safe(data["scores"][mask], 50) if "scores" in data else np.nan,
        "template_positive_dominant": pd.NA, "max_other_template_similarity": np.nan,
        "nearby_similar_template_count": np.nan,
    }
    if data["sorter"] == "kilosort" and 0 <= unit < len(data["templates"]):
        template = np.asarray(data["templates"][unit], dtype=float)
        peak = int(np.argmax(np.ptp(template, axis=0)))
        row["template_positive_dominant"] = bool(abs(np.max(template[:, peak])) > abs(np.min(template[:, peak])))
        sim = np.asarray(data["similarity"][unit], dtype=float).copy()
        sim[unit] = np.nan
        row["max_other_template_similarity"] = float(np.nanmax(sim))
        candidates = np.flatnonzero(sim >= 0.8)
        count = 0
        for other in candidates:
            other_peak = int(np.argmax(np.ptp(np.asarray(data["templates"][other]), axis=0)))
            if np.linalg.norm(data["geom"][peak] - data["geom"][other_peak]) <= 100:
                count += 1
        row["nearby_similar_template_count"] = count
    return row


def unit_metrics(data: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([unit_metric_row(data, int(unit)) for unit in np.unique(data["labels"])])


def qualified_edges(pairs: pd.DataFrame, ks_eligible: set[int], dart_eligible: set[int]) -> pd.DataFrame:
    return pairs.loc[
        pairs.ks_unit.isin(ks_eligible) & pairs.dartsort_unit.isin(dart_eligible)
        & pairs.coincident_events.ge(MIN_PAIR_EVENTS)
        & ((pairs[["ks_coverage", "dartsort_coverage"]].max(axis=1) >= 0.5)
           | (pairs[["ks_coverage", "dartsort_coverage"]].min(axis=1) >= 0.2))
    ].copy()


def components(edges: pd.DataFrame) -> list[tuple[list[int], list[int]]]:
    graph: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
    for row in edges.itertuples():
        a, b = ("ks", int(row.ks_unit)), ("dart", int(row.dartsort_unit))
        graph[a].add(b); graph[b].add(a)
    seen, groups = set(), []
    for start in graph:
        if start in seen:
            continue
        queue, members = deque([start]), []
        seen.add(start)
        while queue:
            node = queue.popleft(); members.append(node)
            for neighbor in graph[node]:
                if neighbor not in seen:
                    seen.add(neighbor); queue.append(neighbor)
        groups.append((sorted(x[1] for x in members if x[0] == "ks"), sorted(x[1] for x in members if x[0] == "dart")))
    return groups


def _partition_stats(data: dict[str, Any], indices: np.ndarray, prefix: str) -> dict[str, Any]:
    if indices.size == 0:
        return {f"{prefix}_count": 0, f"{prefix}_amplitude_median": np.nan,
                f"{prefix}_depth_median_um": np.nan, f"{prefix}_early_fraction": np.nan,
                f"{prefix}_middle_fraction": np.nan, f"{prefix}_late_fraction": np.nan}
    t = data["times"][indices] / FS
    return {
        f"{prefix}_count": int(indices.size),
        f"{prefix}_amplitude_median": _safe(data["amplitude"][indices], 50),
        f"{prefix}_depth_median_um": _safe(data["depth"][indices], 50),
        f"{prefix}_early_fraction": float(np.mean(t < DURATION_S / 3)),
        f"{prefix}_middle_fraction": float(np.mean((t >= DURATION_S / 3) & (t < 2 * DURATION_S / 3))),
        f"{prefix}_late_fraction": float(np.mean(t >= 2 * DURATION_S / 3)),
    }


def family_rows(arm: str, ks: dict[str, Any], dart: dict[str, Any], edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    families, partitions = [], []
    tolerance = int(round(BASE_TOLERANCE_MS * 1e-3 * FS))
    for family_id, (ks_units, dart_units) in enumerate(components(edges)):
        kmask = np.isin(ks["labels"], ks_units)
        dmask = np.isin(dart["labels"], dart_units)
        ksub, dsub = subset(ks, kmask), subset(dart, dmask)
        khit, dhit = exclusive_pairs(ksub, dsub, tolerance, BASE_DEPTH_UM)
        kshared = np.zeros(len(ksub["times"]), dtype=bool); kshared[khit] = True
        dshared = np.zeros(len(dsub["times"]), dtype=bool); dshared[dhit] = True
        krow = unit_metric_row(ksub, -1, np.ones(len(ksub["times"]), dtype=bool))
        drow = unit_metric_row(dsub, -1, np.ones(len(dsub["times"]), dtype=bool))
        matched = int(len(khit)); kn, dn = len(ksub["times"]), len(dsub["times"])
        family = {
            "arm": arm, "family_id": family_id, "ks_units": ";".join(map(str, ks_units)),
            "dartsort_units": ";".join(map(str, dart_units)), "ks_unit_count": len(ks_units),
            "dartsort_unit_count": len(dart_units), "family_shape": f"{len(ks_units)}x{len(dart_units)}",
            "ks_good_unit_count": sum(ks.get("ks_label", {}).get(unit, "").lower() == "good" for unit in ks_units),
            "matched_events": matched, "ks_spikes": kn, "dartsort_spikes": dn,
            "ks_coverage": matched / kn, "dartsort_coverage": matched / dn,
            "jaccard": matched / (kn + dn - matched),
        }
        for field in ("firing_rate_hz", "presence_ratio_10s", "firing_rate_10s_cv", "isi_violations_fraction_1p5ms",
                      "si_rp_contamination_1ms", "depth_excursion_um", "depth_bin_median_range_um",
                      "amplitude_cv", "amplitude_bin_median_cv", "edge_spike_fraction_40um"):
            family[f"ks_{field}"] = krow[field]; family[f"dartsort_{field}"] = drow[field]
        families.append(family)
        partition = {key: family[key] for key in ("arm", "family_id", "family_shape", "ks_units", "dartsort_units")}
        partition.update(_partition_stats(ksub, np.flatnonzero(kshared), "ks_shared"))
        partition.update(_partition_stats(ksub, np.flatnonzero(~kshared), "ks_only"))
        partition.update(_partition_stats(dsub, np.flatnonzero(dshared), "dartsort_shared"))
        partition.update(_partition_stats(dsub, np.flatnonzero(~dshared), "dartsort_only"))
        for sorter in ("ks", "dartsort"):
            shared = partition[f"{sorter}_shared_amplitude_median"]
            exclusive = partition[f"{sorter}_only_amplitude_median"]
            partition[f"{sorter}_only_to_shared_amplitude_ratio"] = exclusive / shared if np.isfinite(shared) and shared else np.nan
            partition[f"{sorter}_only_minus_shared_depth_um"] = partition[f"{sorter}_only_depth_median_um"] - partition[f"{sorter}_shared_depth_median_um"]
            shared_time = np.array([partition[f"{sorter}_shared_{epoch}_fraction"] for epoch in ("early", "middle", "late")])
            only_time = np.array([partition[f"{sorter}_only_{epoch}_fraction"] for epoch in ("early", "middle", "late")])
            partition[f"{sorter}_only_shared_temporal_tv"] = float(0.5 * np.sum(np.abs(only_time - shared_time))) if np.all(np.isfinite(np.r_[shared_time, only_time])) else np.nan
        partitions.append(partition)
    return pd.DataFrame(families), pd.DataFrame(partitions)


def paired_metrics(baseline: pd.DataFrame, ks_metrics: pd.DataFrame, dart_metrics: pd.DataFrame) -> pd.DataFrame:
    result = baseline.copy()
    km = ks_metrics.set_index("unit_id"); dm = dart_metrics.set_index("unit_id")
    fields = [
        "firing_rate_hz", "presence_ratio_10s", "firing_rate_10s_cv",
        "isi_violations_fraction_1p5ms", "si_isi_violations_ratio_1p5ms",
        "si_rp_contamination_1ms", "depth_excursion_um", "depth_bin_median_range_um",
        "amplitude_cv", "amplitude_bin_median_cv", "edge_spike_fraction_40um",
    ]
    for field in fields:
        result[f"ks_{field}"] = result.ks_unit.map(km[field])
        result[f"dartsort_{field}"] = result.dartsort_unit.map(dm[field])
        result[f"dartsort_minus_ks_{field}"] = result[f"dartsort_{field}"] - result[f"ks_{field}"]
    for field in ("dartsort_max_responsibility_median", "dartsort_responsibility_margin_median", "dartsort_gmm_train_fraction", "dartsort_score_median"):
        result[field] = result.dartsort_unit.map(dm[field])
    return result


def summarize_arm(arm: str, ks_metrics: pd.DataFrame, dart_metrics: pd.DataFrame,
                  pairs: pd.DataFrame, paired: pd.DataFrame, families: pd.DataFrame,
                  partitions: pd.DataFrame) -> dict[str, Any]:
    def med(frame, field):
        return float(frame[field].median()) if len(frame) else np.nan
    return {
        "arm": arm, "ks_units": int(len(ks_metrics)), "dartsort_units": int(len(dart_metrics)),
        "strict_matched_pairs": int(len(paired)), "strict_matched_ks_good_pairs": int(paired.ks_good.sum()),
        "qualified_families": int(len(families)), "families_with_ks_good": int((families.ks_good_unit_count > 0).sum()),
        "one_ks_to_multiple_dart_families": int(((families.ks_unit_count == 1) & (families.dartsort_unit_count > 1)).sum()),
        "multiple_ks_to_one_dart_families": int(((families.ks_unit_count > 1) & (families.dartsort_unit_count == 1)).sum()),
        "family_median_ks_coverage": med(families, "ks_coverage"),
        "family_median_dartsort_coverage": med(families, "dartsort_coverage"),
        "family_median_jaccard": med(families, "jaccard"),
        "paired_median_ks_rp_contamination": med(paired, "ks_si_rp_contamination_1ms"),
        "paired_median_dartsort_rp_contamination": med(paired, "dartsort_si_rp_contamination_1ms"),
        "paired_dartsort_lower_rp_fraction": float(np.mean(paired.dartsort_si_rp_contamination_1ms < paired.ks_si_rp_contamination_1ms)),
        "family_median_ks_rp_contamination": med(families, "ks_si_rp_contamination_1ms"),
        "family_median_dartsort_rp_contamination": med(families, "dartsort_si_rp_contamination_1ms"),
        "family_dartsort_lower_rp_fraction": float(np.mean(families.dartsort_si_rp_contamination_1ms < families.ks_si_rp_contamination_1ms)),
        "paired_median_ks_isi_fraction": med(paired, "ks_isi_violations_fraction_1p5ms"),
        "paired_median_dartsort_isi_fraction": med(paired, "dartsort_isi_violations_fraction_1p5ms"),
        "paired_dartsort_lower_isi_fraction": float(np.mean(paired.dartsort_isi_violations_fraction_1p5ms < paired.ks_isi_violations_fraction_1p5ms)),
        "family_median_ks_isi_fraction": med(families, "ks_isi_violations_fraction_1p5ms"),
        "family_median_dartsort_isi_fraction": med(families, "dartsort_isi_violations_fraction_1p5ms"),
        "family_dartsort_lower_isi_fraction": float(np.mean(families.dartsort_isi_violations_fraction_1p5ms < families.ks_isi_violations_fraction_1p5ms)),
        "paired_median_ks_depth_excursion_um": med(paired, "ks_depth_excursion_um"),
        "paired_median_dartsort_depth_excursion_um": med(paired, "dartsort_depth_excursion_um"),
        "family_median_ks_depth_excursion_um": med(families, "ks_depth_excursion_um"),
        "family_median_dartsort_depth_excursion_um": med(families, "dartsort_depth_excursion_um"),
        "ks_only_to_shared_amplitude_ratio_median": med(partitions, "ks_only_to_shared_amplitude_ratio"),
        "dartsort_only_to_shared_amplitude_ratio_median": med(partitions, "dartsort_only_to_shared_amplitude_ratio"),
    }


def comparison_summary(table: pd.DataFrame, metrics: list[str], *, seed: int) -> pd.DataFrame:
    """Paired medians, direction fractions, and bootstrap median-difference CIs."""
    rng = np.random.default_rng(seed)
    rows = []
    for arm in ARMS:
        arm_table = table.loc[table.arm == arm]
        for metric in metrics:
            ks = arm_table[f"ks_{metric}"].to_numpy(float)
            dart = arm_table[f"dartsort_{metric}"].to_numpy(float)
            valid = np.isfinite(ks) & np.isfinite(dart)
            ks, dart = ks[valid], dart[valid]
            delta = dart - ks
            if len(delta):
                draws = rng.integers(0, len(delta), size=(5000, len(delta)))
                bootstrap = np.median(delta[draws], axis=1)
                low, high = np.percentile(bootstrap, [2.5, 97.5])
            else:
                low = high = np.nan
            rows.append({
                "arm": arm, "metric": metric, "n": len(delta),
                "ks_median": float(np.median(ks)) if len(ks) else np.nan,
                "dartsort_median": float(np.median(dart)) if len(dart) else np.nan,
                "dartsort_minus_ks_median": float(np.median(delta)) if len(delta) else np.nan,
                "bootstrap_median_delta_ci95_low": float(low),
                "bootstrap_median_delta_ci95_high": float(high),
                "dartsort_lower_fraction": float(np.mean(dart < ks)) if len(delta) else np.nan,
                "equal_fraction": float(np.mean(dart == ks)) if len(delta) else np.nan,
            })
    return pd.DataFrame(rows)


def partition_summary(partitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fields = (
        "ks_only_to_shared_amplitude_ratio", "dartsort_only_to_shared_amplitude_ratio",
        "ks_only_minus_shared_depth_um", "dartsort_only_minus_shared_depth_um",
        "ks_only_shared_temporal_tv", "dartsort_only_shared_temporal_tv",
    )
    for arm in ARMS:
        frame = partitions.loc[partitions.arm == arm]
        for field in fields:
            values = frame[field].to_numpy(float)
            values = values[np.isfinite(values)]
            rows.append({
                "arm": arm, "metric": field, "n": len(values),
                "median": float(np.median(values)) if len(values) else np.nan,
                "p25": float(np.percentile(values, 25)) if len(values) else np.nan,
                "p75": float(np.percentile(values, 75)) if len(values) else np.nan,
            })
    return pd.DataFrame(rows)


def validation_checks(paired: pd.DataFrame, families: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks["strict_ks_unit_unique_per_arm"] = bool(paired.groupby("arm").ks_unit.apply(lambda x: x.is_unique).all())
    checks["strict_dartsort_unit_unique_per_arm"] = bool(paired.groupby("arm").dartsort_unit.apply(lambda x: x.is_unique).all())
    probability_columns = ["ks_coverage", "dartsort_coverage", "jaccard"]
    checks["family_probabilities_bounded"] = bool(
        ((families[probability_columns] >= 0) & (families[probability_columns] <= 1)).all().all()
    )
    checks["family_ids_unique_per_arm"] = bool(
        ~families.duplicated(["arm", "family_id"]).any()
    )
    prior_path = Path("testing/outputs/luke_kilosort_dartsort_unit_comparison_v2/sorter_pair_summary.csv")
    if prior_path.is_file():
        prior = pd.read_csv(prior_path).set_index("arm")["matched_unit_pairs"].astype(int)
        current = summary.set_index("arm")["strict_matched_pairs"].astype(int)
        checks["strict_counts_match_prior_frozen_analysis"] = bool(current.equals(prior.reindex(current.index)))
        checks["prior_strict_counts"] = prior.to_dict()
    else:
        checks["strict_counts_match_prior_frozen_analysis"] = None
    checks["all_required_checks_pass"] = all(value is True for key, value in checks.items() if key != "prior_strict_counts" and value is not None)
    return checks


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    labels = [x.replace("_", "\n") for x in summary.arm]
    x = np.arange(len(labels)); width = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].bar(x - width / 2, 100 * summary.family_median_ks_coverage, width, label="KS", color="#7C3AED")
    axes[0, 0].bar(x + width / 2, 100 * summary.family_median_dartsort_coverage, width, label="DARTsort", color="#2563EB")
    axes[0, 0].set_title("Family-union event coverage"); axes[0, 0].set_ylabel("Median percent")
    axes[0, 1].bar(x - width / 2, 100 * summary.paired_median_ks_isi_fraction, width, label="KS", color="#7C3AED")
    axes[0, 1].bar(x + width / 2, 100 * summary.paired_median_dartsort_isi_fraction, width, label="DARTsort", color="#2563EB")
    axes[0, 1].set_title("Strict-pair observed ISIs <1.5 ms"); axes[0, 1].set_ylabel("Median percent")
    axes[1, 0].bar(x - width / 2, summary.family_median_ks_depth_excursion_um, width, label="KS", color="#7C3AED")
    axes[1, 0].bar(x + width / 2, summary.family_median_dartsort_depth_excursion_um, width, label="DARTsort", color="#2563EB")
    axes[1, 0].set_title("Family-union depth excursion"); axes[1, 0].set_ylabel("Median µm")
    axes[1, 1].bar(x - width / 2, summary.ks_only_to_shared_amplitude_ratio_median, width, label="KS-only/shared", color="#7C3AED")
    axes[1, 1].bar(x + width / 2, summary.dartsort_only_to_shared_amplitude_ratio_median, width, label="DART-only/shared", color="#2563EB")
    axes[1, 1].axhline(1, color="#111827", lw=1); axes[1, 1].set_title("Exclusive-event amplitude within sorter")
    for axis in axes.flat:
        axis.set_xticks(x, labels); axis.grid(axis="y", alpha=.2)
    axes[0, 0].legend(); axes[0, 1].legend(); axes[1, 0].legend(); axes[1, 1].legend()
    fig.suptitle("Detailed Kilosort–DARTsort QC, 930–1030 s")
    fig.savefig(output / "01_detailed_qc_summary.png", dpi=180)
    fig.savefig(output / "01_detailed_qc_summary.pdf")
    plt.close(fig)


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError("output exists; preserve prior evidence")
    output.mkdir(parents=True)
    matrix = Path("/media/huklab/Data/luke_motion_348ch_matrix_v1")
    lfp = Path("/media/huklab/Data/luke_lfp_native_sg_348ch_v1")
    dart_root = Path("/media/huklab/Data/NPX/Ryansorting/Luke/DARTsort_motion_experiments/luke0804-imec0-930-1230-v1/native_motion")
    fields_path = matrix / "resolved_candidate_fields.npz"
    fields = np.load(fields_path, allow_pickle=False)
    roots = {
        "unwarped": matrix / "arms/unwarped", "ap_rigid": matrix / "arms/ap_rigid",
        "dartsort_native": matrix / "arms/dartsort_native",
        "lfp_native": lfp / "arms/lfp_native", "lfp_savgol": lfp / "arms/lfp_savgol",
    }
    dart_npz, dart_h5, dart_receipt = dart_root / "dartsort_sorting.npz", dart_root / "matching1.h5", dart_root / "receipt.json"
    dart = load_dartsort_detailed(dart_npz, dart_h5, dart_receipt)
    dart_metrics = unit_metrics(dart)
    unit_tables = [dart_metrics]
    pair_tables, family_tables, partition_tables, summaries = [], [], [], []
    tolerance = int(round(BASE_TOLERANCE_MS * 1e-3 * FS))
    for arm in ARMS:
        ks = load_ks_detailed(arm, roots[arm], fields)
        km = unit_metrics(ks); unit_tables.append(km)
        khit, dhit = exclusive_pairs(ks, dart, tolerance, BASE_DEPTH_UM)
        pairs = pair_table(
            {**ks, "name": arm, "good": {u for u, v in ks["ks_label"].items() if v.lower() == "good"}},
            {**dart, "name": "dartsort", "good": set()}, khit, dhit,
        )
        ke = set(km.loc[km.num_spikes >= MIN_UNIT_SPIKES, "unit_id"].astype(int))
        de = set(dart_metrics.loc[dart_metrics.num_spikes >= MIN_UNIT_SPIKES, "unit_id"].astype(int))
        baseline = pairs.loc[
            pairs.mutual_best & pairs.coincident_events.ge(MIN_PAIR_EVENTS)
            & pairs.smaller_unit_fraction.ge(BASE_MATCH_FRACTION)
            & pairs.ks_unit.isin(ke) & pairs.dartsort_unit.isin(de)
        ].copy()
        paired = paired_metrics(baseline, km, dart_metrics); pair_tables.append(paired)
        edges = qualified_edges(pairs, ke, de)
        family, partition = family_rows(arm, ks, dart, edges)
        family_tables.append(family); partition_tables.append(partition)
        summaries.append(summarize_arm(arm, km, dart_metrics, pairs, paired, family, partition))
    units = pd.concat(unit_tables, ignore_index=True)
    paired = pd.concat(pair_tables, ignore_index=True)
    families = pd.concat(family_tables, ignore_index=True)
    partitions = pd.concat(partition_tables, ignore_index=True)
    summary = pd.DataFrame(summaries)
    comparison_metrics = [
        "firing_rate_hz", "presence_ratio_10s", "firing_rate_10s_cv",
        "isi_violations_fraction_1p5ms", "si_rp_contamination_1ms",
        "depth_excursion_um", "depth_bin_median_range_um", "amplitude_cv",
        "amplitude_bin_median_cv", "edge_spike_fraction_40um",
    ]
    pair_summary = comparison_summary(paired, comparison_metrics, seed=20260910)
    pair_good_summary = comparison_summary(paired.loc[paired.ks_good], comparison_metrics, seed=20260912)
    family_summary = comparison_summary(families, comparison_metrics, seed=20260911)
    event_summary = partition_summary(partitions)
    validation = validation_checks(paired, families, summary)
    if not validation["all_required_checks_pass"]:
        raise RuntimeError(f"detailed QC validation failed: {validation}")
    units.to_csv(output / "unit_metrics.csv", index=False)
    paired.to_csv(output / "strict_matched_pair_qc.csv", index=False)
    families.to_csv(output / "correspondence_family_qc.csv", index=False)
    partitions.to_csv(output / "event_partition_qc.csv", index=False)
    summary.to_csv(output / "arm_summary.csv", index=False)
    pair_summary.to_csv(output / "paired_metric_summary.csv", index=False)
    pair_good_summary.to_csv(output / "paired_metric_summary_ks_good.csv", index=False)
    family_summary.to_csv(output / "family_metric_summary.csv", index=False)
    event_summary.to_csv(output / "event_partition_summary.csv", index=False)
    atomic_json(output / "validation.json", validation)
    plot_summary(summary, output)
    result = {
        "schema": SCHEMA, "status": "complete", "interval_s": [930.0, 1030.0],
        "duration_s": DURATION_S, "snippet_profile": {"bin_s": BIN_S, "equal_bins": 10, "drift_min_spikes": DRIFT_MIN_SPIKES},
        "matching": {"tolerance_ms": BASE_TOLERANCE_MS, "depth_um": BASE_DEPTH_UM,
                     "minimum_pair_events": MIN_PAIR_EVENTS, "minimum_unit_spikes": MIN_UNIT_SPIKES,
                     "strict_smaller_unit_fraction": BASE_MATCH_FRACTION},
        "source_hashes": {"fields": sha256(fields_path), "dartsort_npz": sha256(dart_npz),
                          "dartsort_h5": sha256(dart_h5), "dartsort_receipt": sha256(dart_receipt),
                          **{f"{arm}_spikes": sha256(roots[arm] / "kilosort4/sorter_output/spike_times.npy") for arm in ARMS}},
        "row_counts": {"units": len(units), "strict_pairs": len(paired), "families": len(families), "event_partitions": len(partitions),
                       "paired_metric_summaries": len(pair_summary), "family_metric_summaries": len(family_summary)},
        "amplitude_policy": "Within-sorter distributions and exclusive/shared ratios only; KS feature scale and DARTsort denoised PTP are not compared absolutely.",
        "waveform_policy": "No raw voltage read. Cached evidence cannot arbitrate waveform quality of KS-only events absent from DARTsort.",
        "rp_contamination_caveat": "The Hill/SpikeInterface scalar frequently reaches its 1.0 ceiling on dense 100 s family unions; observed 1.5 ms ISI fractions are reported separately and are the primary refractory description.",
        "summaries": json.loads(summary.to_json(orient="records")),
        "validation": validation,
    }
    atomic_json(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("testing/outputs/luke_kilosort_dartsort_detailed_qc_v1"))
    args = parser.parse_args()
    print(json.dumps(run(args.output)["summaries"], indent=2))


if __name__ == "__main__":
    main()
