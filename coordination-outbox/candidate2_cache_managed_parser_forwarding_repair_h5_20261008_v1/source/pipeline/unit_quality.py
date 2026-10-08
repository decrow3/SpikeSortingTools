"""Auditable, sorter-independent unit-quality summaries.

This module intentionally computes only metrics supported by the existing
Kilosort output and legacy QC caches.  Kilosort ``amplitudes.npy`` values are
template-feature scales, not calibrated microvolts, so they are never exposed
as waveform amplitude or used as a proxy for SNR/amplitude cutoff.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .truncation import is_saturated, missing_pct_from_normalisation


STANDARD_QC_SCHEMA = "standard-unit-quality-v1"
DEFAULT_PARAMETERS = {
    "presence_bin_s": 60.0,
    "coarse_presence_bin_s": 300.0,
    "rate_bin_s": 5.0,
    "drift_bin_s": 60.0,
    "drift_min_spikes_per_bin": 100,
    "isi_threshold_ms": 1.5,
    "rp_refractory_ms": 1.0,
    "rp_censored_ms": 0.0,
    "nearby_template_similarity": 0.8,
    "nearby_template_um": 100.0,
    "edge_distance_um": 40.0,
    "truncation_saturation_pct": 50.0,
}


def _load_vector(path: Path, *, dtype=None) -> np.ndarray:
    values = np.load(path, mmap_mode="r").reshape(-1)
    return np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)


def _load_cluster_table(path: Path, value_name: str) -> dict[int, Any]:
    if not path.is_file():
        return {}
    table = pd.read_csv(path, sep="\t")
    if "cluster_id" not in table:
        raise ValueError(f"{path} has no cluster_id column")
    candidates = [column for column in table if column != "cluster_id"]
    if not candidates:
        raise ValueError(f"{path} has no value column")
    column = value_name if value_name in table else candidates[0]
    return dict(zip(table["cluster_id"].astype(int), table[column]))


def _safe_percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else np.nan


def _binned_counts(times_s: np.ndarray, duration_s: float, bin_s: float) -> np.ndarray:
    n_bins = max(1, int(np.ceil(duration_s / bin_s)))
    indices = np.floor(times_s / bin_s).astype(np.int64)
    indices = indices[(indices >= 0) & (indices < n_bins)]
    return np.bincount(indices, minlength=n_bins)


def _pair_violations(times_samples: np.ndarray, threshold_samples: int) -> int:
    """Count all pairs at or below the pinned SI refractory boundary."""
    if threshold_samples <= 0 or times_samples.size < 2:
        return 0
    count = 0
    left = 0
    for right in range(times_samples.size):
        while times_samples[right] - times_samples[left] > threshold_samples:
            left += 1
        count += right - left
    return int(count)


def _isi_metrics(
    times_samples: np.ndarray,
    *,
    sampling_frequency: float,
    duration_s: float,
    isi_threshold_ms: float,
    rp_refractory_ms: float,
    rp_censored_ms: float,
) -> dict[str, float | int]:
    times_samples = np.sort(np.asarray(times_samples, dtype=np.int64))
    n = int(times_samples.size)
    threshold_s = isi_threshold_ms / 1000.0
    censored_s = rp_censored_ms / 1000.0
    # SI 0.102.1 requests spike times in seconds before applying this strict
    # comparison. Preserve that implementation detail, including its floating
    # point boundary semantics, so the named field is parity-testable.
    consecutive = int(np.count_nonzero(np.diff(times_samples / sampling_frequency) < threshold_s))
    violation_time = 2.0 * n * max(0.0, threshold_s - censored_s)
    total_rate = n / duration_s
    isi_ratio = (
        (consecutive / violation_time) / total_rate
        if n and duration_s > 0 and violation_time > 0
        else np.nan
    )

    rp_samples = max(1, int(round(rp_refractory_ms * sampling_frequency / 1000)))
    censored_samples = max(0, int(round(rp_censored_ms * sampling_frequency / 1000)))
    all_rp = _pair_violations(times_samples, rp_samples)
    if censored_samples:
        all_rp -= _pair_violations(times_samples, censored_samples)
    effective_time = duration_s - 2.0 * n * censored_s
    denominator = n * n * max(0.0, rp_refractory_ms / 1000.0 - censored_s)
    discriminant = 1.0 - (all_rp * effective_time / denominator) if denominator > 0 else np.nan
    rp_contamination = (
        1.0 - np.sqrt(discriminant) if np.isfinite(discriminant) and discriminant >= 0 else 1.0
    )
    return {
        "isi_violations_count_1p5ms": consecutive,
        "isi_violations_fraction_1p5ms": consecutive / max(1, n - 1),
        "si_isi_violations_ratio_1p5ms": float(isi_ratio),
        "rp_violations_count_1ms": all_rp,
        "si_rp_contamination_1ms": float(rp_contamination),
    }


def _truncation_by_unit(legacy_qc_dir: Path) -> dict[int, dict[str, Any]]:
    trunc_path = legacy_qc_dir / "amp_truncation/truncation_qc.npz"
    present_path = legacy_qc_dir / "amp_truncation/present_qc.npz"
    if not trunc_path.is_file() or not present_path.is_file():
        return {}
    with np.load(trunc_path) as data:
        cids = np.asarray(data["cid"]).reshape(-1)
        mpcts = np.asarray(data["mpcts"], dtype=float).reshape(-1)
        popts = np.asarray(data["popts"], dtype=float)
    with np.load(present_path) as data:
        present_cids = np.asarray(data["cid"]).reshape(-1)
        valid_blocks = np.asarray(data["valid_blocks"])
    if cids.size != mpcts.size or popts.shape != (cids.size, 3):
        raise ValueError("legacy truncation cache arrays have incompatible shapes")
    if present_cids.size != len(valid_blocks):
        raise ValueError("legacy presence cache arrays have incompatible shapes")
    result: dict[int, dict[str, Any]] = {}
    for cid in np.unique(np.r_[cids, present_cids]).astype(int):
        mask = cids.astype(int) == cid
        values = mpcts[mask]
        saturated = is_saturated(values)
        measured = values[np.isfinite(values) & ~saturated]
        norm = missing_pct_from_normalisation(popts[mask]) if np.any(mask) else np.empty(0)
        disagreements = np.abs(values - norm)
        pmask = present_cids.astype(int) == cid
        eligible_spikes = 0
        for start, stop in np.atleast_2d(valid_blocks[pmask]):
            eligible_spikes += max(0, int(stop) - int(start))
        result[cid] = {
            "legacy_truncation_window_count": int(values.size),
            "legacy_truncation_measured_window_count": int(measured.size),
            "legacy_truncation_saturated_window_count": int(np.count_nonzero(saturated)),
            "legacy_truncation_saturated_fraction": float(np.mean(saturated)) if values.size else np.nan,
            "legacy_truncation_missing_pct_median_uncensored": _safe_percentile(measured, 50),
            "legacy_truncation_missing_pct_p90_uncensored": _safe_percentile(measured, 90),
            "legacy_truncation_fit_disagreement_pct_median": _safe_percentile(disagreements, 50),
            "legacy_truncation_fit_disagreement_gt5_fraction": (
                float(np.mean(disagreements > 5)) if disagreements.size else np.nan
            ),
            "legacy_truncation_eligible_spike_count": eligible_spikes,
        }
    return result


def _template_metrics(
    templates: np.ndarray | None,
    similarity: np.ndarray | None,
    channel_x: np.ndarray,
    channel_y: np.ndarray,
    unit_id: int,
    good_template_ids: set[int],
    params: dict[str, Any],
) -> dict[str, Any]:
    empty = {
        "template_peak_channel": np.nan,
        "template_peak_depth_um": np.nan,
        "template_positive_dominant": pd.NA,
        "max_other_template_similarity": np.nan,
        "nearby_similar_template_count": np.nan,
        "nearby_similar_good_template_count": np.nan,
    }
    if templates is None or unit_id < 0 or unit_id >= len(templates):
        return empty
    template = np.asarray(templates[unit_id], dtype=float)
    if template.ndim != 2 or template.shape[1] != channel_y.size:
        return empty
    channel_ptp = np.ptp(template, axis=0)
    peak_channel = int(np.argmax(channel_ptp))
    trough = abs(float(np.min(template[:, peak_channel])))
    positive = abs(float(np.max(template[:, peak_channel])))
    out = dict(empty)
    out.update({
        "template_peak_channel": peak_channel,
        "template_peak_depth_um": float(channel_y[peak_channel]),
        "template_positive_dominant": bool(positive > trough),
    })
    if similarity is None or unit_id >= similarity.shape[0]:
        return out
    row = np.asarray(similarity[unit_id], dtype=float).copy()
    if unit_id < row.size:
        row[unit_id] = np.nan
    out["max_other_template_similarity"] = float(np.nanmax(row)) if np.any(np.isfinite(row)) else np.nan
    candidates = np.flatnonzero(np.isfinite(row) & (row >= params["nearby_template_similarity"]))
    nearby = []
    for other in candidates:
        if other >= len(templates):
            continue
        other_template = np.asarray(templates[other], dtype=float)
        if other_template.ndim != 2 or other_template.shape[1] != channel_y.size:
            continue
        other_peak = int(np.argmax(np.ptp(other_template, axis=0)))
        distance = np.hypot(channel_x[other_peak] - channel_x[peak_channel], channel_y[other_peak] - channel_y[peak_channel])
        if distance <= params["nearby_template_um"]:
            nearby.append(int(other))
    out["nearby_similar_template_count"] = len(nearby)
    out["nearby_similar_good_template_count"] = sum(other in good_template_ids for other in nearby)
    return out


def metric_definitions(parameters: dict[str, Any]) -> dict[str, Any]:
    """Machine-readable definitions, provenance, units, and caveats."""
    metrics = {
        "unit_id": {"unit": "identifier", "source": "spike_clusters.npy", "definition": "Curated cluster identifier used for all joins."},
        "ks_label": {"unit": "category", "source": "cluster_KSLabel.tsv", "definition": "Kilosort good/mua label carried through curation."},
        "ks_good": {"unit": "boolean", "source": "cluster_KSLabel.tsv", "definition": "True when the normalized Kilosort label is good."},
        "ks_contamination_pct": {"unit": "percent", "source": "cluster_ContamPct.tsv", "definition": "Kilosort contamination estimate in its stored percent units."},
        "ks_contamination_fraction": {"unit": "fraction", "source": "cluster_ContamPct.tsv", "definition": "Kilosort contamination percentage divided by 100."},
        "num_spikes": {"unit": "spikes", "source": "spike arrays", "definition": "Number of spikes assigned to the curated unit."},
        "first_spike_s": {"unit": "s", "source": "spike_times.npy", "definition": "First unit spike relative to recording start."},
        "last_spike_s": {"unit": "s", "source": "spike_times.npy", "definition": "Last unit spike relative to recording start."},
        "unit_lifetime_s": {"unit": "s", "source": "spike_times.npy", "definition": "Last minus first unit spike time."},
        "firing_rate_hz": {"unit": "Hz", "source": "spike_times.npy", "definition": "Spike count divided by full recording duration."},
        "presence_ratio_60s": {"unit": "fraction", "source": "spike_times.npy", "definition": "Fraction of 60 s bins over the full recording containing at least one spike."},
        "presence_ratio_300s": {"unit": "fraction", "source": "spike_times.npy", "definition": "Fraction of 300 s bins over the full recording containing at least one spike."},
        "active_bins_60s": {"unit": "bins", "source": "spike_times.npy", "definition": "Number of occupied 60 s bins."},
        "total_bins_60s": {"unit": "bins", "source": "recording metadata", "definition": "Ceiling of full recording duration divided by 60 s."},
        "firing_rate_5s_p05_hz": {"unit": "Hz", "source": "spike_times.npy", "definition": "5th percentile of spike count per 5 s bin divided by 5 s."},
        "firing_rate_5s_p95_hz": {"unit": "Hz", "source": "spike_times.npy", "definition": "95th percentile of spike count per 5 s bin divided by 5 s."},
        "firing_rate_5s_range_hz": {"unit": "Hz", "source": "spike_times.npy", "definition": "95th minus 5th percentile of 5 s binned firing rate."},
        "firing_rate_300s_cv": {"unit": "ratio", "source": "spike_times.npy", "definition": "Population standard deviation divided by mean of spike counts in 300 s bins."},
        "ks_feature_amplitude_median": {"unit": "Kilosort feature scale", "source": "amplitudes.npy", "definition": "Median Kilosort template-feature amplitude; not calibrated voltage."},
        "ks_feature_amplitude_p05": {"unit": "Kilosort feature scale", "source": "amplitudes.npy", "definition": "5th percentile Kilosort feature amplitude."},
        "ks_feature_amplitude_p95": {"unit": "Kilosort feature scale", "source": "amplitudes.npy", "definition": "95th percentile Kilosort feature amplitude."},
        "ks_feature_amplitude_cv": {"unit": "ratio", "source": "amplitudes.npy", "definition": "Population standard deviation divided by mean feature amplitude."},
        "depth_median_um": {"unit": "um", "source": "spike_positions.npy", "definition": "Median finite per-spike y position."},
        "depth_p05_um": {"unit": "um", "source": "spike_positions.npy", "definition": "5th percentile finite per-spike y position."},
        "depth_p95_um": {"unit": "um", "source": "spike_positions.npy", "definition": "95th percentile finite per-spike y position."},
        "depth_excursion_um": {"unit": "um", "source": "spike_positions.npy", "definition": "95th minus 5th percentile of per-spike y position."},
        "drift_valid_bin_count": {"unit": "bins", "source": "spike positions/times", "definition": "Number of 60 s bins meeting the configured finite-spike minimum."},
        "drift_depth_range_um": {"unit": "um", "source": "spike positions/times", "definition": "Range of median depth across valid 60 s bins; unavailable with fewer than two bins."},
        "drift_depth_std_um": {"unit": "um", "source": "spike positions/times", "definition": "Population standard deviation of median depth across valid 60 s bins."},
        "edge_spike_fraction_40um": {"unit": "fraction", "source": "spike positions and ops geometry", "definition": "Fraction of finite-depth spikes within 40 um of either recorded depth boundary."},
        "isi_violations_count_1p5ms": {"unit": "events", "source": "spike_times.npy", "definition": "Number of consecutive ISIs strictly below 1.5 ms using SpikeInterface 0.102.1 seconds comparison."},
        "isi_violations_fraction_1p5ms": {"unit": "fraction", "source": "spike_times.npy", "definition": "1.5 ms consecutive violation count divided by the number of ISIs."},
        "si_isi_violations_ratio_1p5ms": {"unit": "ratio", "source": "spike_times.npy", "definition": "SpikeInterface 0.102.1 isi_violations ratio formula at 1.5 ms and zero censored period."},
        "rp_violations_count_1ms": {"unit": "pairs", "source": "spike_times.npy", "definition": "All within-unit spike pairs separated by at most 1 ms, matching SpikeInterface 0.102.1 boundary semantics."},
        "si_rp_contamination_1ms": {"unit": "fraction", "source": "spike_times.npy", "definition": "Hill/SpikeInterface 0.102.1 refractory-period contamination formula at 1 ms and zero censored period."},
        "template_peak_channel": {"unit": "zero-based channel index", "source": "templates.npy", "definition": "Channel with largest template peak-to-peak range."},
        "template_peak_depth_um": {"unit": "um", "source": "templates.npy and ops geometry", "definition": "Probe y coordinate of template peak channel."},
        "template_positive_dominant": {"unit": "boolean", "source": "templates.npy", "definition": "Positive peak magnitude exceeds trough magnitude on the peak channel."},
        "max_other_template_similarity": {"unit": "similarity", "source": "similar_templates.npy", "definition": "Maximum similarity to another template, with the diagonal excluded."},
        "nearby_similar_template_count": {"unit": "templates", "source": "templates/similarity/geometry", "definition": "Other templates at or above configured similarity whose peak channels are within configured Euclidean distance."},
        "nearby_similar_good_template_count": {"unit": "templates", "source": "templates/similarity/geometry/labels", "definition": "Nearby similar templates whose Kilosort label is good."},
        "legacy_truncation_window_count": {"unit": "windows", "source": "legacy truncation cache", "definition": "All stored amplitude-truncation windows, including boundary-censored fits."},
        "legacy_truncation_measured_window_count": {"unit": "windows", "source": "legacy truncation cache", "definition": "Finite windows excluding 50 percent boundary-censored fits."},
        "legacy_truncation_saturated_window_count": {"unit": "windows", "source": "legacy truncation cache", "definition": "Fits pinned at the 50 percent upper reporting boundary."},
        "legacy_truncation_saturated_fraction": {"unit": "fraction", "source": "legacy truncation cache", "definition": "Boundary-censored windows divided by all stored windows."},
        "legacy_truncation_missing_pct_median_uncensored": {"unit": "percent", "source": "legacy truncation cache", "definition": "Median fitted missing percentage after excluding 50 percent boundary-pinned estimates."},
        "legacy_truncation_missing_pct_p90_uncensored": {"unit": "percent", "source": "legacy truncation cache", "definition": "90th percentile fitted missing percentage after excluding boundary-pinned estimates."},
        "legacy_truncation_fit_disagreement_pct_median": {"unit": "percentage points", "source": "legacy truncation cache", "definition": "Median absolute disagreement between shape-derived missingness and the independent fit-normalisation estimate."},
        "legacy_truncation_fit_disagreement_gt5_fraction": {"unit": "fraction", "source": "legacy truncation cache", "definition": "Fraction of windows whose two fit-derived missingness estimates differ by more than five percentage points."},
        "legacy_truncation_eligible_spike_count": {"unit": "spikes", "source": "legacy presence cache", "definition": "Total half-open valid-block lengths selected by the legacy 1,000-spike eligibility rule."},
    }
    return {
        "schema_version": STANDARD_QC_SCHEMA,
        "parameters": parameters,
        "global_notes": [
            "Kilosort feature amplitude is dimensionless and is not waveform amplitude in microvolts.",
            "Presence ratios use the full verified recording duration, including empty bins.",
            "SpikeInterface-named ISI/RP fields reproduce the documented formulas without requiring a waveform analyzer.",
            "Legacy truncation estimates at 50 percent are censored lower bounds and are excluded from uncensored summaries.",
        ],
        "metrics": metrics,
    }


def quality_policy(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": STANDARD_QC_SCHEMA,
        "decision_semantics": "Reference profiles are annotations, not automatic curation decisions.",
        "profiles": {
            "kilosort": {"evaluable": True, "fields": ["ks_label", "ks_contamination_fraction"]},
            "allen_like": {"evaluable": False, "missing": ["calibrated_waveform_amplitude_uv", "amplitude_cutoff", "presence_ratio_100_equal_bins"]},
            "ibl_like": {"evaluable": False, "missing": ["sliding_refractory_pass", "noise_cutoff", "calibrated_waveform_amplitude_uv"]},
            "spikeinterface_example": {"evaluable": False, "missing": ["amplitude_cutoff", "snr"]},
            "lab_evidence": {"evaluable": True, "hard_gate": False, "description": "Warnings expose instability, collision, truncation, and duplicate risk without changing unit labels."},
        },
        "parameters": parameters,
    }


def build_unit_quality_tables(
    curated_output: Path,
    legacy_qc_dir: Path,
    *,
    sampling_frequency: float,
    duration_s: float,
    parameters: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build per-unit metric and policy tables without reading raw voltage."""
    curated_output = Path(curated_output)
    legacy_qc_dir = Path(legacy_qc_dir)
    params = {**DEFAULT_PARAMETERS, **(parameters or {})}
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be positive")

    spike_times = _load_vector(curated_output / "spike_times.npy", dtype=np.int64)
    spike_clusters = _load_vector(curated_output / "spike_clusters.npy", dtype=np.int64)
    if spike_times.size != spike_clusters.size:
        raise ValueError("spike_times.npy and spike_clusters.npy have different lengths")
    if np.any(spike_times < 0) or np.any(spike_times >= round(duration_s * sampling_frequency)):
        raise ValueError("spike time falls outside the verified recording duration")
    amplitudes = _load_vector(curated_output / "amplitudes.npy", dtype=float) if (curated_output / "amplitudes.npy").is_file() else None
    positions = np.load(curated_output / "spike_positions.npy", mmap_mode="r") if (curated_output / "spike_positions.npy").is_file() else None
    for name, values in (("amplitudes", amplitudes), ("spike_positions", positions)):
        if values is not None and len(values) != spike_times.size:
            raise ValueError(f"{name} has a different spike count")

    labels = _load_cluster_table(curated_output / "cluster_KSLabel.tsv", "KSLabel")
    contamination = _load_cluster_table(curated_output / "cluster_ContamPct.tsv", "ContamPct")
    templates = np.load(curated_output / "templates.npy", mmap_mode="r") if (curated_output / "templates.npy").is_file() else None
    similarity = np.load(curated_output / "similar_templates.npy", mmap_mode="r") if (curated_output / "similar_templates.npy").is_file() else None
    ops_path = curated_output / "ops.npy"
    ops = np.load(ops_path, allow_pickle=True).item() if ops_path.is_file() else {}
    channel_x = np.asarray(ops.get("xc", []), dtype=float).reshape(-1)
    channel_y = np.asarray(ops.get("yc", []), dtype=float).reshape(-1)
    if templates is not None and (channel_x.size != templates.shape[-1] or channel_y.size != templates.shape[-1]):
        channel_x = np.arange(templates.shape[-1], dtype=float)
        channel_y = np.arange(templates.shape[-1], dtype=float)
    good_ids = {cid for cid, label in labels.items() if str(label).strip().lower() == "good"}
    truncation = _truncation_by_unit(legacy_qc_dir)

    unit_ids = np.unique(spike_clusters)
    rows: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []
    for unit_id in unit_ids:
        mask = spike_clusters == unit_id
        samples = np.sort(spike_times[mask])
        times_s = samples / sampling_frequency
        counts60 = _binned_counts(times_s, duration_s, params["presence_bin_s"])
        counts300 = _binned_counts(times_s, duration_s, params["coarse_presence_bin_s"])
        counts5 = _binned_counts(times_s, duration_s, params["rate_bin_s"])
        rates5 = counts5 / params["rate_bin_s"]
        depth = np.asarray(positions[mask, 1], dtype=float) if positions is not None and positions.ndim == 2 and positions.shape[1] >= 2 else np.empty(0)
        finite_depth = depth[np.isfinite(depth)]
        depth_medians = []
        if depth.size:
            bins = np.floor(times_s / params["drift_bin_s"]).astype(int)
            for bin_id in np.unique(bins):
                values = depth[bins == bin_id]
                values = values[np.isfinite(values)]
                if values.size >= params["drift_min_spikes_per_bin"]:
                    depth_medians.append(float(np.median(values)))
        depth_medians_array = np.asarray(depth_medians)
        amp = np.asarray(amplitudes[mask], dtype=float) if amplitudes is not None else np.empty(0)
        amp = amp[np.isfinite(amp)]
        amp_mean = float(np.mean(amp)) if amp.size else np.nan
        label = str(labels.get(int(unit_id), "unlabelled"))
        contam_pct = pd.to_numeric(pd.Series([contamination.get(int(unit_id), np.nan)]), errors="coerce").iloc[0]
        row: dict[str, Any] = {
            "unit_id": int(unit_id),
            "ks_label": label,
            "ks_good": label.strip().lower() == "good",
            "ks_contamination_pct": float(contam_pct) if np.isfinite(contam_pct) else np.nan,
            "ks_contamination_fraction": float(contam_pct / 100.0) if np.isfinite(contam_pct) else np.nan,
            "num_spikes": int(samples.size),
            "first_spike_s": float(times_s[0]),
            "last_spike_s": float(times_s[-1]),
            "unit_lifetime_s": float(times_s[-1] - times_s[0]),
            "firing_rate_hz": float(samples.size / duration_s),
            "presence_ratio_60s": float(np.mean(counts60 > 0)),
            "presence_ratio_300s": float(np.mean(counts300 > 0)),
            "active_bins_60s": int(np.count_nonzero(counts60)),
            "total_bins_60s": int(counts60.size),
            "firing_rate_5s_p05_hz": _safe_percentile(rates5, 5),
            "firing_rate_5s_p95_hz": _safe_percentile(rates5, 95),
            "firing_rate_5s_range_hz": _safe_percentile(rates5, 95) - _safe_percentile(rates5, 5),
            "firing_rate_300s_cv": float(np.std(counts300) / np.mean(counts300)) if np.mean(counts300) > 0 else np.nan,
            "ks_feature_amplitude_median": _safe_percentile(amp, 50),
            "ks_feature_amplitude_p05": _safe_percentile(amp, 5),
            "ks_feature_amplitude_p95": _safe_percentile(amp, 95),
            "ks_feature_amplitude_cv": float(np.std(amp) / amp_mean) if amp.size and amp_mean != 0 else np.nan,
            "depth_median_um": _safe_percentile(finite_depth, 50),
            "depth_p05_um": _safe_percentile(finite_depth, 5),
            "depth_p95_um": _safe_percentile(finite_depth, 95),
            "depth_excursion_um": _safe_percentile(finite_depth, 95) - _safe_percentile(finite_depth, 5),
            "drift_valid_bin_count": int(depth_medians_array.size),
            "drift_depth_range_um": float(np.ptp(depth_medians_array)) if depth_medians_array.size >= 2 else np.nan,
            "drift_depth_std_um": float(np.std(depth_medians_array)) if depth_medians_array.size >= 2 else np.nan,
            "edge_spike_fraction_40um": float(np.mean((finite_depth <= np.min(channel_y) + params["edge_distance_um"]) | (finite_depth >= np.max(channel_y) - params["edge_distance_um"]))) if finite_depth.size and channel_y.size else np.nan,
        }
        row.update(_isi_metrics(samples, sampling_frequency=sampling_frequency, duration_s=duration_s,
                                isi_threshold_ms=params["isi_threshold_ms"], rp_refractory_ms=params["rp_refractory_ms"],
                                rp_censored_ms=params["rp_censored_ms"]))
        row.update(_template_metrics(templates, similarity, channel_x, channel_y, int(unit_id), good_ids, params))
        row.update(truncation.get(int(unit_id), {
            "legacy_truncation_window_count": 0, "legacy_truncation_measured_window_count": 0,
            "legacy_truncation_saturated_window_count": 0, "legacy_truncation_saturated_fraction": np.nan,
            "legacy_truncation_missing_pct_median_uncensored": np.nan,
            "legacy_truncation_missing_pct_p90_uncensored": np.nan,
            "legacy_truncation_fit_disagreement_pct_median": np.nan,
            "legacy_truncation_fit_disagreement_gt5_fraction": np.nan,
            "legacy_truncation_eligible_spike_count": 0,
        }))
        rows.append(row)

        warnings = []
        if not row["ks_good"]:
            warnings.append("kilosort_not_good")
        if row["presence_ratio_60s"] < 0.9:
            warnings.append("presence_ratio_60s_below_0p9")
        if row["si_rp_contamination_1ms"] > 0.1:
            warnings.append("rp_contamination_1ms_above_0p1")
        if row["nearby_similar_good_template_count"] and row["nearby_similar_good_template_count"] > 0:
            warnings.append("nearby_similar_good_template")
        if row["legacy_truncation_saturated_window_count"] > 0:
            warnings.append("truncation_has_censored_windows")
        if row["legacy_truncation_window_count"] == 0:
            warnings.append("truncation_not_eligible")
        flags.append({
            "unit_id": int(unit_id),
            "ks_status": "pass" if row["ks_good"] else "fail",
            "allen_like_status": "not_evaluable",
            "ibl_like_status": "not_evaluable",
            "spikeinterface_example_status": "not_evaluable",
            "lab_qc_status": "evidence_only",
            "lab_qc_warning_count": len(warnings),
            "lab_qc_warnings": ";".join(warnings),
            "automatic_curation_action": "none",
        })

    metrics = pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)
    flag_table = pd.DataFrame(flags).sort_values("unit_id").reset_index(drop=True)
    summary = {
        "schema_version": STANDARD_QC_SCHEMA,
        "unit_count": int(len(metrics)),
        "spike_count": int(spike_times.size),
        "kilosort_good_unit_count": int(metrics["ks_good"].sum()),
        "units_with_legacy_truncation_windows": int((metrics["legacy_truncation_window_count"] > 0).sum()),
        "units_with_any_lab_warning": int((flag_table["lab_qc_warning_count"] > 0).sum()),
        "recording_duration_s": float(duration_s),
        "sampling_frequency_hz": float(sampling_frequency),
        "automatic_curation_changes": 0,
    }
    return metrics, flag_table, metric_definitions(params), quality_policy(params), summary


def _atomic_csv(path: Path, table: pd.DataFrame) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    table.to_csv(partial, index=False)
    os.replace(partial, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(partial, path)


def write_unit_quality_artifacts(
    output_dir: Path,
    metrics: pd.DataFrame,
    flags: pd.DataFrame,
    definitions: dict[str, Any],
    policy: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output_dir / "unit_quality_metrics.csv", metrics)
    _atomic_csv(output_dir / "unit_quality_flags.csv", flags)
    _atomic_json(output_dir / "metric_definitions.json", definitions)
    _atomic_json(output_dir / "quality_policy.json", policy)
    _atomic_json(output_dir / "quality_summary.json", summary)
