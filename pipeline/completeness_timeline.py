"""Time-resolved amplitude-completeness screening with explicit fit trust."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .truncation import is_saturated, missing_pct_from_normalisation


COMPLETENESS_TIMELINE_SCHEMA = "amplitude-completeness-timeline-v2"
DEFAULT_TIMELINE_PARAMETERS = {
    "fit_disagreement_warning_pp": 5.0,
    "deterioration_screen_pp": 10.0,
    "baseline_quantile": 0.10,
    "historical_window_semantics": "stored stop index is inclusive for time/support, while the historical fitter consumed the stop-exclusive amplitude slice",
}
TIMELINE_COLUMNS = (
    "unit_id", "cache_row", "unit_window_index", "start_spike_index",
    "stop_spike_index_inclusive", "nominal_window_spikes",
    "historical_fit_sample_count", "start_s", "stop_s", "missing_pct_shape",
    "missing_pct_normalization", "fit_disagreement_pp", "measurement_status",
)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(text)
    os.replace(partial, path)


def classify_fit_trust(shape_missing, popts, *, disagreement_warning_pp: float = 5.0):
    """Shared fit-trust policy for the screening sidecar and paired comparisons."""
    if not np.isfinite(disagreement_warning_pp) or disagreement_warning_pp < 0:
        raise ValueError("fit disagreement boundary must be finite and nonnegative")
    shape_missing = np.asarray(shape_missing, dtype=float).reshape(-1)
    popts = np.asarray(popts, dtype=float)
    if popts.shape != (len(shape_missing), 3):
        raise ValueError("fit parameters must have one three-parameter row per estimate")
    normalization = missing_pct_from_normalisation(popts)
    disagreement = np.abs(shape_missing - normalization)
    status = np.full(len(shape_missing), "measured", dtype=object)
    status[disagreement > disagreement_warning_pp] = "poor_fit_disagreement"
    status[is_saturated(shape_missing)] = "censored_at_least_50pct"
    status[~np.isfinite(shape_missing) | ~np.isfinite(normalization)] = "nonfinite"
    return normalization, disagreement, status


def build_completeness_timeline(
    curated_output: Path,
    legacy_qc_dir: Path,
    *,
    sampling_frequency: float,
    parameters: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Map historical truncation fits onto physical time without overstating them."""
    params = {**DEFAULT_TIMELINE_PARAMETERS, **(parameters or {})}
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive")
    curated_output, legacy_qc_dir = Path(curated_output), Path(legacy_qc_dir)
    spike_times = np.load(curated_output / "spike_times.npy", mmap_mode="r").reshape(-1)
    spike_clusters = np.load(curated_output / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    if len(spike_times) != len(spike_clusters):
        raise ValueError("spike time and cluster arrays differ in length")
    if not np.all(np.isfinite(spike_clusters)) or not np.all(spike_clusters == np.rint(spike_clusters)):
        raise ValueError("spike cluster ids must be finite integers")
    with np.load(legacy_qc_dir / "amp_truncation/truncation_qc.npz") as cache:
        cids = np.asarray(cache["cid"]).reshape(-1)
        blocks = np.asarray(cache["window_blocks"], dtype=np.int64)
        popts = np.asarray(cache["popts"], dtype=float)
        shape_missing = np.asarray(cache["mpcts"], dtype=float).reshape(-1)
    with np.load(legacy_qc_dir / "amp_truncation/present_qc.npz") as cache:
        present_cids = np.asarray(cache["cid"]).reshape(-1)
        valid_blocks = np.asarray(cache["valid_blocks"], dtype=np.int64)
    if blocks.shape != (len(cids), 2) or popts.shape != (len(cids), 3) or len(shape_missing) != len(cids):
        raise ValueError("truncation cache arrays have incompatible shapes")
    if not np.all(np.isfinite(cids)) or not np.all(cids == np.rint(cids)):
        raise ValueError("truncation cache unit ids must be finite integers")
    if valid_blocks.shape != (len(present_cids), 2):
        raise ValueError("presence cache arrays have incompatible shapes")
    if not np.all(np.isfinite(present_cids)) or not np.all(present_cids == np.rint(present_cids)):
        raise ValueError("presence cache unit ids must be finite integers")
    normalization_missing, disagreement, fit_status = classify_fit_trust(
        shape_missing, popts,
        disagreement_warning_pp=float(params["fit_disagreement_warning_pp"]),
    )
    rows: list[dict[str, Any]] = []
    per_unit_times: dict[int, np.ndarray] = {}
    for row_index, cid_value in enumerate(cids):
        cid = int(cid_value)
        if cid not in per_unit_times:
            per_unit_times[cid] = np.asarray(spike_times[spike_clusters == cid], dtype=np.int64)
        times = per_unit_times[cid]
        first, last = map(int, blocks[row_index])
        if first < 0 or last < first or last >= len(times):
            raise ValueError(f"truncation window outside unit {cid} spike train")
        status = fit_status[row_index]
        rows.append({
            "unit_id": cid,
            "cache_row": row_index,
            "unit_window_index": 0,
            "start_spike_index": first,
            "stop_spike_index_inclusive": last,
            "nominal_window_spikes": last - first + 1,
            "historical_fit_sample_count": last - first,
            "start_s": float(times[first] / sampling_frequency),
            "stop_s": float(times[last] / sampling_frequency),
            "missing_pct_shape": float(shape_missing[row_index]),
            "missing_pct_normalization": float(normalization_missing[row_index]),
            "fit_disagreement_pp": float(disagreement[row_index]),
            "measurement_status": status,
        })
    timeline = pd.DataFrame(rows, columns=TIMELINE_COLUMNS)
    if not timeline.empty:
        timeline["unit_window_index"] = timeline.groupby("unit_id", sort=False).cumcount()
    unit_rows = []
    baselines: dict[int, float] = {}
    all_unit_ids = np.unique(spike_clusters).astype(int)
    for cid in all_unit_ids:
        frame = timeline[timeline.unit_id == cid]
        measured = frame.loc[frame.measurement_status == "measured", "missing_pct_shape"].to_numpy(float)
        baseline = float(np.quantile(measured, float(params["baseline_quantile"]))) if len(measured) else np.nan
        baselines[int(cid)] = baseline
        unit_spikes = np.asarray(spike_times[spike_clusters == cid], dtype=np.int64)
        present_mask = present_cids.astype(int) == cid
        eligible_spikes = int(np.sum(
            np.maximum(0, valid_blocks[present_mask, 1] - valid_blocks[present_mask, 0])
        )) if np.any(present_mask) else 0
        unit_rows.append({
            "unit_id": int(cid), "spike_count": int(len(unit_spikes)),
            "first_spike_s": float(unit_spikes[0] / sampling_frequency) if len(unit_spikes) else np.nan,
            "last_spike_s": float(unit_spikes[-1] / sampling_frequency) if len(unit_spikes) else np.nan,
            "eligible_spike_count": eligible_spikes,
            "window_count": int(len(frame)), "has_supported_window": bool(len(frame)),
            "measured_window_count": int(np.count_nonzero(frame.measurement_status == "measured")),
            "censored_window_count": int(np.count_nonzero(frame.measurement_status == "censored_at_least_50pct")),
            "poor_fit_window_count": int(np.count_nonzero(frame.measurement_status == "poor_fit_disagreement")),
            "nonfinite_window_count": int(np.count_nonzero(frame.measurement_status == "nonfinite")),
            "screening_baseline_missing_pct": baseline,
        })
    units = pd.DataFrame(unit_rows)
    if timeline.empty:
        timeline["screening_baseline_missing_pct"] = pd.Series(dtype=float)
        timeline["deterioration_from_baseline_pp"] = pd.Series(dtype=float)
        timeline["deterioration_screen"] = pd.Series(dtype=bool)
    else:
        timeline["screening_baseline_missing_pct"] = timeline.unit_id.map(baselines)
        timeline["deterioration_from_baseline_pp"] = (
            timeline.missing_pct_shape - timeline.screening_baseline_missing_pct
        )
        timeline["deterioration_screen"] = (
            (timeline.measurement_status == "measured")
            & (timeline.deterioration_from_baseline_pp >= float(params["deterioration_screen_pp"]))
        )
    policy = {
        "schema_version": COMPLETENESS_TIMELINE_SCHEMA,
        "parameters": params,
        "interpretation": {
            "measured": "Both fit-derived estimates are finite, the shape estimate is not pinned at 50%, and their disagreement is within the warning boundary.",
            "censored_at_least_50pct": "The fit reached its reporting boundary; severity is at least 50% and is not a measured value.",
            "poor_fit_disagreement": "The two estimates disagree beyond the boundary; do not use the shape estimate as a reliability value.",
            "nonfinite": "The cached fit is not numerically interpretable.",
            "deterioration_screen": "A nomination flag relative to the unit's trustworthy-window baseline. It is not proof of lost spikes and must not automatically exclude data or alter labels.",
            "absence_of_window": "Unsupported by amplitude completeness. No row means no fit, not zero missingness or reliable detection.",
        },
    }
    summary = {
        "schema_version": COMPLETENESS_TIMELINE_SCHEMA,
        "unit_count": int(len(units)), "window_count": int(len(timeline)),
        "units_without_supported_windows": int(np.count_nonzero(~units.has_supported_window)) if len(units) else 0,
        "measured_window_count": int(np.count_nonzero(timeline.measurement_status == "measured")) if len(timeline) else 0,
        "censored_window_count": int(np.count_nonzero(timeline.measurement_status == "censored_at_least_50pct")) if len(timeline) else 0,
        "poor_fit_window_count": int(np.count_nonzero(timeline.measurement_status == "poor_fit_disagreement")) if len(timeline) else 0,
        "nonfinite_window_count": int(np.count_nonzero(timeline.measurement_status == "nonfinite")) if len(timeline) else 0,
        "deterioration_screen_count": int(timeline.deterioration_screen.sum()) if len(timeline) else 0,
        "screening_only": True,
    }
    return timeline, units, policy, summary


def write_completeness_timeline(output_dir: Path, timeline: pd.DataFrame, units: pd.DataFrame,
                                policy: dict[str, Any], summary: dict[str, Any]) -> None:
    output_dir = Path(output_dir)
    _atomic_text(output_dir / "amplitude_completeness_timeline.csv", timeline.to_csv(index=False))
    _atomic_text(output_dir / "amplitude_completeness_units.csv", units.to_csv(index=False))
    _atomic_text(output_dir / "amplitude_completeness_policy.json", json.dumps(policy, indent=2) + "\n")
    _atomic_text(output_dir / "amplitude_completeness_summary.json", json.dumps(summary, indent=2) + "\n")
