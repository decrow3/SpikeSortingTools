"""Screen whether external-nonrigid losses are confined to high-field periods."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.config import fingerprint
from testing.sort_comparison import exclusive_count


SCHEMA = "luke-improved-nonrigid-regime-screen-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_trains(times: np.ndarray, clusters: np.ndarray, wanted: np.ndarray) -> dict[int, np.ndarray]:
    """Group selected units once while preserving each train's time order."""
    times = np.asarray(times).reshape(-1)
    clusters = np.asarray(clusters).reshape(-1)
    wanted = np.unique(np.asarray(wanted, dtype=np.int64))
    if len(times) != len(clusters) or np.any(np.diff(times) < 0):
        raise ValueError("spike arrays must be aligned and globally time-sorted")
    keep = np.isin(clusters, wanted)
    kept_times, kept_clusters = times[keep], clusters[keep]
    order = np.argsort(kept_clusters, kind="stable")
    grouped_clusters, grouped_times = kept_clusters[order], kept_times[order]
    return {
        int(unit): grouped_times[
            np.searchsorted(grouped_clusters, unit, side="left") :
            np.searchsorted(grouped_clusters, unit, side="right")
        ]
        for unit in wanted
    }


def field_exposure(field_path: Path, *, duration_s: float, bin_s: float) -> pd.DataFrame:
    """Define full, nonoverlapping field-only quiet/high exposure bins."""
    bins = int(np.floor(duration_s / bin_s))
    with np.load(field_path, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=float)
        displacement = np.asarray(data["nonrigid_displacement_um"], dtype=float)
    if displacement.shape[0] != len(time_s) or displacement.ndim != 2:
        raise ValueError("invalid field axes")
    index = np.floor(time_s / bin_s).astype(int)
    keep = (index >= 0) & (index < bins)
    index = index[keep]
    rms_sample = np.sqrt(np.mean(np.square(displacement[keep]), axis=1))
    spread_sample = np.ptp(displacement[keep], axis=1)
    counts = np.bincount(index, minlength=bins)
    if np.any(counts == 0):
        raise ValueError("field has an empty full-duration exposure bin")
    rms = np.bincount(index, weights=rms_sample, minlength=bins) / counts
    spread = np.bincount(index, weights=spread_sample, minlength=bins) / counts
    return pd.DataFrame({
        "bin": np.arange(bins),
        "start_s": np.arange(bins) * bin_s,
        "end_s": (np.arange(bins) + 1) * bin_s,
        "field_rms_um": rms,
        "field_spread_um": spread,
    })


def _regime_metrics(a: np.ndarray, b: np.ndarray, intervals: np.ndarray,
                    *, tolerance: int, refractory: int) -> dict[str, float | int]:
    acount = bcount = matched = arv = brv = aisi = bisi = 0
    for start, stop in intervals:
        alo, ahi = np.searchsorted(a, (start, stop))
        blo, bhi = np.searchsorted(b, (start, stop))
        aa, bb = a[alo:ahi], b[blo:bhi]
        acount += len(aa)
        bcount += len(bb)
        matched += exclusive_count(aa, bb, tolerance)
        arv += int(np.sum(np.diff(aa) < refractory))
        brv += int(np.sum(np.diff(bb) < refractory))
        aisi += max(len(aa) - 1, 0)
        bisi += max(len(bb) - 1, 0)
    return {
        "baseline_events": acount,
        "candidate_events": bcount,
        "matched_events": matched,
        "baseline_retention": matched / acount if acount else np.nan,
        "candidate_retention": matched / bcount if bcount else np.nan,
        "candidate_baseline_count_ratio": bcount / acount if acount else np.nan,
        "baseline_refractory_fraction": arv / max(aisi, 1),
        "candidate_refractory_fraction": brv / max(bisi, 1),
    }


def analyse(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fs = float(config["sampling_frequency_hz"])
    exposure = field_exposure(
        Path(config["field_file"]), duration_s=float(config["duration_s"]),
        bin_s=float(config["bin_s"]),
    )
    quiet_cut = float(exposure.field_rms_um.quantile(config["quiet_quantile"]))
    high_cut = float(exposure.field_rms_um.quantile(config["high_quantile"]))
    exposure["regime"] = "middle"
    exposure.loc[exposure.field_rms_um <= quiet_cut, "regime"] = "quiet"
    exposure.loc[exposure.field_rms_um >= high_cut, "regime"] = "high"

    matches = pd.read_csv(config["primary_matches"])
    matches = matches[matches.interior_primary.astype(bool)].copy()
    baseline_units = matches.baseline_cluster.astype(int).to_numpy()
    candidate_units = matches.candidate_cluster.astype(int).to_numpy()
    baseline_path, candidate_path = Path(config["baseline_curated"]), Path(config["candidate_curated"])
    baseline = selected_trains(
        np.load(baseline_path / "spike_times.npy", mmap_mode="r"),
        np.load(baseline_path / "spike_clusters.npy", mmap_mode="r"), baseline_units,
    )
    candidate = selected_trains(
        np.load(candidate_path / "spike_times.npy", mmap_mode="r"),
        np.load(candidate_path / "spike_clusters.npy", mmap_mode="r"), candidate_units,
    )
    tolerance = int(round(config["correspondence_tolerance_ms"] * 1e-3 * fs))
    refractory = int(round(config["refractory_period_ms"] * 1e-3 * fs))
    rows = []
    for match in matches.itertuples(index=False):
        for regime in ("quiet", "high"):
            selected = exposure[exposure.regime == regime]
            intervals = np.c_[selected.start_s.to_numpy() * fs, selected.end_s.to_numpy() * fs].astype(np.int64)
            metrics = _regime_metrics(
                baseline[int(match.baseline_cluster)], candidate[int(match.candidate_cluster)],
                intervals, tolerance=tolerance, refractory=refractory,
            )
            rows.append({
                "baseline_cluster": int(match.baseline_cluster),
                "candidate_cluster": int(match.candidate_cluster),
                "regime": regime,
                **metrics,
            })
    unit_regime = pd.DataFrame(rows)
    wide = unit_regime.pivot(index=["baseline_cluster", "candidate_cluster"], columns="regime")
    minimum = int(config["minimum_events_per_arm_per_regime"])
    supported = (
        (wide[("baseline_events", "quiet")] >= minimum)
        & (wide[("candidate_events", "quiet")] >= minimum)
        & (wide[("baseline_events", "high")] >= minimum)
        & (wide[("candidate_events", "high")] >= minimum)
    )
    pair_rows = []
    for index, row in wide.iterrows():
        pair_rows.append({
            "baseline_cluster": int(index[0]),
            "candidate_cluster": int(index[1]),
            "supported": bool(supported.loc[index]),
            "high_minus_quiet_baseline_retention": row[("baseline_retention", "high")] - row[("baseline_retention", "quiet")],
            "high_minus_quiet_candidate_retention": row[("candidate_retention", "high")] - row[("candidate_retention", "quiet")],
            "high_to_quiet_candidate_baseline_count_ratio": row[("candidate_baseline_count_ratio", "high")] / row[("candidate_baseline_count_ratio", "quiet")],
            "high_candidate_minus_baseline_refractory": row[("candidate_refractory_fraction", "high")] - row[("baseline_refractory_fraction", "high")],
        })
    pair_summary = pd.DataFrame(pair_rows)
    qualified = pair_summary[pair_summary.supported]
    medians = {
        column: (float(qualified[column].median()) if len(qualified) else None)
        for column in pair_summary.columns if column.startswith("high_")
    }
    gate = config["selective_followup_gate"]
    checks = {
        "support": len(qualified) >= int(config["minimum_supported_pairs"]),
        "baseline_retention": medians["high_minus_quiet_baseline_retention"] is not None and medians["high_minus_quiet_baseline_retention"] >= gate["minimum_high_minus_quiet_baseline_retention"],
        "refractory": medians["high_candidate_minus_baseline_refractory"] is not None and medians["high_candidate_minus_baseline_refractory"] <= gate["maximum_high_candidate_minus_baseline_refractory"],
        "count_ratio": medians["high_to_quiet_candidate_baseline_count_ratio"] is not None and medians["high_to_quiet_candidate_baseline_count_ratio"] >= gate["minimum_high_to_quiet_candidate_baseline_count_ratio"],
    }
    summary = {
        "schema_version": SCHEMA,
        "interior_primary_pairs": int(len(matches)),
        "supported_pairs": int(len(qualified)),
        "quiet_bins": int((exposure.regime == "quiet").sum()),
        "high_bins": int((exposure.regime == "high").sum()),
        "quiet_field_rms_cut_um": quiet_cut,
        "high_field_rms_cut_um": high_cut,
        "medians": medians,
        "gate_checks": checks,
        "decision": "selective_followup_supported" if all(checks.values()) else "selective_followup_not_supported",
        "limitations": config["interpretation"],
    }
    return exposure, unit_regime.merge(pair_summary, on=["baseline_cluster", "candidate_cluster"]), summary


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unsupported config schema")
    field = Path(config["field_file"])
    if sha256(field) != config["field_sha256"]:
        raise RuntimeError("field identity differs")
    comparison = json.loads(Path(config["comparison_summary"]).read_text())
    if comparison.get("status") != "complete":
        raise RuntimeError("full-session comparison is incomplete")
    request = {
        "schema_version": SCHEMA,
        "config": config,
        "source_sha256": sha256(Path(__file__)),
        "comparison_request_digest": comparison["request_digest"],
    }
    request["request_digest"] = fingerprint(request)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    request_path = output / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()).get("request_digest") != request["request_digest"]:
        raise RuntimeError("existing output belongs to another request")
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    exposure, rows, summary = analyse(config)
    exposure.to_csv(output / "field_exposure_bins.csv", index=False)
    rows.to_csv(output / "unit_regime_metrics.csv", index=False)
    final = {**summary, "request_digest": request["request_digest"]}
    (output / "summary.json").write_text(json.dumps(final, indent=2) + "\n")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
