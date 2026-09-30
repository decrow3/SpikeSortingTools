#!/usr/bin/env python3
"""Vectorized whole-session descriptive audit for rounded kriging and rescue KS."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from testing.de_common_outcome_scorecard import DOMAINS, intervals
from testing.em2g_slice_full_sort import stable_time_order


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def mask_membership(times_s: np.ndarray, spans: np.ndarray) -> np.ndarray:
    """Membership in sorted, nonoverlapping half-open spans without span loops."""
    times_s = np.asarray(times_s, dtype=np.float64)
    spans = np.asarray(spans, dtype=np.float64)
    if spans.size == 0:
        return np.zeros(times_s.shape, dtype=bool)
    if np.any(spans[:, 1] <= spans[:, 0]) or np.any(spans[1:, 0] < spans[:-1, 1]):
        raise ValueError("mask spans must be positive, sorted, and nonoverlapping")
    index = np.searchsorted(spans[:, 0], times_s, side="right") - 1
    valid = index >= 0
    output = np.zeros(times_s.shape, dtype=bool)
    output[valid] = times_s[valid] < spans[index[valid], 1]
    return output


def domain_codes(
    times_samples: np.ndarray,
    sampling_frequency: float,
    field_times: np.ndarray,
    field_deviation: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    times_s = np.asarray(times_samples, dtype=np.float64) / sampling_frequency
    value = np.interp(times_s, field_times, field_deviation)
    masked = mask_membership(times_s, mask)
    codes = np.full(times_s.shape, 2, dtype=np.int8)
    codes[(~masked) & (np.abs(value) < 20.0)] = 1
    codes[value < -120.0] = 0
    return codes


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return None
    result = spearmanr(x[valid], y[valid])
    value = float(result.statistic if hasattr(result, "statistic") else result[0])
    return value if math.isfinite(value) else None


def unit_domain_metrics(
    labels: np.ndarray,
    domains: np.ndarray,
    keep: np.ndarray,
    exposure: np.ndarray,
    minimum_flat_events: int,
) -> tuple[dict[str, object], pd.DataFrame, np.ndarray]:
    selected_labels = np.asarray(labels)[keep]
    selected_domains = np.asarray(domains)[keep]
    units, inverse = np.unique(selected_labels, return_inverse=True)
    counts = np.bincount(
        inverse.astype(np.int64) * len(DOMAINS) + selected_domains,
        minlength=len(units) * len(DOMAINS),
    ).reshape(len(units), len(DOMAINS))
    eligible = counts[:, 1] >= minimum_flat_events
    log_rates = np.log((counts + 0.5) / exposure[None, :])
    rho = safe_spearman(log_rates[eligible, 1], log_rates[eligible, 0])
    table = pd.DataFrame({"unit_id": units, "eligible_flat": eligible})
    for index, name in enumerate(DOMAINS):
        table[f"count_{name}"] = counts[:, index]
        table[f"log_rate_{name}"] = log_rates[:, index]
    summary = {
        "units": int(len(units)),
        "events": int(keep.sum()),
        "eligible_units": int(eligible.sum()),
        "rho_negative_vs_flat": rho,
        "events_by_domain": {
            name: int(counts[:, index].sum()) for index, name in enumerate(DOMAINS)
        },
        "assigned_rate_hz_by_domain": {
            name: float(counts[:, index].sum() / exposure[index])
            for index, name in enumerate(DOMAINS)
        },
    }
    return summary, table, inverse


def temporal_blocks(
    times_samples: np.ndarray,
    labels: np.ndarray,
    keep: np.ndarray,
    sampling_frequency: float,
    end_frame: int,
    block_seconds: float,
) -> pd.DataFrame:
    block_samples = int(round(block_seconds * sampling_frequency))
    n_blocks = int(math.ceil(end_frame / block_samples))
    selected_times = np.asarray(times_samples)[keep]
    selected_labels = np.asarray(labels)[keep]
    units, inverse = np.unique(selected_labels, return_inverse=True)
    block = np.minimum(selected_times // block_samples, n_blocks - 1).astype(np.int64)
    event_counts = np.bincount(block, minlength=n_blocks)
    pairs = np.unique(block * max(1, len(units)) + inverse)
    active = np.bincount(pairs // max(1, len(units)), minlength=n_blocks)
    starts = np.arange(n_blocks, dtype=np.int64) * block_samples
    ends = np.minimum(starts + block_samples, end_frame)
    duration = (ends - starts) / sampling_frequency
    return pd.DataFrame(
        {
            "block": np.arange(n_blocks),
            "start_frame": starts,
            "end_frame": ends,
            "duration_s": duration,
            "events": event_counts,
            "event_rate_hz": event_counts / duration,
            "active_units": active,
        }
    )


def segment_safe_isi(
    times_samples: np.ndarray,
    labels: np.ndarray,
    keep: np.ndarray,
    interval_table: pd.DataFrame,
    sampling_frequency: float,
) -> list[dict[str, object]]:
    times = np.asarray(times_samples)[keep]
    selected_labels = np.asarray(labels)[keep]
    order = np.lexsort((times, selected_labels))
    times = times[order]
    selected_labels = selected_labels[order]
    seconds = times.astype(np.float64) / sampling_frequency
    starts = interval_table.start_s.to_numpy(np.float64)
    ends = interval_table.end_s.to_numpy(np.float64)
    interval_domain = interval_table.domain.to_numpy(str)
    segment = np.searchsorted(starts, seconds, side="right") - 1
    if np.any(segment < 0) or np.any(seconds >= ends[segment]):
        raise RuntimeError("event does not map to an exact domain interval")
    same = (selected_labels[1:] == selected_labels[:-1]) & (segment[1:] == segment[:-1])
    deltas = np.diff(times)
    pair_domain = interval_domain[segment[:-1]]
    rows = []
    for name in DOMAINS:
        selected = same & (pair_domain == name)
        values = deltas[selected]
        rows.append(
            {
                "domain": name,
                "denominator": int(values.size),
                "lag8": int(np.count_nonzero(values == 8)),
                "lag9_29": int(np.count_nonzero((values >= 9) & (values <= 29))),
                "lag30": int(np.count_nonzero(values == 30)),
                "fraction9_29": (
                    float(np.mean((values >= 9) & (values <= 29))) if values.size else None
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    contract = json.loads(args.contract.read_text())
    fs = float(contract["scope"]["sampling_frequency_hz"])
    start_frame, end_frame = map(int, contract["scope"]["frames_half_open"])
    if start_frame != 0:
        raise RuntimeError("this audit expects the full source clock")

    for value in contract["domain_inputs"].values():
        if sha256(Path(value["path"])) != value["sha256"]:
            raise RuntimeError("domain input hash differs")
    candidate_run = Path(contract["candidate"]["run"])
    receipt = json.loads((candidate_run / "receipt.json").read_text())
    expected_config = contract["candidate"]["resolved_config_sha256"]
    if receipt.get("status") != "complete" or receipt.get("exit_status") != 0:
        raise RuntimeError("candidate run is not complete and successful")
    if receipt.get("config_sha256") != expected_config or sha256(candidate_run / "config.json") != expected_config:
        raise RuntimeError("candidate configuration differs")
    candidate_path = Path(contract["candidate"]["sorting"])
    candidate_sha = sha256(candidate_path)
    with np.load(candidate_path, allow_pickle=False) as saved:
        candidate_times = np.asarray(saved["times_samples"], dtype=np.int64)
        candidate_labels = np.asarray(saved["labels"], dtype=np.int64)
        candidate_fs = float(saved["sampling_frequency"])
    candidate_times, (candidate_labels,), candidate_time_ordering = stable_time_order(
        candidate_times, candidate_labels
    )
    if not np.isclose(candidate_fs, fs, rtol=0, atol=1e-9):
        raise RuntimeError("candidate sampling frequency differs")

    reference_folder = Path(contract["reference"]["folder"])
    for name, expected in contract["reference"]["files"].items():
        if sha256(reference_folder / name) != expected:
            raise RuntimeError(f"reference file hash differs: {name}")
    reference_times = np.asarray(
        np.load(reference_folder / "spike_times.npy", mmap_mode="r").reshape(-1),
        dtype=np.int64,
    )
    reference_labels = np.asarray(
        np.load(reference_folder / "spike_clusters.npy", mmap_mode="r").reshape(-1),
        dtype=np.int64,
    )
    label_table = pd.read_csv(reference_folder / "cluster_KSLabel.tsv", sep="\t")
    good_ids = set(
        label_table.loc[label_table.KSLabel.astype(str).str.lower() == "good", "cluster_id"].astype(int)
    )
    reference_good = np.isin(
        reference_labels, np.fromiter(sorted(good_ids), dtype=np.int64)
    )

    if candidate_times.size and (candidate_times.min() < 0 or candidate_times.max() >= end_frame):
        raise RuntimeError("candidate event outside support")
    if reference_times.size and (reference_times.min() < 0 or reference_times.max() >= end_frame):
        raise RuntimeError("reference event outside support")
    if np.any(np.diff(reference_times) < 0):
        raise RuntimeError("reference event times are not sorted")

    with np.load(contract["domain_inputs"]["field"]["path"], allow_pickle=False) as saved:
        field_times = np.asarray(saved["time_s"], dtype=np.float64)
        displacement = np.asarray(saved["displacement_um"], dtype=np.float64)[:, 0]
    baseline = pd.Series(displacement).rolling(480, center=True, min_periods=1).median().to_numpy()
    deviation = displacement - baseline
    mask = pd.read_csv(contract["domain_inputs"]["mask"]["path"])[
        ["start_s", "end_s"]
    ].to_numpy(np.float64)
    exact_intervals = intervals(field_times, deviation, mask, 0.0, end_frame / fs)
    interval_table = pd.DataFrame(exact_intervals, columns=["start_s", "end_s", "domain"])
    interval_table["duration_s"] = interval_table.end_s - interval_table.start_s
    exposure = (
        interval_table.groupby("domain").duration_s.sum().reindex(DOMAINS).to_numpy(np.float64)
    )
    if not np.isclose(exposure.sum(), end_frame / fs, atol=1e-9):
        raise RuntimeError("domain exposure does not close full session")

    candidate_domains = domain_codes(candidate_times, fs, field_times, deviation, mask)
    reference_domains = domain_codes(reference_times, fs, field_times, deviation, mask)
    subsets = [
        ("rounded_kriging", "assigned_units", candidate_times, candidate_labels, candidate_labels >= 0, candidate_domains),
        ("rescue_kilosort", "all_units", reference_times, reference_labels, np.ones(len(reference_labels), bool), reference_domains),
        ("rescue_kilosort", "ks_good_units", reference_times, reference_labels, reference_good, reference_domains),
    ]
    summaries = []
    unit_tables = []
    block_tables = []
    isi_rows = []
    minimum = int(contract["metrics"]["minimum_flat_events"])
    for arm, subset, times, labels, keep, domains in subsets:
        summary, units, _ = unit_domain_metrics(labels, domains, keep, exposure, minimum)
        summary.update({"arm": arm, "subset": subset})
        summaries.append(summary)
        units.insert(0, "subset", subset)
        units.insert(0, "arm", arm)
        unit_tables.append(units)
        blocks = temporal_blocks(
            times,
            labels,
            keep,
            fs,
            end_frame,
            float(contract["metrics"]["time_block_seconds"]),
        )
        blocks.insert(0, "subset", subset)
        blocks.insert(0, "arm", arm)
        block_tables.append(blocks)
        for row in segment_safe_isi(times, labels, keep, interval_table, fs):
            isi_rows.append({"arm": arm, "subset": subset, **row})

    args.output.mkdir(parents=True)
    interval_table.to_csv(args.output / "DOMAIN_INTERVALS.csv", index=False)
    pd.DataFrame(summaries).to_json(args.output / "ARM_SUMMARIES.json", orient="records", indent=2)
    pd.concat(unit_tables, ignore_index=True).to_csv(args.output / "UNIT_DOMAIN_METRICS.csv", index=False)
    pd.concat(block_tables, ignore_index=True).to_csv(args.output / "TEMPORAL_BLOCKS.csv", index=False)
    pd.DataFrame(isi_rows).to_csv(args.output / "SEGMENT_SAFE_ISI.csv", index=False)
    resources = {
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "voltage_bytes_read": 0,
        "gpu_used": False,
    }
    result = {
        "schema": "em2h-full-session-descriptive-result-v1",
        "status": "complete",
        "contract_sha256": sha256(args.contract),
        "candidate_sorting_sha256": candidate_sha,
        "candidate_time_ordering": candidate_time_ordering,
        "domain_exposure_seconds": {
            name: float(exposure[index]) for index, name in enumerate(DOMAINS)
        },
        "summaries": summaries,
        "interpretation": contract["interpretation"],
        "resources": resources,
        "resource_limits_pass": (
            resources["cpu_seconds"] <= contract["resources"]["cpu_seconds_max"]
            and resources["peak_rss_gib"] <= contract["resources"]["memory_gib_max"]
        ),
    }
    result_path = args.output / "RESULT.json"
    atomic_json(result_path, result)
    products = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(args.output.iterdir())
        if path.name not in {"MANIFEST.json", "COMPLETE.json"}
    ]
    manifest_path = args.output / "MANIFEST.json"
    atomic_json(manifest_path, {"products": products})
    complete_status = "complete" if result["resource_limits_pass"] else "resource_limit_exceeded"
    atomic_json(
        args.output / "COMPLETE.json",
        {"status": complete_status, "manifest_sha256": sha256(manifest_path)},
    )
    if not result["resource_limits_pass"]:
        raise RuntimeError(f"resource limit exceeded: {resources}")


if __name__ == "__main__":
    main()
