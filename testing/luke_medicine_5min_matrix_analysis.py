"""Evaluate five-minute correction sorts against frozen lighthouse events."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


ARMS_ROOT = Path("/media/huklab/Data/luke_medicine_5min_matrix_v1/arms")
EVENTS = Path("testing/outputs/luke_lighthouse_extension_300s_v1/overlay_events.csv")
FIELD = Path("testing/outputs/luke_screened_medicine_300s_v1/fit_6sigma_full_10000/field.npz")
OUTPUT = Path("testing/outputs/luke_medicine_5min_matrix_analysis_v1")
START_S = 930.0
TRAIN_STOP_S = 940.0
TOLERANCE_MS = 0.5
DEPTH_TOLERANCE_UM = 100.0


def arm_spec(name: str) -> dict | None:
    if name == "unwarped":
        return None
    tokens = name.split("_")
    spatial = tokens[0]
    gain = 0.25 if "g025" in tokens else 1.0
    return {"spatial": spatial, "gain": gain}


def expected_depth(events: pd.DataFrame, field_path: Path, spec: dict | None) -> np.ndarray:
    observed = events.centroid_um.to_numpy(float)
    if spec is None:
        return observed
    field = np.load(field_path)
    displacement = np.asarray(field["displacement_um"], dtype=float)
    if spec["spatial"] == "rigid":
        values = np.interp(events.time_s, field["time_s"], np.median(displacement, axis=1))
    else:
        sampler = RegularGridInterpolator(
            (np.asarray(field["time_s"], float), np.asarray(field["depth_um"], float)),
            displacement, bounds_error=False, fill_value=None,
        )
        values = sampler(np.column_stack((events.time_s, observed)))
    return observed - float(spec["gain"]) * values


def match_events(
    event_samples: np.ndarray,
    event_depths: np.ndarray,
    spike_samples: np.ndarray,
    spike_depths: np.ndarray,
    spike_clusters: np.ndarray,
    tolerance_samples: int,
    depth_tolerance_um: float,
) -> pd.DataFrame:
    """Greedily match events to unused nearby spikes for one identity."""
    rows = []
    used: set[int] = set()
    order = np.argsort(event_samples, kind="stable")
    for event_index in order:
        sample = int(event_samples[event_index])
        depth = float(event_depths[event_index])
        lo = int(np.searchsorted(spike_samples, sample - tolerance_samples, side="left"))
        hi = int(np.searchsorted(spike_samples, sample + tolerance_samples, side="right"))
        candidates = np.arange(lo, hi, dtype=int)
        if candidates.size:
            candidates = candidates[np.abs(spike_depths[candidates] - depth) <= depth_tolerance_um]
        distinct_clusters = int(np.unique(spike_clusters[candidates]).size) if candidates.size else 0
        available = np.array([index for index in candidates if int(index) not in used], dtype=int)
        if available.size:
            rank = np.lexsort((np.abs(spike_depths[available] - depth),
                               np.abs(spike_samples[available] - sample)))
            chosen = int(available[rank[0]])
            used.add(chosen)
            cluster = int(spike_clusters[chosen])
            delta = int(spike_samples[chosen] - sample)
        else:
            cluster, delta = -1, np.iinfo(np.int32).min
        rows.append(dict(event_index=int(event_index), recovered=cluster >= 0,
                         cluster=cluster, delta_samples=delta,
                         nearby_spikes=int(candidates.size), nearby_clusters=distinct_clusters))
    return pd.DataFrame(rows).sort_values("event_index").reset_index(drop=True)


def load_sort(arm_dir: Path) -> dict:
    root = arm_dir / "kilosort4/sorter_output"
    ops = np.load(root / "ops.npy", allow_pickle=True).item()
    return {
        "fs": float(ops["fs"]),
        "times": np.load(root / "spike_times.npy", mmap_mode="r").reshape(-1),
        "clusters": np.load(root / "spike_clusters.npy", mmap_mode="r").reshape(-1),
        "depths": np.load(root / "spike_positions.npy", mmap_mode="r")[:, 1],
        "labels": pd.read_csv(root / "cluster_KSLabel.tsv", sep="\t"),
        "contam": pd.read_csv(root / "cluster_ContamPct.tsv", sep="\t"),
    }


def unit_metrics(matched: pd.DataFrame) -> dict:
    recovered = matched.loc[matched.recovered]
    n = len(matched)
    counts = recovered.cluster.value_counts()
    modal = int(counts.iloc[0]) if len(counts) else 0
    fragments = int((counts >= max(2, 0.05 * len(recovered))).sum()) if len(counts) else 0
    return {
        "events": n,
        "recovered": int(len(recovered)),
        "recovery_fraction": float(len(recovered) / n) if n else np.nan,
        "single_cluster_fraction_all_events": float(modal / n) if n else np.nan,
        "single_cluster_fraction_recovered": float(modal / len(recovered)) if len(recovered) else np.nan,
        "fragment_clusters": fragments,
        "duplicate_event_fraction": float((matched.nearby_clusters > 1).mean()) if n else np.nan,
        "median_abs_timing_samples": float(np.median(np.abs(recovered.delta_samples))) if len(recovered) else np.nan,
    }


def analyze_arm(name: str, arm_dir: Path, events: pd.DataFrame, field_path: Path):
    sort = load_sort(arm_dir)
    fs = sort["fs"]
    tolerance = int(round(TOLERANCE_MS * 1e-3 * fs))
    depths = expected_depth(events, field_path, arm_spec(name))
    full_start = int(round(START_S * fs))
    samples = events.frame.to_numpy(np.int64) - full_start
    event_rows, unit_rows = [], []
    for unit_id, indices in events.groupby("unit_id", sort=False).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        matched = match_events(samples[idx], depths[idx], sort["times"], sort["depths"],
                               sort["clusters"], tolerance, DEPTH_TOLERANCE_UM)
        matched["event_row"] = idx
        matched["unit_id"] = int(unit_id)
        matched["arm"] = name
        event_rows.append(matched)
        unit_rows.append({"arm": name, "unit_id": int(unit_id), **unit_metrics(matched)})
    event_table = pd.concat(event_rows, ignore_index=True)
    unit_table = pd.DataFrame(unit_rows)
    labels = sort["labels"]
    label_col = next(column for column in labels.columns if column != "cluster_id")
    contamination = sort["contam"]
    contam_col = next(column for column in contamination.columns if column != "cluster_id")
    good_ids = set(labels.loc[labels[label_col].astype(str).str.lower().eq("good"), "cluster_id"].astype(int))
    isi = []
    for cluster in np.unique(sort["clusters"]):
        train = np.sort(sort["times"][sort["clusters"] == cluster])
        if train.size > 1:
            isi.append(float(np.mean(np.diff(train) < int(round(1.5e-3 * fs)))))
    aggregate = {
        "arm": name,
        "spikes": int(len(sort["times"])),
        "units": int(labels.cluster_id.nunique()),
        "good_units": int(len(good_ids)),
        "median_contamination_pct": float(contamination[contam_col].median()),
        "median_refractory_fraction": float(np.median(isi)),
        "macro_lighthouse_recovery": float(unit_table.recovery_fraction.mean()),
        "macro_single_cluster_all_events": float(unit_table.single_cluster_fraction_all_events.mean()),
        "macro_single_cluster_recovered": float(unit_table.single_cluster_fraction_recovered.mean()),
        "median_fragment_clusters": float(unit_table.fragment_clusters.median()),
        "macro_duplicate_event_fraction": float(unit_table.duplicate_event_fraction.mean()),
        "lighthouses": int(len(unit_table)),
    }
    return aggregate, unit_table, event_table


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    labels = summary.arm.str.replace("_", "\n")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    metrics = [
        ("macro_lighthouse_recovery", "Lighthouse recovery"),
        ("macro_single_cluster_all_events", "Single-cluster recovery"),
        ("macro_duplicate_event_fraction", "Events near >1 cluster"),
        ("good_units", "KS-good units"),
    ]
    for ax, (column, title) in zip(axes.flat, metrics):
        ax.bar(np.arange(len(summary)), summary[column], color=["#555555"] + ["#348ABD"] * (len(summary) - 1))
        ax.set(xticks=np.arange(len(summary)), xticklabels=labels, title=title)
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    fig.suptitle("Five-minute MEDiCINe voltage-correction matrix\nFrozen strict lighthouse events, 940–1230 s")
    fig.tight_layout()
    fig.savefig(output / "01_matrix_summary.png", dpi=180)
    fig.savefig(output / "01_matrix_summary.pdf")
    plt.close(fig)


def run(arms_root: Path, events_path: Path, field_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    events = pd.read_csv(events_path)
    primary = events[(events.evidence == "strict_accepted") & (~events.training.astype(bool))].copy()
    primary = primary.sort_values(["unit_id", "frame", "score"], ascending=[True, True, False])
    primary = primary.drop_duplicates(["unit_id", "frame"]).reset_index(drop=True)
    if primary.empty:
        raise RuntimeError("no frozen held-out strict lighthouse events")
    aggregates, units, matched = [], [], []
    names = ["unwarped"] + sorted(path.name for path in arms_root.iterdir() if path.is_dir() and path.name != "unwarped")
    for name in names:
        aggregate, unit_table, event_table = analyze_arm(name, arms_root / name, primary, field_path)
        aggregates.append(aggregate); units.append(unit_table); matched.append(event_table)
    summary = pd.DataFrame(aggregates)
    baseline = summary.set_index("arm").loc["unwarped"]
    for column in ["macro_lighthouse_recovery", "macro_single_cluster_all_events",
                   "macro_duplicate_event_fraction", "median_contamination_pct",
                   "median_refractory_fraction", "good_units"]:
        summary[f"delta_vs_unwarped_{column}"] = summary[column] - baseline[column]
    unit_table = pd.concat(units, ignore_index=True)
    matched_table = pd.concat(matched, ignore_index=True)
    summary.to_csv(output / "arm_summary.csv", index=False)
    unit_table.to_csv(output / "lighthouse_unit_metrics.csv", index=False)
    matched_table.to_csv(output / "matched_events.csv", index=False)
    plot_summary(summary, output)
    result = {
        "status": "complete",
        "primary": "strict_accepted frozen lighthouse events after 930–940 s training interval",
        "events": int(len(primary)),
        "lighthouses": int(primary.unit_id.nunique()),
        "time_tolerance_ms": TOLERANCE_MS,
        "depth_tolerance_um": DEPTH_TOLERANCE_UM,
        "depth_policy": "observed centroid for unwarped; observed minus arm-applied MEDiCINe displacement for corrected arms",
        "identity_policy": "greedy one-to-one event/spike matching within each frozen lighthouse; no sorter labels select lighthouse identities",
        "interpretation": "development comparison; raw unit count is not an efficacy endpoint",
        "arms": json.loads(summary.to_json(orient="records")),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms-root", type=Path, default=ARMS_ROOT)
    parser.add_argument("--events", type=Path, default=EVENTS)
    parser.add_argument("--field", type=Path, default=FIELD)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.arms_root, args.events, args.field, args.output), indent=2))


if __name__ == "__main__":
    main()
