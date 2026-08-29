"""Definition-matched Luke/Yates audit at the Kilosort input and output stages.

The audit replays each sort's saved Kilosort high-pass, common-average
reference, and whitening transform on 60 two-second batches.  Luke uses the
complete 120 s no-external-motion diagnostic; Yates uses 60 evenly spaced
batches spanning the known-good session.  A simple detector then counts
spatiotemporally deduplicated extrema at fixed whitened amplitudes and asks
whether each negative event has a final sorted spike nearby.

This is intentionally not a replacement spike sorter.  Its purpose is to
locate the Luke/Yates deficit before or after Kilosort's learned-template pass
using the actual binary and exact preprocessing matrix supplied to each sort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


OUT = Path("testing/outputs/luke_motion_candidate_results")
LUKE_ROOT = Path(
    "/mnt/NPX/Luke/20250804/dredge_pipeline_results_"
    "Luke0804_V2V1_g0_imec1/upstream_sorter_ablation"
)
YATES_ROOT = Path(
    "/media/huklab/Data/Yates_session_copy/processed/Allen_2022-02-16"
)

DATASETS = {
    "Luke pathological, current conditioning": {
        "binary": LUKE_ROOT / "recordings/current_no_motion/traces_cached_seg0.raw",
        "sorter": LUKE_ROOT / "sorts/current_no_motion/sorter_output",
        "params": LUKE_ROOT / "sorts/current_no_motion/spikeinterface_params.json",
        "all_batches": True,
    },
    "Luke shared, current conditioning": {
        "binary": (
            LUKE_ROOT.parent
            / "motion_candidate_replication/shared_template/recordings/no_external_correction/traces_cached_seg0.raw"
        ),
        "sorter": LUKE_ROOT.parent
        / "motion_candidate_replication/shared_template/sorts/no_external_correction/sorter_output",
        "params": LUKE_ROOT.parent
        / "motion_candidate_replication/shared_template/sorts/no_external_correction/spikeinterface_params.json",
        "all_batches": False,
    },
    "Luke shared, single KS preprocessing": {
        "binary": LUKE_ROOT.parent
        / "motion_candidate_replication/shared_template/recordings/single_ks_preprocessing/traces_cached_seg0.raw",
        "sorter": LUKE_ROOT.parent
        / "motion_candidate_replication/shared_template/sorts/single_ks_preprocessing/sorter_output",
        "params": LUKE_ROOT.parent
        / "motion_candidate_replication/shared_template/sorts/single_ks_preprocessing/spikeinterface_params.json",
        "all_batches": False,
    },
    "Yates known-good": {
        "binary": YATES_ROOT / "preprocessed.dat",
        "sorter": YATES_ROOT / "ks4/sorter_output",
        "params": YATES_ROOT / "ks4/spikeinterface_params.json",
        "all_batches": False,
    },
}


def spatial_neighbors(positions: np.ndarray, radius_um: float) -> list[np.ndarray]:
    delta = positions[:, None, :] - positions[None, :, :]
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    return [np.flatnonzero(distance[i] <= radius_um) for i in range(len(positions))]


def collapse_candidates(
    times: np.ndarray,
    channels: np.ndarray,
    scores: np.ndarray,
    neighbors: list[np.ndarray],
    n_samples: int,
    temporal_radius: int,
) -> np.ndarray:
    """Greedy nonmaximum suppression in time and physical channel space."""
    if times.size == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(scores)[::-1]
    suppressed = np.zeros((len(neighbors), n_samples), dtype=bool)
    keep: list[int] = []
    for idx in order:
        t = int(times[idx])
        ch = int(channels[idx])
        if suppressed[ch, t]:
            continue
        keep.append(int(idx))
        lo = max(0, t - temporal_radius)
        hi = min(n_samples, t + temporal_radius + 1)
        suppressed[neighbors[ch], lo:hi] = True
    keep_arr = np.asarray(keep, dtype=np.int64)
    return keep_arr[np.argsort(times[keep_arr])]


def extrema_candidates(x: np.ndarray, threshold: float, negative: bool) -> tuple[np.ndarray, ...]:
    center = x[:, 1:-1]
    if negative:
        mask = (center < x[:, :-2]) & (center <= x[:, 2:]) & (center <= -threshold)
        scores = -center[mask]
    else:
        mask = (center > x[:, :-2]) & (center >= x[:, 2:]) & (center >= threshold)
        scores = center[mask]
    channels, times = np.nonzero(mask)
    return times.astype(np.int64) + 1, channels.astype(np.int64), scores.astype(np.float32)


def nearby_fraction(
    event_times: np.ndarray,
    event_channels: np.ndarray,
    positions: np.ndarray,
    spike_times: np.ndarray,
    spike_positions: np.ndarray,
    temporal_radius: int,
    spatial_radius_um: float,
) -> float:
    if not len(event_times):
        return float("nan")
    order = np.argsort(spike_times)
    st = spike_times[order]
    sp = spike_positions[order]
    matched = 0
    for t, ch in zip(event_times, event_channels):
        lo = np.searchsorted(st, t - temporal_radius, side="left")
        hi = np.searchsorted(st, t + temporal_radius, side="right")
        if hi <= lo:
            continue
        delta = sp[lo:hi] - positions[ch]
        if np.any(np.sqrt(np.sum(delta * delta, axis=1)) <= spatial_radius_um):
            matched += 1
    return matched / len(event_times)


def probe_depth_exposure_mm(positions: np.ndarray, shanks: np.ndarray) -> float:
    """Sum the sampled depth spans, counting each shank once."""
    shanks = np.asarray(shanks).copy()
    if np.unique(shanks).size == 1:
        unique_x = np.unique(positions[:, 0])
        gaps = np.diff(unique_x)
        if np.any(gaps > 100.0):
            boundaries = unique_x[:-1][gaps > 100.0]
            shanks = np.searchsorted(boundaries, positions[:, 0], side="left")
    exposure_um = 0.0
    for shank in np.unique(shanks):
        y = positions[shanks == shank, 1]
        exposure_um += float(np.max(y) - np.min(y))
    return exposure_um / 1000.0


def choose_batches(n_batches: int, n: int, all_batches: bool) -> np.ndarray:
    if all_batches or n_batches <= n:
        return np.arange(n_batches, dtype=int)
    return np.unique(np.linspace(0, n_batches - 1, n, dtype=int))


def make_filtered(binary: Path, ops: dict, device: torch.device):
    from kilosort.io import BinaryFiltered

    prep = ops["preprocessing"]
    return BinaryFiltered(
        binary,
        n_chan_bin=int(ops["n_chan_bin"]),
        fs=float(ops["fs"]),
        NT=int(ops["batch_size"]),
        nt=int(ops["nt"]),
        nt0min=int(ops["nt0min"]),
        chan_map=np.asarray(ops["chanMap"]),
        hp_filter=torch.as_tensor(prep["hp_filter"], device=device),
        whiten_mat=torch.as_tensor(prep["whiten_mat"], device=device),
        device=device,
        do_CAR=bool(ops["do_CAR"]),
        invert_sign=bool(ops["invert_sign"]),
        artifact_threshold=float(ops["artifact_threshold"]),
        dtype="int16",
    )


def analyze_dataset(label: str, cfg: dict, device: torch.device) -> tuple[pd.DataFrame, dict]:
    sorter = Path(cfg["sorter"])
    ops = np.load(sorter / "ops.npy", allow_pickle=True).item()
    filtered = make_filtered(Path(cfg["binary"]), ops, device)
    positions = np.load(sorter / "channel_positions.npy").astype(float)
    shanks = np.load(sorter / "channel_shanks.npy").reshape(-1)
    depth_exposure_mm = probe_depth_exposure_mm(positions, shanks)
    neighbors = spatial_neighbors(positions, 100.0)
    batches = choose_batches(filtered.n_batches, 60, bool(cfg["all_batches"]))
    nt = int(ops["nt"])
    batch_size = int(ops["batch_size"])
    fs = float(ops["fs"])

    final_times = np.load(sorter / "spike_times.npy").astype(np.int64)
    final_pos = np.load(sorter / "spike_positions.npy").astype(float)
    learned_times = np.load(sorter / "full_st.npy", mmap_mode="r")[:, 0].astype(np.int64)
    learned_amp = np.load(sorter / "full_amp.npy", mmap_mode="r")

    rows = []
    for ibatch in batches:
        with torch.no_grad():
            x = filtered.padded_batch_to_torch(int(ibatch))[:, nt:-nt]
        x = x.detach().cpu().numpy()
        start = int(ibatch) * batch_size
        stop = min(start + x.shape[1], filtered.n_samples)
        x = x[:, : stop - start]
        duration = x.shape[1] / fs
        row = {
            "dataset": label,
            "batch": int(ibatch),
            "start_s": start / fs,
            "duration_s": duration,
            "n_channels": x.shape[0],
            "whitened_rms": float(np.sqrt(np.mean(x * x))),
            "whitened_mad_sigma": float(np.median(np.abs(x)) / 0.67448975),
            "exact_zero_fraction": float(np.mean(x == 0)),
            "channel_median_abs_correlation": float(
                np.median(np.abs(np.corrcoef(x[:, ::30])[np.triu_indices(x.shape[0], 1)]))
            ),
        }
        fmask = (final_times >= start) & (final_times < stop)
        lmask = (learned_times >= start) & (learned_times < stop)
        row["learned_detections_per_s"] = int(lmask.sum()) / duration
        row["learned_detections_per_channel_s"] = int(lmask.sum()) / duration / x.shape[0]
        row["learned_detections_per_depth_mm_s"] = int(lmask.sum()) / duration / depth_exposure_mm
        row["final_spikes_per_s"] = int(fmask.sum()) / duration
        row["final_spikes_per_channel_s"] = int(fmask.sum()) / duration / x.shape[0]
        row["final_spikes_per_depth_mm_s"] = int(fmask.sum()) / duration / depth_exposure_mm
        row["median_learned_amplitude"] = float(np.median(learned_amp[lmask])) if lmask.any() else np.nan

        for threshold in (6.0, 8.0):
            event_sets = {}
            for negative in (True, False):
                sign = "negative" if negative else "positive"
                times, channels, scores = extrema_candidates(x, threshold, negative)
                keep = collapse_candidates(
                    times, channels, scores, neighbors, x.shape[1], int(round(0.0005 * fs))
                )
                event_times = times[keep]
                event_channels = channels[keep]
                event_sets[sign] = (event_times, event_channels)
                prefix = f"{sign}_{int(threshold)}sigma"
                row[f"{prefix}_channel_crossings_per_channel_s"] = len(times) / duration / x.shape[0]
                row[f"{prefix}_events_per_s"] = len(keep) / duration
                row[f"{prefix}_events_per_depth_mm_s"] = len(keep) / duration / depth_exposure_mm
                row[f"{prefix}_final_recovery"] = nearby_fraction(
                    event_times + start,
                    event_channels,
                    positions,
                    final_times[fmask],
                    final_pos[fmask],
                    int(round(0.0005 * fs)),
                    100.0,
                )
            neg_t, neg_c = event_sets["negative"]
            pos_t, pos_c = event_sets["positive"]
            pair_radius = int(round(0.001 * fs))
            row[f"positive_{int(threshold)}sigma_near_negative_fraction"] = nearby_fraction(
                pos_t, pos_c, positions, neg_t, positions[neg_c], pair_radius, 100.0
            )
            row[f"negative_{int(threshold)}sigma_near_positive_fraction"] = nearby_fraction(
                neg_t, neg_c, positions, pos_t, positions[pos_c], pair_radius, 100.0
            )
            row[f"positive_{int(threshold)}sigma_unpaired_events_per_s"] = (
                len(pos_t)
                * (1.0 - row[f"positive_{int(threshold)}sigma_near_negative_fraction"])
                / duration
            )
        rows.append(row)
        print(f"{label}: batch {ibatch} ({len(rows)}/{len(batches)})", flush=True)

    params = json.loads(Path(cfg["params"]).read_text())["sorter_params"]
    metadata = {
        "dataset": label,
        "binary": str(cfg["binary"]),
        "sorter": str(sorter),
        "n_batches_available": int(filtered.n_batches),
        "batches_analyzed": batches.tolist(),
        "duration_analyzed_s": float(sum(r["duration_s"] for r in rows)),
        "sample_rate": fs,
        "batch_size": batch_size,
        "n_channels": int(positions.shape[0]),
        "depth_exposure_mm": depth_exposure_mm,
        "sorter_params": params,
    }
    return pd.DataFrame(rows), metadata


def summarize(batches: pd.DataFrame) -> pd.DataFrame:
    numeric = [c for c in batches.columns if c not in {"dataset", "batch", "start_s"}]
    rows = []
    for dataset, group in batches.groupby("dataset", sort=False):
        for metric in numeric:
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "median": float(group[metric].median()),
                    "q25": float(group[metric].quantile(0.25)),
                    "q75": float(group[metric].quantile(0.75)),
                    "mean": float(group[metric].mean()),
                    "n_batches": int(group[metric].notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_summary(batches: pd.DataFrame, out: Path) -> None:
    colors = {
        "Luke pathological, current conditioning": "#c44e52",
        "Luke shared, current conditioning": "#dd8452",
        "Luke shared, single KS preprocessing": "#55a868",
        "Yates known-good": "#4c72b0",
    }
    metrics = [
        ("negative_6sigma_events_per_depth_mm_s", "Negative events/mm/s"),
        ("negative_6sigma_final_recovery", "6σ events with final spike"),
        ("learned_detections_per_depth_mm_s", "Learned detections/mm/s"),
        ("final_spikes_per_depth_mm_s", "Final spikes/mm/s"),
        ("whitened_mad_sigma", "Whitened MAD scale"),
        ("channel_median_abs_correlation", "Median |channel correlation|"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, metrics):
        labels = list(colors)
        values = [batches.loc[batches.dataset.eq(label), metric].dropna().to_numpy() for label in labels]
        bp = ax.boxplot(
            values,
            labels=["Luke\npath", "Luke\ncurrent", "Luke\nsingle", "Yates"],
            patch_artist=True,
            showfliers=False,
        )
        for patch, label in zip(bp["boxes"], labels):
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.75)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Luke–Yates Kilosort-input and detection-stage audit", fontsize=15)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def cross_unit_near_coincident_fraction(
    times: np.ndarray,
    clusters: np.ndarray,
    depths: np.ndarray,
    temporal_radius: int,
    depth_radius_um: float = 75.0,
) -> float:
    order = np.argsort(times)
    times = np.asarray(times)[order]
    clusters = np.asarray(clusters)[order]
    depths = np.asarray(depths)[order]
    marked = np.zeros(len(times), dtype=bool)
    for left in range(len(times)):
        right = left + 1
        while right < len(times) and times[right] - times[left] <= temporal_radius:
            if clusters[right] != clusters[left] and abs(depths[right] - depths[left]) <= depth_radius_um:
                marked[left] = True
                marked[right] = True
            right += 1
    return float(marked.mean()) if len(marked) else 0.0


def dataset_metrics(metadata: list[dict]) -> pd.DataFrame:
    rows = []
    for meta in metadata:
        sorter = Path(meta["sorter"])
        times = np.load(sorter / "spike_times.npy").reshape(-1).astype(np.int64)
        clusters = np.load(sorter / "spike_clusters.npy").reshape(-1)
        depths = np.load(sorter / "spike_positions.npy")[:, 1].astype(float)
        remapped_times = []
        sampled_clusters = []
        sampled_depths = []
        batch_size = int(meta["batch_size"])
        for ordinal, batch in enumerate(meta["batches_analyzed"]):
            start = int(batch) * batch_size
            stop = start + batch_size
            mask = (times >= start) & (times < stop)
            remapped_times.append(ordinal * batch_size + times[mask] - start)
            sampled_clusters.append(clusters[mask])
            sampled_depths.append(depths[mask])
        times_sampled = np.concatenate(remapped_times)
        clusters_sampled = np.concatenate(sampled_clusters)
        depths_sampled = np.concatenate(sampled_depths)
        active_units = np.unique(clusters_sampled)
        n_units = int(active_units.size)
        labels = pd.read_csv(sorter / "cluster_KSLabel.tsv", sep="\t")
        n_good = int(
            (
                labels["KSLabel"].astype(str).str.lower().eq("good")
                & labels["cluster_id"].isin(active_units)
            ).sum()
        )
        contamination = pd.read_csv(sorter / "cluster_ContamPct.tsv", sep="\t")
        contam_col = next(c for c in contamination if c != "cluster_id")
        active_contam = contamination.loc[contamination.cluster_id.isin(active_units), contam_col]
        refractory = []
        isi_limit = int(round(0.0015 * meta["sample_rate"]))
        for unit in active_units:
            unit_times = np.sort(times_sampled[clusters_sampled == unit])
            if len(unit_times) > 1:
                refractory.append(float(np.mean(np.diff(unit_times) < isi_limit)))
        exposure = float(meta["depth_exposure_mm"])
        rows.append(
            {
                "dataset": meta["dataset"],
                "n_units": n_units,
                "n_ks_good": n_good,
                "depth_exposure_mm": exposure,
                "units_per_depth_mm": n_units / exposure,
                "ks_good_per_depth_mm": n_good / exposure,
                "contacts_per_depth_mm": meta["n_channels"] / exposure,
                "sampled_final_spikes": int(len(times_sampled)),
                "mean_spikes_per_active_unit_s": len(times_sampled)
                / meta["duration_analyzed_s"]
                / n_units,
                "median_contamination_pct": float(np.median(active_contam)),
                "cross_unit_near_coincident_fraction": cross_unit_near_coincident_fraction(
                    times_sampled,
                    clusters_sampled,
                    depths_sampled,
                    int(round(0.0005 * meta["sample_rate"])),
                ),
                "median_unit_refractory_violation_fraction": float(np.median(refractory)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    metadata = []
    for label, cfg in DATASETS.items():
        frame, meta = analyze_dataset(label, cfg, device)
        frames.append(frame)
        metadata.append(meta)
    batches = pd.concat(frames, ignore_index=True)
    summary = summarize(batches)
    batches.to_csv(OUT / "luke_yates_detection_stage_batches.csv", index=False)
    summary.to_csv(OUT / "luke_yates_detection_stage_summary.csv", index=False)
    dataset_metrics(metadata).to_csv(OUT / "luke_yates_detection_stage_dataset_metrics.csv", index=False)
    (OUT / "luke_yates_detection_stage_manifest.json").write_text(
        json.dumps({"device": str(device), "datasets": metadata}, indent=2)
    )
    plot_summary(batches, OUT / "luke_yates_detection_stage_audit.png")


if __name__ == "__main__":
    main()
