"""Fresh-snippet timing audit for RESCUE-good to MEDiCINe-MUA matches.

This reads a bounded, deterministic sample of matched events from both persisted
recording binaries.  The primary endpoint is spike-time precision, not voltage
amplitude: cross-run timestamp differences and within-run residual alignment
jitter of high-pass-filtered multichannel snippets.  Early/middle/late waveform
medians are retained so temporal stability can be inspected directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


BASE = Path("/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0")
CAND = Path("/media/huklab/Data/luke_medicine_rigid_sort_20260909_v1")
PAIRS = Path("testing/outputs/luke_medicine_rigid_vs_rescue_fair_v1/reciprocal_primary_matches.csv")
FS = 29999.835983263598
TOLERANCE_SAMPLES = 15
SNIPPET_HALF = 70
REFERENCE_HALF = 30
SEARCH_LAGS = np.arange(-6, 7, dtype=int)
SAMPLES_PER_EPOCH = 80
PRIMARY_DEPTH_RANGE_UM = (300.0, 3540.0)
COLORS = {"rescue": "#1f5a94", "medicine": "#c56a1a"}


def exclusive_pairs(a: np.ndarray, b: np.ndarray, tolerance: int) -> tuple[np.ndarray, np.ndarray]:
    """Greedy chronological one-to-one pairing, identical to fair comparison."""
    ai: list[int] = []
    bi: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j] - tolerance:
            i += 1
        elif b[j] < a[i] - tolerance:
            j += 1
        else:
            ai.append(i)
            bi.append(j)
            i += 1
            j += 1
    return np.asarray(ai, dtype=np.int64), np.asarray(bi, dtype=np.int64)


def load_unit_times(root: Path, cluster_ids: set[int]) -> dict[int, np.ndarray]:
    curated = root / "cur/cur_output"
    times = np.load(curated / "spike_times.npy", mmap_mode="r").reshape(-1)
    clusters = np.load(curated / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    out = {}
    for cid in sorted(cluster_ids):
        out[cid] = np.asarray(times[clusters == cid], dtype=np.int64)
    return out


def recording_spec(root: Path) -> dict:
    path = root / "recording/binary.json"
    spec = json.loads(path.read_text())["kwargs"]
    binary = root / "recording" / spec["file_paths"][0]
    n_channels = int(spec["num_channels"])
    dtype = np.dtype(spec["dtype"])
    n_frames, remainder = divmod(binary.stat().st_size, dtype.itemsize * n_channels)
    if remainder:
        raise ValueError(f"binary size is not frame aligned: {binary}")
    return {
        "path": binary,
        "n_channels": n_channels,
        "dtype": dtype,
        "n_frames": n_frames,
        "channel_ids": list(spec["channel_ids"]),
        "sampling_frequency": float(spec["sampling_frequency"]),
    }


def template_channel_sets(root: Path, cluster_ids: set[int], count: int = 12) -> dict[int, np.ndarray]:
    curated = root / "cur/cur_output"
    templates = np.load(curated / "templates.npy", mmap_mode="r")
    inverse = np.load(curated / "whitening_mat_inv.npy")
    positions = np.load(curated / "channel_positions.npy")
    out = {}
    for cid in cluster_ids:
        physical = np.asarray(templates[cid], dtype=np.float64) @ inverse
        energy = np.sum(physical * physical, axis=0)
        peak = int(np.argmax(energy))
        near = np.flatnonzero(np.abs(positions[:, 1] - positions[peak, 1]) <= 160.0)
        ranked = near[np.argsort(energy[near])[::-1]]
        out[cid] = np.sort(ranked[: min(count, len(ranked))]).astype(int)
    return out


def stratified_indices(times: np.ndarray, n_frames: int, per_epoch: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0, n_frames, 4)
    chosen = []
    epochs = []
    for epoch in range(3):
        eligible = np.flatnonzero((times >= edges[epoch]) & (times < edges[epoch + 1]))
        if not len(eligible):
            continue
        take = eligible[np.linspace(0, len(eligible) - 1, min(per_epoch, len(eligible)), dtype=int)]
        chosen.append(take)
        epochs.append(np.full(len(take), epoch, dtype=int))
    return np.concatenate(chosen), np.concatenate(epochs)


def extract_filtered(spec: dict, centers: np.ndarray, channels: np.ndarray) -> np.ndarray:
    raw = np.memmap(spec["path"], mode="r", dtype=spec["dtype"],
                    shape=(spec["n_frames"], spec["n_channels"]))
    width = 2 * SNIPPET_HALF + 1
    waves = np.empty((len(centers), width, len(channels)), dtype=np.float32)
    for row, center in enumerate(centers):
        start, stop = int(center) - SNIPPET_HALF, int(center) + SNIPPET_HALF + 1
        if start < 0 or stop > spec["n_frames"]:
            raise ValueError(f"snippet center outside safe range: {center}")
        waves[row] = raw[start:stop, channels]
    del raw
    sos = butter(3, (300.0, 6000.0), btype="bandpass", fs=FS, output="sos")
    waves = sosfiltfilt(sos, waves, axis=1).astype(np.float32)
    return waves


def best_alignment(waves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sub-sample residual lag, cosine, and the all-event reference."""
    center = SNIPPET_HALF
    reference = np.median(
        waves[:, center - REFERENCE_HALF:center + REFERENCE_HALF + 1, :], axis=0
    )
    reference = reference - reference.mean(axis=0, keepdims=True)
    rv = reference.reshape(-1).astype(np.float64)
    rn = np.linalg.norm(rv)
    scores = np.empty((len(waves), len(SEARCH_LAGS)), dtype=np.float64)
    for column, lag in enumerate(SEARCH_LAGS):
        segment = waves[:, center - REFERENCE_HALF + lag:center + REFERENCE_HALF + 1 + lag, :]
        segment = segment - segment.mean(axis=1, keepdims=True)
        flat = segment.reshape(len(waves), -1).astype(np.float64)
        denom = np.linalg.norm(flat, axis=1) * rn
        scores[:, column] = np.divide(flat @ rv, denom, out=np.zeros(len(waves)), where=denom > 0)
    best = np.argmax(scores, axis=1)
    refined = SEARCH_LAGS[best].astype(float)
    interior = (best > 0) & (best < len(SEARCH_LAGS) - 1)
    rows = np.flatnonzero(interior)
    y0, y1, y2 = scores[rows, best[rows] - 1], scores[rows, best[rows]], scores[rows, best[rows] + 1]
    curvature = y0 - 2 * y1 + y2
    correction = np.divide(0.5 * (y0 - y2), curvature, out=np.zeros_like(curvature), where=np.abs(curvature) > 1e-12)
    refined[rows] += np.clip(correction, -0.5, 0.5)
    return refined, scores[np.arange(len(waves)), best], reference


def robust_mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.median(np.abs(values - np.median(values))))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    av, bv = a.reshape(-1).astype(float), b.reshape(-1).astype(float)
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    return float(np.dot(av, bv) / denom) if denom else np.nan


def epoch_medians(waves: np.ndarray, epochs: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, list[float]]:
    center = SNIPPET_HALF
    medians, similarities = [], []
    for epoch in range(3):
        median = np.median(waves[epochs == epoch], axis=0)
        crop = median[center - REFERENCE_HALF:center + REFERENCE_HALF + 1]
        crop = crop - crop.mean(axis=0, keepdims=True)
        medians.append(crop)
        similarities.append(cosine(crop, reference))
    return np.asarray(medians), similarities


def render_summary(pair_metrics: pd.DataFrame, event_metrics: pd.DataFrame, output: Path) -> None:
    interior = pair_metrics[pair_metrics.primary_interior]
    events = event_metrics[event_metrics.primary_interior]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    bins = np.arange(-15.5, 16.5, 1)
    ax.hist(events.timestamp_delta_samples, bins=bins, color="#536a7a", edgecolor="white", lw=0.4)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title="Matched-event timestamp differences", xlabel="MEDiCINe − RESCUE (samples)", ylabel="sampled matched events")

    ax = axes[0, 1]
    heat = interior[[f"timestamp_median_epoch_{i}_samples" for i in range(3)]].to_numpy()
    vmax = max(1.0, float(np.nanmax(np.abs(heat))))
    image = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set(xticks=range(3), xticklabels=["early", "middle", "late"],
           yticks=np.arange(len(interior)),
           yticklabels=[f"{int(r.baseline_cluster)}→{int(r.candidate_cluster)}" for r in interior.itertuples()],
           title="Median timestamp difference by recording third")
    fig.colorbar(image, ax=ax, label="samples", shrink=0.8)

    ax = axes[1, 0]
    ax.scatter(interior.rescue_alignment_mad_samples, interior.medicine_alignment_mad_samples,
               color="#536a7a", edgecolor="white", s=42, linewidth=0.5)
    limit = max(1.0, float(np.nanmax(interior[["rescue_alignment_mad_samples", "medicine_alignment_mad_samples"]].to_numpy())) * 1.1)
    ax.plot([0, limit], [0, limit], color="#9ca3af", ls="--", lw=0.9)
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="RESCUE residual alignment MAD (samples)",
           ylabel="MEDiCINe residual alignment MAD (samples)", title="Waveform-based timing jitter")

    ax = axes[1, 1]
    ax.scatter(interior.rescue_median_event_cosine, interior.medicine_median_event_cosine,
               color="#536a7a", edgecolor="white", s=42, linewidth=0.5)
    lo = min(0.5, float(np.nanmin(interior[["rescue_median_event_cosine", "medicine_median_event_cosine"]].to_numpy())) - 0.02)
    ax.plot([lo, 1], [lo, 1], color="#9ca3af", ls="--", lw=0.9)
    ax.set(xlim=(lo, 1), ylim=(lo, 1), xlabel="RESCUE event-to-median cosine",
           ylabel="MEDiCINe event-to-median cosine", title="Fresh-snippet waveform consistency")
    for ax in axes.flat:
        ax.grid(color="#e5e7eb", lw=0.5, zorder=0)
        ax.tick_params(labelsize=8)
    fig.suptitle("Good→MUA timing audit · 16 fair-interior reciprocal pairs\nFresh snippets; amplitude is not a quality gate", fontsize=13)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_gallery(pair_metrics: pd.DataFrame, wave_payload: dict, output: Path) -> None:
    interior = pair_metrics[pair_metrics.primary_interior].reset_index(drop=True)
    fig, axes = plt.subplots(4, 4, figsize=(13, 10.5), constrained_layout=True)
    epoch_alpha = [0.32, 0.62, 1.0]
    t_ms = (np.arange(2 * REFERENCE_HALF + 1) - REFERENCE_HALF) / FS * 1000
    for ax, row in zip(axes.flat, interior.itertuples(index=False)):
        key = f"b{int(row.baseline_cluster)}_c{int(row.candidate_cluster)}"
        for arm, linestyle in (("rescue", "-"), ("medicine", "--")):
            waves = wave_payload[f"{key}_{arm}_epoch_medians"]
            peak_channel = int(np.argmax(np.ptp(np.median(waves, axis=0), axis=0)))
            scale = np.max(np.abs(waves[:, :, peak_channel])) or 1.0
            for epoch in range(3):
                ax.plot(t_ms, waves[epoch, :, peak_channel] / scale,
                        color=COLORS[arm], ls=linestyle, alpha=epoch_alpha[epoch], lw=1.1)
        ax.axvline(0, color="#d1d5db", lw=0.6)
        ax.axhline(0, color="#d1d5db", lw=0.6)
        ax.set_title(f"{int(row.baseline_cluster)}→{int(row.candidate_cluster)}  Δt MAD {row.timestamp_mad_samples:.2f}", fontsize=8)
        ax.set_xlim(t_ms[0], t_ms[-1])
        ax.set_ylim(-1.08, 1.08)
        ax.tick_params(labelsize=7)
    fig.supxlabel("time from each run's spike timestamp (ms)")
    fig.supylabel("normalized voltage on each unit's dominant fresh-snippet channel")
    fig.suptitle("Early / middle / late fresh waveform medians (light → dark)\n"
                 "All 16 fair-interior good→MUA pairs · blue solid: RESCUE · orange dashed: MEDiCINe",
                 fontsize=13)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    all_pairs = pd.read_csv(PAIRS)
    pairs = all_pairs[(all_pairs.baseline_label == "good") & (all_pairs.candidate_label == "mua")].copy()
    pairs["primary_interior"] = (
        pairs.baseline_median_depth_um.between(*PRIMARY_DEPTH_RANGE_UM)
        & pairs.candidate_median_depth_um.between(*PRIMARY_DEPTH_RANGE_UM)
    )
    b_ids, c_ids = set(pairs.baseline_cluster.astype(int)), set(pairs.candidate_cluster.astype(int))
    b_times, c_times = load_unit_times(BASE, b_ids), load_unit_times(CAND, c_ids)
    b_spec, c_spec = recording_spec(BASE), recording_spec(CAND)
    if b_spec["n_frames"] != c_spec["n_frames"] or b_spec["sampling_frequency"] != c_spec["sampling_frequency"]:
        raise ValueError("recording durations or sampling rates differ")
    b_channels, c_channels = template_channel_sets(BASE, b_ids), template_channel_sets(CAND, c_ids)

    pair_rows, event_rows, payload = [], [], {}
    for pair_number, row in enumerate(pairs.itertuples(index=False), start=1):
        bid, cid = int(row.baseline_cluster), int(row.candidate_cluster)
        ai, ci = exclusive_pairs(b_times[bid], c_times[cid], TOLERANCE_SAMPLES)
        if len(ai) != int(row.matched_events):
            raise ValueError(f"match count changed for {bid}→{cid}: {len(ai)} != {row.matched_events}")
        bt, ct = b_times[bid][ai], c_times[cid][ci]
        select, epochs = stratified_indices(bt, b_spec["n_frames"], SAMPLES_PER_EPOCH)
        sbt, sct = bt[select], ct[select]
        print(f"[{pair_number:02d}/{len(pairs)}] {bid}→{cid}: extracting {len(select)} paired snippets", flush=True)
        bw = extract_filtered(b_spec, sbt, b_channels[bid])
        cw = extract_filtered(c_spec, sct, c_channels[cid])
        blag, bcos, bref = best_alignment(bw)
        clag, ccos, cref = best_alignment(cw)
        bmed, b_epoch_cos = epoch_medians(bw, epochs, bref)
        cmed, c_epoch_cos = epoch_medians(cw, epochs, cref)
        key = f"b{bid}_c{cid}"
        payload[f"{key}_rescue_epoch_medians"] = bmed.astype(np.float32)
        payload[f"{key}_medicine_epoch_medians"] = cmed.astype(np.float32)
        payload[f"{key}_rescue_channels"] = b_channels[bid]
        payload[f"{key}_medicine_channels"] = c_channels[cid]
        delta = sct - sbt
        item = {
            "baseline_cluster": bid,
            "candidate_cluster": cid,
            "primary_interior": bool(row.primary_interior),
            "matched_events_all": int(len(bt)),
            "sampled_events": int(len(select)),
            "timestamp_median_samples": float(np.median(delta)),
            "timestamp_mad_samples": robust_mad(delta),
            "timestamp_fraction_exact": float(np.mean(delta == 0)),
            "timestamp_fraction_within_1_sample": float(np.mean(np.abs(delta) <= 1)),
            "timestamp_fraction_within_2_samples": float(np.mean(np.abs(delta) <= 2)),
            "rescue_alignment_median_samples": float(np.median(blag)),
            "rescue_alignment_mad_samples": robust_mad(blag),
            "medicine_alignment_median_samples": float(np.median(clag)),
            "medicine_alignment_mad_samples": robust_mad(clag),
            "rescue_median_event_cosine": float(np.median(bcos)),
            "medicine_median_event_cosine": float(np.median(ccos)),
        }
        for epoch in range(3):
            take = epochs == epoch
            item[f"timestamp_median_epoch_{epoch}_samples"] = float(np.median(delta[take]))
            item[f"timestamp_mad_epoch_{epoch}_samples"] = robust_mad(delta[take])
            item[f"rescue_epoch_{epoch}_waveform_cosine"] = float(b_epoch_cos[epoch])
            item[f"medicine_epoch_{epoch}_waveform_cosine"] = float(c_epoch_cos[epoch])
        pair_rows.append(item)
        for event_index in range(len(select)):
            event_rows.append({
                "baseline_cluster": bid, "candidate_cluster": cid,
                "primary_interior": bool(row.primary_interior), "epoch": int(epochs[event_index]),
                "baseline_sample": int(sbt[event_index]), "candidate_sample": int(sct[event_index]),
                "timestamp_delta_samples": int(delta[event_index]),
                "rescue_alignment_lag_samples": float(blag[event_index]),
                "medicine_alignment_lag_samples": float(clag[event_index]),
                "rescue_waveform_cosine": float(bcos[event_index]),
                "medicine_waveform_cosine": float(ccos[event_index]),
            })

    pair_metrics = pd.DataFrame(pair_rows)
    event_metrics = pd.DataFrame(event_rows)
    pair_metrics.to_csv(output / "pair_timing_metrics.csv", index=False)
    event_metrics.to_csv(output / "sampled_event_metrics.csv", index=False)
    np.savez_compressed(output / "fresh_epoch_waveforms.npz", **payload)
    render_summary(pair_metrics, event_metrics, output / "timing_quality_summary.png")
    render_gallery(pair_metrics, payload, output / "fresh_waveform_gallery_interior.png")

    interior = pair_metrics[pair_metrics.primary_interior]
    interior_events = event_metrics[event_metrics.primary_interior]
    summary = {
        "schema_version": "luke-medicine-fresh-timing-waveforms-v1",
        "primary_question": "Does MEDiCINe amplitude attenuation degrade spike-time precision?",
        "primary_interior_pairs": int(len(interior)),
        "preserved_suspect_edge_pairs": int((~pair_metrics.primary_interior).sum()),
        "fresh_paired_snippets": int(len(event_metrics)),
        "fresh_primary_interior_paired_snippets": int(len(interior_events)),
        "sampling_frequency_hz": FS,
        "match_tolerance_samples": TOLERANCE_SAMPLES,
        "match_tolerance_ms": TOLERANCE_SAMPLES / FS * 1000,
        "aggregate_timestamp_difference": {
            "median_samples": float(interior_events.timestamp_delta_samples.median()),
            "mad_samples": robust_mad(interior_events.timestamp_delta_samples.to_numpy()),
            "fraction_exact": float(interior_events.timestamp_delta_samples.eq(0).mean()),
            "fraction_within_1_sample": float(interior_events.timestamp_delta_samples.abs().le(1).mean()),
            "fraction_within_2_samples": float(interior_events.timestamp_delta_samples.abs().le(2).mean()),
        },
        "unit_median_metrics": {
            column: float(interior[column].median()) for column in (
                "timestamp_mad_samples", "rescue_alignment_mad_samples", "medicine_alignment_mad_samples",
                "rescue_median_event_cosine", "medicine_median_event_cosine",
            )
        },
        "pair_counts": {
            "medicine_alignment_mad_le_rescue": int((interior.medicine_alignment_mad_samples <= interior.rescue_alignment_mad_samples).sum()),
            "medicine_event_cosine_ge_rescue": int((interior.medicine_median_event_cosine >= interior.rescue_median_event_cosine).sum()),
            "all_epoch_timestamp_medians_within_1_sample": int((interior[[f"timestamp_median_epoch_{i}_samples" for i in range(3)]].abs() <= 1).all(axis=1).sum()),
        },
        "method": {
            "event_selection": "80 deterministic evenly spaced exclusive matches per recording third per pair",
            "waveform_preprocessing": "third-order zero-phase 300-6000 Hz bandpass on 141-sample fresh snippets",
            "spatial_support": "12 highest-energy channels within 160 um of each run's dewhitened template peak",
            "alignment": "multichannel cosine against each run's own fresh median, +/-6 samples with parabolic sub-sample refinement",
            "amplitude_gate": "none",
        },
        "limitations": [
            "Cross-run timestamp agreement is not external ground truth; shared errors can agree.",
            "Within-run residual alignment uses each run's own median waveform and measures precision, not absolute timing accuracy.",
            "The shallow- and deep-edge pairs are preserved in files but excluded from the 300-3540 um primary summary.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
