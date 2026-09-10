"""Cheap depth-blind linkage of cached Kilosort CIDs in the 930--1030 s pilot.

This is deliberately not a spike sort and does not read voltage.  It recenters
each cached Kilosort template on an exact 40-um repeating Neuropixels patch,
compares relative multichannel waveforms without absolute depth, freezes a
conservative reciprocal-nearest-neighbour linkage, and only then reveals the
cached template and per-spike depths for plausibility review.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    "/mnt/NPX/Luke/20250804/"
    "rescue_pipeline_results_Luke0804_V2V1_g0_imec0/cur/cur_output"
)
EVENTS = ROOT / "testing/outputs/luke_kilosort_cid_depth_scatter_v1/spikes_930_1030.npz"
EVENT_SUMMARY = ROOT / "testing/outputs/luke_kilosort_cid_depth_scatter_v1/summary.json"
OUT = ROOT / "testing/outputs/luke_kilosort_depthblind_tracklet_linkage_v1"

INTERVAL_S = (930.0, 1030.0)
MIN_SNIPPET_SPIKES = 20
MIN_LOCAL_ENERGY_FRACTION = 0.80
PATCH_Y_MIN_UM = -60.0
PATCH_Y_MAX_UM = 80.0
WAVEFORM_SAMPLES = 49
MAX_TEMPORAL_LAG_SAMPLES = 3
THRESHOLDS = (0.90, 0.925, 0.95, 0.975, 0.985, 0.99, 0.995)
STRICT_COSINE = 0.99
SENSITIVITY_COSINE = 0.975
COINCIDENCE_S = 0.0003
REFRACTORY_S = 0.001


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def centered_crop(wave: np.ndarray, peak_sample: int, samples: int) -> np.ndarray:
    out = np.zeros((samples, wave.shape[1]), dtype=np.float32)
    source_start = peak_sample - samples // 2
    a = max(0, source_start)
    b = min(wave.shape[0], source_start + samples)
    out[a - source_start:b - source_start] = wave[a:b]
    return out


def shifted(waves: np.ndarray, lag: int) -> np.ndarray:
    out = np.zeros_like(waves)
    if lag < 0:
        out[:, :lag] = waves[:, -lag:]
    elif lag > 0:
        out[:, lag:] = waves[:, :-lag]
    else:
        out[:] = waves
    return out


def normalized_rows(waves: np.ndarray) -> np.ndarray:
    flat = waves.reshape(len(waves), -1).astype(np.float64)
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    return flat / np.maximum(norm, np.finfo(float).tiny)


def best_lagged_cosine(waves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise cosine, shifting only the right-hand template at each lag."""
    reference = normalized_rows(waves)
    n = len(waves)
    best = np.full((n, n), -np.inf, dtype=np.float32)
    best_lag = np.zeros((n, n), dtype=np.int8)
    for lag in range(-MAX_TEMPORAL_LAG_SAMPLES, MAX_TEMPORAL_LAG_SAMPLES + 1):
        score = reference @ normalized_rows(shifted(waves, lag)).T
        take = score > best
        best[take] = score[take]
        best_lag[take] = lag

    # Numerical ties can make the two directions differ slightly.  Keep the
    # better direction and express lag as the shift applied to the second row.
    transpose_wins = best.T > best
    symmetric = np.maximum(best, best.T)
    symmetric_lag = best_lag.copy()
    symmetric_lag[transpose_wins] = -best_lag.T[transpose_wins]
    np.fill_diagonal(symmetric, 1.0)
    np.fill_diagonal(symmetric_lag, 0)
    return symmetric, symmetric_lag


def nearest_cross_count(a: np.ndarray, b: np.ndarray, tolerance_s: float) -> int:
    """Number of events in the smaller train with any event in the other train."""
    if len(a) > len(b):
        a, b = b, a
    j = np.searchsorted(b, a)
    left = np.maximum(j - 1, 0)
    right = np.minimum(j, len(b) - 1)
    distance = np.minimum(np.abs(a - b[left]), np.abs(a - b[right]))
    return int(np.count_nonzero(distance <= tolerance_s))


def violation_fraction(times_s: np.ndarray, interval_s: float) -> float:
    if len(times_s) < 2:
        return np.nan
    return float(np.mean(np.diff(np.sort(times_s)) < interval_s))


def make_templates(
    templates: np.ndarray,
    positions: np.ndarray,
    event_cids: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    active, counts = np.unique(event_cids, return_counts=True)
    labels_path = SOURCE / "cluster_KSLabel.tsv"
    labels = {}
    if labels_path.exists():
        label_df = pd.read_csv(labels_path, sep="\t")
        labels = dict(zip(label_df["cluster_id"].astype(int), label_df["KSLabel"].astype(str)))

    records: list[dict] = []
    canonical: list[np.ndarray] = []
    for cid, count in zip(active.astype(int), counts.astype(int)):
        if count < MIN_SNIPPET_SPIKES or cid < 0 or cid >= len(templates):
            continue
        full = np.asarray(templates[cid], dtype=np.float32)
        peak_sample_full, peak_channel = np.unravel_index(np.argmax(np.abs(full)), full.shape)
        peak_y = float(positions[peak_channel, 1])
        patch_origin = 40.0 * np.floor(peak_y / 40.0)
        keep = np.flatnonzero(
            (positions[:, 1] >= patch_origin + PATCH_Y_MIN_UM - 1e-6)
            & (positions[:, 1] <= patch_origin + PATCH_Y_MAX_UM + 1e-6)
        )
        keep = keep[np.lexsort((positions[keep, 0], positions[keep, 1] - patch_origin))]
        expected_channels = 16
        total_energy = float(np.sum(full.astype(np.float64) ** 2))
        local_energy = float(np.sum(full[:, keep].astype(np.float64) ** 2)) if len(keep) else 0.0
        local_fraction = local_energy / total_energy if total_energy > 0 else np.nan
        reason = "eligible"
        if len(keep) != expected_channels:
            reason = "incomplete_probe_edge_patch"
        elif not np.isfinite(local_fraction) or local_fraction < MIN_LOCAL_ENERGY_FRACTION:
            reason = "diffuse_template"

        record = {
            "cid": cid,
            "snippet_spikes": count,
            "KSLabel": labels.get(cid, "unknown"),
            "eligibility": reason,
            "local_energy_fraction": local_fraction,
            "template_peak_abs_ks_units": float(np.max(np.abs(full))),
            "peak_sample_full": int(peak_sample_full),
            # Depth-bearing fields are saved separately after matching.
            "peak_channel": int(peak_channel),
            "template_peak_x_um": float(positions[peak_channel, 0]),
            "template_peak_depth_um": peak_y,
            "patch_origin_depth_um": patch_origin,
        }
        records.append(record)
        if reason == "eligible":
            patch = full[:, keep]
            patch_peak_sample = int(np.unravel_index(np.argmax(np.abs(patch)), patch.shape)[0])
            canonical.append(centered_crop(patch, patch_peak_sample, WAVEFORM_SAMPLES))

    table = pd.DataFrame(records)
    eligible = table[table["eligibility"] == "eligible"].copy().reset_index(drop=True)
    # records and canonical were both appended in CID order, but ineligible rows
    # are interspersed; rebuild the eligible order explicitly to guard refactors.
    assert len(eligible) == len(canonical)
    return table, np.stack(canonical)


def pair_event_metrics(
    cid_a: int,
    cid_b: int,
    frames: np.ndarray,
    event_cids: np.ndarray,
    fs: float,
) -> dict:
    a = np.sort(frames[event_cids == cid_a].astype(np.float64) / fs)
    b = np.sort(frames[event_cids == cid_b].astype(np.float64) / fs)
    merged = np.sort(np.concatenate([a, b]))
    cross = nearest_cross_count(a, b, COINCIDENCE_S)
    duration = INTERVAL_S[1] - INTERVAL_S[0]
    expected_cross = 2 * COINCIDENCE_S * len(a) * len(b) / duration
    bins = np.arange(INTERVAL_S[0], INTERVAL_S[1] + 1.0, 1.0)
    ca = np.histogram(a, bins=bins)[0]
    cb = np.histogram(b, bins=bins)[0]
    if np.std(ca) > 0 and np.std(cb) > 0:
        count_correlation = float(pearsonr(ca, cb).statistic)
    else:
        count_correlation = np.nan
    occupied_a = set(np.flatnonzero(ca))
    occupied_b = set(np.flatnonzero(cb))
    occupied_overlap = len(occupied_a & occupied_b) / max(1, min(len(occupied_a), len(occupied_b)))
    rate_hz = len(merged) / duration
    poisson_refractory = 1.0 - np.exp(-rate_hz * REFRACTORY_S)
    merged_refractory = violation_fraction(merged, REFRACTORY_S)
    return {
        "events_a": len(a),
        "events_b": len(b),
        "median_time_a_s": float(np.median(a)),
        "median_time_b_s": float(np.median(b)),
        "one_second_bin_count_correlation": count_correlation,
        "occupied_second_overlap_fraction": occupied_overlap,
        "cross_coincidences_within_0p3ms": cross,
        "expected_cross_coincidences_poisson": expected_cross,
        "cross_coincidence_ratio_to_poisson": cross / expected_cross if expected_cross > 0 else np.nan,
        "merged_refractory_violation_fraction_1ms": merged_refractory,
        "expected_poisson_short_isi_fraction_1ms": poisson_refractory,
        "merged_to_poisson_short_isi_ratio": merged_refractory / poisson_refractory if poisson_refractory > 0 else np.nan,
        "within_a_refractory_violation_fraction_1ms": violation_fraction(a, REFRACTORY_S),
        "within_b_refractory_violation_fraction_1ms": violation_fraction(b, REFRACTORY_S),
    }


def reveal_pairs(
    links: pd.DataFrame,
    eligible: pd.DataFrame,
    frames: np.ndarray,
    event_cids: np.ndarray,
    xy: np.ndarray,
    fs: float,
) -> pd.DataFrame:
    depth_by_cid = eligible.set_index("cid")[[
        "template_peak_x_um", "template_peak_depth_um", "patch_origin_depth_um"
    ]].to_dict("index")
    rows = []
    for row in links.itertuples(index=False):
        a, b = int(row.cid_a), int(row.cid_b)
        y = xy[(event_cids == a) | (event_cids == b), 1]
        record = row._asdict()
        record.update(
            {
                "template_peak_x_a_um": depth_by_cid[a]["template_peak_x_um"],
                "template_peak_x_b_um": depth_by_cid[b]["template_peak_x_um"],
                "template_peak_depth_a_um": depth_by_cid[a]["template_peak_depth_um"],
                "template_peak_depth_b_um": depth_by_cid[b]["template_peak_depth_um"],
                "template_depth_delta_um": abs(depth_by_cid[a]["template_peak_depth_um"] - depth_by_cid[b]["template_peak_depth_um"]),
                "combined_spike_depth_p05_um": float(np.quantile(y, 0.05)),
                "combined_spike_depth_p95_um": float(np.quantile(y, 0.95)),
                "combined_spike_depth_p90_span_um": float(np.quantile(y, 0.95) - np.quantile(y, 0.05)),
            }
        )
        record.update(pair_event_metrics(a, b, frames, event_cids, fs))
        rows.append(record)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["template_depth_delta_um", "cosine"], ascending=False).reset_index(drop=True)
    return result


def save_figures(
    threshold_table: pd.DataFrame,
    revealed: pd.DataFrame,
    sensitivity_revealed: pd.DataFrame,
    eligible: pd.DataFrame,
    waves: np.ndarray,
    frames: np.ndarray,
    event_cids: np.ndarray,
    xy: np.ndarray,
    fs: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5), layout="constrained")
    ax.plot(threshold_table["cosine_threshold"], threshold_table["all_pairs"], "o-", label="all waveform pairs")
    ax.plot(threshold_table["cosine_threshold"], threshold_table["reciprocal_nearest_pairs"], "o-", label="reciprocal nearest pairs")
    ax.axvline(STRICT_COSINE, color="0.35", ls="--", lw=1, label=f"review threshold {STRICT_COSINE:g}")
    ax.set(xlabel="Depth-blind relative-waveform cosine", ylabel="Pair count", yscale="log")
    ax.set_title("Similarity threshold exposes a dense lookalike bank")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(OUT / "01_similarity_threshold.png", dpi=180)
    fig.savefig(OUT / "01_similarity_threshold.pdf")
    plt.close(fig)

    if revealed.empty:
        return
    plot = revealed.sort_values(["template_depth_delta_um", "cosine"], ascending=False).head(12)
    fig, axes = plt.subplots(len(plot), 1, figsize=(14, max(4, 2.05 * len(plot))), sharex=True, layout="constrained", squeeze=False)
    times = frames.astype(np.float64) / fs
    for ax, row in zip(axes[:, 0], plot.itertuples(index=False)):
        for cid, color in [(int(row.cid_a), "#1f77b4"), (int(row.cid_b), "#d62728")]:
            take = event_cids == cid
            ax.scatter(times[take], xy[take, 1], s=3, alpha=0.55, color=color, linewidths=0, rasterized=True, label=f"CID {cid}")
        ax.set_ylabel("depth (µm)")
        ax.set_title(
            f"cos={row.cosine:.4f}, template Δ={row.template_depth_delta_um:.0f} µm, "
            f"merged/Poisson <1 ms={row.merged_to_poisson_short_isi_ratio:.2f}",
            fontsize=9,
        )
        ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=8)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[-1, 0].set_xlabel("Recording time (s)")
    fig.suptitle(
        "Cached Kilosort positions revealed after waveform-only reciprocal linkage\n"
        "Pairs ordered for display by template-depth separation; positions are sorter-derived, not independent localization",
        fontsize=12,
    )
    fig.savefig(OUT / "02_strict_link_depth_time.png", dpi=180)
    fig.savefig(OUT / "02_strict_link_depth_time.pdf")
    plt.close(fig)

    good = sensitivity_revealed[
        (sensitivity_revealed["KSLabel_a"] == "good")
        & (sensitivity_revealed["KSLabel_b"] == "good")
    ].sort_values("cosine", ascending=False)
    if good.empty:
        return
    wave_by_cid = {int(cid): waves[i] for i, cid in enumerate(eligible["cid"])}
    fig, axes = plt.subplots(len(good), 4, figsize=(15, 3.2 * len(good)), layout="constrained", squeeze=False)
    times = frames.astype(np.float64) / fs
    for row_axes, row in zip(axes, good.itertuples(index=False)):
        ax = row_axes[0]
        for cid, color in [(int(row.cid_a), "#1f77b4"), (int(row.cid_b), "#d62728")]:
            take = event_cids == cid
            ax.scatter(times[take], xy[take, 1], s=9, alpha=0.65, color=color, linewidths=0, label=f"CID {cid}")
        ax.set(xlim=INTERVAL_S, xlabel="time (s)", ylabel="cached depth (µm)")
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.legend(frameon=False, fontsize=8)
        a = wave_by_cid[int(row.cid_a)]
        b = wave_by_cid[int(row.cid_b)]
        limit = float(max(np.max(np.abs(a)), np.max(np.abs(b))))
        difference_limit = float(max(np.max(np.abs(a - b)), np.finfo(float).eps))
        for panel, data, title, lim in [
            (row_axes[1], a, f"CID {int(row.cid_a)}", limit),
            (row_axes[2], b, f"CID {int(row.cid_b)}", limit),
            (row_axes[3], a - b, "raw difference", difference_limit),
        ]:
            panel.imshow(data, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
            panel.set(xlabel="relative channel", ylabel="centered sample", title=title)
        row_axes[0].set_title(
            f"cos={row.cosine:.4f}; template Δ={row.template_depth_delta_um:.0f} µm; "
            f"events={row.events_a}+{row.events_b}", fontsize=9
        )
    fig.suptitle(
        f"Good–good reciprocal pairs in the exploratory cosine ≥{SENSITIVITY_COSINE:g} tier\n"
        "Depth/time was not used for linkage; sparse trains provide weak refractory evidence",
        fontsize=12,
    )
    fig.savefig(OUT / "03_good_label_sensitivity_pairs.png", dpi=180)
    fig.savefig(OUT / "03_good_label_sensitivity_pairs.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    event_summary = json.loads(EVENT_SUMMARY.read_text())
    fs = float(event_summary["sampling_frequency_hz"])
    events = np.load(EVENTS)
    frames = np.asarray(events["frames"])
    event_cids = np.asarray(events["cid"]).astype(int)
    xy = np.asarray(events["xy_um"])
    templates_path = SOURCE / "templates.npy"
    positions_path = SOURCE / "channel_positions.npy"
    templates = np.load(templates_path, mmap_mode="r")
    positions = np.load(positions_path)

    all_templates, waves = make_templates(templates, positions, event_cids)
    eligible = all_templates[all_templates["eligibility"] == "eligible"].copy().reset_index(drop=True)
    similarity, lag = best_lagged_cosine(waves)
    np.savez_compressed(
        OUT / "waveform_similarity_matrix.npz",
        cid=eligible["cid"].to_numpy(int),
        cosine=similarity,
        best_lag_samples=lag,
    )

    ranking = similarity.copy()
    np.fill_diagonal(ranking, -np.inf)
    nearest = np.argmax(ranking, axis=1)
    pair_rows = []
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            score = float(similarity[i, j])
            if score < min(THRESHOLDS):
                continue
            pair_rows.append(
                {
                    "cid_a": int(eligible.loc[i, "cid"]),
                    "cid_b": int(eligible.loc[j, "cid"]),
                    "cosine": score,
                    "best_lag_samples": int(lag[i, j]),
                    "reciprocal_nearest": bool(nearest[i] == j and nearest[j] == i),
                    "KSLabel_a": eligible.loc[i, "KSLabel"],
                    "KSLabel_b": eligible.loc[j, "KSLabel"],
                }
            )
    pairs = pd.DataFrame(pair_rows).sort_values("cosine", ascending=False).reset_index(drop=True)
    pairs.to_csv(OUT / "waveform_pairs_no_depth.csv", index=False)

    threshold_rows = []
    for threshold in THRESHOLDS:
        above = pairs["cosine"] >= threshold
        threshold_rows.append(
            {
                "cosine_threshold": threshold,
                "all_pairs": int(above.sum()),
                "reciprocal_nearest_pairs": int((above & pairs["reciprocal_nearest"]).sum()),
                "templates_in_all_pairs": int(pd.unique(pairs.loc[above, ["cid_a", "cid_b"]].to_numpy().ravel()).size),
                "templates_in_reciprocal_pairs": int(pd.unique(pairs.loc[above & pairs["reciprocal_nearest"], ["cid_a", "cid_b"]].to_numpy().ravel()).size),
            }
        )
    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table.to_csv(OUT / "similarity_threshold_sensitivity.csv", index=False)

    strict = pairs[(pairs["cosine"] >= STRICT_COSINE) & pairs["reciprocal_nearest"]].copy()
    strict.to_csv(OUT / "strict_links_waveform_only.csv", index=False)
    revealed = reveal_pairs(strict, eligible, frames, event_cids, xy, fs)
    revealed.to_csv(OUT / "strict_links_depth_time_revealed.csv", index=False)
    sensitivity = pairs[(pairs["cosine"] >= SENSITIVITY_COSINE) & pairs["reciprocal_nearest"]].copy()
    sensitivity_revealed = reveal_pairs(sensitivity, eligible, frames, event_cids, xy, fs)
    sensitivity_revealed.to_csv(OUT / "sensitivity_links_depth_time_revealed.csv", index=False)

    no_depth_columns = [
        "cid", "snippet_spikes", "KSLabel", "eligibility", "local_energy_fraction",
        "template_peak_abs_ks_units", "peak_sample_full",
    ]
    all_templates[no_depth_columns].to_csv(OUT / "template_eligibility_no_depth.csv", index=False)
    all_templates.to_csv(OUT / "template_metadata_depth_revealed.csv", index=False)
    save_figures(threshold_table, revealed, sensitivity_revealed, eligible, waves, frames, event_cids, xy, fs)

    strict_depth_ge_40 = int((revealed["template_depth_delta_um"] >= 40).sum()) if len(revealed) else 0
    strict_depth_ge_100 = int((revealed["template_depth_delta_um"] >= 100).sum()) if len(revealed) else 0
    good_sensitivity = sensitivity_revealed[
        (sensitivity_revealed["KSLabel_a"] == "good")
        & (sensitivity_revealed["KSLabel_b"] == "good")
    ]
    summary = {
        "status": "complete",
        "purpose": "cheap cached-data screen for depth-blind linkage of final Kilosort CIDs; not a new sort or validated cell tracker",
        "interval_s": list(INTERVAL_S),
        "sampling_frequency_hz": fs,
        "raw_voltage_read": False,
        "new_spike_sort_run": False,
        "input_spikes": int(len(frames)),
        "active_cids": int(np.unique(event_cids).size),
        "minimum_snippet_spikes": MIN_SNIPPET_SPIKES,
        "candidate_templates_before_geometry_and_locality": int(len(all_templates)),
        "eligible_complete_local_templates": int(len(eligible)),
        "minimum_local_energy_fraction": MIN_LOCAL_ENERGY_FRACTION,
        "patch_relative_y_um": [PATCH_Y_MIN_UM, PATCH_Y_MAX_UM],
        "patch_channels": int(waves.shape[2]),
        "waveform_samples": WAVEFORM_SAMPLES,
        "temporal_lag_samples": [-MAX_TEMPORAL_LAG_SAMPLES, MAX_TEMPORAL_LAG_SAMPLES],
        "strict_link_rule": f"cosine >= {STRICT_COSINE} and reciprocal nearest neighbour; absolute depth and spike time excluded",
        "strict_links": int(len(strict)),
        "strict_links_template_depth_delta_ge_40_um": strict_depth_ge_40,
        "strict_links_template_depth_delta_ge_100_um": strict_depth_ge_100,
        "sensitivity_link_rule": f"cosine >= {SENSITIVITY_COSINE} and reciprocal nearest neighbour; exploratory tier",
        "sensitivity_links": int(len(sensitivity)),
        "sensitivity_good_good_links": int(((sensitivity["KSLabel_a"] == "good") & (sensitivity["KSLabel_b"] == "good")).sum()),
        "promoted_lighthouse_links": 0,
        "review_outcome": (
            "No pair is promoted. All strict links are MUA/MUA and overlap in at least "
            f"{revealed['occupied_second_overlap_fraction'].min():.3f} of the less-occupied train's active seconds. "
            "The three good/good sensitivity pairs are exploratory: one has extreme excess near-coincident events, "
            "while the other two are too sparse for their zero near-coincidence counts to be informative "
            f"(Poisson expectations {good_sensitivity['expected_cross_coincidences_poisson'].min():.4f}--"
            f"{good_sensitivity['expected_cross_coincidences_poisson'].max():.4f})."
        ),
        "identity_features": "signed, amplitude-normalized 49x16 relative multichannel waveform with x geometry retained and absolute y origin removed",
        "posthoc_only": "template/per-spike depth, temporal overlap, cross-coincidence and refractory metrics",
        "limitations": [
            "Templates and CIDs are final Kilosort outputs learned from the full recording, so this is not snippet-only training or held-out identity validation.",
            "Per-spike depths are Kilosort PC-feature-weighted positions, not independent raw-voltage localization.",
            "High waveform similarity can join different neurons or artifacts; reciprocal linkage is a conservative review device, not biological identity proof.",
            "Exact 40-um recentering does not solve intermediate spatial phase or recover events that Kilosort failed to assign.",
        ],
        "source_sha256": {
            str(EVENTS): sha256(EVENTS),
            str(templates_path): sha256(templates_path),
            str(positions_path): sha256(positions_path),
        },
        "script_sha256": sha256(Path(__file__)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    readme = "# Cached Kilosort depth-blind tracklet linkage\n\n"
    readme += "This is the free first-pass experiment: no voltage was read and no sorter was run. "
    readme += "It compares cached final Kilosort templates after removing their absolute vertical origin.\n\n"
    readme += f"- Eligible templates: **{len(eligible)}**\n"
    readme += f"- Strict reciprocal links at cosine >= {STRICT_COSINE}: **{len(strict)}**\n"
    readme += f"- Strict links separated by >=40 um after depth reveal: **{strict_depth_ge_40}**\n"
    readme += f"- Strict links separated by >=100 um after depth reveal: **{strict_depth_ge_100}**\n\n"
    readme += f"- Exploratory reciprocal links at cosine >= {SENSITIVITY_COSINE}: **{len(sensitivity)}**, including **{summary['sensitivity_good_good_links']}** good–good pairs\n\n"
    readme += "## Initial review\n\n"
    readme += "**No pair is promoted as a lighthouse cell.** All six strict pairs are MUA/MUA and remain active at the same time at separated depths. "
    readme += "Of the three good–good sensitivity pairs, CID 319/379 has 47 near-coincident events versus 0.17 expected under the simple Poisson reference, consistent with duplicate/shared artifact detections rather than a depth handoff. "
    readme += "CID 216/231 and 356/399 are too sparse for their absence of near coincidences or refractory violations to validate identity.\n\n"
    readme += "`strict_links_waveform_only.csv` is the frozen identity screen. "
    readme += "`strict_links_depth_time_revealed.csv` adds sorter-derived depth and spike-train plausibility checks afterward. "
    readme += "The figures are diagnostics, not a count of verified lighthouse cells. See `summary.json` for limitations and hashes.\n"
    (OUT / "README.md").write_text(readme)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
