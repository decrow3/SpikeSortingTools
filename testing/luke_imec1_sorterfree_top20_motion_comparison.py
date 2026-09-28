#!/usr/bin/env python
"""Compare frozen sorter-free imec1 candidates with existing motion fields.

Candidate rank is inherited from the depth-blind seed screen.  Motion estimates
are consulted only after identities are frozen.  Comparisons use recording-
relative time, the saved physical sign, no lag/sign/scale optimization, and one
observation per candidate per one-second bin.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DISCOVERY = ROOT / "testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v2"
MOTION_ROOT = Path("/mnt/NPX/Luke/20250804/dredge_pipeline_results_Luke0804_V2V1_g0_imec1/motion")
OUT = ROOT / "testing/outputs/luke_imec1_sorterfree_top20_motion_comparison_v2"
ORIGIN_S = 3057.6775463558583
SEED_INTERVAL = (36.457968473, 56.457968473)
WINDOWS = [(935,941),(974,986),(998,1002),(1014,1025),(1071,1075),(1097,1112)]
ESTIMATORS = {
    "DREDGE": "dredge-motion",
    "KS sidecar": "ks-motion",
    "Decentralized": "decentralized-motion",
    "MEDiCINe": "medicine",
}


def load_field(folder: str):
    path = MOTION_ROOT / folder
    time = np.load(path / "time_bins.npy").reshape(-1) - ORIGIN_S
    depth = np.load(path / "depth_bins.npy").reshape(-1)
    motion = np.load(path / "motion.npy")
    if motion.shape != (len(time), len(depth)) or not np.isfinite(motion).all():
        raise RuntimeError(f"Invalid motion field {folder}")
    return time, depth, motion, RegularGridInterpolator((time, depth), motion, bounds_error=False, fill_value=np.nan)


def sample(interpolator, time_s: np.ndarray, depth_um: float) -> np.ndarray:
    return interpolator(np.c_[time_s, np.full(len(time_s), depth_um)])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    families = pd.read_csv(DISCOVERY / "families_after_depth_reveal.csv")
    top = families.sort_values("candidate_rank_depth_blind").head(20).copy()
    events = pd.read_csv(DISCOVERY / "heldout_events.csv")
    strict = events[events.status.eq("strict")].copy()
    fields = {name: load_field(folder) for name, folder in ESTIMATORS.items()}

    inventory = []
    track_rows = []
    metric_rows = []
    for family in top.itertuples(index=False):
        q = strict[strict.family_id.eq(family.family_id)].copy()
        q["second_bin"] = np.floor(q.time_s).astype(int) if len(q) else pd.Series(dtype=int)
        binned = q.groupby("second_bin", as_index=False).agg(
            time_s=("time_s", "median"), observed_depth_um=("waveform_centroid_um", "median"),
            strict_events=("time_s", "size"),
        ) if len(q) else pd.DataFrame(columns=["second_bin","time_s","observed_depth_um","strict_events"])
        windows = sum(bool(((q.time_s >= a) & (q.time_s < b)).any()) for a,b in WINDOWS) if len(q) else 0
        inventory.append({
            "rank": int(family.candidate_rank_depth_blind), "family_id": family.family_id,
            "candidate_tier": family.candidate_tier, "seed_events": int(family.seed_events),
            "seed_depth_median_um": family.seed_depth_median_um,
            "seed_depth_p90_span_um": family.seed_depth_p90_span_um,
            "strict_events": len(q), "strict_one_second_bins": len(binned), "heldout_windows": windows,
        })
        observed = binned.observed_depth_um.to_numpy(dtype=float) - family.seed_depth_median_um
        for estimator, (_, _, _, interp) in fields.items():
            seed_times = np.arange(SEED_INTERVAL[0] + .5, SEED_INTERVAL[1], 1.0)
            baseline = float(np.nanmedian(sample(interp, seed_times, family.seed_depth_median_um)))
            predicted = sample(interp, binned.time_s.to_numpy(dtype=float), family.seed_depth_median_um) - baseline if len(binned) else np.array([], dtype=float)
            for row, obs, pred in zip(binned.itertuples(index=False), observed, predicted):
                track_rows.append({
                    "rank": int(family.candidate_rank_depth_blind), "family_id": family.family_id,
                    "time_s": row.time_s, "strict_events": row.strict_events,
                    "observed_displacement_um": obs, "estimator": estimator,
                    "predicted_displacement_um": pred,
                })
            valid = np.isfinite(observed) & np.isfinite(predicted)
            n = int(valid.sum())
            pearson = float(np.corrcoef(observed[valid], predicted[valid])[0,1]) if n >= 5 and np.std(observed[valid]) > 0 and np.std(predicted[valid]) > 0 else np.nan
            spearman = float(spearmanr(observed[valid], predicted[valid]).statistic) if n >= 5 else np.nan
            slope = float(np.polyfit(predicted[valid], observed[valid], 1)[0]) if n >= 5 and np.std(predicted[valid]) > 0 else np.nan
            mae = float(np.mean(np.abs(observed[valid] - predicted[valid]))) if n else np.nan
            metric_rows.append({
                "rank": int(family.candidate_rank_depth_blind), "family_id": family.family_id,
                "estimator": estimator, "one_second_bins": n, "pearson_r_fixed_sign": pearson,
                "spearman_rho_fixed_sign": spearman, "observed_per_predicted_slope": slope,
                "unfitted_mae_um": mae,
            })

    inventory = pd.DataFrame(inventory)
    tracks = pd.DataFrame(track_rows)
    metrics = pd.DataFrame(metric_rows)
    inventory.to_csv(OUT / "top20_candidate_inventory.csv", index=False)
    tracks.to_csv(OUT / "one_second_tracks_and_predictions.csv", index=False)
    metrics.to_csv(OUT / "candidate_estimator_metrics.csv", index=False)

    active = inventory[inventory.strict_one_second_bins > 0]
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True, layout="constrained")
    colors = {"DREDGE":"#007f73", "KS sidecar":"#2265ac", "Decentralized":"#aa7600", "MEDiCINe":"#6e59a5"}
    for ax, candidate in zip(axes.flat, active.itertuples(index=False)):
        g = tracks[tracks.family_id.eq(candidate.family_id)]
        observed = g.drop_duplicates(["time_s"])
        ax.scatter(observed.time_s, observed.observed_displacement_um, c="black", s=18, label="Waveform depth")
        for estimator in ESTIMATORS:
            h = g[g.estimator.eq(estimator)]
            for start, stop in WINDOWS:
                z = h[(h.time_s >= start) & (h.time_s < stop)]
                if len(z): ax.plot(z.time_s, z.predicted_displacement_um, marker=".", lw=1, color=colors[estimator], label=estimator if start==WINDOWS[0][0] else None)
        ax.axhline(0, color="gray", lw=.5)
        ax.set_title(f"#{candidate.rank} {candidate.family_id}: {candidate.strict_events} strict events / {candidate.strict_one_second_bins} bins")
        ax.set_ylabel("Relative depth / displacement (µm)")
    for ax in axes.flat[len(active):]: ax.axis("off")
    axes.flat[0].legend(fontsize=7, ncol=2)
    fig.supxlabel("Recording-relative time (s)")
    fig.suptitle("Top-20 waveform candidates versus existing imec1 motion estimates\nFixed physical sign; seed-referenced; no fitted lag, sign, or scale")
    fig.savefig(OUT / "01_active_top20_motion_overlay.png", dpi=170)
    fig.savefig(OUT / "01_active_top20_motion_overlay.pdf")
    plt.close(fig)

    evaluable = metrics[metrics.one_second_bins >= 5].copy()
    matrix = evaluable.pivot(index=["rank","family_id"], columns="estimator", values="pearson_r_fixed_sign").reindex(columns=list(ESTIMATORS))
    fig, ax = plt.subplots(figsize=(9, max(3, .55*len(matrix)+1.6)), layout="constrained")
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix)), [f"#{int(r)} {f}" for r,f in matrix.index])
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            value = matrix.iloc[i,j]
            ax.text(j,i,"—" if not np.isfinite(value) else f"{value:.2f}",ha="center",va="center",fontsize=9)
    fig.colorbar(image, ax=ax, label="Pearson r, fixed physical sign")
    ax.set_title("Candidate–motion agreement (one-second medians; at least five bins)")
    fig.savefig(OUT / "02_candidate_estimator_correlation.png", dpi=170)
    fig.savefig(OUT / "02_candidate_estimator_correlation.pdf")
    plt.close(fig)

    summary = {
        "schema": "luke0804-imec1-sorterfree-top20-motion-comparison-v1",
        "status": "complete", "candidate_ranking_used_motion": False,
        "top20_candidates": inventory.family_id.tolist(),
        "top20_with_strict_events": int((inventory.strict_events > 0).sum()),
        "top20_evaluable_candidates": int(inventory.strict_one_second_bins.ge(5).sum()),
        "motion_estimators": list(ESTIMATORS), "recording_time_origin_s": ORIGIN_S,
        "comparison": "One-second median waveform depths versus fields sampled at frozen seed depth; seed-referenced; saved physical sign; no sign, scale, or lag optimization.",
        "limitations": [
            "Waveform families are candidates, not verified cells.",
            "Sparse held-out windows and multimodal families can inflate or obscure correlations.",
            "The motion estimates share some detection inputs and are not independent ground truth.",
            "Unfitted waveform-centroid excursion need not have unit slope relative to tissue displacement.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
