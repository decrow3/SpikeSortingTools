"""Compare full-session 10k/30k MEDiCINe fits with frozen lighthouse tracks.

No estimator enters waveform identity matching.  This script reuses the frozen
whole-probe, depth-aware lighthouse observations through 1230 s, then evaluates
both completed fields on identical events.  Each field is referenced only to
its own median at the original 930--940 s seed spike times; no later offset is
fit.  Strict, lower-score, ambiguous, and unmatched evidence remain separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0")
FIELD_10K = Path("/media/huklab/Data/luke_full_session_medicine_20260909_v1/fit/field.npz")
FIELD_30K = Path("/media/huklab/Data/luke_full_session_medicine_30000_20260909_v1/fit/field.npz")
TRACKS = ROOT / "testing/outputs/luke_lighthouse_extension_300s_v1/overlay_events.csv"
COHORT = ROOT / "testing/outputs/luke_waveform_only_expansion_v1/candidate_audit.csv"
TEMPLATES = ROOT / "testing/outputs/luke_population_depth_v2/templates.npz"
PEAKS = ROOT / "testing/outputs/luke_screened_medicine_300s_v1/input_5sigma_relaxed/peaks.npy"
LOCATIONS = ROOT / "testing/outputs/luke_screened_medicine_300s_v1/input_5sigma_relaxed/locations.npy"
FS = 29999.835983263598
COLORS = {"10k": "#1f5a94", "30k": "#c56a1a"}
INTERVALS = {"post_seed_940_1230": (940.0, 1230.0), "extension_1030_1230": (1030.0, 1230.0)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def field_curve(field: np.lib.npyio.NpzFile, depth_um: float, seed_times_s: np.ndarray) -> np.ndarray:
    if not field["depth_um"][0] <= depth_um <= field["depth_um"][-1]:
        raise ValueError(f"candidate depth {depth_um} outside field support")
    curve = np.asarray([np.interp(depth_um, field["depth_um"], row)
                        for row in field["displacement_um"]], dtype=float)
    seed = seed_times_s[(seed_times_s >= field["time_s"][0]) & (seed_times_s <= field["time_s"][-1])]
    if not len(seed):
        raise ValueError("no seed spikes on field support")
    curve -= np.median(np.interp(seed, field["time_s"], curve))
    return curve


def build_predictions() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    cohort = pd.read_csv(COHORT)
    cohort = cohort[cohort.selected.astype(bool)].sort_values("seed_centroid_um").copy()
    events = pd.read_csv(TRACKS)
    events = events[events.unit_id.isin(cohort.unit_id)].copy()
    fields = {"10k": np.load(FIELD_10K), "30k": np.load(FIELD_30K)}
    seeds = np.load(TEMPLATES)
    curves: dict[tuple[str, int], np.ndarray] = {}
    predictions = []
    for cell in cohort.itertuples(index=False):
        seed_times = seeds[f"unit_{int(cell.unit_id)}_seed_frames"] / FS
        unit_events = events[events.unit_id == cell.unit_id]
        for fit, field in fields.items():
            curve = field_curve(field, float(cell.seed_centroid_um), seed_times)
            curves[(fit, int(cell.unit_id))] = curve
            part = unit_events.copy()
            part["fit"] = fit
            part["prediction_um"] = np.interp(part.time_s, field["time_s"], curve)
            part["difference_um"] = part.relative_um - part.prediction_um
            predictions.append(part)
    return cohort, pd.concat(predictions, ignore_index=True), fields, curves


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for interval, (lo, hi) in INTERVALS.items():
        q = predictions[predictions.time_s.between(lo, hi, inclusive="left")]
        for (fit, unit, evidence), part in q.groupby(["fit", "unit_id", "evidence"]):
            for large in (False, True):
                take = part[np.abs(part.relative_um) >= 120.0] if large else part
                rows.append({
                    "interval": interval, "fit": fit, "unit_id": int(unit), "evidence": evidence,
                    "large_excursions": large, "events": int(len(take)),
                    "median_abs_difference_um": float(np.median(np.abs(take.difference_um))) if len(take) >= 5 else np.nan,
                    "median_signed_difference_um": float(np.median(take.difference_um)) if len(take) >= 5 else np.nan,
                })
    return pd.DataFrame(rows)


def aggregate_summary(metrics: pd.DataFrame) -> list[dict]:
    rows = []
    for keys, part in metrics.groupby(["interval", "fit", "evidence", "large_excursions"]):
        valid = part.dropna(subset=["median_abs_difference_um"])
        rows.append({
            "interval": keys[0], "fit": keys[1], "evidence": keys[2],
            "large_excursions": bool(keys[3]), "events": int(part.events.sum()),
            "candidates_with_at_least_5_events": int(len(valid)),
            "candidate_median_abs_difference_um": float(valid.median_abs_difference_um.median()) if len(valid) else None,
        })
    return rows


def plot_summary(metrics: pd.DataFrame, predictions: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    panels = [("post_seed_940_1230", False, axes[0, 0], "Strict events, 940–1230 s"),
              ("extension_1030_1230", True, axes[0, 1], "Strict large excursions, 1030–1230 s")]
    for interval, large, ax, title in panels:
        q = metrics[(metrics.interval == interval) & (metrics.evidence == "strict_accepted")
                    & (metrics.large_excursions == large)]
        wide = q.pivot(index="unit_id", columns="fit", values="median_abs_difference_um").dropna()
        ax.scatter(wide["10k"], wide["30k"], color="#536a7a", s=42, edgecolor="white", linewidth=0.5)
        for unit, row in wide.iterrows():
            ax.annotate(str(int(unit)), (row["10k"], row["30k"]), xytext=(3, 3),
                        textcoords="offset points", fontsize=7)
        limit = max(10.0, float(wide.max().max()) * 1.08) if len(wide) else 10.0
        ax.plot([0, limit], [0, limit], color="#9ca3af", ls="--", lw=0.9)
        lower = max(1.0, float(wide.min().min()) * 0.8) if len(wide) else 1.0
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set(xlim=(lower, limit), ylim=(lower, limit), xlabel="10k median absolute difference (µm)",
               ylabel="30k median absolute difference (µm)", title=f"{title} · n={len(wide)} candidates")

    ax = axes[1, 0]
    q = metrics[(metrics.interval == "post_seed_940_1230") & (metrics.evidence == "strict_accepted")
                & (~metrics.large_excursions)]
    wide = q.pivot(index="unit_id", columns="fit", values="median_abs_difference_um").dropna()
    change = (wide["30k"] - wide["10k"]).sort_values()
    colors = np.where(change <= 0, "#1f5a94", "#c56a1a")
    ax.barh(np.arange(len(change)), change, color=colors)
    ax.axvline(0, color="#374151", lw=0.8)
    ax.set(yticks=np.arange(len(change)), yticklabels=change.index.astype(int),
           xlabel="30k − 10k median absolute difference (µm)", ylabel="candidate unit",
           title="Per-candidate change on strict post-seed events")

    ax = axes[1, 1]
    q = predictions[(predictions.evidence == "strict_accepted") & predictions.time_s.between(940, 1230, inclusive="left")].copy()
    q["time_bin_s"] = np.floor(q.time_s / 10) * 10
    per_unit = (q.groupby(["fit", "unit_id", "time_bin_s"]).difference_um
                .apply(lambda x: np.median(np.abs(x))).rename("mad_um").reset_index())
    equal_unit = per_unit.groupby(["fit", "time_bin_s"]).mad_um.median().reset_index()
    for fit in ("10k", "30k"):
        line = equal_unit[equal_unit.fit == fit]
        ax.plot(line.time_bin_s + 5, line.mad_um, color=COLORS[fit], lw=1.5,
                ls="-" if fit == "10k" else "--", label=fit)
    ax.axvline(1030, color="#9ca3af", ls=":", lw=0.9)
    ax.set(xlabel="recording time (s)", ylabel="median absolute difference (µm)",
           title="10-s bins; candidate medians receive equal weight")
    ax.legend(frameon=False)
    for ax in axes.flat:
        ax.grid(color="#e5e7eb", lw=0.5, zorder=0)
        ax.tick_params(labelsize=8)
    fig.suptitle("Full-session MEDiCINe 10k versus 30k · frozen lighthouse comparison\n"
                 "Offsets fixed by 930–940 s seed spikes; identity matching did not use either field", fontsize=13)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_selected(cohort: pd.DataFrame, predictions: pd.DataFrame, fields: dict, curves: dict, output: Path) -> None:
    requested = [161, 555, 657, 673]
    available = [unit for unit in requested if unit in set(cohort.unit_id)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, unit in zip(axes.flat, available):
        q = predictions[(predictions.unit_id == unit) & (predictions.fit == "10k")]
        for evidence, marker, color, fill in (
            ("strict_accepted", "o", "#111111", True),
            ("lower_score", "o", "#1f5a94", False),
            ("identity_ambiguous", "x", "#c56a1a", True),
        ):
            part = q[q.evidence == evidence]
            kwargs = {"c": color} if fill else {"facecolors": "none", "edgecolors": color}
            ax.scatter(part.time_s, part.relative_um, s=12, marker=marker, linewidths=0.55,
                       label=evidence, **kwargs)
        for fit in ("10k", "30k"):
            field = fields[fit]
            keep = (field["time_s"] >= 930) & (field["time_s"] <= 1230)
            ax.plot(field["time_s"][keep], curves[(fit, unit)][keep], color=COLORS[fit],
                    ls="-" if fit == "10k" else "--", lw=1.4, label=f"full-session {fit}")
        ax.axvline(1030, color="#9ca3af", ls=":", lw=0.8)
        ax.set(xlim=(930, 1230), xlabel="recording time (s)", ylabel="seed-relative depth (µm)",
               title=f"candidate {unit}")
        ax.grid(color="#e5e7eb", lw=0.5)
    axes.flat[0].legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    fig.suptitle("Selected frozen lighthouse tracks and full-session MEDiCINe fields\n"
                 "Black: strict · open blue: lower score · orange ×: ambiguous · blue/orange lines: 10k/30k",
                 fontsize=13)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_all(cohort: pd.DataFrame, predictions: pd.DataFrame, fields: dict, curves: dict, output: Path) -> None:
    peaks = np.load(PEAKS)
    locations = np.load(LOCATIONS)
    peak_time = 930 + peaks["sample_index"] / FS
    with PdfPages(output) as pdf:
        for cell in cohort.itertuples(index=False):
            unit = int(cell.unit_id)
            q = predictions[(predictions.unit_id == unit) & (predictions.fit == "10k")]
            low = min(float(q.centroid_um.min()), float(cell.seed_centroid_um)) - 80
            high = max(float(q.centroid_um.max()), float(cell.seed_centroid_um)) + 80
            fig, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)
            bg = (locations["y"] >= low) & (locations["y"] <= high)
            axes[0].scatter(peak_time[bg], locations["y"][bg], s=0.3, color="#9ca3af", alpha=0.22, rasterized=True)
            for evidence, marker, color, fill in (
                ("identity_ambiguous", "x", "#c56a1a", True),
                ("lower_score", "o", "#1f5a94", False),
                ("strict_accepted", "o", "#111111", True),
            ):
                part = q[q.evidence == evidence]
                kwargs = {"c": color} if fill else {"facecolors": "none", "edgecolors": color}
                axes[0].scatter(part.time_s, part.centroid_um, s=12, marker=marker, linewidths=0.55,
                                label=evidence, **kwargs)
                axes[1].scatter(part.time_s, part.relative_um, s=12, marker=marker, linewidths=0.55, **kwargs)
            for fit in ("10k", "30k"):
                field = fields[fit]
                keep = (field["time_s"] >= 930) & (field["time_s"] <= 1230)
                axes[1].plot(field["time_s"][keep], curves[(fit, unit)][keep], color=COLORS[fit],
                             ls="-" if fit == "10k" else "--", lw=1.4, label=f"full-session {fit}")
            for ax in axes:
                ax.axvline(940, color="#9ca3af", ls=":", lw=0.8)
                ax.axvline(1030, color="#9ca3af", ls="--", lw=0.8)
                ax.set_xlim(930, 1230)
                ax.grid(color="#e5e7eb", lw=0.45)
                ax.legend(frameon=False, fontsize=8, ncol=3)
            axes[0].set(ylim=(low, high), ylabel="absolute waveform depth (µm)",
                        title="Waveform depths over independent 5σ/relaxed peak background")
            axes[1].set(xlabel="recording time (s)", ylabel="seed-relative displacement (µm)",
                        title="Frozen waveform observations; vertical lines mark seed end and extension start")
            fig.suptitle(f"Candidate {unit} · full-session 10k/30k MEDiCINe comparison\n"
                         "No depth prior in identity matching; no later offset fitting", fontsize=13)
            pdf.savefig(fig)
            plt.close(fig)


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    cohort, predictions, fields, curves = build_predictions()
    metrics = summarize(predictions)
    predictions.to_csv(output / "event_predictions.csv", index=False)
    metrics.to_csv(output / "candidate_metrics.csv", index=False)
    aggregates = aggregate_summary(metrics)
    pd.DataFrame(aggregates).to_csv(output / "aggregate_metrics.csv", index=False)
    plot_summary(metrics, predictions, output / "summary.png")
    plot_selected(cohort, predictions, fields, curves, output / "selected_tracks.png")
    plot_all(cohort, predictions, fields, curves, output / "all17_tracks.pdf")

    def primary(fit: str, interval: str, large: bool) -> dict:
        q = metrics[(metrics.fit == fit) & (metrics.interval == interval)
                    & (metrics.evidence == "strict_accepted") & (metrics.large_excursions == large)]
        q = q.dropna(subset=["median_abs_difference_um"])
        return {"candidates": int(len(q)), "events": int(q.events.sum()),
                "candidate_median_abs_difference_um": float(q.median_abs_difference_um.median())}

    paired = metrics[(metrics.interval == "post_seed_940_1230") &
                     (metrics.evidence == "strict_accepted") & (~metrics.large_excursions)]
    wide = paired.pivot(index="unit_id", columns="fit", values="median_abs_difference_um").dropna()
    summary = {
        "schema_version": "luke-full-session-medicine-30k-lighthouse-v1",
        "status": "complete",
        "candidates": int(len(cohort)),
        "identity_selection": "frozen waveform-only whole-probe candidates selected at 930-940 s",
        "offset_rule": "each field referenced to its own median at unchanged 930-940 s seed spike times",
        "strict_post_seed": {fit: primary(fit, "post_seed_940_1230", False) for fit in ("10k", "30k")},
        "strict_extension": {fit: primary(fit, "extension_1030_1230", False) for fit in ("10k", "30k")},
        "strict_large_extension": {fit: primary(fit, "extension_1030_1230", True) for fit in ("10k", "30k")},
        "paired_strict_post_seed": {
            "candidates": int(len(wide)),
            "30k_better": int((wide["30k"] < wide["10k"]).sum()),
            "30k_worse": int((wide["30k"] > wide["10k"]).sum()),
            "median_30k_minus_10k_um": float((wide["30k"] - wide["10k"]).median()),
        },
        "source_sha256": {str(path): sha256(path) for path in (FIELD_10K, FIELD_30K, TRACKS, COHORT, TEMPLATES)},
        "interpretation_limits": [
            "Lighthouse identities are provisional and strict/lower-score/ambiguous evidence is not pooled.",
            "The motion estimator saw the recording peak population; post-seed means identity-selection holdout, not estimator-training holdout.",
            "Candidate medians give each identity one vote; no aggregate alone establishes ground-truth motion.",
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
