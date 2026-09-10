"""Cached lighthouse review of the completed full-session MEDiCINe field.

This report never reads voltage, detects spikes, changes lighthouse identities, or
fits motion.  It samples the already published field at the frozen candidate
depths and event times and retains the five-minute fit as a contextual control.
"""
from __future__ import annotations

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
OUT = ROOT / "testing/outputs/luke_full_session_medicine_lighthouse_review_v1"
EVENT_DIR = ROOT / "testing/outputs/luke_lighthouse_extension_300s_v1"
COHORT_DIR = ROOT / "testing/outputs/luke_waveform_only_expansion_v1"
TEMPLATE_PATH = ROOT / "testing/outputs/luke_population_depth_v2/templates.npz"
SNIPPET_PATH = ROOT / "testing/outputs/luke_screened_medicine_300s_v1/fit_5sigma_relaxed_10000/field.npz"
SNIPPET_INPUT = ROOT / "testing/outputs/luke_screened_medicine_300s_v1/input_5sigma_relaxed"
FULL_PATH = Path("/media/huklab/Data/luke_full_session_medicine_20260909_v1/candidate_fields.npz")
FULL_SUMMARY = Path("/media/huklab/Data/luke_full_session_medicine_20260909_v1/summary.json")
FS = 29999.835983263598


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def depth_curve(values: np.ndarray, depths: np.ndarray, depth: float) -> np.ndarray:
    if values.ndim == 1:
        return values.astype(float)
    j = int(np.searchsorted(depths, depth))
    j = min(max(j, 1), len(depths) - 1)
    weight = (depth - depths[j - 1]) / (depths[j] - depths[j - 1])
    return values[:, j - 1] * (1 - weight) + values[:, j] * weight


def seed_reference(time_s: np.ndarray, curve: np.ndarray, seed_frames: np.ndarray) -> np.ndarray:
    seed_s = seed_frames / FS
    seed_s = seed_s[(seed_s >= time_s[0]) & (seed_s <= time_s[-1])]
    if not len(seed_s):
        raise ValueError("No seed events in field support")
    return curve - np.median(np.interp(seed_s, time_s, curve))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [
        EVENT_DIR / "overlay_events.csv", COHORT_DIR / "candidate_audit.csv",
        TEMPLATE_PATH, SNIPPET_PATH, SNIPPET_INPUT / "peaks.npy",
        SNIPPET_INPUT / "locations.npy", FULL_PATH, FULL_SUMMARY,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    full_summary = json.loads(FULL_SUMMARY.read_text())
    if full_summary.get("status") != "complete":
        raise RuntimeError("Full-session MEDiCINe output is not complete")

    events = pd.read_csv(EVENT_DIR / "overlay_events.csv")
    candidates = pd.read_csv(COHORT_DIR / "candidate_audit.csv")
    candidates = candidates[candidates.selected].sort_values("seed_centroid_um")
    if len(candidates) != 17 or not events.unit_id.isin(candidates.unit_id).all():
        raise ValueError("Frozen 17-candidate cohort mismatch")
    templates = np.load(TEMPLATE_PATH)
    full = np.load(FULL_PATH)
    snippet = np.load(SNIPPET_PATH)
    fields = {
        "full_nonrigid": (full["time_s"], full["depth_um"], full["nonrigid_displacement_um"]),
        "full_rigid": (full["time_s"], full["depth_um"], full["rigid_displacement_um"]),
        "five_minute_nonrigid_control": (snippet["time_s"], snippet["depth_um"], snippet["displacement_um"]),
    }
    common_start = max(x[0][0] for x in fields.values())
    common_stop = min(x[0][-1] for x in fields.values())
    colors = {
        "full_nonrigid": "#0072B2", "full_rigid": "#D55E00",
        "five_minute_nonrigid_control": "#777777",
    }
    styles = {"full_nonrigid": "-", "full_rigid": "-", "five_minute_nonrigid_control": "--"}

    predictions = []
    metrics = []
    curves = {}
    intervals = {"post_seed_940_1230": (940, 1230), "new_holdout_1030_1230": (1030, 1230)}
    for cell in candidates.itertuples():
        seed_frames = templates[f"unit_{cell.unit_id}_seed_frames"]
        for field_name, (time_s, depths, values) in fields.items():
            curve = seed_reference(time_s, depth_curve(values, depths, cell.seed_centroid_um), seed_frames)
            curves[(cell.unit_id, field_name)] = (time_s, curve)
            q = events[events.unit_id == cell.unit_id].copy()
            q["field"] = field_name
            q["predicted_um"] = np.interp(q.time_s, time_s, curve, left=np.nan, right=np.nan)
            q["difference_um"] = q.relative_um - q.predicted_um
            predictions.append(q[["field", "unit_id", "time_s", "evidence", "relative_um", "predicted_um", "difference_um"]])
            for interval_name, (start, stop) in intervals.items():
                for evidence in ["strict_accepted", "lower_score", "identity_ambiguous"]:
                    a = q[(q.time_s >= max(start, common_start)) &
                          (q.time_s <= min(stop, common_stop)) &
                          (q.evidence == evidence) & q.predicted_um.notna()]
                    for scope in ["all_observations", "large_excursions"]:
                        b = a if scope == "all_observations" else a[abs(a.relative_um) >= 120]
                        supported = len(b) >= 5
                        metrics.append(dict(
                            field=field_name, interval=interval_name, unit_id=cell.unit_id,
                            evidence=evidence, scope=scope, events=len(b), supported=supported,
                            median_abs_difference_um=float(np.median(abs(b.difference_um))) if supported else np.nan,
                            median_signed_difference_um=float(np.median(b.difference_um)) if supported else np.nan,
                        ))

    predictions = pd.concat(predictions, ignore_index=True)
    metrics = pd.DataFrame(metrics)
    aggregate = (metrics[metrics.supported]
                 .groupby(["field", "interval", "evidence", "scope"])
                 .agg(candidates=("unit_id", "nunique"), events=("events", "sum"),
                      median_candidate_abs_difference_um=("median_abs_difference_um", "median"),
                      mean_candidate_abs_difference_um=("median_abs_difference_um", "mean"),
                      median_candidate_signed_difference_um=("median_signed_difference_um", "median"))
                 .reset_index())
    predictions.to_csv(OUT / "event_predictions.csv", index=False)
    metrics.to_csv(OUT / "per_candidate_metrics.csv", index=False)
    aggregate.to_csv(OUT / "aggregate_metrics.csv", index=False)

    peak = np.load(SNIPPET_INPUT / "peaks.npy", mmap_mode="r")
    loc = np.load(SNIPPET_INPUT / "locations.npy", mmap_mode="r")
    peak_time = 930 + peak["sample_index"] / FS
    with PdfPages(OUT / "01_all17_absolute_depth_and_motion.pdf") as pdf:
        for cell in candidates.itertuples():
            q = events[events.unit_id == cell.unit_id]
            low = min(q.centroid_um.min(), cell.seed_centroid_um) - 70
            high = max(q.centroid_um.max(), cell.seed_centroid_um) + 70
            bg = (loc["y"] >= low) & (loc["y"] <= high)
            fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True, constrained_layout=True)
            axes[0].scatter(peak_time[bg], loc["y"][bg], s=.25, c="#BBBBBB", alpha=.2, rasterized=True)
            for evidence, marker, color in [
                ("identity_ambiguous", "x", "#D55E00"),
                ("lower_score", "o", "#0072B2"),
                ("strict_accepted", "o", "#111111"),
            ]:
                a = q[q.evidence == evidence]
                kwargs = dict(facecolors="none", edgecolors=color) if evidence == "lower_score" else dict(c=color)
                axes[0].scatter(a.time_s, a.centroid_um, s=10, marker=marker, linewidths=.5,
                                label=evidence, rasterized=True, **kwargs)
                axes[1].scatter(a.time_s, a.relative_um, s=10, marker=marker, linewidths=.5,
                                alpha=.65, rasterized=True, **kwargs)
            for name in fields:
                time_s, curve = curves[(cell.unit_id, name)]
                axes[1].plot(time_s, curve, color=colors[name], ls=styles[name], lw=1.1, label=name)
            for ax in axes:
                ax.axvspan(930, 940, color="gray", alpha=.1)
                ax.axvline(1030, color="#555555", ls=":", lw=.8)
                ax.legend(fontsize=7, ncol=3)
                ax.grid(alpha=.12)
            axes[0].set(ylabel="Absolute waveform depth (µm)", ylim=(low, high))
            axes[1].set(xlabel="Recording time (s)", ylabel="Seed-relative displacement (µm)", xlim=(930, 1230))
            fig.suptitle(f"Frozen unit {cell.unit_id} · seed depth {cell.seed_centroid_um:.0f} µm")
            fig.supxlabel("Waveforms were matched without depth or motion priors. Dotted boundary starts the new 200 s temporal holdout.", fontsize=9)
            pdf.savefig(fig)
            if cell.unit_id in [161, 555, 632, 673]:
                fig.savefig(OUT / f"unit_{cell.unit_id}.png", dpi=140)
            plt.close(fig)

    strict = aggregate[(aggregate.interval == "new_holdout_1030_1230") &
                       (aggregate.evidence == "strict_accepted") &
                       (aggregate.scope == "all_observations")].set_index("field")
    per = metrics[(metrics.interval == "new_holdout_1030_1230") &
                  (metrics.evidence == "strict_accepted") &
                  (metrics.scope == "all_observations")]
    pivot = per.pivot(index="unit_id", columns="field", values="median_abs_difference_um")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    order = ["full_nonrigid", "full_rigid", "five_minute_nonrigid_control"]
    axes[0].bar(range(3), [strict.loc[x, "median_candidate_abs_difference_um"] for x in order],
                color=[colors[x] for x in order])
    axes[0].set_xticks(range(3), ["Full nonrigid", "Full rigid", "5-min control"], rotation=15)
    axes[0].set_ylabel("Median of candidate median |difference| (µm)")
    axes[0].set_title("Strict waveform evidence · 1030–1230 s")
    for unit_id, row in pivot.iterrows():
        axes[1].plot([0, 1], [row.full_nonrigid, row.full_rigid], "o-", alpha=.55,
                     label=str(unit_id) if unit_id in [161, 555, 632] else None)
    axes[1].plot([0, 1], [pivot.full_nonrigid.median(), pivot.full_rigid.median()], "ko-", lw=3, label="candidate median")
    axes[1].set_xticks([0, 1], ["Full nonrigid", "Full rigid"])
    axes[1].set_ylabel("Per-candidate median |difference| (µm)")
    axes[1].set_yscale("symlog", linthresh=20)
    axes[1].set_title("Each line is one supported candidate")
    axes[1].legend(fontsize=8)
    fig.suptitle("Cached lighthouse check of completed full-session MEDiCINe")
    fig.savefig(OUT / "00_summary.png", dpi=170)
    fig.savefig(OUT / "00_summary.pdf")
    plt.close(fig)

    result = {
        "status": "complete",
        "scientific_status": "descriptive_requires_review",
        "full_session_status": full_summary["status"],
        "candidate_count": int(len(candidates)),
        "new_holdout_strict_supported_candidates": int(strict.loc["full_nonrigid", "candidates"]),
        "common_field_support_s": [float(common_start), float(common_stop)],
        "new_holdout_strict_supported_events": int(strict.loc["full_nonrigid", "events"]),
        "new_holdout_median_candidate_abs_difference_um": {
            name: float(strict.loc[name, "median_candidate_abs_difference_um"]) for name in order
        },
        "full_nonrigid_candidates_over_100um_median_difference": [
            int(x) for x in pivot.index[pivot.full_nonrigid > 100]
        ],
        "full_rigid_better_than_full_nonrigid_candidates": int((pivot.full_rigid < pivot.full_nonrigid).sum()),
        "full_rigid_worse_than_full_nonrigid_candidates": int((pivot.full_rigid > pivot.full_nonrigid).sum()),
        "interpretation": (
            "Direct cached comparison only. Candidate identities are provisional and related families are not "
            "independent cells. Missing or ambiguous tracks are not stationary tissue, and waveform-centroid "
            "differences are not calibrated physical motion error."
        ),
        "source_sha256": {str(path): sha256(path) for path in required},
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# Full-session MEDiCINe lighthouse review\n\n"
        "The completed full-session 5σ/relaxed MEDiCINe field and its equal-depth rigid projection are "
        "sampled at the frozen 17 waveform-only candidate depths and actual event times. The existing "
        "five-minute fit is a contextual control. No voltage was read, no event was redetected, no identity "
        "was selected with motion or depth, and no field, sign, gain, lag, or later offset was fitted. Curves "
        "use the original seed-event reference only.\n\n"
        "The 1030–1230 s interval is the newer temporal holdout. Metrics require at least five events per "
        "candidate/evidence/scope and aggregate with equal candidate weight. Strict, lower-score, ambiguous, "
        "large-excursion, and unsupported evidence remain separate. Absolute waveform depths are plotted over "
        "the cached screened peak background after identity matching.\n\n"
        "These are descriptive discrepancies from provisional waveform centroids, not ground-truth motion "
        "errors. Related candidate families are not independent cells; fixed seed templates can lose identity "
        "under large translation. Unmatched events, gaps, and ambiguity do not imply stationarity. The report "
        "can reject obvious field/cell inconsistencies but cannot by itself certify correction or a full sort.\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
