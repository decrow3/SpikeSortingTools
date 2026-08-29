"""Aligned scientific visualizations for the Luke 120 s motion benchmark.

Chart contract
--------------
Question: which temporal/spatial structures are shared across estimators, and
which corrections visibly distort the localized-peak raster?
Surface: reproducible static PNG research figures, inspected at final size.
Families: time-depth heatmaps, faceted line traces, annotated correlation
matrices, and time-depth density rasters.  Diverging fields use one symmetric
scale; estimator traces use explicit colors plus line styles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator

from testing.luke_motion_scale_characterization import (
    PROBES,
    WINDOW,
    decompose_spatial_field,
    recording_t_start,
)
from testing.luke_motion_scale_sweep import (
    DEFAULT_OUTPUT,
    _window_peaks,
    candidate_by_name,
    run_dir,
)


OUTPUT = Path("testing/outputs/luke_motion_visualizations")
DEPTHS = np.arange(310.0, 3510.1, 200.0)
RELATIVE_TIMES = np.arange(1.0, WINDOW.duration_s, 2.0)
SEED = 20250804

LABELS = {
    "dredge_nr_current_exact": "DREDGE AP 150/100\ncap 37.5",
    "dredge_nr_current_max80": "DREDGE AP 150/100\ncap 80",
    "dredge_nr_200_300": "DREDGE AP 300/200",
    "decentralized_nr_200_300": "Decentralized 300/200",
    "ks_pipeline_default": "KS repository default",
    "medicine_pipeline_default": "MEDiCINe default",
    "medicine_nr_8bin_t20": "MEDiCINe 8-bin / 20 s",
    "dredge_lfp_rigid_100hz": "DREDGE LFP rigid",
    "dredge_lfp_rigid_max20_100hz": "DREDGE LFP rigid, cap 20",
    "dredge_lfp_nr_200_300_100hz": "DREDGE LFP 300/200",
}

AP_FIELDS = [
    "dredge_nr_current_exact",
    "dredge_nr_current_max80",
    "dredge_nr_200_300",
    "decentralized_nr_200_300",
    "ks_pipeline_default",
    "medicine_pipeline_default",
    "medicine_nr_8bin_t20",
]
AGREEMENT_FIELDS = [
    "dredge_nr_200_300",
    "decentralized_nr_200_300",
    "ks_pipeline_default",
    "medicine_pipeline_default",
    "medicine_nr_8bin_t20",
    "dredge_lfp_rigid_100hz",
]

COLORS = {
    "dredge_nr_200_300": "#3569a8",
    "decentralized_nr_200_300": "#d08b27",
    "ks_pipeline_default": "#71864a",
    "medicine_pipeline_default": "#b85c7a",
    "medicine_nr_8bin_t20": "#6f5aa8",
    "dredge_lfp_rigid_100hz": "#333333",
    "dredge_lfp_rigid_max20_100hz": "#d1495b",
}


def completed_run(probe: str, candidate_name: str) -> Path | None:
    candidate = candidate_by_name(candidate_name)
    target = run_dir(DEFAULT_OUTPUT, probe, candidate, "full", SEED)
    return target if (target / "motion.npy").exists() else None


def load_native(probe: str, candidate_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = completed_run(probe, candidate_name)
    if target is None:
        raise FileNotFoundError(f"Missing completed run: {probe}/{candidate_name}")
    return (
        np.asarray(np.load(target / "motion.npy"), dtype=float),
        np.asarray(np.load(target / "time_bins.npy"), dtype=float),
        np.asarray(np.load(target / "depth_bins.npy"), dtype=float),
    )


def sample_field(probe: str, candidate_name: str) -> np.ndarray:
    motion, times, depths = load_native(probe, candidate_name)
    target_times = recording_t_start(probe) + WINDOW.start_s + RELATIVE_TIMES
    target_times = np.clip(target_times, times.min(), times.max())
    if motion.shape[1] == 1:
        trace = np.interp(target_times, times, motion[:, 0])
        return np.repeat(trace[:, None], len(DEPTHS), axis=1)
    # Edge-hold is intentional for MEDiCINe: its 2-bin repository default is
    # applied across the whole probe by interpolation.  Other candidates cover
    # the common grid natively.
    depth_sampled = np.vstack([np.interp(DEPTHS, depths, row) for row in motion])
    return np.column_stack(
        [np.interp(target_times, times, depth_sampled[:, index]) for index in range(len(DEPTHS))]
    )


def dynamic(field: np.ndarray) -> np.ndarray:
    return field - np.median(field, axis=0, keepdims=True)


def save_ap_fields(fields: dict[tuple[str, str], np.ndarray], output: Path) -> None:
    values = np.concatenate([dynamic(fields[(p, c)]).ravel() for p in PROBES for c in AP_FIELDS])
    limit = float(np.percentile(np.abs(values), 99.0))
    limit = max(10.0, np.ceil(limit / 10.0) * 10.0)
    fig, axes = plt.subplots(2, len(AP_FIELDS), figsize=(18, 7.2), sharex=True, sharey=True)
    for row, probe in enumerate(PROBES):
        for col, candidate in enumerate(AP_FIELDS):
            ax = axes[row, col]
            image = ax.imshow(
                dynamic(fields[(probe, candidate)]).T,
                origin="lower",
                aspect="auto",
                extent=[RELATIVE_TIMES[0], RELATIVE_TIMES[-1], DEPTHS[0], DEPTHS[-1]],
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
            )
            if row == 0:
                ax.set_title(LABELS[candidate], fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{probe}\nDepth (µm)")
            if row == 1:
                ax.set_xlabel("Window time (s)")
    fig.suptitle("AP-derived motion fields on a common displacement scale", fontsize=14, y=0.99)
    fig.text(0.5, 0.955, f"Luke pathological 120 s window; temporal median removed per depth; shared range ±{limit:.0f} µm", ha="center", fontsize=10)
    fig.subplots_adjust(left=0.055, right=0.90, bottom=0.09, top=0.88, wspace=0.08, hspace=0.10)
    cax = fig.add_axes([0.925, 0.15, 0.012, 0.67])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Dynamic displacement (µm)")
    fig.savefig(output / "ap_motion_fields.png", dpi=180)
    plt.close(fig)


def save_rigid_traces(fields: dict[tuple[str, str], np.ndarray], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5), sharex=True)
    ap_methods = AGREEMENT_FIELDS[:-1]
    for row, probe in enumerate(PROBES):
        ax = axes[row, 0]
        for candidate in ap_methods:
            trace = decompose_spatial_field(fields[(probe, candidate)], DEPTHS)["rigid"]
            trace = trace - np.median(trace)
            ax.plot(RELATIVE_TIMES, trace, color=COLORS[candidate], lw=1.6, label=LABELS[candidate].replace("\n", " "))
        ax.axhline(0, color="#777777", lw=0.8)
        ax.set_ylabel(f"{probe}\nDisplacement (µm)")
        ax.set_title("AP-peak estimators", fontsize=11)
        ax.grid(axis="y", color="#dddddd", lw=0.6)

        lfp = decompose_spatial_field(fields[(probe, "dredge_lfp_rigid_100hz")], DEPTHS)["rigid"]
        lfp = lfp - np.median(lfp)
        lfp_capped = decompose_spatial_field(fields[(probe, "dredge_lfp_rigid_max20_100hz")], DEPTHS)["rigid"]
        lfp_capped = lfp_capped - np.median(lfp_capped)
        ax = axes[row, 1]
        ax.plot(RELATIVE_TIMES, lfp, color=COLORS["dredge_lfp_rigid_100hz"], lw=1.5, label="cap 80 µm")
        ax.plot(RELATIVE_TIMES, lfp_capped, color=COLORS["dredge_lfp_rigid_max20_100hz"], lw=1.3, ls="--", label="cap 20 µm")
        for boundary in np.arange(10, WINDOW.duration_s, 10):
            ax.axvline(boundary, color="#bbbbbb", lw=0.7, ls="--")
        ax.axhline(0, color="#777777", lw=0.8)
        ax.set_title("DREDGE-LFP rigid (independent scale)", fontsize=11)
        ax.grid(axis="y", color="#dddddd", lw=0.6)
    for ax in axes[-1]:
        ax.set_xlabel("Window time (s)")
    axes[0, 0].legend(loc="upper left", ncol=2, fontsize=8, frameon=False)
    axes[0, 1].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Rigid-component motion traces", fontsize=14)
    fig.text(0.5, 0.945, "AP-derived local consensus versus 100 Hz LFP registration; dashed lines mark LFP chunk boundaries", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output / "rigid_trace_comparison.png", dpi=180)
    plt.close(fig)


def finite_corr(first: np.ndarray, second: np.ndarray) -> float:
    x, y = first.ravel(), second.ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def save_agreement(fields: dict[tuple[str, str], np.ndarray], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    short = ["DREDGE AP", "Decentralized", "KS default", "MED default", "MED 8-bin", "DREDGE LFP"]
    for row, probe in enumerate(PROBES):
        decomposed = {c: decompose_spatial_field(fields[(probe, c)], DEPTHS) for c in AGREEMENT_FIELDS}
        for col, component in enumerate(("rigid", "residual")):
            matrix = np.empty((len(AGREEMENT_FIELDS), len(AGREEMENT_FIELDS)))
            for i, left in enumerate(AGREEMENT_FIELDS):
                for j, right in enumerate(AGREEMENT_FIELDS):
                    matrix[i, j] = finite_corr(decomposed[left][component], decomposed[right][component])
            ax = axes[row, col]
            image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
            for i in range(len(short)):
                for j in range(len(short)):
                    value = matrix[i, j]
                    label = "—" if not np.isfinite(value) else f"{value:.2f}"
                    color = "white" if np.isfinite(value) and abs(value) > 0.55 else "#222222"
                    ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)
            ax.set_xticks(range(len(short)), short, rotation=42, ha="right", fontsize=8)
            ax.set_yticks(range(len(short)), short, fontsize=8)
            ax.set_title(f"{probe}: {'rigid trace' if component == 'rigid' else 'residual non-rigid field'}")
    fig.suptitle("Pairwise estimator agreement", fontsize=14)
    fig.text(0.5, 0.955, "Pearson correlation on the shared 2 s × 200 µm grid; MEDiCINe uses edge-hold outside native centers", ha="center", fontsize=10)
    fig.subplots_adjust(left=0.15, right=0.88, bottom=0.12, top=0.90, wspace=0.35, hspace=0.40)
    cax = fig.add_axes([0.915, 0.18, 0.014, 0.64])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Correlation (r)")
    fig.savefig(output / "estimator_agreement_matrices.png", dpi=180)
    plt.close(fig)


def interpolate_at_peaks(probe: str, candidate: str, times_rel: np.ndarray, depths_um: np.ndarray) -> np.ndarray:
    motion, times, depths = load_native(probe, candidate)
    query_times = recording_t_start(probe) + WINDOW.start_s + times_rel
    query_times = np.clip(query_times, times.min(), times.max())
    if motion.shape[1] == 1:
        return np.interp(query_times, times, motion[:, 0])
    query_depths = np.clip(depths_um, depths.min(), depths.max())
    interpolator = RegularGridInterpolator((times, depths), motion, bounds_error=False, fill_value=None)
    return interpolator(np.column_stack((query_times, query_depths)))


def save_corrected_rasters(output: Path) -> None:
    candidates = [
        None,
        "dredge_nr_current_exact",
        "dredge_nr_200_300",
        "decentralized_nr_200_300",
        "ks_pipeline_default",
        "medicine_nr_8bin_t20",
    ]
    labels = ["Uncorrected", "DREDGE current", "DREDGE 300/200", "Decentralized", "KS default", "MEDiCINe 8-bin"]
    fig, axes = plt.subplots(2, len(candidates), figsize=(18, 7.2), sharex=True, sharey=True)
    histograms = {}
    for row, probe in enumerate(PROBES):
        reference_run = completed_run(probe, "dredge_nr_current_exact")
        manifest = json.loads((reference_run / "manifest.json").read_text())
        fs = float(manifest["sampling_frequency_hz"])
        peaks, locations, _, _ = _window_peaks(probe, fs)
        take = np.arange(0, len(peaks), 8)
        times_rel = peaks["sample_index"][take] / fs
        raw_depths = locations["y"][take]
        for col, candidate in enumerate(candidates):
            corrected = raw_depths if candidate is None else raw_depths - interpolate_at_peaks(probe, candidate, times_rel, raw_depths)
            hist, _, _ = np.histogram2d(times_rel, corrected, bins=(120, 192), range=((0, 120), (0, 3840)))
            histograms[(row, col)] = hist.T
    vmax = max(np.percentile(h[h > 0], 99.5) for h in histograms.values())
    for row, probe in enumerate(PROBES):
        for col, label in enumerate(labels):
            ax = axes[row, col]
            image = ax.imshow(
                histograms[(row, col)], origin="lower", aspect="auto", extent=[0, 120, 0, 3840],
                cmap="magma", norm=LogNorm(vmin=1, vmax=max(2, vmax)), interpolation="nearest"
            )
            if row == 0:
                ax.set_title(label, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{probe}\nCorrected depth (µm)")
            if row == 1:
                ax.set_xlabel("Window time (s)")
    fig.suptitle("Localized-peak density before and after candidate corrections", fontsize=14, y=0.99)
    fig.text(0.5, 0.945, "Same deterministic 1-in-8 peak sample and shared log-count scale across all panels", ha="center", fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.09, top=0.87, wspace=0.08, hspace=0.10)
    cax = fig.add_axes([0.925, 0.15, 0.012, 0.66])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Peaks per 1 s × 20 µm bin (log scale)")
    fig.savefig(output / "corrected_peak_density.png", dpi=180)
    plt.close(fig)


def save_lfp_diagnostic(fields: dict[tuple[str, str], np.ndarray], output: Path) -> None:
    candidate = "dredge_lfp_nr_200_300_100hz"
    if completed_run("imec1", candidate) is None:
        return
    field = dynamic(fields[("imec1", candidate)])
    full_limit = float(np.ceil(np.percentile(np.abs(field), 99) / 50) * 50)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True, sharey=True)
    settings = [
        (field, -full_limit, full_limit, "Full robust range"),
        (field, -60, 60, "Clipped to AP comparison scale"),
        (dynamic(fields[("imec1", "dredge_nr_200_300")]), -60, 60, "AP DREDGE 300/200 reference"),
    ]
    for ax, (values, low, high, title) in zip(axes, settings):
        image = ax.imshow(values.T, origin="lower", aspect="auto", extent=[1, 119, 310, 3510], cmap="RdBu_r", vmin=low, vmax=high)
        for boundary in np.arange(10, 120, 10):
            ax.axvline(boundary, color="#555555", lw=0.5, ls="--", alpha=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Window time (s)")
        fig.colorbar(image, ax=ax, shrink=0.80, label="Dynamic displacement (µm)")
    axes[0].set_ylabel("Depth (µm)")
    fig.suptitle("imec1 DREDGE-LFP non-rigid diagnostic", fontsize=14, y=0.99)
    fig.text(0.5, 0.935, "100 Hz paired-depth LFP; dashed lines mark independent 10 s online chunks", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(output / "dredge_lfp_diagnostic_imec1.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    needed = set(AP_FIELDS + AGREEMENT_FIELDS + ["dredge_lfp_rigid_max20_100hz", "dredge_lfp_nr_200_300_100hz"])
    fields = {}
    for probe in PROBES:
        for candidate in sorted(needed):
            if completed_run(probe, candidate) is not None:
                fields[(probe, candidate)] = sample_field(probe, candidate)

    save_ap_fields(fields, args.output_dir)
    save_rigid_traces(fields, args.output_dir)
    save_agreement(fields, args.output_dir)
    save_corrected_rasters(args.output_dir)
    save_lfp_diagnostic(fields, args.output_dir)
    chart_map = {
        "ap_motion_fields.png": "Common-scale dynamic displacement heatmaps for AP-derived estimators.",
        "rigid_trace_comparison.png": "AP rigid-component traces and separately scaled LFP rigid traces.",
        "estimator_agreement_matrices.png": "Annotated rigid and residual-field correlation matrices.",
        "corrected_peak_density.png": "Localized-peak density before and after candidate correction.",
        "dredge_lfp_diagnostic_imec1.png": "Full/clipped non-rigid LFP field with chunk boundaries and AP reference.",
    }
    (args.output_dir / "chart_map.json").write_text(json.dumps(chart_map, indent=2) + "\n")


if __name__ == "__main__":
    main()
