"""Add the validated LFP candidate to the frozen early peak-motion raster.

This is a plotting-only comparison.  It reuses the exact raster, DREDGE field,
and MEDiCINe field from the prior v8 figure and does not fit or modify motion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "testing/outputs"
OUT = SRC / "luke_early_motion_raster_with_lfp_v1"
HIST_PATH = SRC / "luke_peak_population_review_v1/02_long_histograms_histograms.npz"
DREDGE_DIR = SRC / "luke_early_medicine_peak_comparison_v8"
MEDICINE_DIR = SRC / "luke_early_medicine_peak_comparison_v6"
LFP_TRACE_PATH = SRC / "luke_lfp_lighthouse_validation_v1/lfp_package/validation_traces.csv"
WINDOW = (930.0, 1030.0)
REFERENCE = (970.0, 975.0)
LFP_ARM = "band_0p5_8"
ROWS = [
    (0, 3840, [320, 960, 1600, 2240, 2920, 3380]),
    (1950, 2350, [2100, 2250]),
    (2849, 2989, [2919]),
    (3311, 3451, [3381]),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def anchored_nonrigid_curves(path: Path, anchors: list[float]) -> list[tuple[float, np.ndarray, np.ndarray]]:
    z = np.load(path)
    time_s = z["time_s"].astype(float)
    depth_um = z["depth_um"].astype(float)
    displacement_um = z["displacement_um"].astype(float)
    interp = RegularGridInterpolator(
        (time_s, depth_um), displacement_um, bounds_error=False, fill_value=np.nan
    )
    curves = []
    for anchor in anchors:
        trace = interp(np.column_stack([time_s, np.full(len(time_s), anchor)]))
        ref = np.nanmedian(trace[(time_s >= REFERENCE[0]) & (time_s < REFERENCE[1])])
        curves.append((float(anchor), time_s, trace - ref))
    return curves


def anchored_medicine_curves(arm: str, anchors: list[float]) -> list[tuple[float, np.ndarray, np.ndarray]]:
    time_s = np.load(MEDICINE_DIR / arm / "time_bins.npy").astype(float)
    depth_um = np.load(MEDICINE_DIR / arm / "depth_bins.npy").astype(float)
    displacement_um = np.load(MEDICINE_DIR / arm / "motion.npy").astype(float)
    interp = RegularGridInterpolator(
        (time_s, depth_um), displacement_um, bounds_error=False, fill_value=np.nan
    )
    curves = []
    for anchor in anchors:
        trace = interp(np.column_stack([time_s, np.full(len(time_s), anchor)]))
        ref = np.nanmedian(trace[(time_s >= REFERENCE[0]) & (time_s < REFERENCE[1])])
        curves.append((float(anchor), time_s, trace - ref))
    return curves


def anchored_lfp_curve() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pd.read_csv(LFP_TRACE_PATH)
    d = table.loc[
        table.window.eq("930_1030") & table.arm.eq(LFP_ARM),
        ["time_s", "pred_um", "supported"],
    ].sort_values("time_s")
    if len(d) < 300 or d.time_s.min() != WINDOW[0] or d.time_s.max() < WINDOW[1] - 0.5:
        raise ValueError("Unexpected LFP validation trace support")
    time_s = d.time_s.to_numpy(float)
    supported = d.supported.to_numpy(bool) & np.isfinite(d.pred_um.to_numpy(float))
    reference = supported & (time_s >= REFERENCE[0]) & (time_s < REFERENCE[1])
    if reference.sum() < 10:
        raise ValueError("Insufficient supported LFP reference samples")
    trace = d.pred_um.to_numpy(float) - np.median(d.loc[reference, "pred_um"])
    trace[~supported] = np.nan
    return time_s, trace, supported


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [HIST_PATH, LFP_TRACE_PATH]
    for arm in ["original", "compensated"]:
        required.extend(
            [
                DREDGE_DIR / f"{arm}_dredge_motion.npz",
                MEDICINE_DIR / arm / "time_bins.npy",
                MEDICINE_DIR / arm / "depth_bins.npy",
                MEDICINE_DIR / arm / "motion.npy",
            ]
        )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    hist = np.load(HIST_PATH)
    counts = [hist["arm0_counts"], hist["arm1_counts"]]
    if counts[0].shape != counts[1].shape:
        raise ValueError("Raster arm shapes differ")
    color_max = float(np.quantile(np.concatenate([np.log1p(x).ravel() for x in counts]), 0.995))
    lfp_time, lfp_trace, lfp_supported = anchored_lfp_curve()

    # Chart contract: dense depth/time matrix with three overlaid trajectory
    # families.  Categorical colors plus solid/dashed/dotted styles preserve
    # identity in grayscale. The LFP trace gets a dark keyline so its bright
    # chartreuse remains legible over both dim and saturated raster regions.
    # All curves use the same 970--975 s anchor rule.
    styles = {
        "DREDGE": dict(color="#56E0E8", lw=2.8, ls="-", zorder=5),
        "MEDiCINe": dict(color="#FFFFFF", lw=1.8, ls="--", zorder=6),
        "LFP 0.5–8 Hz d1": dict(color="#A3E635", lw=2.1, ls=":", zorder=8),
    }
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True, layout="constrained")
    saved_curves: list[dict[str, float | str | bool]] = []

    for col, arm in enumerate(["original", "compensated"]):
        dredge_path = DREDGE_DIR / f"{arm}_dredge_motion.npz"
        for row, (low, high, anchors) in enumerate(ROWS):
            ax = axes[row, col]
            ax.imshow(
                np.log1p(counts[col]),
                origin="lower",
                aspect="auto",
                extent=[930, 1030, 0, 3840],
                cmap="magma",
                vmin=0,
                vmax=color_max,
                interpolation="nearest",
                rasterized=True,
            )
            method_curves = {
                "DREDGE": anchored_nonrigid_curves(dredge_path, anchors),
                "MEDiCINe": anchored_medicine_curves(arm, anchors),
                "LFP 0.5–8 Hz d1": [(float(anchor), lfp_time, lfp_trace) for anchor in anchors],
            }
            for method, curves in method_curves.items():
                for curve_index, (anchor, time_s, displacement) in enumerate(curves):
                    if method == "LFP 0.5–8 Hz d1":
                        ax.plot(
                            time_s,
                            anchor + displacement,
                            color="#111827",
                            lw=4.0,
                            ls=":",
                            alpha=0.9,
                            zorder=7,
                        )
                    ax.plot(
                        time_s,
                        anchor + displacement,
                        label=method if curve_index == 0 else None,
                        **styles[method],
                    )
                    for t, value in zip(time_s, displacement):
                        saved_curves.append(
                            {
                                "raster_arm": arm,
                                "panel_low_um": low,
                                "panel_high_um": high,
                                "method": method,
                                "anchor_depth_um": anchor,
                                "time_s": float(t),
                                "displacement_um": float(value) if np.isfinite(value) else np.nan,
                                "supported": bool(np.isfinite(value)),
                            }
                        )
            ax.set(
                xlim=WINDOW,
                ylim=(low, high),
                title=f"{arm} peaks · {low}–{high} µm",
                ylabel="Depth (µm)",
            )
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            if row == 0:
                legend = ax.legend(
                    fontsize=8,
                    ncol=3,
                    loc="upper left",
                    facecolor="#231F20",
                    edgecolor="#6B7280",
                    labelcolor="white",
                    framealpha=0.92,
                )
                for line in legend.get_lines():
                    line.set_linewidth(2.6)
        axes[-1, col].set_xlabel("Recording time (s)")

    fig.suptitle(
        "Peak depth/time raster with DREDGE, MEDiCINe, and LFP motion\n"
        "930–1030 s · curves independently zeroed at 970–975 s · LFP is rigid and repeated at each anchor",
        fontsize=15,
    )
    fig.savefig(OUT / "01_peak_motion_overlay_with_lfp.png", dpi=180)
    fig.savefig(OUT / "01_peak_motion_overlay_with_lfp.pdf", dpi=180)
    plt.close(fig)

    pd.DataFrame(saved_curves).to_csv(OUT / "overlay_curves.csv.gz", index=False)
    manifest = {
        "status": "complete",
        "plot_only": True,
        "window_s": list(WINDOW),
        "reference_window_s": list(REFERENCE),
        "lfp_arm": LFP_ARM,
        "lfp_supported_fraction": float(lfp_supported.mean()),
        "lfp_reference_rule": "Subtract median supported 250 ms trace over 970 <= t < 975 s",
        "nonrigid_reference_rule": "At each displayed anchor depth, subtract median field value over 970 <= t < 975 s",
        "lfp_spatial_rule": "Rigid trace repeated at each displayed anchor depth",
        "raster": "Exact frozen log1p peak-count histograms from the prior v8 plot",
        "palette": {
            "DREDGE": "#56E0E8",
            "MEDiCINe": "#FFFFFF",
            "LFP 0.5–8 Hz d1": "#A3E635",
            "LFP keyline": "#111827",
        },
        "sources_sha256": {str(path): sha256(path) for path in required},
        "outputs": [
            "01_peak_motion_overlay_with_lfp.png",
            "01_peak_motion_overlay_with_lfp.pdf",
            "overlay_curves.csv.gz",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
