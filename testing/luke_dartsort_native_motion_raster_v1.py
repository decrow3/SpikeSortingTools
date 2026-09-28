"""Plot the DARTsort native nonrigid motion on the frozen early peak raster.

This is plotting-only. It reads the completed DARTsort motion estimate and the
same frozen original-voltage peak raster used by
``luke_early_motion_raster_with_lfp_v1.py``. No motion is fit or modified.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "testing/outputs"
OUT = SRC / "luke_dartsort_native_motion_raster_v1"
HIST_PATH = SRC / "luke_peak_population_review_v1/02_long_histograms_histograms.npz"
MOTION_PATH = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/DARTsort_motion_experiments/"
    "luke0804-imec0-930-1230-v1/native_motion/motion.pkl"
)
WINDOW = (930.0, 1030.0)
SORT_WINDOW_START_S = 930.0
REFERENCE = (970.0, 975.0)
ROWS = [
    (0, 3840, [320, 960, 1600, 2240, 2920, 3380]),
    (1950, 2350, [2100, 2250]),
    (2849, 2989, [2919]),
    (3311, 3451, [3381]),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class MotionUnpickler(pickle.Unpickler):
    """Read a SciPy 1.18 pickle with the plotting environment's SciPy 1.15."""

    def find_class(self, module: str, name: str):
        if (module, name) == (
            "scipy._external.array_api_compat.numpy._aliases",
            "asarray",
        ):
            return np.asarray
        return super().find_class(module, name)


def load_motion():
    # This is a trusted local artifact produced by the completed DARTsort job.
    with MOTION_PATH.open("rb") as stream:
        payload = MotionUnpickler(stream).load()
    if not isinstance(payload, dict) or "dredge_motion_est" not in payload:
        raise ValueError("motion.pkl lacks the expected dredge_motion_est")
    motion = payload["dredge_motion_est"]
    if motion is None or np.asarray(motion.displacement).ndim != 2:
        raise ValueError("DARTsort motion estimate is missing or malformed")
    return motion


def anchored_curves(motion, anchors: list[float]):
    relative_time_s = np.asarray(motion.time_bin_centers_s, dtype=float)
    recording_time_s = SORT_WINDOW_START_S + relative_time_s
    keep = (recording_time_s >= WINDOW[0]) & (recording_time_s < WINDOW[1])
    reference = (recording_time_s >= REFERENCE[0]) & (recording_time_s < REFERENCE[1])
    if reference.sum() < 2:
        raise ValueError("Insufficient DARTsort samples in the reference window")

    # Rebuild the interpolator from the stored numeric field. The interpolator
    # object's private pickle state changed between the producer's SciPy 1.18
    # and this plotting environment's SciPy 1.15.
    interpolate = RegularGridInterpolator(
        (
            np.asarray(motion.spatial_bin_centers_um, dtype=float),
            relative_time_s,
        ),
        np.asarray(motion.displacement, dtype=float),
        bounds_error=False,
        fill_value=None,
    )

    curves = []
    for anchor in anchors:
        clipped_anchor = np.clip(anchor, motion.d_low, motion.d_high)
        displacement_um = interpolate(
            np.column_stack(
                [np.full(relative_time_s.shape, clipped_anchor), relative_time_s]
            )
        )
        displacement_um = np.asarray(displacement_um, dtype=float)
        zeroed_um = displacement_um - np.nanmedian(displacement_um[reference])
        curves.append((float(anchor), recording_time_s[keep], zeroed_um[keep]))
    return curves


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (HIST_PATH, MOTION_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    hist = np.load(HIST_PATH)
    counts = np.asarray(hist["arm0_counts"])
    time_edges = np.asarray(hist["time_edges"], dtype=float)
    depth_edges = np.asarray(hist["depth_edges"], dtype=float)
    if counts.shape != (len(depth_edges) - 1, len(time_edges) - 1):
        raise ValueError("Frozen raster shape does not match its bin edges")
    if not np.allclose(time_edges[[0, -1]], WINDOW):
        raise ValueError("Frozen raster does not cover the requested window")

    motion = load_motion()
    if not np.isfinite(np.asarray(motion.displacement)).all():
        raise ValueError("DARTsort displacement field contains non-finite values")

    color_max = float(np.quantile(np.log1p(counts).ravel(), 0.995))
    style = dict(color="#56E0E8", lw=2.8, ls="-", zorder=5)
    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True, layout="constrained")
    saved_curves: list[dict[str, float]] = []

    for row_index, (ax, (low, high, anchors)) in enumerate(zip(axes, ROWS)):
        ax.imshow(
            np.log1p(counts),
            origin="lower",
            aspect="auto",
            extent=[time_edges[0], time_edges[-1], depth_edges[0], depth_edges[-1]],
            cmap="magma",
            vmin=0,
            vmax=color_max,
            interpolation="nearest",
            rasterized=True,
        )
        for curve_index, (anchor, time_s, displacement_um) in enumerate(
            anchored_curves(motion, anchors)
        ):
            ax.plot(
                time_s,
                anchor + displacement_um,
                label="DARTsort native motion" if curve_index == 0 else None,
                **style,
            )
            saved_curves.extend(
                {
                    "panel_low_um": float(low),
                    "panel_high_um": float(high),
                    "anchor_depth_um": anchor,
                    "time_s": float(time),
                    "displacement_um": float(displacement),
                }
                for time, displacement in zip(time_s, displacement_um)
            )
        ax.set(
            xlim=WINDOW,
            ylim=(low, high),
            title=f"original peaks · {low}–{high} µm",
            ylabel="Depth (µm)",
        )
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        if row_index == 0:
            legend = ax.legend(
                fontsize=8,
                loc="upper left",
                facecolor="#231F20",
                edgecolor="#6B7280",
                labelcolor="white",
                framealpha=0.92,
            )
            for line in legend.get_lines():
                line.set_linewidth(2.8)

    axes[-1].set_xlabel("Recording time (s)")
    fig.suptitle(
        "Peak depth/time raster with DARTsort native motion\n"
        "930–1030 s · nonrigid trajectories independently zeroed at 970–975 s",
        fontsize=15,
    )
    png_path = OUT / "01_dartsort_native_motion_raster.png"
    pdf_path = OUT / "01_dartsort_native_motion_raster.pdf"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path, dpi=180)
    plt.close(fig)

    curves_path = OUT / "dartsort_native_motion_curves.csv.gz"
    pd.DataFrame(saved_curves).to_csv(curves_path, index=False)
    displacement = np.asarray(motion.displacement, dtype=float)
    manifest = {
        "status": "complete",
        "plot_only": True,
        "motion_source": str(MOTION_PATH),
        "raster_source": str(HIST_PATH),
        "raster_arm": "original unwarped voltage",
        "window_s": list(WINDOW),
        "sort_window_start_s": SORT_WINDOW_START_S,
        "reference_window_s": list(REFERENCE),
        "reference_rule": (
            "At each displayed anchor depth, subtract the median DARTsort field "
            "value over 970 <= recording time < 975 s"
        ),
        "motion_time_bins": int(displacement.shape[1]),
        "motion_depth_bins": int(displacement.shape[0]),
        "motion_depth_centers_um": np.asarray(motion.spatial_bin_centers_um).tolist(),
        "raw_displacement_range_um": [float(displacement.min()), float(displacement.max())],
        "palette": {"DARTsort native motion": "#56E0E8"},
        "sources_sha256": {
            str(HIST_PATH): sha256(HIST_PATH),
            str(MOTION_PATH): sha256(MOTION_PATH),
        },
        "outputs": [png_path.name, pdf_path.name, curves_path.name],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
