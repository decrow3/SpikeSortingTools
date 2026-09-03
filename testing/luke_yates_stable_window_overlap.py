"""Increment 1 of the Luke--Yates stable-period comparison.

Prespec: ``docs/luke_yates_stable_period_comparison_plan.md`` (§3).

Motion-signature-only characterisation of candidate 120 s windows in
Luke_20250804 and the known-good Yates session, plus the prespecified
**feasibility gate** that decides whether a motion-matched cross-animal
comparison is viable as designed.

This script reads only motion-estimator arrays (``motion.npy`` /
``time_bins.npy`` / ``depth_bins.npy``). It never reads sorter labels and never
touches voltage. It selects nothing about biology; it only measures how much of
Yates's own motion-quiet regime Luke can reach.

The gate is a *feasibility / calibration* screen, not a matching rule: it asks
whether enough Luke windows fall inside Yates's own everyday-quiet region for a
matched comparison to be built at all. The actual matched-window selection is a
separate rule frozen in increment 2/3.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

SCHEMA = "luke-yates-stable-window-overlap-v2"

LUKE_ROOT = Path("/mnt/NPX/Luke/20250804")
YATES_ROOT = Path(
    "/media/huklab/Data/Yates_session_copy/processed/Allen_2022-02-16"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs/luke_yates_stable_window_overlap"

WINDOW_S = 120.0
STRIDE_S = 120.0
MIN_BINS_PER_WINDOW = 20
MIN_TIME_COVERAGE = 0.9  # in-window bins must span >= this fraction of WINDOW_S
MAX_GAP_FACTOR = 3.0  # no internal gap may exceed this * the source's median bin spacing

COMMON_ESTIMATORS = ("medicine", "ks-motion", "decentralized-motion")
PRIMARY_ESTIMATOR = "medicine"
LUKE_ONLY_ESTIMATORS = ("dredge-motion",)

LUKE_PROBES = ("imec0", "imec1")
YATES_SHANKS = ("shank1", "shank2")
LUKE_PRIMARY_PROBE = "imec0"

# Overlap box: upper edge on each axis = this quantile of the pooled Yates
# window distribution, per estimator (prespec §3). The non-rigid axis is
# depth-span-normalised (µm of across-depth range per mm of probe depth) so it
# is comparable between a ~4 mm Luke span and a ~1.2 mm Yates span.
YATES_BOX_QUANTILE = 0.75
OVERLAP_AXES = ("rigid_excursion_um", "nonrigid_grad_um_per_mm", "rigid_speed_um_s")
FINITE_FRACTION_FLOOR = 0.90

GATE_MIN_LUKE = 6
GATE_MIN_YATES = 6  # counted as unique session time intervals quiet on *all* shanks

HIGH_MOTION_LUKE_QUANTILE = 0.90

TIME_INTERVAL_ROUND_S = 1.0  # granularity for matching Yates shank windows to the same time


class SourceValidationError(RuntimeError):
    """A required motion source is missing or internally inconsistent."""


@dataclass(frozen=True)
class MotionSource:
    """One estimator's motion arrays for one probe/shank of one dataset."""

    dataset: str  # "Luke" | "Yates"
    probe: str  # "imec0"/"imec1" | "shank1"/"shank2"
    estimator: str
    motion_dir: Path

    def load(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            motion = np.load(self.motion_dir / "motion.npy")
            times = np.load(self.motion_dir / "time_bins.npy")
            depths = np.load(self.motion_dir / "depth_bins.npy")
        except FileNotFoundError as exc:
            raise SourceValidationError(f"{self}: {exc}") from exc

        motion = np.atleast_2d(np.asarray(motion, dtype=float))
        times = np.asarray(times, dtype=float).ravel()
        depths = np.asarray(depths, dtype=float).ravel()
        if motion.shape[0] != times.shape[0] and motion.shape[1] == times.shape[0]:
            motion = motion.T

        if motion.shape[0] != times.shape[0]:
            raise SourceValidationError(
                f"{self}: motion rows {motion.shape[0]} != time bins {times.shape[0]}"
            )
        if motion.shape[1] != depths.shape[0]:
            raise SourceValidationError(
                f"{self}: motion cols {motion.shape[1]} != depth bins {depths.shape[0]}"
            )
        if times.size < 2 or not np.all(np.diff(times) > 0):
            raise SourceValidationError(f"{self}: time bins are not strictly increasing")
        return motion, times, depths


def default_sources(
    luke_root: Path = LUKE_ROOT, yates_root: Path = YATES_ROOT
) -> list[MotionSource]:
    """The full required source matrix plus the Luke-only DREDGE arms.

    Missing *required* sources are still returned so that
    :func:`validate_source_matrix` can report exactly what is absent rather than
    the gate silently passing on a partial matrix.
    """

    sources: list[MotionSource] = []
    for probe in LUKE_PROBES:
        base = luke_root / f"dredge_pipeline_results_Luke0804_V2V1_g0_{probe}" / "motion"
        for estimator in (*COMMON_ESTIMATORS, *LUKE_ONLY_ESTIMATORS):
            if estimator in LUKE_ONLY_ESTIMATORS and not (base / estimator / "motion.npy").exists():
                continue
            sources.append(MotionSource("Luke", probe, estimator, base / estimator))
    for shank in YATES_SHANKS:
        base = yates_root / f"{shank}-motion"
        for estimator in COMMON_ESTIMATORS:
            sources.append(MotionSource("Yates", shank, estimator, base / estimator))
    return sources


def validate_source_matrix(sources: list[MotionSource]) -> None:
    """Fail loudly unless every required probe/shank x common-estimator arm loads."""

    have = {(s.dataset, s.probe, s.estimator) for s in sources}
    required = {
        ("Luke", probe, est) for probe in LUKE_PROBES for est in COMMON_ESTIMATORS
    } | {
        ("Yates", shank, est) for shank in YATES_SHANKS for est in COMMON_ESTIMATORS
    }
    missing = sorted(required - have)
    if missing:
        raise SourceValidationError(f"missing required motion sources: {missing}")

    for source in sources:
        if (source.dataset, source.probe, source.estimator) in required:
            source.load()  # raises SourceValidationError on any inconsistency


def window_signature(
    motion: np.ndarray, times: np.ndarray, depths: np.ndarray
) -> dict[str, float]:
    """Per-window motion signature (prespec §3). ``motion`` is (n_time, n_depth)."""

    motion = np.atleast_2d(motion)
    depths = np.asarray(depths, dtype=float).ravel()
    depth_span_um = float(depths.max() - depths.min()) if depths.size > 1 else 0.0

    row_finite = np.isfinite(motion).all(axis=1)
    finite_fraction = float(row_finite.mean()) if motion.shape[0] else 0.0

    usable = motion[row_finite]
    usable_t = times[row_finite]

    nan = float("nan")
    base = {
        "rigid_excursion_um": nan,
        "nonrigid_grad_um_per_mm": nan,
        "p95_nonrigid_grad_um_per_mm": nan,
        "rigid_speed_um_s": nan,
        "finite_fraction": finite_fraction,
        "n_time_bins": int(usable.shape[0]),
        "n_depth_bins": int(motion.shape[1]),
        "depth_span_um": depth_span_um,
        "dt_median_s": nan,
        "max_time_gap_s": nan,
    }
    if usable.shape[0] < 2:
        return base

    diffs = np.diff(usable_t)
    base["dt_median_s"] = float(np.median(diffs))
    base["max_time_gap_s"] = float(diffs.max())

    rigid = usable.mean(axis=1)
    base["rigid_excursion_um"] = float(np.percentile(rigid, 95) - np.percentile(rigid, 5))

    if usable.shape[1] > 1 and depth_span_um > 0:
        depth_range = usable.max(axis=1) - usable.min(axis=1)  # µm across depth, per time bin
        grad = depth_range / (depth_span_um / 1000.0)  # µm per mm of probe depth
        base["nonrigid_grad_um_per_mm"] = float(np.median(grad))
        base["p95_nonrigid_grad_um_per_mm"] = float(np.percentile(grad, 95))
    else:
        base["nonrigid_grad_um_per_mm"] = 0.0
        base["p95_nonrigid_grad_um_per_mm"] = 0.0

    good = diffs > 0
    speed = np.abs(np.diff(rigid)[good] / diffs[good])
    base["rigid_speed_um_s"] = float(np.percentile(speed, 95)) if speed.size else nan
    return base


def enumerate_windows(
    source: MotionSource,
    loader: Callable[[MotionSource], tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> list[dict]:
    """Every non-overlapping 120 s window in ``source`` with its signature.

    A window is emitted when it holds >= ``MIN_BINS_PER_WINDOW`` bins whose
    timestamps span >= ``MIN_TIME_COVERAGE`` * ``WINDOW_S`` (prespec §3). Both
    a native-clock start and a recording-relative start are recorded.
    """

    motion, times, depths = (loader(source) if loader is not None else source.load())
    if times.size == 0:
        return []
    t0, t1 = float(times.min()), float(times.max())
    rows: list[dict] = []
    start = t0
    while start < t1:
        keep = (times >= start) & (times < start + WINDOW_S)
        in_t = times[keep]
        covered = float(in_t.max() - in_t.min()) if in_t.size else 0.0
        if keep.sum() >= MIN_BINS_PER_WINDOW and covered >= MIN_TIME_COVERAGE * WINDOW_S:
            sig = window_signature(motion[keep], in_t, depths)
            rows.append(
                {
                    "dataset": source.dataset,
                    "probe": source.probe,
                    "estimator": source.estimator,
                    "window_start_native_s": float(start),
                    "window_start_recording_s": float(start - t0),
                    "time_origin_native_s": t0,
                    "window_duration_s": WINDOW_S,
                    "time_interval_id": round((start - t0) / TIME_INTERVAL_ROUND_S),
                    **sig,
                }
            )
        start += STRIDE_S
    return rows


def build(sources: list[MotionSource]) -> pd.DataFrame:
    rows: list[dict] = []
    for source in sources:
        rows.extend(enumerate_windows(source))
    return pd.DataFrame(rows)


def overlap_box(yates_windows: pd.DataFrame) -> dict[str, float]:
    """Upper edge per axis = ``YATES_BOX_QUANTILE`` of the pooled Yates windows."""

    box: dict[str, float] = {}
    for axis in OVERLAP_AXES:
        values = yates_windows[axis].to_numpy(float)
        if not np.all(np.isfinite(values)):
            raise SourceValidationError(
                f"non-finite Yates values on axis {axis!r}; cannot build the box"
            )
        box[axis] = float(np.quantile(values, YATES_BOX_QUANTILE))
    return box


def mark_overlap(windows: pd.DataFrame, box: dict[str, float]) -> pd.Series:
    inside = windows["finite_fraction"] >= FINITE_FRACTION_FLOOR
    inside &= windows["max_time_gap_s"] <= MAX_GAP_FACTOR * windows["dt_median_s"]
    for axis, edge in box.items():
        inside &= windows[axis] <= edge
    return inside.fillna(False)


def _yates_unique_quiet_intervals(yates: pd.DataFrame) -> int:
    """Count session time intervals that are in-overlap on *every* Yates shank present."""

    if yates.empty:
        return 0
    shanks_present = yates["probe"].nunique()
    per_interval = yates.groupby("time_interval_id")["in_overlap"].agg(["sum", "count"])
    quiet = per_interval[
        (per_interval["count"] == shanks_present) & (per_interval["sum"] == shanks_present)
    ]
    return int(len(quiet))


def evaluate_gate(windows: pd.DataFrame) -> dict:
    """Apply the prespecified overlap box + feasibility gate, per estimator (prespec §3)."""

    result: dict = {
        "schema": SCHEMA,
        "window_s": WINDOW_S,
        "yates_box_quantile": YATES_BOX_QUANTILE,
        "overlap_axes": list(OVERLAP_AXES),
        "finite_fraction_floor": FINITE_FRACTION_FLOOR,
        "max_gap_factor": MAX_GAP_FACTOR,
        "gate_min_luke": GATE_MIN_LUKE,
        "gate_min_yates": GATE_MIN_YATES,
        "gate_semantics": (
            "feasibility screen: >= gate_min_luke Luke imec0 windows AND "
            ">= gate_min_yates unique Yates session time intervals quiet on all "
            "shanks, inside the Yates-Q75 box under the primary estimator"
        ),
        "primary_estimator": PRIMARY_ESTIMATOR,
        "luke_primary_probe": LUKE_PRIMARY_PROBE,
        "by_estimator": {},
    }

    for estimator in COMMON_ESTIMATORS:
        est = windows[windows["estimator"] == estimator].copy()
        yates = est[est["dataset"] == "Yates"]
        if yates.empty:
            result["by_estimator"][estimator] = {"error": "no Yates windows"}
            continue
        box = overlap_box(yates)
        est["in_overlap"] = mark_overlap(est, box)
        yates = est[est["dataset"] == "Yates"]

        luke_primary = est[(est["dataset"] == "Luke") & (est["probe"] == LUKE_PRIMARY_PROBE)]
        luke_imec1 = est[(est["dataset"] == "Luke") & (est["probe"] == "imec1")]
        n_luke = int(luke_primary["in_overlap"].sum())
        n_yates_intervals = _yates_unique_quiet_intervals(yates)

        result["by_estimator"][estimator] = {
            "box": box,
            "n_luke_imec0_overlap": n_luke,
            "n_luke_imec1_overlap": int(luke_imec1["in_overlap"].sum()),
            "n_yates_shank_windows_overlap": int(yates["in_overlap"].sum()),
            "n_yates_unique_quiet_intervals": n_yates_intervals,
            "n_luke_imec0_total": int(len(luke_primary)),
            "n_yates_shank_windows_total": int(len(yates)),
            "pass": bool(n_luke >= GATE_MIN_LUKE and n_yates_intervals >= GATE_MIN_YATES),
        }

    primary = result["by_estimator"].get(PRIMARY_ESTIMATOR, {})
    result["overall_pass"] = bool(primary.get("pass", False))
    return result


def high_motion_luke_controls(windows: pd.DataFrame) -> list[dict]:
    est = windows[
        (windows["estimator"] == PRIMARY_ESTIMATOR)
        & (windows["dataset"] == "Luke")
        & (windows["probe"] == LUKE_PRIMARY_PROBE)
        & np.isfinite(windows["rigid_excursion_um"])
    ]
    if est.empty:
        return []
    edge = float(np.quantile(est["rigid_excursion_um"].to_numpy(float), HIGH_MOTION_LUKE_QUANTILE))
    hi = est[est["rigid_excursion_um"] >= edge].sort_values("rigid_excursion_um", ascending=False)
    cols = [
        "window_start_native_s",
        "window_start_recording_s",
        "rigid_excursion_um",
        "nonrigid_grad_um_per_mm",
        "rigid_speed_um_s",
    ]
    return hi[cols].to_dict("records")


def render(windows: pd.DataFrame, gate: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    est = windows[windows["estimator"] == PRIMARY_ESTIMATOR]
    box = gate["by_estimator"].get(PRIMARY_ESTIMATOR, {}).get("box")
    fig, ax = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    palette = {
        ("Yates", "shank1"): ("#32688e", "o"),
        ("Yates", "shank2"): ("#4f97c9", "o"),
        ("Luke", "imec0"): ("#d17a22", "s"),
        ("Luke", "imec1"): ("#a24b12", "^"),
    }
    for (dataset, probe), (color, marker) in palette.items():
        sel = est[(est["dataset"] == dataset) & (est["probe"] == probe)]
        ax.scatter(
            sel["rigid_excursion_um"],
            sel["nonrigid_grad_um_per_mm"],
            s=26,
            color=color,
            marker=marker,
            alpha=0.6,
            label=f"{dataset} {probe} (n={len(sel)})",
        )
    if box:
        ax.axvline(box["rigid_excursion_um"], color="0.4", ls="--", lw=1)
        ax.axhline(box["nonrigid_grad_um_per_mm"], color="0.4", ls="--", lw=1)
        ax.add_patch(
            plt.Rectangle(
                (0, 0),
                box["rigid_excursion_um"],
                box["nonrigid_grad_um_per_mm"],
                facecolor="#dfe8d8",
                edgecolor="none",
                alpha=0.5,
                zorder=0,
            )
        )
    ax.set_xlabel("rigid excursion P95−P5 (µm)")
    ax.set_ylabel("median non-rigid gradient (µm per mm of depth)")
    verdict = "PASS" if gate["overall_pass"] else "FAIL"
    ax.set_title(
        f"Luke–Yates 120 s motion windows — {PRIMARY_ESTIMATOR}\n"
        f"feasibility gate: {verdict}  (rigid-speed axis not shown)"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--luke-root", type=Path, default=LUKE_ROOT)
    parser.add_argument("--yates-root", type=Path, default=YATES_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = default_sources(args.luke_root, args.yates_root)
    validate_source_matrix(sources)

    windows = build(sources)
    windows.to_csv(args.output_dir / "window_signatures.csv", index=False)

    gate = evaluate_gate(windows)
    gate["high_motion_luke_controls"] = high_motion_luke_controls(windows)
    (args.output_dir / "overlap_gate.json").write_text(json.dumps(gate, indent=2))

    render(windows, gate, args.output_dir / "overlap_scatter.png")

    print(json.dumps({k: v for k, v in gate.items() if k != "high_motion_luke_controls"}, indent=2))
    print(f"\n{len(windows)} windows over {len(sources)} sources -> {args.output_dir}")
    print("OVERALL GATE:", "PASS" if gate["overall_pass"] else "FAIL")


if __name__ == "__main__":
    main()
