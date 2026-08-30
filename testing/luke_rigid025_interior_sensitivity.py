"""Compare no-motion and rigid-0.25 sorts away from depth-strip boundaries.

This is a sensitivity analysis, not an advancement test: the no-halo motion
sort can affect template learning globally.  Restricting to an 80-um interior
tests whether its favorable metrics are confined to the extrapolated edges.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_claimmask_window_sweep import (
    cross_unit_near_coincident_fraction,
    load_reference_settings,
    local_match_mask,
)
from testing.luke_two_axis_pilot import (
    DEFAULT_REVIEW,
    circular_shift_coincidence_null,
    temporal_bin_metrics,
)

ROOT = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/"
    "Luke0804_two_axis_pilot_imec1/sorts/core_depth_strip"
)
SORTERS = {
    "no_motion": ROOT / "single_ks_preprocessing_claim_off/sorter_output",
    "rigid025_p2_no_halo": ROOT / "rigid025_p2_single_ks_preprocessing_claim_off/sorter_output",
}
OUTPUT = Path("testing/outputs/luke_rigid025_depth_strip/interior_sensitivity.json")
N_FRAMES = 314_204_094


def recovery_with_shared_jitter(
    events: pd.DataFrame,
    times: np.ndarray,
    depths: np.ndarray,
    fs: float,
    duration_frames: int,
    seed: int,
) -> tuple[float, float]:
    tolerance = int(round(0.5e-3 * fs))
    samples = events.sample_index.to_numpy(np.int64)
    event_depths = events.peak_depth_um.to_numpy(float)
    observed = float(local_match_mask(samples, event_depths, times, depths, tolerance, 100.0).mean())
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(100):
        offsets = rng.uniform(0.020 * fs, 0.500 * fs, len(events))
        offsets *= rng.choice((-1.0, 1.0), len(events))
        jittered = np.mod(samples + np.rint(offsets).astype(np.int64), duration_frames)
        null.append(float(local_match_mask(jittered, event_depths, times, depths, tolerance, 100.0).mean()))
    return observed, float(np.mean(null))


def score_interior(sorter: Path, margin_um: float, events: pd.DataFrame) -> dict:
    _, fs = load_reference_settings()
    positions = np.load(sorter / "spike_positions.npy", mmap_mode="r")[:, 1].astype(float)
    channel_depths = np.load(sorter / "channel_positions.npy")[:, 1].astype(float)
    lo, hi = float(channel_depths.min() + margin_um), float(channel_depths.max() - margin_um)
    all_clusters = np.load(sorter / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    unit_median_depth = pd.Series(positions).groupby(all_clusters).median()
    interior_units = unit_median_depth[
        unit_median_depth.between(lo, hi)
    ].index.to_numpy(int)
    keep = (positions >= lo) & (positions <= hi)
    times = np.load(sorter / "spike_times.npy", mmap_mode="r").reshape(-1).astype(np.int64)[keep]
    clusters = all_clusters[keep]
    depths = positions[keep]
    duration_frames = N_FRAMES
    duration_s = duration_frames / fs
    tolerance = int(round(0.5e-3 * fs))
    unit_ids = np.unique(clusters).astype(int)
    labels = pd.read_csv(sorter / "cluster_KSLabel.tsv", sep="\t").set_index("cluster_id")
    contamination = pd.read_csv(sorter / "cluster_ContamPct.tsv", sep="\t").set_index("cluster_id")
    label_col, contam_col = labels.columns[0], contamination.columns[0]
    refractory = []
    for unit in unit_ids:
        unit_times = np.sort(times[clusters == unit])
        if len(unit_times) > 1:
            refractory.append(float(np.mean(np.diff(unit_times) < int(round(1.5e-3 * fs)))))
    bins = temporal_bin_metrics(times, fs, duration_s, 300.0)
    rate_mean = float(bins.spikes_per_s.mean())
    scoped_events = events[
        events.peak_depth_um.between(lo, hi)
        & events.review_label.eq("neural")
        & events.status.eq("unmatched")
    ]
    recovery, jitter = recovery_with_shared_jitter(
        scoped_events, times, depths, fs, duration_frames, seed=20250804
    )
    coincidence = cross_unit_near_coincident_fraction(times, clusters, depths, tolerance)
    coincidence_null = circular_shift_coincidence_null(
        times, clusters, depths, duration_frames, tolerance, seed=20250804
    )
    return {
        "interior_depth_range_um": [lo, hi],
        "margin_um_each_edge": margin_um,
        "n_spikes": int(len(times)),
        "n_units_with_interior_spikes": int(len(unit_ids)),
        "n_ks_good_units_with_median_depth_interior": int(
            labels.loc[labels.index.intersection(interior_units), label_col]
            .astype(str).str.lower().eq("good").sum()
        ),
        "median_contamination_pct_units_with_median_depth_interior": float(
            contamination.loc[contamination.index.intersection(interior_units), contam_col].median()
        ),
        "cross_unit_coincidence_excess": float(coincidence - coincidence_null),
        "median_refractory_violation_fraction": float(np.median(refractory)),
        "spike_rate_cv_across_300s_bins": float(bins.spikes_per_s.std(ddof=0) / rate_mean),
        "n_neural_unmatched_events": int(len(scoped_events)),
        "neural_unmatched_recovery": recovery,
        "neural_unmatched_jitter_recovery": jitter,
        "neural_unmatched_recovery_excess": recovery - jitter,
    }


def run(margin_um: float, review_path: Path, output: Path) -> dict:
    if margin_um <= 0 or margin_um >= 470:
        raise ValueError("margin_um must leave a nonempty strip interior")
    events = pd.read_csv(review_path)
    scores = {name: score_interior(path, margin_um, events) for name, path in SORTERS.items()}
    keys = [key for key, value in scores["no_motion"].items() if isinstance(value, (int, float)) and key != "margin_um_each_edge"]
    comparison = {
        "status": "sensitivity_only; no-halo correction remains ineligible for advancement",
        "reason": "interior restriction cannot undo global effects on template learning",
        "scores": scores,
        "rigid025_minus_no_motion": {key: scores["rigid025_p2_no_halo"][key] - scores["no_motion"][key] for key in keys},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2) + "\n")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margin-um", type=float, default=80.0)
    parser.add_argument("--review-events", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.margin_um, args.review_events, args.output), indent=2))


if __name__ == "__main__":
    main()
