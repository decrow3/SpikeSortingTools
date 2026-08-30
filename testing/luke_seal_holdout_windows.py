"""Seal prospective Luke finalist-validation windows without sorter access.

Windows are selected from the pre-existing full-session DREDGE fields only.
The completed no-motion imec1 sort already spans the session, so this is not a
pristine retrospective test of that baseline.  It is a prospective holdout for
all subsequent artifact, motion, and finalist comparisons.  Candidate events
must later be drawn under the sealed polarity/amplitude/depth protocol without
changing these windows or using sorter output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path("testing/outputs/luke_prospective_holdout")
MOTION_ROOTS = {
    probe: Path(
        "/mnt/NPX/Luke/20250804/"
        f"dredge_pipeline_results_Luke0804_V2V1_g0_{probe}/motion/dredge-motion"
    )
    for probe in ("imec0", "imec1")
}
WINDOW_DURATION_S = 120.0
WINDOW_STEP_S = 120.0
EXCLUSION_GUARD_S = 300.0
# Named discovery/calibration intervals already used to inspect or tune choices.
DISCOVERY_INTERVALS = {
    "quiet_negative_control": (3951.0, 4071.0),
    "good_neutral_shared": (7095.0, 7335.0),
    "pathological_and_runtime_calibration": (7800.0, 8400.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def overlaps_exclusion(start: float, stop: float) -> bool:
    for discovery_start, discovery_stop in DISCOVERY_INTERVALS.values():
        if start < discovery_stop + EXCLUSION_GUARD_S and stop > discovery_start - EXCLUSION_GUARD_S:
            return True
    return False


def load_motion(probe: str) -> pd.DataFrame:
    root = MOTION_ROOTS[probe]
    times = np.load(root / "time_bins.npy").astype(float)
    depths = np.load(root / "depth_bins.npy").astype(float)
    field = np.load(root / "motion.npy", mmap_mode="r")
    local_time = times - times[0] + 1.0
    cortical = (depths >= 400.0) & (depths <= 3600.0)
    selected = np.asarray(field[:, cortical], dtype=float)
    rigid = np.nanmedian(selected, axis=1)
    spread = np.nanpercentile(selected, 95, axis=1) - np.nanpercentile(
        selected, 5, axis=1
    )
    step = np.r_[np.nan, np.abs(np.diff(rigid))]
    return pd.DataFrame(
        {
            "time_s": local_time,
            "rigid_um": rigid,
            "spread_um": spread,
            "abs_step_um": step,
        }
    )


def robust_percentile(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def candidate_windows() -> tuple[pd.DataFrame, float]:
    motion = {probe: load_motion(probe) for probe in MOTION_ROOTS}
    duration_s = min(frame.time_s.max() for frame in motion.values()) + 1.0
    rows = []
    for start in np.arange(600.0, duration_s - WINDOW_DURATION_S - 600.0, WINDOW_STEP_S):
        stop = start + WINDOW_DURATION_S
        if overlaps_exclusion(start, stop):
            continue
        row = {"start_s": float(start), "stop_s": float(stop)}
        for probe, frame in motion.items():
            selected = frame[frame.time_s.between(start, stop, inclusive="left")]
            row[f"{probe}_rigid_excursion_um"] = float(
                selected.rigid_um.quantile(0.95) - selected.rigid_um.quantile(0.05)
            )
            row[f"{probe}_median_nonrigid_spread_um"] = float(
                selected.spread_um.median()
            )
            row[f"{probe}_p99_abs_rigid_step_um"] = float(
                selected.abs_step_um.quantile(0.99)
            )
        rows.append(row)
    candidates = pd.DataFrame(rows)
    score_fields = []
    for probe in MOTION_ROOTS:
        for metric in (
            "rigid_excursion_um",
            "median_nonrigid_spread_um",
            "p99_abs_rigid_step_um",
        ):
            field = f"{probe}_{metric}"
            rank_field = f"{field}_percentile"
            candidates[rank_field] = robust_percentile(candidates[field])
            score_fields.append(rank_field)
    candidates["combined_motion_score"] = candidates[score_fields].max(axis=1)
    candidates["time_third"] = np.minimum(
        (candidates.start_s / (duration_s / 3.0)).astype(int), 2
    )
    return candidates, duration_s


def select_windows(candidates: pd.DataFrame) -> pd.DataFrame:
    selections = []
    used_starts: set[float] = set()
    for third in range(3):
        group = candidates[candidates.time_third == third]
        for stratum, ascending in (("relative_quiet", True), ("high_motion", False)):
            ordered = group.sort_values(
                ["combined_motion_score", "start_s"],
                ascending=[ascending, True],
            )
            selected = next(
                row for row in ordered.itertuples(index=False) if row.start_s not in used_starts
            )
            used_starts.add(float(selected.start_s))
            row = dict(selected._asdict())
            row["motion_stratum"] = stratum
            row["window_id"] = f"T{third + 1}_{stratum}"
            selections.append(row)
    return pd.DataFrame(selections).sort_values("start_s").reset_index(drop=True)


def seal_holdout(output_dir: Path) -> dict:
    candidates, duration_s = candidate_windows()
    selected = select_windows(candidates)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / "candidate_window_motion_metrics_v2.csv", index=False)
    selected.to_csv(output_dir / "sealed_windows_v2.csv", index=False)
    window_records = []
    for window in selected.itertuples(index=False):
        for probe in MOTION_ROOTS:
            window_records.append(
                {
                    "window_id": window.window_id,
                    "probe": probe,
                    "start_s": window.start_s,
                    "stop_s": window.stop_s,
                    "motion_stratum": window.motion_stratum,
                    "time_third": int(window.time_third) + 1,
                    "rigid_excursion_um": getattr(window, f"{probe}_rigid_excursion_um"),
                    "median_nonrigid_spread_um": getattr(
                        window, f"{probe}_median_nonrigid_spread_um"
                    ),
                    "p99_abs_rigid_step_um": getattr(
                        window, f"{probe}_p99_abs_rigid_step_um"
                    ),
                }
            )
    manifest = {
        "version": 2,
        "purpose": "Prospective holdout for all post-baseline artifact, motion, and finalist comparisons",
        "not_valid_for": "Pristine retrospective validation of the already completed full-session imec1 no-motion baseline",
        "selection_inputs": {
            "allowed": "Pre-existing full-session DREDGE motion fields and recording coordinates only",
            "forbidden": "Any Kilosort output, reviewed-event recovery, unit yield, contamination, waveform, residual, or claim metric",
        },
        "recording_duration_s": duration_s,
        "window_duration_s": WINDOW_DURATION_S,
        "window_step_s": WINDOW_STEP_S,
        "exclusion_guard_s": EXCLUSION_GUARD_S,
        "excluded_discovery_intervals": DISCOVERY_INTERVALS,
        "selection_rule": "Within each session time third, select the minimum and maximum combined percentile score across imec0/imec1 rigid excursion, nonrigid spread, and rigid step; break ties by earliest start.",
        "windows": window_records,
        "event_sampling_protocol": {
            "status": "sealed protocol; event indices not yet drawn",
            "reference": "matched local reference fixed before candidate drawing",
            "depth_strata": ["shallow third", "middle third", "deep third"],
            "polarity_strata": ["negative", "positive"],
            "fixed_amplitude_uv_strata": ["50_to_75", "75_to_100", "at_least_100"],
            "events_per_probe_window_depth_polarity_amplitude_cell": 4,
            "deduplication": "0.5 ms and 100 um",
            "ranking": "SHA256(seed, probe, window_id, sample_index, physical_channel); take lowest hashes",
            "seed": "luke-20250804-prospective-holdout-v2",
            "review": "blind labels and sorter conditions; keep key separate until all finalist outputs are frozen",
            "no_replacement_rule": "If a cell has fewer than four candidates, retain all and report the deficit; do not borrow from another stratum.",
        },
        "immutability_rule": "Any change to windows, strata, seed, thresholds, or counts creates a new version and may not replace v1.",
    }
    manifest_path = output_dir / "holdout_manifest_v2.json"
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    manifest_path.write_bytes(manifest_bytes)
    seal = {
        "manifest": str(manifest_path),
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "sealed": True,
        "event_indices_drawn": False,
    }
    (output_dir / "seal_v2.json").write_text(json.dumps(seal, indent=2) + "\n")
    v1_seal = output_dir / "seal_v1.json"
    if v1_seal.exists():
        (output_dir / "v1_superseded.json").write_text(
            json.dumps(
                {
                    "superseded": True,
                    "reason": "Before any event indices were drawn, the label quiet was corrected to relative_quiet because the minimum-motion candidate in time third 3 remained at combined percentile 0.72.",
                    "v1_event_indices_drawn": False,
                    "replacement": str(output_dir / "seal_v2.json"),
                },
                indent=2,
            )
            + "\n"
        )
    return seal


def main() -> None:
    args = parse_args()
    seal = seal_holdout(args.output_dir)
    print(json.dumps(seal, indent=2))


if __name__ == "__main__":
    main()
