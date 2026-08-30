"""Apply the selected Luke motion field and score a fixed-event Kilosort run.

This is deliberately isolated from production.  It reuses the already saved
120 s current-conditioned/no-motion recording, applies the cache-safe DREDGE
300/200 field with an explicitly named interpolation implementation, runs the
same claim-off Kilosort diagnostic, and scores the prespecified reviewed raw
events plus the existing collision/contamination safeguards.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_claimmask_window_sweep import (
    ClaimSetting,
    build_sorter_params,
    cross_unit_near_coincident_fraction,
    event_local_samples,
    events_in_window,
    load_reference_settings,
    local_match_mask,
)
from testing.luke_motion_scale_sweep import (
    DEFAULT_OUTPUT as MOTION_SWEEP_ROOT,
    candidate_by_name,
    run_dir,
)
from testing.luke_upstream_sorter_ablation import (
    DEFAULT_REVIEW,
    OUTPUT_ROOT,
    WINDOW,
    parse_extraction_counts,
)


SOURCE_CANDIDATE = "dredge_nr_200_300_split"
MEDICINE_CANDIDATE = "medicine_pipeline_default"
RIGID_GAINS = {
    "rigid_gain_025": 0.25,
    "rigid_gain_050": 0.50,
    "rigid_gain_075": 0.75,
}
P2_RIGID_GAINS = {
    "rigid_gain_025_p2_extrapolate": 0.25,
    "rigid_gain_100_p2_extrapolate": 1.0,
}
CONDITIONS = {
    "nonrigid": SOURCE_CANDIDATE,
    "nonrigid_p2_extrapolate": f"{SOURCE_CANDIDATE}_p2_extrapolate",
    "nonrigid_p2_sigma28_extrapolate": f"{SOURCE_CANDIDATE}_p2_sigma28_extrapolate",
    "rigid": "dredge_rigid_from_300_200",
    "identity": "zero_displacement_identity",
    "ks_internal_rigid": "kilosort_internal_rigid",
    "medicine_sigma10": "medicine_default_sigma10",
    **{field: f"dredge_rigid_gain_{int(gain * 100):03d}" for field, gain in RIGID_GAINS.items()},
    **{
        field: f"dredge_rigid_gain_{int(gain * 100):03d}_p2_extrapolate"
        for field, gain in P2_RIGID_GAINS.items()
    },
}
SEED = 20250804
SOURCE_RECORDING = OUTPUT_ROOT / "recordings/current_no_motion"
CLAIM_OFF = ClaimSetting("claim_off", 0.0, 0.0)
LOCAL_RIGID_100_ROOT = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/"
    "Luke0804_rigid100_p2_pathological_imec1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument(
        "--field",
        choices=tuple(CONDITIONS),
        default="nonrigid",
        help=(
            "Apply the native selected field, its depth-mean rigid component, "
            "an all-zero identity field, or Kilosort's internal rigid correction."
        ),
    )
    parser.add_argument("--review-events", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--n-jitters", type=int, default=250)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def motion_run_path(field: str = "nonrigid") -> Path:
    candidate_name = MEDICINE_CANDIDATE if field == "medicine_sigma10" else SOURCE_CANDIDATE
    candidate = candidate_by_name(candidate_name)
    return run_dir(MOTION_SWEEP_ROOT, "imec1", candidate, "full", SEED)


def condition_paths(field: str) -> tuple[str, Path, Path]:
    condition = CONDITIONS[field]
    root = LOCAL_RIGID_100_ROOT if field == "rigid_gain_100_p2_extrapolate" else OUTPUT_ROOT
    recording = SOURCE_RECORDING if field == "ks_internal_rigid" else OUTPUT_ROOT / f"recordings/{condition}"
    if field == "rigid_gain_100_p2_extrapolate":
        recording = root / f"recordings/{condition}"
    return (
        condition,
        recording,
        root / f"sorts/{condition}",
    )


def field_displacement(displacement: np.ndarray, field: str) -> np.ndarray:
    """Return the native, depth-mean rigid, or zero-displacement field."""
    if field in (
        "nonrigid",
        "nonrigid_p2_extrapolate",
        "nonrigid_p2_sigma28_extrapolate",
        "medicine_sigma10",
    ):
        return displacement
    if field == "rigid":
        rigid = np.mean(displacement, axis=1, keepdims=True)
        return np.repeat(rigid, displacement.shape[1], axis=1)
    if field in RIGID_GAINS or field in P2_RIGID_GAINS:
        gain = RIGID_GAINS.get(field, P2_RIGID_GAINS.get(field))
        rigid = gain * np.mean(displacement, axis=1, keepdims=True)
        return np.repeat(rigid, displacement.shape[1], axis=1)
    if field == "identity":
        return np.zeros_like(displacement)
    raise ValueError(f"Unknown field: {field}")


def interpolation_spec(field: str) -> dict:
    """Return explicit interpolation parameters for a diagnostic condition."""
    if field == "nonrigid_p2_extrapolate" or field in P2_RIGID_GAINS:
        return {"border_mode": "force_extrapolate", "sigma_um": 20.0, "p": 2}
    if field == "nonrigid_p2_sigma28_extrapolate":
        return {
            "border_mode": "force_extrapolate",
            "sigma_um": float(np.sqrt(2.0) * 20.0),
            "p": 2,
        }
    return {
        "border_mode": "force_zeros",
        "sigma_um": 10.0 if field == "medicine_sigma10" else 20.0,
        "p": 1,
    }


def prepare_recording(field: str) -> None:
    import spikeinterface.core as sc
    from spikeinterface.core.motion import Motion
    from spikeinterface.sortingcomponents.motion import interpolate_motion

    if not SOURCE_RECORDING.exists():
        raise FileNotFoundError(SOURCE_RECORDING)
    if field == "ks_internal_rigid":
        print(f"Kilosort internal correction uses the untouched source recording: {SOURCE_RECORDING}")
        return
    motion_path = motion_run_path(field)
    required = [motion_path / name for name in ("motion.npy", "time_bins.npy", "depth_bins.npy", "manifest.json")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete selected motion run: {missing}")
    condition, target_recording, _ = condition_paths(field)
    metadata = target_recording / "window_manifest.json"
    if target_recording.exists():
        if not metadata.exists():
            raise RuntimeError(f"Ambiguous existing recording directory: {target_recording}")
        print(f"Reusing {target_recording}")
        return

    recording = sc.load(SOURCE_RECORDING)
    source_displacement = np.load(motion_path / "motion.npy")
    motion = Motion(
        displacement=field_displacement(source_displacement, field),
        temporal_bins_s=np.load(motion_path / "time_bins.npy"),
        spatial_bins_um=np.load(motion_path / "depth_bins.npy"),
    )
    interpolation = interpolation_spec(field)
    corrected = interpolate_motion(
        recording.astype("float"),
        motion,
        spatial_interpolation_method="kriging",
        **interpolation,
    ).astype("int16")
    target_recording.parent.mkdir(parents=True, exist_ok=True)
    corrected.save(folder=target_recording, n_jobs=-1, chunk_duration="1s", progress_bar=True)
    metadata.write_text(
        json.dumps(
            {
                "condition": condition,
                "source_recording": str(SOURCE_RECORDING),
                "motion_run": str(motion_path),
                "motion_spec_hash": motion_path.name.removeprefix("full_"),
                "field_transform": {
                    "nonrigid": "native",
                    "nonrigid_p2_extrapolate": "native",
                    "nonrigid_p2_sigma28_extrapolate": "native",
                    "medicine_sigma10": "native_medicine",
                    "rigid": "depth_mean_repeated",
                    "identity": "all_zeros",
                }.get(field, "scaled_depth_mean_repeated"),
                "rigid_gain": (
                    RIGID_GAINS.get(field, P2_RIGID_GAINS.get(field))
                    if field != "rigid"
                    else 1.0
                ),
                "rigid_definition": (
                    "gain times arithmetic mean across native depth bins at each time bin"
                    if field == "rigid" or field in RIGID_GAINS or field in P2_RIGID_GAINS
                    else None
                ),
                "interpolation_border_mode": interpolation["border_mode"],
                "spatial_interpolation_method": "kriging",
                "spatial_interpolation_sigma_um": interpolation["sigma_um"],
                "spatial_interpolation_p": interpolation["p"],
                "interpolation_input_dtype": "float",
                "saved_dtype": "int16",
                "window": WINDOW.name,
                "claim_mask": "off for downstream diagnostic sort",
            },
            indent=2,
        )
        + "\n"
    )


def assert_gpu_and_patch() -> None:
    import torch
    from kilosort.parameters import DEFAULT_SETTINGS

    missing = {"cross_peel_claim_ms", "cross_peel_claim_um"} - set(DEFAULT_SETTINGS)
    if missing:
        raise RuntimeError(f"Patched Kilosort settings are missing: {sorted(missing)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this phase outside the sandbox")


def sorter_params(field: str) -> dict:
    params = build_sorter_params(CLAIM_OFF)
    if field == "ks_internal_rigid":
        params.update(do_correction=True, nblocks=1)
    return params


def run_sort(field: str) -> None:
    import spikeinterface.core as sc
    from spikeinterface.sorters import run_sorter

    assert_gpu_and_patch()
    condition, target_recording, target_sort = condition_paths(field)
    result = target_sort / "sorter_output/spike_times.npy"
    if result.exists():
        print(f"Reusing completed sort {target_sort}")
        return
    if target_sort.exists():
        raise RuntimeError(f"Partial or ambiguous sort directory: {target_sort}")
    if not target_recording.exists():
        raise FileNotFoundError(f"Prepare the recording first: {target_recording}")
    recording = sc.load(target_recording)
    params = sorter_params(field)
    target_sort.parent.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_ROOT / "logs" / f"{condition}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("[%(levelname)s] - %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        with log_path.open("a") as log_file, contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            run_sorter(
                "kilosort4",
                recording,
                folder=str(target_sort),
                verbose=True,
                remove_existing_folder=False,
                **params,
            )
    finally:
        root_logger.removeHandler(handler)
        handler.close()


def score_sort(field: str, review_path: Path, n_jitters: int, seed: int) -> pd.DataFrame:
    condition, _, target_sort = condition_paths(field)
    result_dir = target_sort / "sorter_output"
    if not (result_dir / "spike_times.npy").exists():
        raise FileNotFoundError(result_dir)
    _, fs = load_reference_settings()
    events = events_in_window(pd.read_csv(review_path), WINDOW)
    populations = {
        "visual_neural_unmatched": (events["review_label"] == "neural") & (events["status"] == "unmatched"),
        "all_reviewed": pd.Series(True, index=events.index),
    }
    duration_samples = int(round(WINDOW.duration_s * fs))
    tolerance = int(round(0.5e-3 * fs))
    times = np.load(result_dir / "spike_times.npy").reshape(-1).astype(np.int64)
    clusters = np.load(result_dir / "spike_clusters.npy").reshape(-1)
    positions = np.load(result_dir / "spike_positions.npy")
    valid = (times >= 0) & (times < duration_samples)
    times, clusters, depths = times[valid], clusters[valid], positions[valid, 1]
    full_count = len(np.load(result_dir / "full_st.npy", mmap_mode="r"))
    labels = pd.read_csv(result_dir / "cluster_KSLabel.tsv", sep="\t")
    label_column = next(column for column in labels if column != "cluster_id")
    contamination = pd.read_csv(result_dir / "cluster_ContamPct.tsv", sep="\t")
    contam_column = next(column for column in contamination if column != "cluster_id")
    log_path = OUTPUT_ROOT / "logs" / f"{condition}.log"
    universal_count = learned_log_count = None
    if log_path.exists():
        universal_count, learned_log_count = parse_extraction_counts(log_path.read_text())
    if learned_log_count is not None and learned_log_count != full_count:
        raise RuntimeError(f"Learned count mismatch: log={learned_log_count}, full_st={full_count}")
    refractory = []
    isi_limit = int(round(1.5e-3 * fs))
    for unit in np.unique(clusters):
        unit_times = np.sort(times[clusters == unit])
        if len(unit_times) > 1:
            refractory.append(float(np.mean(np.diff(unit_times) < isi_limit)))
    rng = np.random.default_rng(seed)
    rows = []
    for population, mask in populations.items():
        subset = events[mask]
        samples = event_local_samples(subset["sample_index"].to_numpy(), WINDOW, fs)
        event_depths = subset["peak_depth_um"].to_numpy(float)
        observed = float(local_match_mask(samples, event_depths, times, depths, tolerance, 100.0).mean())
        null = []
        for _ in range(n_jitters):
            offsets_ms = rng.uniform(20.0, 500.0, len(subset))
            offsets_ms *= rng.choice((-1.0, 1.0), len(subset))
            shifted = samples + np.rint(offsets_ms * fs / 1000.0).astype(np.int64)
            null.append(float(local_match_mask(shifted, event_depths, times, depths, tolerance, 100.0).mean()))
        rows.append(
            {
                "condition": condition,
                "population": population,
                "n_events": len(subset),
                "observed_recovery": observed,
                "jitter_null_mean": float(np.mean(null)),
                "recovery_above_null": observed - float(np.mean(null)),
                "universal_detection_count": universal_count,
                "learned_detection_count": full_count,
                "learned_to_universal_expansion": full_count / universal_count if universal_count else np.nan,
                "n_final_spikes": len(times),
                "n_units": len(np.unique(clusters)),
                "n_ks_good": int(labels[label_column].astype(str).str.lower().eq("good").sum()),
                "median_contamination_pct": float(np.median(contamination[contam_column].to_numpy(float))),
                "cross_unit_near_coincident_fraction": cross_unit_near_coincident_fraction(times, clusters, depths, tolerance),
                "median_unit_refractory_violation_fraction": float(np.median(refractory)),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_ROOT / f"{condition}_scores.csv", index=False)
    return frame


def main() -> None:
    args = parse_args()
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-motion-candidate-numba-cache")
    if not (args.prepare or args.run or args.score):
        raise SystemExit("Choose at least one of --prepare, --run, or --score")
    if args.prepare:
        prepare_recording(args.field)
    if args.run:
        run_sort(args.field)
    if args.score:
        print(score_sort(args.field, args.review_events, args.n_jitters, args.seed).to_string(index=False))


if __name__ == "__main__":
    main()
