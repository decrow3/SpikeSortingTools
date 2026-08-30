"""Millisecond-scale known-drift residual screen for Luke interpolation ideas.

The empirical snippet lab is limited by unstable provisional cluster identity.
This complementary screen uses the already extracted discovery templates as
exact signals, imposes the measured Luke time/depth displacement, applies a
candidate inverse field, and scores the recovered waveform against the known
input.  It is an interpolation/mechanics test, not a biological or sorter
benchmark, and it never opens the prospective holdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_motion_snippet_residual_lab import (
    MOTION_DIR,
    SNIPPETS,
    SOURCE,
    best_scaled_residual,
    decompose_motion,
)
from testing.luke_rigid025_depth_strip import relative_motion_bins


TEMPLATES = Path("testing/outputs/luke_injected_ground_truth_pilot/donor_templates.npz")
OUTPUT = Path("testing/outputs/luke_synthetic_motion_residual_lab")
DEPTH_CHANNELS = np.arange(176, 272)


@dataclass(frozen=True)
class Kernel:
    name: str
    method: str
    sigma_um: float = 20.0
    p: float = 2.0
    num_closest: int = 4


@dataclass(frozen=True)
class Candidate:
    name: str
    rigid_gain: float
    residual_gain: float
    kernel: Kernel


GENERATORS = (
    Kernel("truth_kriging_p2_sigma10", "kriging", 10.0, 2.0),
    Kernel("truth_kriging_p2_sigma20", "kriging", 20.0, 2.0),
    Kernel("truth_idw4", "idw", num_closest=4),
)
K10 = Kernel("kriging_p2_sigma10", "kriging", 10.0, 2.0)
K20 = Kernel("kriging_p2_sigma20", "kriging", 20.0, 2.0)
IDW4 = Kernel("idw4", "idw", num_closest=4)
CANDIDATES = (
    Candidate("no_motion", 0.0, 0.0, K20),
    Candidate("rigid025_sigma20", 0.25, 0.0, K20),
    Candidate("rigid025_nr010_sigma20", 0.25, 0.10, K20),
    Candidate("rigid025_nr025_sigma20", 0.25, 0.25, K20),
    Candidate("rigid025_nr010_sigma10", 0.25, 0.10, K10),
    Candidate("rigid025_nr025_sigma10", 0.25, 0.25, K10),
    Candidate("full_nonrigid_sigma10", 1.0, 1.0, K10),
    Candidate("full_nonrigid_sigma20", 1.0, 1.0, K20),
    Candidate("full_nonrigid_idw4", 1.0, 1.0, IDW4),
)


def interpolate_field_at(
    displacement: np.ndarray,
    temporal_bins_s: np.ndarray,
    spatial_bins_um: np.ndarray,
    time_s: float,
    channel_depths_um: np.ndarray,
) -> np.ndarray:
    """Linearly evaluate a time-depth field at one time and channel depths."""
    field_at_depth_bins = np.asarray(
        [
            np.interp(time_s, temporal_bins_s, displacement[:, index])
            for index in range(displacement.shape[1])
        ]
    )
    return np.interp(
        channel_depths_um,
        spatial_bins_um,
        field_at_depth_bins,
        left=field_at_depth_bins[0],
        right=field_at_depth_bins[-1],
    )


def spatial_warp(
    waveform: np.ndarray,
    locations: np.ndarray,
    displacement_um: np.ndarray,
    kernel: Kernel,
) -> np.ndarray:
    """Sample a waveform at depth-shifted target contact locations."""
    from spikeinterface.preprocessing.preprocessing_tools import (
        get_spatial_interpolation_kernel,
    )

    values = np.asarray(waveform, dtype=np.float32)
    target = np.asarray(locations, dtype=float).copy()
    shift = np.asarray(displacement_um, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(target) or shift.shape != (len(target),):
        raise ValueError("waveform, locations and displacement dimensions disagree")
    target[:, 1] += shift
    matrix = get_spatial_interpolation_kernel(
        np.asarray(locations, dtype=float),
        target,
        method=kernel.method,
        sigma_um=kernel.sigma_um,
        p=kernel.p,
        num_closest=kernel.num_closest,
        force_extrapolate=True,
        dtype="float32",
    )
    return values @ matrix


def template_subset(template_path: Path, locations: np.ndarray, maximum: int) -> dict[str, np.ndarray]:
    arrays = np.load(template_path)
    selected = {}
    depth_min, depth_max = float(locations[:, 1].min()), float(locations[:, 1].max())
    for name in arrays.files:
        waveform = np.asarray(arrays[name], dtype=np.float32)[:, DEPTH_CHANNELS]
        peak = np.unravel_index(np.argmax(np.abs(waveform)), waveform.shape)[1]
        peak_depth = float(locations[peak, 1])
        if peak_depth - depth_min < 160 or depth_max - peak_depth < 160:
            continue
        if np.max(np.abs(waveform)) < 20:
            continue
        selected[name] = waveform
        if len(selected) == maximum:
            break
    if not selected:
        raise RuntimeError("No donor template has adequate depth-strip support")
    return selected


def summarize_candidates(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates while guarding against residual-only over-smoothing."""
    values = metrics.copy()
    keys = ["snippet", "template_id", "generator"]
    baseline = values[values.candidate == "no_motion"].set_index(keys)
    case_index = pd.MultiIndex.from_frame(values[keys])
    baseline_residual = baseline.residual_fraction.reindex(case_index).to_numpy()
    baseline_amplitude_error = (
        baseline.amplitude_retention.sub(1).abs().reindex(case_index).to_numpy()
    )
    baseline_cosine = baseline.template_cosine.reindex(case_index).to_numpy()
    values["delta_residual_vs_no_motion"] = (
        values.residual_fraction.to_numpy() - baseline_residual
    )
    values["delta_absolute_amplitude_error_vs_no_motion"] = (
        values.amplitude_retention.sub(1).abs().to_numpy() - baseline_amplitude_error
    )
    values["delta_cosine_vs_no_motion"] = (
        values.template_cosine.to_numpy() - baseline_cosine
    )
    by_generator = (
        values.groupby(["candidate", "generator"], observed=True)
        .agg(
            generator_median_delta_residual=("delta_residual_vs_no_motion", "median"),
            generator_median_delta_amplitude_error=(
                "delta_absolute_amplitude_error_vs_no_motion",
                "median",
            ),
            generator_median_delta_cosine=("delta_cosine_vs_no_motion", "median"),
        )
        .reset_index()
        .groupby("candidate", observed=True)
        .agg(
            worst_generator_median_delta_residual=(
                "generator_median_delta_residual",
                "max",
            ),
            worst_generator_median_delta_amplitude_error=(
                "generator_median_delta_amplitude_error",
                "max",
            ),
            worst_generator_median_delta_cosine=(
                "generator_median_delta_cosine",
                "min",
            ),
        )
    )
    summary = (
        values.groupby("candidate", observed=True)
        .agg(
            cases=("template_id", "size"),
            median_residual_fraction=("residual_fraction", "median"),
            p90_residual_fraction=("residual_fraction", lambda x: x.quantile(0.9)),
            median_delta_residual_vs_no_motion=("delta_residual_vs_no_motion", "median"),
            median_waveform_cosine=("template_cosine", "median"),
            p10_waveform_cosine=("template_cosine", lambda x: x.quantile(0.1)),
            median_amplitude_retention=("amplitude_retention", "median"),
            p10_amplitude_retention=("amplitude_retention", lambda x: x.quantile(0.1)),
            median_delta_absolute_amplitude_error_vs_no_motion=(
                "delta_absolute_amplitude_error_vs_no_motion",
                "median",
            ),
            fraction_peak_channel_error=("peak_channel_error", lambda x: np.mean(np.abs(x) > 0)),
        )
        .join(by_generator)
        .reset_index()
    )
    # Advance only candidates whose gain is robust to the unknown continuous
    # waveform between contacts and is not bought with amplitude/cosine loss.
    summary["robust_screen_pass"] = (
        (summary.candidate != "no_motion")
        & (summary.worst_generator_median_delta_residual <= -0.005)
        & (summary.worst_generator_median_delta_amplitude_error <= 0.005)
        & (summary.worst_generator_median_delta_cosine >= 0.0)
    )
    return summary.sort_values(
        ["robust_screen_pass", "median_residual_fraction", "p90_residual_fraction"],
        ascending=[False, True, True],
    )


def run(maximum_templates: int, output_dir: Path) -> dict:
    import spikeinterface.core as sc

    source = sc.load(SOURCE)
    locations = np.asarray(source.get_channel_locations(), dtype=float)
    templates = template_subset(TEMPLATES, locations, maximum_templates)
    displacement = np.load(MOTION_DIR / "motion.npy")
    absolute_bins = np.load(MOTION_DIR / "time_bins.npy")
    spatial_bins = np.load(MOTION_DIR / "depth_bins.npy")
    relative_bins, acquisition_start_s = relative_motion_bins(absolute_bins)
    rigid, residual = decompose_motion(displacement)
    rows = []
    for snippet in SNIPPETS:
        time_s = snippet.start_s + snippet.duration_s / 2
        full_at_channels = interpolate_field_at(
            displacement, relative_bins, spatial_bins, time_s, locations[:, 1]
        )
        rigid_at_channels = interpolate_field_at(
            rigid,
            relative_bins,
            np.asarray([np.nanmedian(spatial_bins)]),
            time_s,
            locations[:, 1],
        )
        residual_at_channels = full_at_channels - rigid_at_channels
        for template_id, original in templates.items():
            peak = np.unravel_index(np.argmax(np.abs(original)), original.shape)[1]
            local = np.flatnonzero(np.abs(locations[:, 1] - locations[peak, 1]) <= 160)
            reference = original[:, local]
            reference_peak = float(np.max(np.abs(reference)))
            for generator in GENERATORS:
                observed = spatial_warp(original, locations, full_at_channels, generator)
                for candidate in CANDIDATES:
                    correction = (
                        candidate.rigid_gain * rigid_at_channels
                        + candidate.residual_gain * residual_at_channels
                    )
                    recovered = (
                        observed
                        if candidate.name == "no_motion"
                        else spatial_warp(observed, locations, -correction, candidate.kernel)
                    )
                    metrics = best_scaled_residual(recovered[:, local], reference)
                    recovered_peak = float(np.max(np.abs(recovered[:, local])))
                    recovered_peak_channel = int(
                        local[np.unravel_index(np.argmax(np.abs(recovered[:, local])), recovered[:, local].shape)[1]]
                    )
                    rows.append(
                        {
                            "snippet": snippet.name,
                            "motion_class": snippet.motion_class,
                            "time_s": time_s,
                            "template_id": template_id,
                            "generator": generator.name,
                            "candidate": candidate.name,
                            "rigid_gain": candidate.rigid_gain,
                            "residual_gain": candidate.residual_gain,
                            "candidate_kernel": candidate.kernel.name,
                            "field_rigid_um": float(np.median(rigid_at_channels)),
                            "field_residual_p95_p5_um": float(
                                np.quantile(residual_at_channels, 0.95)
                                - np.quantile(residual_at_channels, 0.05)
                            ),
                            "amplitude_retention": recovered_peak / reference_peak,
                            "peak_channel_error": recovered_peak_channel - peak,
                            **metrics,
                        }
                    )
    metrics = pd.DataFrame(rows)
    summary = summarize_candidates(metrics)
    keys = ["snippet", "template_id", "generator"]
    baseline = metrics[metrics.candidate == "no_motion"].set_index(keys)
    case_index = pd.MultiIndex.from_frame(metrics[keys])
    metrics["delta_residual_vs_no_motion"] = (
        metrics.residual_fraction.to_numpy()
        - baseline.residual_fraction.reindex(case_index).to_numpy()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "case_metrics.csv", index=False)
    summary.to_csv(output_dir / "candidate_summary.csv", index=False)
    result = {
        "status": "discovery_template_known_drift_interpolation_screen",
        "templates": list(templates),
        "snippets": [asdict(value) for value in SNIPPETS],
        "generators": [asdict(value) for value in GENERATORS],
        "candidates": [
            {**asdict(value), "kernel": asdict(value.kernel)} for value in CANDIDATES
        ],
        "inferred_acquisition_start_s": acquisition_start_s,
        "cases": int(len(metrics)),
        "prospective_holdout_accessed": False,
        "sorter_run": False,
        "robust_screen_pass": summary.loc[
            summary.robust_screen_pass, "candidate"
        ].tolist(),
        "screen_gate": {
            "worst_generator_median_delta_residual_max": -0.005,
            "worst_generator_median_delta_absolute_amplitude_error_max": 0.005,
            "worst_generator_median_delta_cosine_min": 0.0,
        },
        "limitations": [
            "The continuous spatial waveform between contacts is unknown; three generator kernels test sensitivity to that assumption.",
            "This isolates field/interpolation recovery and does not test detection, clustering, collisions, or biological unit identity.",
            "Donor morphology comes from discovery templates, but the recovery target for each imposed waveform is exact.",
        ],
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maximum-templates", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.maximum_templates, args.output_dir)


if __name__ == "__main__":
    main()
