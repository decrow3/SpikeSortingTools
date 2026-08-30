"""Classify edge-dominated and longitudinally unstable Luke strip units.

The audit is read-only and does not curate clusters.  It combines the existing
unit-continuity table with exported template geometry so that strip truncation,
low-amplitude/high-rate clusters, and plausible motion-fragmentation candidates
are not conflated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SORTER = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/"
    "Luke0804_two_axis_pilot_imec1/sorts/core_depth_strip/"
    "single_ks_preprocessing_claim_off/sorter_output"
)
UNIT_METRICS = Path(
    "testing/outputs/luke_full_strip_diagnostic_audit/unit_continuity_metrics.csv"
)
OUTPUT = Path("testing/outputs/luke_targeted_unit_audit")


def template_metrics(template: np.ndarray, locations: np.ndarray) -> dict:
    template = np.asarray(template, dtype=float)
    locations = np.asarray(locations, dtype=float)
    energy = np.sum(template * template, axis=0)
    total = float(energy.sum())
    peak_channel = int(np.argmax(energy))
    depth_min, depth_max = locations[:, 1].min(), locations[:, 1].max()
    peak_depth = float(locations[peak_channel, 1])
    distance = float(min(peak_depth - depth_min, depth_max - peak_depth))
    edge = np.minimum(locations[:, 1] - depth_min, depth_max - locations[:, 1]) <= 60
    waveform = template[:, peak_channel]
    peak_index = int(np.argmax(np.abs(waveform)))
    signed_peak = float(waveform[peak_index])
    opposite = float(np.max(waveform) if signed_peak < 0 else -np.min(waveform))
    dominant = abs(signed_peak)
    return {
        "template_peak_channel": peak_channel,
        "template_peak_depth_um": peak_depth,
        "template_peak_distance_to_strip_edge_um": distance,
        "template_energy_within_60um_edge_fraction": float(energy[edge].sum() / total) if total else np.nan,
        "template_peak_time_sample": peak_index,
        "template_signed_peak": signed_peak,
        "template_opposite_to_dominant_ratio": opposite / dominant if dominant else np.nan,
        "template_channels_over_20pct_peak_energy": int(np.sum(energy >= 0.2 * energy.max())) if total else 0,
    }


def classify(row: pd.Series) -> str:
    if row.edge_spike_fraction > 0.5 and row.template_peak_distance_to_strip_edge_um <= 60:
        return "strip_boundary_truncation_candidate"
    if row.first_last_pc_cosine < 0.8 and row.median_amplitude < 30 and row.mean_rate_hz > 1:
        return "low_amplitude_high_rate_template_instability"
    if row.first_last_pc_cosine < 0.8 and row.depth_excursion_p95_p5_um >= 40:
        return "motion_or_family_fragmentation_candidate"
    if row.first_last_pc_cosine < 0.8:
        return "waveform_feature_instability_requires_review"
    return "edge_localized_other"


def run_audit(sorter: Path, unit_metrics_path: Path, output_dir: Path) -> dict:
    units = pd.read_csv(unit_metrics_path)
    targets = units[(units.edge_spike_fraction > 0.5) | (units.first_last_pc_cosine < 0.8)].copy()
    templates = np.load(sorter / "templates.npy", mmap_mode="r")
    locations = np.load(sorter / "channel_positions.npy")
    details = pd.DataFrame(
        [template_metrics(templates[int(unit)], locations) for unit in targets.unit_id]
    )
    targets = pd.concat([targets.reset_index(drop=True), details], axis=1)
    targets["diagnostic_class"] = targets.apply(classify, axis=1)
    targets["automatic_curation_allowed"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output_dir / "targeted_unit_metrics.csv", index=False)
    counts = targets.diagnostic_class.value_counts().to_dict()
    summary = {
        "target_units": int(len(targets)),
        "edge_dominated_units": int((targets.edge_spike_fraction > 0.5).sum()),
        "weak_first_last_pc_units": int((targets.first_last_pc_cosine < 0.8).sum()),
        "class_counts": {str(key): int(value) for key, value in counts.items()},
        "ks_good_targets": int(targets.ks_good.sum()),
        "automatic_curation_allowed": False,
        "interpretation": (
            "Boundary classification identifies units whose evidence is censored by the "
            "96-channel strip; it does not establish artifact. Low PC cosine is a review "
            "trigger, not proof of motion fragmentation."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sorter", type=Path, default=SORTER)
    parser.add_argument("--unit-metrics", type=Path, default=UNIT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.sorter, args.unit_metrics, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
