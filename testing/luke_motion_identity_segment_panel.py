"""Freeze non-overlapping Luke segments spanning motion and failure regimes.

Selection uses only the existing input/motion-estimator feature tables. It does
not read sorter labels, apply motion, or inspect identity-continuity outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SOURCE = Path("testing/outputs/luke_motion_regime_windows")
OUTPUT = Path("testing/outputs/luke_motion_identity_segment_panel")
ANCHOR_DURATION_S = 120.0
SEGMENT_DURATION_S = 600.0
SELECTIONS = (
    ("sustained_noise_control", 1170.0, "control"),
    ("relative_quiet", 4080.0, "primary_motion_gradient"),
    ("moderate_supported_motion", 4800.0, "primary_motion_gradient"),
    ("large_supported_motion", 5910.0, "primary_motion_gradient"),
    ("support_dropout_control", 6540.0, "control"),
    ("large_motion_with_input_anomaly", 7230.0, "motion_input_interaction"),
)


def build_panel(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    probe_rows = []
    for segment_id, anchor_start_s, role in SELECTIONS:
        selected = features.loc[features.start_s.eq(anchor_start_s)].copy()
        if set(selected.probe) != {"imec0", "imec1"}:
            raise ValueError(f"missing simultaneous probe rows for {segment_id}")
        segment_start_s = anchor_start_s - (SEGMENT_DURATION_S - ANCHOR_DURATION_S) / 2
        segment_end_s = segment_start_s + SEGMENT_DURATION_S
        selected.insert(0, "segment_id", segment_id)
        selected.insert(1, "panel_role", role)
        selected.insert(2, "segment_start_s", segment_start_s)
        selected.insert(3, "segment_end_s", segment_end_s)
        probe_rows.append(selected)

        rows.append(
            {
                "segment_id": segment_id,
                "panel_role": role,
                "segment_start_s": segment_start_s,
                "segment_end_s": segment_end_s,
                "segment_duration_s": SEGMENT_DURATION_S,
                "anchor_start_s": anchor_start_s,
                "anchor_end_s": anchor_start_s + ANCHOR_DURATION_S,
                "anchor_median_dredge_excursion_um": float(
                    selected.dredge_excursion_um.median()
                ),
                "anchor_min_dredge_excursion_um": float(selected.dredge_excursion_um.min()),
                "anchor_max_dredge_excursion_um": float(selected.dredge_excursion_um.max()),
                "anchor_median_decentralized_excursion_um": float(
                    selected.decentralized_excursion_um.median()
                ),
                "anchor_min_cross_method_r": float(selected.dredge_decentralized_r.min()),
                "anchor_max_input_anomaly_score": float(selected.input_anomaly_score.max()),
                "anchor_max_support_instability_score": float(
                    selected.support_instability_score.max()
                ),
                "selection_uses_sorter_outcomes": False,
            }
        )

    panel = pd.DataFrame(rows).sort_values("segment_start_s").reset_index(drop=True)
    intervals = panel[["segment_start_s", "segment_end_s"]].to_numpy(float)
    if any(intervals[i, 1] > intervals[i + 1, 0] for i in range(len(intervals) - 1)):
        raise ValueError("frozen segments overlap")
    return panel, pd.concat(probe_rows, ignore_index=True)


def run(source: Path = SOURCE, output: Path = OUTPUT) -> dict:
    features = pd.read_csv(source / "candidate_window_features.csv")
    panel, probe_rows = build_panel(features)
    output.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output / "segment_panel.csv", index=False)
    probe_rows.to_csv(output / "anchor_probe_features.csv", index=False)
    manifest = {
        "schema_version": "luke-motion-identity-segment-panel-v1",
        "source_feature_table": str(source / "candidate_window_features.csv"),
        "segment_duration_s": SEGMENT_DURATION_S,
        "anchor_duration_s": ANCHOR_DURATION_S,
        "segments": panel.to_dict(orient="records"),
        "selection_basis": (
            "Existing simultaneous-probe input and motion-estimator features; "
            "no sorter or identity-continuity outcome used."
        ),
        "sorter_labels_accessed": False,
        "motion_applied": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
