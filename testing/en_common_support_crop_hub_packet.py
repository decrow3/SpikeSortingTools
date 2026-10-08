#!/usr/bin/env python3
"""Build a compact inspectable hub packet from completed crop-screen outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from testing.en_common_support_crop_panel import crop_field_cells_from_global_midpoints
from testing.en_rounded_field import rounded_rigid_field


EXPECTED_RESULT = "febf2cff7caae50fbc10f409e18fef926a7610c2303c20e859dbb8731d05424f"
EXPECTED_EVALUATOR = "d7f0c9baebf33a2481a7ae05ca4ba3820ef494e1b1179238e85a807dcdf39bc6"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def excerpt(path: Path, start: int, stop: int) -> list[dict[str, Any]]:
    lines = path.read_text().splitlines()
    return [{"line": number, "text": lines[number - 1]} for number in range(start, stop + 1)]


def field_support(screen: dict[str, Any], panel: dict[str, Any]) -> dict[str, Any]:
    path = Path(screen["window_selection"]["field_path"])
    with np.load(path, allow_pickle=False) as saved:
        centers = np.asarray(saved["time_s"], dtype=float)
        rounded, reference = rounded_rigid_field(np.asarray(saved["rigid_displacement_um"], dtype=float))
    fs = float(screen["crop"]["sampling_frequency_hz"])
    start, stop = map(int, screen["crop"]["effective_frames_half_open"])
    cells = crop_field_cells_from_global_midpoints(
        centers, rounded, float(panel["scope"]["duration_s"]), start / fs, stop / fs, 300.0)
    states = {}
    for state_index, state in enumerate(cells.states_um):
        segment_ids = np.unique(cells.knot_segment_id[cells.knot_state_index == state_index])
        durations = []
        for segment_id in segment_ids:
            indices = np.flatnonzero(cells.knot_segment_id == segment_id)
            durations.append(float(cells.knot_edges_s[indices[-1] + 1] - cells.knot_edges_s[indices[0]]))
        states[str(float(state))] = {
            "total_exposure_s": float(cells.state_exposure_s[state_index]),
            "block_exposure_s_0_300_and_300_600": cells.block_exposure_s[:, state_index].tolist(),
            "contiguous_segments": int(len(durations)),
            "segment_duration_median_s": float(np.median(durations)) if durations else None,
            "segment_duration_max_s": float(max(durations)) if durations else None,
        }
    return {
        "field_path": str(path), "field_sha256": sha256(path),
        "reference_median_um": float(reference), "block_edges_s": cells.block_edges_s.tolist(),
        "states": states,
        "confounding": (
            "All negative-state exposure occurs in the second 300-s block. Positive exposure is split "
            "69.75 s / 24.8750027949 s. State sign and crop time are therefore confounded, and the "
            "two-block bootstrap cannot establish time-independent state effects."
        ),
    }


def build(root: Path, ops_path: Path, screen_path: Path, panel_path: Path) -> dict[str, Any]:
    result_path = root / "RESULT.json"
    result = json.loads(result_path.read_text())
    evaluator_path = Path(result["contracts"]["executed_evaluator"]["path"])
    if sha256(result_path) != EXPECTED_RESULT or sha256(evaluator_path) != EXPECTED_EVALUATOR:
        raise RuntimeError("result or evaluator source differs from the reviewed artifact")
    state = pd.read_csv(root / "STATE_METRICS.csv")
    state = state[state.exposure_s > 0].copy()
    state_columns = [
        "arm", "scope", "state_um", "exposure_s", "spikes",
        "segment_safe_positive_intervals", "short_interval_count_1_29",
        "short_interval_fraction_1_29", "state_rate_eligible_units", "state_rate_rho_vs_q0",
    ]
    state_records = state[state_columns].to_dict("records")
    if len(state_records) != 32:
        raise RuntimeError("expected 32 nonzero-exposure arm/scope/state rows")
    contrasts = pd.read_csv(root / "FACTORIAL_STATE_CONTRASTS.csv")
    contrasts = contrasts[contrasts.state_um != -120.0]
    contrast_records = contrasts.to_dict("records")
    if len(contrast_records) != 16:
        raise RuntimeError("expected 16 nonzero-exposure scope/state/metric contrast rows")

    ops = json.loads(ops_path.read_text())
    scalar_names = (
        "fs", "n_chan_bin", "Nchan", "tmin", "tmax", "nblocks", "do_CAR",
        "Th_universal", "Th_learned", "Th_single_ch", "duplicate_spike_ms",
        "nearest_chans", "nearest_templates", "max_channel_distance", "nt", "nskip",
        "whitening_range",
    )
    compact_ops = {}
    for arm, arm_row in ops["arms"].items():
        effective = arm_row["effective"]
        compact_ops[arm] = {
            "ops_sha256": arm_row["ops"]["sha256"],
            "scalars": {key: effective[key] for key in scalar_names},
            "dshift": effective["dshift"],
            "arrays": {key: {name: effective[key][name] for name in ("shape", "dtype", "sha256_memory")}
                       for key in ("chanMap", "xc", "yc")},
            "saved_spike_times": arm_row["saved_spike_times"],
            "detection_template_ids": arm_row["detection_template_ids"],
            "snapshot_shapes": {name: row["shape"] for name, row in arm_row["snapshots"].items()},
            "ancestry_checks": arm_row["ancestry_checks"],
        }

    screen, panel = json.loads(screen_path.read_text()), json.loads(panel_path.read_text())
    helper = Path(result["contracts"]["tier1_helper"]["path"])
    return {
        "schema": "en-common-support-crop-hub-packet-v1", "status": "complete",
        "bindings": {
            "result": {"path": str(result_path.resolve()), "sha256": sha256(result_path)},
            "evaluator": {"path": str(evaluator_path.resolve()), "sha256": sha256(evaluator_path)},
            "tier1_helper": {"path": str(helper.resolve()), "sha256": sha256(helper)},
            "ops_audit": {"path": str(ops_path.resolve()), "sha256": sha256(ops_path)},
        },
        "state_metrics_32_nonzero_exposure_rows": state_records,
        "factorial_state_contrasts_16_nonzero_exposure_rows": contrast_records,
        "effective_ops_compact": {
            "arms": compact_ops,
            "invariant_checks": ops["invariant_effective_values_equal"],
            "support_geometry_checks": ops["support_matched_geometry_equal"],
            "intended_support_differences": ops["intended_support_differences"],
        },
        "field_time_support": field_support(screen, panel),
        "source_excerpts": {
            "rho_fixed_eligibility_and_calculation": excerpt(helper, 511, 545),
            "correspondence_matching": excerpt(helper, 210, 260),
            "correspondence_loss_new_ambiguity": excerpt(helper, 272, 300),
        },
        "interpretation_scope": (
            "Common-domain panels are arm-local unit-median populations, not matched-neuron effects. "
            "Near-total correspondence ambiguity further prevents a claim that a surviving biological "
            "population worsened. The defensible claim is limited to arm-local output metrics."
        ),
        "raw_voltage_bytes_read": 0, "sorts_launched": 0, "rf_accesses": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("testing/outputs/en_common_support_crop_evaluation_v1"))
    parser.add_argument("--ops-audit", type=Path, default=Path("testing/outputs/en_common_support_crop_ops_audit_v1.json"))
    parser.add_argument("--screen-config", type=Path, default=Path("configs/en_common_support_crop_screen.v1.json"))
    parser.add_argument("--panel-config", type=Path, default=Path("configs/en_tier1_panel.v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    packet = clean(build(args.root, args.ops_audit, args.screen_config, args.panel_config))
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(packet, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"status": packet["status"], "output": str(args.output.resolve()),
                      "bytes": args.output.stat().st_size, "sha256": sha256(args.output),
                      "state_rows": len(packet["state_metrics_32_nonzero_exposure_rows"]),
                      "contrast_rows": len(packet["factorial_state_contrasts_16_nonzero_exposure_rows"]),
                      "field_time_support": packet["field_time_support"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
