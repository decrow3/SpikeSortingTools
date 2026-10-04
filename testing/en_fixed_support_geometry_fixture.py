#!/usr/bin/env python3
"""Independent known-answer fixture for EN Arm-A fixed-support crop order.

No production remap helper and no voltage are used.  The fixture independently
implements the documented target (x, y) -> source (x, y + q) lookup.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


POSITIONS = Path(
    "/mnt/NPX/Luke/20250804/"
    "rescue_pipeline_results_Luke0804_V2V1_g0_imec0/kilosort4/"
    "sorter_output/channel_positions.npy"
)
OUT = (
    Path(__file__).resolve().parent
    / "outputs/en_executed_provenance_mechanism_audit_20260930_v1/"
    "fixed_support_geometry_fixture.json"
)
A_STATES_UM = (-280, -240, -200, -160, -120, -80, -40, 0, 40)
LOW_UM = 280
HIGH_UM = 3780


def coordinate_lookup(source: np.ndarray, targets: np.ndarray, q_um: int) -> np.ndarray:
    """Return exact source row for each target, or -1, independently."""
    lookup = {(float(x), float(y)): i for i, (x, y) in enumerate(source)}
    if len(lookup) != len(source):
        raise AssertionError("source geometry coordinates are not unique")
    return np.asarray(
        [lookup.get((float(x), float(y + q_um)), -1) for x, y in targets],
        dtype=np.int64,
    )


def missing_rows(targets: np.ndarray, mapping: np.ndarray, q_um: int) -> list[dict]:
    rows = []
    for target_index in np.flatnonzero(mapping < 0):
        x, y = targets[target_index]
        rows.append(
            {
                "target_index_within_crop": int(target_index),
                "target_x_um": float(x),
                "target_y_um": float(y),
                "required_source_x_um": float(x),
                "required_source_y_um": float(y + q_um),
            }
        )
    return rows


def main() -> None:
    full_parent = np.load(POSITIONS)
    retained_mask = (full_parent[:, 1] >= LOW_UM) & (full_parent[:, 1] <= HIGH_UM)
    retained_targets = full_parent[retained_mask]
    prematurely_cropped_parent = retained_targets.copy()

    state_rows = []
    for q_um in A_STATES_UM:
        # Correct execution order: map all full-geometry targets from the full
        # parent, then keep the selected output rows.
        full_mapping = coordinate_lookup(full_parent, full_parent, q_um)
        correct = full_mapping[retained_mask]

        # Independent equivalent formulation: retained targets still look up
        # sources in the full parent.
        direct = coordinate_lookup(full_parent, retained_targets, q_um)
        if not np.array_equal(correct, direct):
            raise AssertionError("full-remap-then-restrict disagrees with direct lookup")

        # Negative control: source was incorrectly cropped before remapping.
        wrong = coordinate_lookup(
            prematurely_cropped_parent, retained_targets, q_um
        )
        state_rows.append(
            {
                "q_um": q_um,
                "correct_order_missing_count": int(np.sum(correct < 0)),
                "wrong_order_missing_count": int(np.sum(wrong < 0)),
                "wrong_order_missing": missing_rows(retained_targets, wrong, q_um),
            }
        )

    if retained_targets.shape != (352, 2):
        raise AssertionError(f"unexpected retained geometry {retained_targets.shape}")
    if any(row["correct_order_missing_count"] != 0 for row in state_rows):
        raise AssertionError("correct order has missing retained targets")
    if not any(
        row["q_um"] != 0 and row["wrong_order_missing_count"] > 0
        for row in state_rows
    ):
        raise AssertionError("negative control did not fail")

    result = {
        "schema": "en-fixed-support-geometry-known-answer-v1",
        "scope": "geometry only; no voltage and no event/detection claim",
        "position_source": str(POSITIONS),
        "lookup_contract": "target (x,y) reads source (x,y+q)",
        "a_states_um": list(A_STATES_UM),
        "retained_target_y_um_inclusive": [LOW_UM, HIGH_UM],
        "full_parent_channel_count": int(full_parent.shape[0]),
        "retained_target_channel_count": int(retained_targets.shape[0]),
        "correct_order": "full-parent remap, then target-channel restriction",
        "wrong_control": "crop parent source first, then remap",
        "states": state_rows,
        "verdict": {
            "correct_order_all_states_supported": True,
            "wrong_order_fails_nonzero_states": True,
        },
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
