#!/usr/bin/env python3
"""No-voltage kernel audit for rounded W3 states not exercised in EM.2b W2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from testing.em2a_operator_screen import (  # noqa: E402
    ARM_SPECS,
    effective_kernel,
    interpolation_insertion,
)
from testing.em_huklaban5_voltage_equivalence import dd_exact_coordinate_mapping  # noqa: E402
from testing.em_huklaban5_voltage_equivalence_arms import array_sha256, preflight  # noqa: E402


W2 = Path(
    "/home/huklaban5/DARTsort_experiment_scratch/dd_lattice_w2_20260928/"
    "huklaban5_v2/lattice_W2_pair.csv"
)
W3 = Path(
    "/home/huklaban5/DARTsort_experiment_scratch/ef_w3_lattice_20260928/"
    "lattice_W3_pair.csv"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_row(
    q_um: float,
    source_geometry: np.ndarray,
    target_geometry: np.ndarray,
    get_kernel,
    insertion: np.ndarray,
    full_geometry: np.ndarray,
) -> dict[str, object]:
    mapping = dd_exact_coordinate_mapping(source_geometry, target_geometry, q_um)
    exact = np.zeros((len(source_geometry), len(target_geometry)), dtype=np.float32)
    supported = mapping >= 0
    exact[mapping[supported], np.flatnonzero(supported)] = 1.0
    effective, structural_zero = effective_kernel(
        get_kernel,
        insertion,
        full_geometry,
        target_geometry,
        q_um,
        ARM_SPECS["rounded_kriging"],
    )
    delta = effective.astype(np.float64) - exact.astype(np.float64)
    column_l1 = np.abs(delta).sum(axis=0)
    return {
        "q_um": float(q_um),
        "exact_zero_targets": int(np.count_nonzero(~supported)),
        "kriging_structural_zero_targets": int(np.count_nonzero(structural_zero)),
        "identical_columns_at_1e_6": int(
            np.count_nonzero(np.max(np.abs(delta), axis=0) <= 1e-6)
        ),
        "kernel_rms_delta": float(np.sqrt(np.mean(np.square(delta)))),
        "kernel_max_abs_delta": float(np.max(np.abs(delta))),
        "column_l1_p50": float(np.quantile(column_l1, 0.5)),
        "column_l1_p95": float(np.quantile(column_l1, 0.95)),
        "column_l1_max": float(np.max(column_l1)),
        "effective_column_sum_min": float(effective.sum(axis=0).min()),
        "effective_column_sum_max": float(effective.sum(axis=0).max()),
        "exact_kernel_sha256": array_sha256(exact),
        "effective_kernel_sha256": array_sha256(effective),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    import spikeinterface as si
    from spikeinterface.preprocessing import get_spatial_interpolation_kernel
    from spikeinterface.preprocessing.preprocessing_tools import get_kriging_channel_weights

    state = preflight(si)
    insertion, insertion_receipt = interpolation_insertion(
        state["parent_ids"],
        state["full_ids"],
        state["source_geometry"],
        state["full_geometry"],
        get_kriging_channel_weights,
    )
    w2_states = sorted(pd.read_csv(W2).q_um.unique().astype(float))
    w3_states = sorted(pd.read_csv(W3).q_um.unique().astype(float))
    rows = [
        state_row(
            q,
            state["source_geometry"],
            state["target_geometry"],
            get_spatial_interpolation_kernel,
            insertion,
            state["full_geometry"],
        )
        for q in sorted(set(w2_states) | set(w3_states))
    ]
    table = pd.DataFrame(rows)
    table["in_w2"] = table.q_um.isin(w2_states)
    table["in_w3"] = table.q_um.isin(w3_states)
    table.to_csv(args.output / "STATE_KERNELS.csv", index=False)
    new_rows = table[table.in_w3 & ~table.in_w2]
    w2_rows = table[table.in_w2]
    envelope = {}
    for metric in (
        "exact_zero_targets",
        "kriging_structural_zero_targets",
        "kernel_rms_delta",
        "kernel_max_abs_delta",
        "column_l1_p95",
        "column_l1_max",
    ):
        low, high = float(w2_rows[metric].min()), float(w2_rows[metric].max())
        envelope[metric] = {
            "w2_min": low,
            "w2_max": high,
            "new_w3_values": [float(value) for value in new_rows[metric]],
            "inside_w2_range": bool(new_rows[metric].between(low, high).all()),
        }
    result = {
        "schema": "em2c-w3-new-rounded-state-kernel-audit-v1",
        "status": "complete",
        "python": sys.executable,
        "spikeinterface": si.__version__,
        "voltage_read": False,
        "sort_launched": False,
        "w2_pair": {"path": str(W2), "sha256": sha256(W2), "states_um": w2_states},
        "w3_pair": {"path": str(W3), "sha256": sha256(W3), "states_um": w3_states},
        "new_w3_states_um": [float(value) for value in new_rows.q_um],
        "interpolation_insertion": insertion_receipt,
        "w2_metric_envelope": envelope,
        "new_state_inside_w2_envelope_all_metrics": all(
            item["inside_w2_range"] for item in envelope.values()
        ),
        "limitations": (
            "Geometry/kernel audit only. It does not establish W3 voltage equivalence, "
            "sorting equivalence, biological identity, or purity."
        ),
    }
    (args.output / "RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    products = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(args.output.iterdir())
    ]
    (args.output / "MANIFEST.json").write_text(
        json.dumps({"products": products}, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "COMPLETE.json").write_text(
        json.dumps(
            {"status": "complete", "manifest_sha256": sha256(args.output / "MANIFEST.json")},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
