#!/usr/bin/env python3
"""Compact, voltage-free REF provenance/matrix audit for the EN review.

This intentionally reads only metadata, saved Kilosort ops, channel maps, probe
positions, and whitening matrices. It does not read the recording binary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


REF = Path(
    "/mnt/NPX/Luke/20250804/"
    "rescue_pipeline_results_Luke0804_V2V1_g0_imec0/kilosort4"
)
SORTER = REF / "sorter_output"
OUT = (
    Path(__file__).resolve().parent
    / "outputs/en_executed_provenance_mechanism_audit_20260930_v1/inventory.json"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scalar(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    ops = np.load(SORTER / "ops.npy", allow_pickle=True).item()
    channel_map = np.load(SORTER / "channel_map.npy").reshape(-1)
    positions = np.load(SORTER / "channel_positions.npy")
    w = np.load(SORTER / "whitening_mat.npy")
    wdat = np.load(SORTER / "whitening_mat_dat.npy")
    winv = np.load(SORTER / "whitening_mat_inv.npy")

    # Kilosort applies Y = W @ X.  Summaries below characterize how much an
    # output channel can depend on other input channels; they do not claim that
    # any resulting signal crosses a detection threshold.
    diag = np.diag(w)
    off = w.copy()
    np.fill_diagonal(off, 0)
    off_l2 = np.linalg.norm(off, axis=1)
    total_l2 = np.linalg.norm(w, axis=1)
    eps = 1e-7
    support = np.sum(np.abs(off) > eps, axis=1)
    identity_error = w @ winv - np.eye(w.shape[0])

    y = positions[:, 1]
    edge_mask = (y <= 280) | (y >= 3540)
    lower_a_extreme_mask = y <= 260
    top_two_rows_mask = y >= 3800

    settings_keys = [
        "fs",
        "batch_size",
        "nblocks",
        "Th_universal",
        "Th_learned",
        "Th_single_ch",
        "nskip",
        "whitening_range",
        "highpass_cutoff",
        "artifact_threshold",
        "position_limit",
        "nearest_chans",
        "max_channel_distance",
    ]
    settings = ops.get("settings", {})
    result = {
        "schema": "en-executed-provenance-mechanism-audit-v1",
        "scope": "REF saved metadata and matrices only; no voltage read",
        "paths": {
            "ref": str(REF),
            "sorter_output": str(SORTER),
        },
        "hashes": {
            name: sha256(SORTER / name)
            for name in [
                "ops.npy",
                "channel_map.npy",
                "channel_positions.npy",
                "whitening_mat.npy",
                "whitening_mat_dat.npy",
                "whitening_mat_inv.npy",
            ]
        },
        "ops": {
            "fs": scalar(ops.get("fs")),
            "nblocks": scalar(ops.get("nblocks")),
            "do_CAR": scalar(ops.get("do_CAR")),
            "artifact_threshold": str(scalar(ops.get("artifact_threshold"))),
            "torch_device": scalar(ops.get("torch_device")),
            "filename": scalar(ops.get("filename")),
            "settings": {k: scalar(settings.get(k)) for k in settings_keys},
            "probe_chanMap_equal_saved_channel_map": bool(
                np.array_equal(np.asarray(ops["probe"]["chanMap"]), channel_map)
            ),
            "probe_x_equal_saved_positions": bool(
                np.array_equal(np.asarray(ops["xc"]), positions[:, 0])
            ),
            "probe_y_equal_saved_positions": bool(
                np.array_equal(np.asarray(ops["yc"]), positions[:, 1])
            ),
        },
        "channels": {
            "count": int(channel_map.size),
            "map_min": int(channel_map.min()),
            "map_max": int(channel_map.max()),
            "map_is_zero_based_permutation": bool(
                np.array_equal(np.sort(channel_map), np.arange(channel_map.size))
            ),
            "position_shape": list(positions.shape),
            "x_unique_um": np.unique(positions[:, 0]).tolist(),
            "y_min_um": float(y.min()),
            "y_max_um": float(y.max()),
            "y_unique_count": int(np.unique(y).size),
        },
        "whitening": {
            "shape": list(w.shape),
            "saved_W_equals_saved_Wdat_exactly": bool(np.array_equal(w, wdat)),
            "inverse_product_max_abs_error": float(np.max(np.abs(identity_error))),
            "diagonal_abs_median": float(np.median(np.abs(diag))),
            "offdiag_row_l2_median": float(np.median(off_l2)),
            "offdiag_fraction_of_row_l2_median": float(
                np.median(off_l2 / total_l2)
            ),
            "offdiag_support_gt_1e-7_min_median_max": [
                int(support.min()),
                float(np.median(support)),
                int(support.max()),
            ],
            "all_rows_have_nonzero_offdiag": bool(np.all(off_l2 > 0)),
            "edge_offdiag_fraction_median": float(
                np.median((off_l2 / total_l2)[edge_mask])
            ),
            "lower_A_extreme_offdiag_fraction_median": float(
                np.median((off_l2 / total_l2)[lower_a_extreme_mask])
            ),
            "top_3800_3820_offdiag_fraction_median": float(
                np.median((off_l2 / total_l2)[top_two_rows_mask])
            ),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
