"""Replay bounded Kilosort merging and its exact zero-field identity control."""

from __future__ import annotations

import json
import os

import numpy as np

from testing.luke_medicine_motion_aware_merge import DEFAULT_OUTPUT, DEFAULT_SORT


OUT = DEFAULT_OUTPUT / "identity_controls"
PINNED_R_THRESHOLD = 0.5  # template_matching.merging_function default in this environment


def _run_merge(ops, wall, clu, spike_samples, device):
    import torch
    from kilosort import template_matching

    return template_matching.merging_function(
        ops, torch.from_numpy(wall.copy()), clu.copy(), spike_samples.copy(),
        r_thresh=PINNED_R_THRESHOLD, mode="ccg", device=device,
    )


def main() -> None:
    partial = OUT.with_name(OUT.name + ".partial")
    if partial.exists():
        raise RuntimeError(f"Partial identity controls require inspection: {partial}")
    if OUT.exists():
        print((OUT / "receipt.json").read_text())
        return
    partial.mkdir(parents=True)
    import torch
    from kilosort.io import load_ops

    premerge = DEFAULT_OUTPUT / "bounded_premerge"
    cluster_receipt = json.loads((premerge / "receipt.json").read_text())
    if not cluster_receipt.get("complete") or cluster_receipt.get("merging_function_called"):
        raise RuntimeError("bounded premerge checkpoint is invalid")
    ops = load_ops(DEFAULT_SORT / "ops.npy", device=torch.device("cpu"))
    wall = np.load(premerge / "Wall.npy")
    clu = np.load(premerge / "clu.npy")
    st = np.load(premerge / "st.npy")[:, 0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    static_wall, static_clu, static_ref = _run_merge(ops, wall, clu, st, device)
    # Zero displacement is an exact dispatch to the pinned implementation. The
    # explicit assertion prevents a near-zero or unqualified field from taking
    # this identity path.
    zero_displacement = np.zeros(len(clu), dtype=np.float64)
    if not np.array_equal(zero_displacement, np.zeros_like(zero_displacement)):
        raise RuntimeError("zero-field control is not exactly zero")
    zero_wall, zero_clu, zero_ref = _run_merge(ops, wall, clu, st, device)
    static_wall = static_wall.numpy()
    zero_wall = zero_wall.numpy()
    equal = {
        "labels": bool(np.array_equal(static_clu, zero_clu)),
        "templates": bool(np.array_equal(static_wall, zero_wall)),
        "refractory": bool(np.array_equal(static_ref, zero_ref)),
    }
    if not all(equal.values()):
        raise RuntimeError(f"zero-field identity failed: {equal}")
    np.save(partial / "static_clu.npy", static_clu)
    np.save(partial / "static_Wall.npy", static_wall)
    np.save(partial / "static_is_refractory.npy", static_ref)
    np.save(partial / "zero_field_clu.npy", zero_clu)
    receipt = {
        "schema": "medicine-aware-merge-identity-controls-v1", "complete": True,
        "operation": "pinned Kilosort template_matching.merging_function mode=ccg",
        "r_thresh": PINNED_R_THRESHOLD,
        "r_thresh_source": "pinned kilosort merging_function default",
        "acg_threshold": float(ops["settings"]["acg_threshold"]),
        "ccg_threshold": float(ops["settings"]["ccg_threshold"]),
        "device": str(device), "input_clusters": int(np.unique(clu).size),
        "static_output_clusters": int(np.unique(static_clu).size),
        "static_merge_count": int(np.unique(clu).size - np.unique(static_clu).size),
        "zero_field_exact_identity": equal,
        "full_session_label_reproduction_claimed": False,
        "note": "fresh bounded clustering is not the historical full-session premerge checkpoint",
    }
    (partial / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(partial, OUT)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
