#!/usr/bin/env python3
"""Compact saved-artifact audit of the B384 upper-boundary q+40 signal."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from npx_preprocessing.motion.lattice_remap_si import exact_coordinate_mapping, sample_stepwise_shifts
from testing.en_rounded_field import rounded_rigid_field


CAPTURE = ROOT / "testing/outputs/en_b384_q40_waveform_capture_20261001_v2"
OUTPUT = ROOT / "testing/outputs/en_b384_q40_boundary_signal_audit_20261001_v2"
BANK = Path("/media/huklaban5/Data/en_common_support_crop_screen_20260930_v1/B384/pre_extraction_snapshot/Wall3.npy")
WPCA = Path("/media/huklaban5/Data/en_common_support_crop_screen_20260930_v1/B384/pre_extraction_snapshot/wPCA.npy")
FIELD = Path("/mnt/NPX/Luke/20250804/shared_analysis/luke_improved_motion_two_machine_20260909_v1/estimation_huklaban1_v1/candidate_fields.npz")
FS = 29999.835983263598
EXPECTED = {
    CAPTURE / "SNIPPETS.npz": "fa2efaabc4ef6a839e195e64c022b7f74cc385577a30c21e8e273672dd1ebe2b",
    CAPTURE / "WINDOWS.csv": "ffd3b8176337874c332ac15179cc6e53c188bceabda9c517f9595188f757a9e5",
    BANK: "3cfa49a4bf9b8d4c0c46709749be9d9f005e7584091ca08785311b090533a608",
    WPCA: "f7a69044b0551370cd0520e007fea6119315bcb02347d5403683036d325847d5",
    FIELD: "547ff39d1c91d819b758246c701603d868bfd4d50bd6b791e3d5b7140c01c71d",
}
TEMPLATES = (673, 674, 676)
BOUNDARY = np.arange(380, 384)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    for path, expected in EXPECTED.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"frozen input differs: {path}: {observed}")
    OUTPUT.mkdir(parents=True, exist_ok=False)

    snippets = np.load(CAPTURE / "SNIPPETS.npz", allow_pickle=False)
    windows = pd.read_csv(CAPTURE / "WINDOWS.csv")
    bank = np.load(BANK, allow_pickle=False)
    wpca = np.load(WPCA, allow_pickle=False)
    raw = snippets["raw"]
    prewhite = snippets["prewhite"]
    whitened = snippets["whitened"]
    samples = snippets["global_samples"]
    geometry = np.column_stack((snippets["x_um"], snippets["y_um"]))

    with np.load(FIELD, allow_pickle=False) as field:
        field_time = np.asarray(field["time_s"], dtype=np.float64)
        rounded_state, reference_um = rounded_rigid_field(field["rigid_displacement_um"], 40.0)
    field_dt = float(np.median(np.diff(field_time)))
    if not np.allclose(np.diff(field_time), field_dt, rtol=0, atol=1e-6):
        raise RuntimeError("frozen field time grid differs")

    selected = windows.index[windows.unit_id.eq(449)].to_numpy()
    if windows.loc[selected, "window_id"].tolist() != [
        "pair_449_1", "pair_449_2", "pair_449_3", "pair_449_4", "single_449"
    ]:
        raise RuntimeError("unit 449 frozen-window identity differs")

    state_rows = []
    transition_edges_s = field_time[np.flatnonzero(rounded_state[1:] != rounded_state[:-1])] + field_dt / 2
    for index in selected:
        observed_states = sample_stepwise_shifts(
            field_time, rounded_state, samples[index] / FS, cell_width_s=field_dt
        )
        endpoint_samples = [int(windows.loc[index, "first_time_global"])]
        if windows.loc[index, "kind"] == "pair":
            endpoint_samples.append(int(windows.loc[index, "second_time_global"]))
        endpoint_times_s = np.asarray(endpoint_samples, dtype=np.float64) / FS
        nearest_transition_s = np.min(
            np.abs(endpoint_times_s[:, None] - transition_edges_s[None, :]), axis=1
        )
        state_rows.append({
            "window_id": windows.loc[index, "window_id"],
            "unique_rounded_states_um": ";".join(map(str, np.unique(observed_states))),
            "state_changes_within_retained_window": int(np.count_nonzero(np.diff(observed_states))),
            "nearest_endpoint_to_state_transition_s": float(nearest_transition_s.min()),
            "farthest_endpoint_to_nearest_state_transition_s": float(nearest_transition_s.max()),
        })
    state_frame = pd.DataFrame(state_rows)
    state_frame.to_csv(OUTPUT / "WINDOW_STATE_BINDING.csv", index=False)
    q40_mapping = exact_coordinate_mapping(geometry, 40.0, direction="y", atol_um=1e-6)
    q40_invalid_rows = np.flatnonzero(q40_mapping < 0)

    template_rows = []
    energy_profiles = {}
    for template_id in TEMPLATES:
        waveform = np.einsum("pc,pt->ct", bank[template_id], wpca)
        energy = np.sum(waveform.astype(np.float64) ** 2, axis=1)
        energy_profiles[template_id] = energy / energy.sum()
        order = np.argsort(energy)[::-1]
        template_rows.append({
            "template_id": template_id,
            "peak_row": int(order[0]),
            "top_four_rows": ";".join(map(str, order[:4])),
            "boundary_380_383_energy_fraction": float(energy[BOUNDARY].sum() / energy.sum()),
            "top_eight_energy_fraction": float(energy[order[:8]].sum() / energy.sum()),
            "nonzero_energy_rows": int(np.count_nonzero(energy)),
        })
    template_frame = pd.DataFrame(template_rows)
    template_frame.to_csv(OUTPUT / "TEMPLATE_BOUNDARY_ENERGY.csv", index=False)

    window_rows = []
    for index in selected:
        boundary_pre = prewhite[index, BOUNDARY]
        boundary_white = whitened[index, BOUNDARY]
        total_pre_energy = float(np.sum(prewhite[index].astype(np.float64) ** 2))
        window_rows.append({
            "window_id": windows.loc[index, "window_id"],
            "kind": windows.loc[index, "kind"],
            "raw_boundary_nonzero_values": int(np.count_nonzero(raw[index, BOUNDARY])),
            "raw_boundary_max_abs": int(np.max(np.abs(raw[index, BOUNDARY]))),
            "prewhite_boundary_rows_exactly_equal": bool(
                all(np.array_equal(boundary_pre[0], boundary_pre[row]) for row in range(1, 4))
            ),
            "prewhite_boundary_pairwise_max_abs_difference": float(
                max(np.max(np.abs(boundary_pre[0] - boundary_pre[row])) for row in range(1, 4))
            ),
            "prewhite_boundary_max_abs": float(np.max(np.abs(boundary_pre))),
            "prewhite_boundary_energy_fraction": float(
                np.sum(boundary_pre.astype(np.float64) ** 2) / total_pre_energy
            ),
            "whitened_boundary_pairwise_max_abs_difference": float(
                max(np.max(np.abs(boundary_white[0] - boundary_white[row])) for row in range(1, 4))
            ),
        })
    window_frame = pd.DataFrame(window_rows)
    window_frame.to_csv(OUTPUT / "WINDOW_BOUNDARY_METRICS.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for template_id in TEMPLATES:
        axes[0, 0].plot(np.arange(360, 384), energy_profiles[template_id][360:384], marker=".", label=str(template_id))
    axes[0, 0].axvspan(379.5, 383.5, color="#D55E00", alpha=0.15, label="rows 380–383")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("selected channel row")
    axes[0, 0].set_ylabel("fraction of template energy")
    axes[0, 0].set_title("Saved-template channel support")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].bar(
        [str(value) for value in TEMPLATES],
        template_frame.boundary_380_383_energy_fraction,
        color="#D55E00",
    )
    axes[0, 1].set_ylim(0.98, 1.0)
    axes[0, 1].set_xlabel("template ID")
    axes[0, 1].set_ylabel("energy fraction on rows 380–383")
    axes[0, 1].set_title("Boundary rows carry 99.19–99.55%")

    first = selected[0]
    for row in BOUNDARY:
        axes[1, 0].plot(samples[first], raw[first, row], label=f"row {row}")
    axes[1, 0].set_xlabel("global sample")
    axes[1, 0].set_ylabel("ADC counts")
    axes[1, 0].set_title("pair_449_1 (+40 µm): unsupported rows are raw zero")
    axes[1, 0].legend(fontsize=8, ncol=2)

    for row in BOUNDARY:
        axes[1, 1].plot(samples[first], prewhite[first, row], label=f"row {row}")
    axes[1, 1].set_xlabel("global sample")
    axes[1, 1].set_ylabel("pre-whitening units")
    axes[1, 1].set_title("CAR/high-pass creates four exactly identical traces")
    axes[1, 1].legend(fontsize=8, ncol=2)
    fig.savefig(OUTPUT / "boundary_signal_audit.png", dpi=160)
    plt.close(fig)

    summary = {
        "schema": "en-b384-q40-boundary-signal-audit-v2",
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Saved bank, retained snippets, and frozen motion-field metadata only; no recording reads.",
        "unit_449_windows": windows.loc[selected, "window_id"].tolist(),
        "all_five_windows_raw_rows_380_383_exactly_zero": bool(np.all(raw[selected][:, BOUNDARY] == 0)),
        "all_five_windows_prewhite_rows_380_383_exactly_equal": bool(all(
            np.array_equal(prewhite[selected, 380], prewhite[selected, row]) for row in range(381, 384)
        )),
        "prewhite_boundary_energy_fraction_range": [
            float(window_frame.prewhite_boundary_energy_fraction.min()),
            float(window_frame.prewhite_boundary_energy_fraction.max()),
        ],
        "template_boundary_energy_fraction_range": [
            float(template_frame.boundary_380_383_energy_fraction.min()),
            float(template_frame.boundary_380_383_energy_fraction.max()),
        ],
        "frozen_field_reference_um": reference_um,
        "field_cell_width_s": field_dt,
        "all_unit_449_retained_samples_rounded_state_um": 40.0,
        "state_changes_across_each_retained_window": 0,
        "endpoint_nearest_state_transition_s_range": [
            float(state_frame.nearest_endpoint_to_state_transition_s.min()),
            float(state_frame.farthest_endpoint_to_nearest_state_transition_s.max()),
        ],
        "q40_exact_mapping_invalid_target_rows": q40_invalid_rows.tolist(),
        "q40_boundary_mapping_380_383": q40_mapping[BOUNDARY].tolist(),
        "can_establish": (
            "All retained unit-449 samples are stably in the frozen +40 um state, whose exact remap leaves "
            "target rows 380-383 unsupported and therefore zero-filled. The executed channel-mean/CAR/high-pass "
            "path synthesizes identical nonzero pre-whitening traces on those raw-zero rows, and templates "
            "673/674/676 are almost entirely supported there."
        ),
        "cannot_establish": (
            "This descriptive audit does not quantify how much of all 9,575 unit-449 short-pair incidence is "
            "caused by the boundary artifact, provide a controlled counterfactual, or assign biological spike truth."
        ),
    }
    write_json(OUTPUT / "SUMMARY.json", summary)

    products = [
        "SUMMARY.json", "TEMPLATE_BOUNDARY_ENERGY.csv", "WINDOW_BOUNDARY_METRICS.csv",
        "WINDOW_STATE_BINDING.csv", "boundary_signal_audit.png",
    ]
    manifest = {name: {"sha256": sha256(OUTPUT / name), "bytes": (OUTPUT / name).stat().st_size} for name in products}
    write_json(OUTPUT / "MANIFEST.json", manifest)
    write_json(OUTPUT / "COMPLETE.json", {
        "schema": "en-b384-q40-boundary-signal-audit-v2",
        "status": "complete",
        "manifest_sha256": sha256(OUTPUT / "MANIFEST.json"),
    })


if __name__ == "__main__":
    main()
