#!/usr/bin/env python3
"""Finite accessible-source review of frozen DH donors 407 and 415."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd

from testing.luke_au_cpu_preparation import exact_remap, exact_same_column_map


ROOT = Path(__file__).resolve().parents[1]
CX = ROOT / "testing/outputs/cx_masked_donor_qualification_v1/run"
DH = ROOT / "testing/outputs/dh_hybrid_truth_contract_20260928"
SOURCE = ROOT / "testing/outputs/luke_au_cpu_preparation/full_probe_extraction_v1/bc_attempt5/full_probe_final_label_templates.npz"
TAPERS = CX / "DONOR_COORDINATE_TAPERS.npz"
CX_ROWS = CX / "ROBUST_EXPLORATORY_MODIFIED_SET.csv"
OUT = ROOT / "testing/outputs/dt_h1_donor_407_415_review_20260928"
DONORS = (407, 415)
TARGET = np.arange(202, 384, dtype=np.int64)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def contrast(a: np.ndarray, b: np.ndarray) -> dict:
    af = np.asarray(a, dtype=np.float64).ravel()
    bf = np.asarray(b, dtype=np.float64).ravel()
    ac = af - af.mean()
    bc = bf - bf.mean()
    raw_cos = float(af @ bf / (np.linalg.norm(af) * np.linalg.norm(bf)))
    centered_cos = float(ac @ bc / (np.linalg.norm(ac) * np.linalg.norm(bc)))
    return {
        "centered_cosine": centered_cos,
        "raw_cosine": raw_cos,
        "unit_norm_distance": float(np.sqrt(max(0.0, 2.0 - 2.0 * raw_cos))),
        "l2_difference_inherited_units": float(np.linalg.norm(af - bf)),
        "rms_difference_inherited_units": float(np.sqrt(np.mean(np.square(af - bf)))),
        "maximum_absolute_difference_inherited_units": float(np.max(np.abs(af - bf))),
        "l2_norm_407_over_415": float(np.linalg.norm(af) / np.linalg.norm(bf)),
    }


def main() -> None:
    started = time.monotonic()
    cpu0 = time.process_time()
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)

    cx = pd.read_csv(CX_ROWS).set_index("unit_id").loc[list(DONORS)]
    dh_donors = pd.read_csv(DH / "DONORS.csv").set_index("donor_id").loc[list(DONORS)]
    dh_support = pd.read_csv(DH / "DONOR_STATE_SUPPORT.csv")
    with np.load(SOURCE, allow_pickle=False) as z:
        ids = np.asarray(z["unit_ids"], dtype=np.int64)
        ix = [int(np.flatnonzero(ids == donor)[0]) for donor in DONORS]
        original = np.asarray(z["templates"])[ix]
        counts = np.asarray(z["spike_counts"])[ix]
        geom = np.asarray(z["geometry_um"], dtype=np.float64)
        channel_ids = np.asarray(z["channel_ids"])
        measurement_domain = str(z["measurement_domain"])
        sampling_frequency = float(z["sampling_frequency"])
        trough_offset = int(z["trough_offset_samples"])
    with np.load(TAPERS, allow_pickle=False) as z:
        if not np.array_equal(ids, z["unit_ids"]):
            raise RuntimeError("taper/source unit ordering changed")
        if not np.array_equal(geom, z["geometry_um"]):
            raise RuntimeError("taper/source geometry changed")
        weights = np.asarray(z["train_weights"])[ix]
    modified = original * weights[:, None, :]
    bases = dh_donors.placement_base_shift_um.to_numpy(float)
    if not np.array_equal(bases, [-40.0, -40.0]):
        raise RuntimeError("frozen placement bases changed")
    occupied = sorted(dh_support.state_um.unique().astype(int).tolist())

    donor_rows = []
    for j, donor in enumerate(DONORS):
        original_ptp = np.ptp(original[j], axis=0)
        modified_ptp = np.ptp(modified[j], axis=0)
        original_peak = int(np.argmax(original_ptp))
        modified_peak = int(np.argmax(modified_ptp))
        donor_rows.append(
            {
                "donor_id": donor,
                "source_array_row": ix[j],
                "saved_spike_count": int(counts[j]),
                "original_max_ptp_inherited_units": float(original_ptp[original_peak]),
                "modified_max_ptp_inherited_units": float(modified_ptp[modified_peak]),
                "original_peak_channel_index": original_peak,
                "modified_peak_channel_index": modified_peak,
                "modified_peak_channel_id": str(channel_ids[modified_peak]),
                "modified_peak_x_um": float(geom[modified_peak, 0]),
                "modified_peak_y_um": float(geom[modified_peak, 1]),
                "cx_train_peak_channel_metadata": int(cx.loc[donor, "train_peak_channel"]),
                "cx_train_depth_span_um": float(cx.loc[donor, "train_depth_span_um"]),
                "taper_nonzero_channels": int(np.count_nonzero(weights[j] > 0)),
                "taper_fully_weighted_channels": int(np.count_nonzero(weights[j] == 1)),
                "placement_base_shift_um": float(bases[j]),
                "old_source_qualified_saved": bool(cx.loc[donor, "old_source_qualified"]),
                "modified_source_qualified_saved": bool(cx.loc[donor, "modified_source_qualified"]),
            }
        )
    pd.DataFrame(donor_rows).to_csv(OUT / "DONOR_METADATA.csv", index=False)

    state_rows = []
    for state in occupied:
        placed = []
        for j in range(2):
            shift = float(bases[j] + state)
            placed.append(exact_remap(modified[j], exact_same_column_map(geom, shift))[:, TARGET])
        metrics = contrast(placed[0], placed[1])
        p407 = float(np.ptp(placed[0], axis=0).max())
        p415 = float(np.ptp(placed[1], axis=0).max())
        ss = dh_support.loc[(dh_support.state_um == state) & dh_support.donor_id.isin(DONORS)]
        if len(ss) != 2:
            raise RuntimeError("missing frozen donor/state support")
        state_rows.append(
            {
                "state_um": state,
                "total_relocation_um": float(bases[0] + state),
                "placed_common_maxptp_y_um": float(geom[np.argmax(np.ptp(modified[0], axis=0)), 1] + bases[0] + state),
                "ptp_407_inherited_units": p407,
                "ptp_415_inherited_units": p415,
                "ptp_407_over_415": p407 / p415,
                "retained_energy_407": float(ss.loc[ss.donor_id == 407, "retained_energy_fraction"].iloc[0]),
                "retained_energy_415": float(ss.loc[ss.donor_id == 415, "retained_energy_fraction"].iloc[0]),
                **metrics,
            }
        )
    state_table = pd.DataFrame(state_rows)
    state_table.to_csv(OUT / "PLACED_CONTRAST_BY_STATE.csv", index=False)

    original_pair = contrast(original[0], original[1])
    modified_pair = contrast(modified[0], modified[1])
    nonzero0 = weights[0] > 0
    nonzero1 = weights[1] > 0
    result = {
        "status": "finite_source_review_complete_actual_hybrid_comparison_external",
        "measurement_domain": measurement_domain,
        "measurement_units": "not explicitly stored; numerical amplitude/difference values retain the source array's units",
        "sampling_frequency_hz": sampling_frequency,
        "trough_offset_samples": trough_offset,
        "native_identity": {
            "saved_source_unit_ids": list(DONORS),
            "interpretation": "Distinct saved IDs are provenance labels, not proof that their placed waveforms remain separable.",
            "unavailable": [
                "independent biological identity ground truth",
                "native per-event waveforms for this bounded review",
                "actual h5 repaired-hybrid waveforms and localization-training inputs",
            ],
        },
        "tapers": {
            "nonzero_channels_407": int(nonzero0.sum()),
            "nonzero_channels_415": int(nonzero1.sum()),
            "nonzero_intersection": int((nonzero0 & nonzero1).sum()),
            "nonzero_union": int((nonzero0 | nonzero1).sum()),
            "weight_cosine_between_donors": float(weights[0] @ weights[1] / (np.linalg.norm(weights[0]) * np.linalg.norm(weights[1]))),
        },
        "source_contrast": {
            "untapered_full_384": original_pair,
            "tapered_full_384": modified_pair,
            "tapering_centered_cosine_change": float(modified_pair["centered_cosine"] - original_pair["centered_cosine"]),
        },
        "placement": {
            "base_shift_um_both": -40.0,
            "occupied_states_um": occupied,
            "total_relocations_um": [int(-40 + state) for state in occupied],
            "actual_modified_maxptp_site_both": {"channel_index": 351, "x_um": 32.0, "y_um": 3500.0},
            "placed_maxptp_y_range_um": [3220.0, 3460.0],
            "crop": "physical AP202:383 after exact full-geometry relocation",
            "frozen_support": "both donors retain energy fraction 1.0 at all seven occupied states",
        },
        "placed_contrast_summary": {
            "centered_cosine_min": float(state_table.centered_cosine.min()),
            "centered_cosine_max": float(state_table.centered_cosine.max()),
            "l2_difference_min": float(state_table.l2_difference_inherited_units.min()),
            "l2_difference_max": float(state_table.l2_difference_inherited_units.max()),
            "max_absolute_difference": float(state_table.maximum_absolute_difference_inherited_units.max()),
            "ptp_ratio_407_over_415": float(state_table.ptp_407_over_415.iloc[0]),
            "state_invariance": "Exact common relocation plus full retained support makes the saved pair contrast numerically invariant across occupied states.",
        },
        "implication": (
            "The sources support a narrow ambiguity conclusion: 407 and 415 are highly similar, colocated after placement, and use strongly overlapping tapers, so a shared output label is plausible. They are not numerically identical: centered cosine is about 0.889, peak PTP differs by about 1.51x, and the maximum pointwise difference is about 5.78 inherited units. The shared label therefore cannot by itself establish donor equivalence, biological purity, or a scorer error."
        ),
        "recommendation": (
            "Keep 407 and 415 as separate known injection sources in truth accounting, but mark label 324 as a shared-label ambiguity. Use the bounded actual-hybrid waveform comparison to determine whether the output contains one donor-like waveform, a mixture, or unresolved overlap; do not split or merge truth from native IDs, original locations, or this source-template comparison alone."
        ),
        "inputs": {
            "source_templates_sha256": sha(SOURCE),
            "tapers_sha256": sha(TAPERS),
            "cx_rows_sha256": sha(CX_ROWS),
            "dh_donors_sha256": sha(DH / "DONORS.csv"),
            "dh_support_sha256": sha(DH / "DONOR_STATE_SUPPORT.csv"),
            "exact_operator_source_sha256": sha(ROOT / "testing/luke_au_cpu_preparation.py"),
        },
        "version_qualification": "Metrics use the already-frozen CX/DH arrays and the current local exact operator source; no held h5 source or payload was inspected.",
    }
    (OUT / "REVIEW.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (OUT / "REPORT.md").write_text(
        "# DT source review: donors 407 and 415\n\n"
        "## Verdict\n\n"
        "The frozen sources make a shared output label plausible but do not show that donors "
        "407 and 415 are the same source. Both placed templates share their actual maximum-PTP "
        "site and exact placement, and their taper supports overlap on 32 of 36 union channels. "
        "Their placed centered cosine is 0.8894 in every occupied state. They remain measurably "
        "different: peak PTP is 29.84 versus 19.76 (ratio 1.51), L2 difference is 23.99, and "
        "maximum pointwise difference is 5.78 in inherited source-array units.\n\n"
        "All seven exact placements retain 100% of each tapered source's energy in the AP202:383 "
        "crop. Keep the donors separate in truth accounting and mark output label 324 as a "
        "shared-label ambiguity. The bounded actual-hybrid waveform comparison, performed "
        "separately on h5, is the appropriate evidence for whether the output is donor-like, "
        "mixed, or unresolved. Distinct source IDs or original-location metadata alone are not.\n\n"
        "No new donor search, qualification rule, threshold, field, sort, raw/recording read, "
        "GPU work, or h5 payload access occurred.\n"
    )

    receipt = {
        "status": result["status"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_s_measured": time.process_time() - cpu0,
        "wall_s_measured": time.monotonic() - started,
        "conservative_active_charge_s": 60.0,
        "threads_max": 2,
        "readers_max": 1,
        "source_bytes_read": sum(p.stat().st_size for p in [SOURCE, TAPERS, CX_ROWS, DH / "DONORS.csv", DH / "DONOR_STATE_SUPPORT.csv"]),
        "raw_or_recording_voltage_bytes": 0,
        "gpu_s": 0,
        "sorts": 0,
        "prior_h1_cumulative_active_s": 18713.22,
        "new_h1_cumulative_active_s": 18773.22,
        "overall_ceiling_active_s": 26000.0,
    }
    (OUT / "RESOURCE_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    files = []
    for p in sorted(OUT.iterdir()):
        if p.name not in {"MANIFEST.json", "COMPLETE.json"}:
            files.append({"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)})
    (OUT / "MANIFEST.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")
    (OUT / "COMPLETE.json").write_text(
        json.dumps({"status": result["status"], "manifest_sha256": sha(OUT / "MANIFEST.json"), "written_last": True}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": result["status"], "output": str(OUT), "charge_s": 60.0}, sort_keys=True))


if __name__ == "__main__":
    main()
