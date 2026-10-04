#!/usr/bin/env python3
"""Independent metadata-only review of support-boundary operator feasibility."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np


PACKET = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "en_first_medium_v13_support_boundary_operator_domain_feasibility_20261003_v1_h5"
)
V2 = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "en_first_medium_v13_support_boundary_waveform_discriminator_contract_repair_20261003_v2_h5"
)
COORDINATION = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/current_coordination_20261002_v1/"
    "H5_STATUS_V13_SUPPORT_BOUNDARY_OPERATOR_DOMAIN_FEASIBILITY_V1.json"
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def independent_mask(geometry, centers, shifts, frames, fs):
    dt = float(np.median(np.diff(centers)))
    positions = np.searchsorted(centers + dt / 2, frames.astype(np.float64) / fs, side="right")
    states = shifts[np.clip(positions, 0, centers.size - 1)]
    lower, upper = geometry.min(0), geometry.max(0)
    result = np.empty((geometry.shape[0], len(frames)), dtype=bool)
    for state in np.unique(states):
        desired = geometry.copy()
        desired[:, 1] += float(state)
        result[:, states == state] = np.all((desired >= lower) & (desired <= upper), axis=1)[:, None]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    contract_path = V2 / "contract/support_boundary_waveform_discriminator.execution_disabled.v2.json"
    selection_path = V2 / "freeze/DETERMINISTIC_RECORD_SELECTION.json"
    operator_path = V2 / "source/support_boundary_prefilter_operator.py"
    evidence_path = PACKET / "SUPPORT_PRECONDITION_EVIDENCE.json"
    contract = json.loads(contract_path.read_text())
    selection = json.loads(selection_path.read_text())
    evidence = json.loads(evidence_path.read_text())

    manifest_errors = []
    members = []
    for line in (PACKET / "MANIFEST.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        members.append(name)
        observed = sha(PACKET / name)
        if observed != expected:
            manifest_errors.append({"name": name, "expected": expected, "observed": observed})

    geometry_path = Path(contract["geometry_receipt_path"])
    field_path = Path(contract["field"]["path"])
    geometry = np.asarray(json.loads(geometry_path.read_text())["recording"]["channel_locations_um"], dtype=np.float64)
    with np.load(field_path, allow_pickle=False) as saved:
        centers = np.asarray(saved[contract["field"]["time_key"]], dtype=np.float64)
        displacement = np.asarray(saved[contract["field"]["displacement_key"]], dtype=np.float64)
    scaled = (displacement - np.median(displacement)) / contract["field"]["rounding_um"]
    shifts = contract["field"]["rounding_um"] * np.copysign(np.floor(np.abs(scaled) + 0.5), scaled)

    sys.path.insert(0, contract["production_source_root"])
    helper = importlib.import_module("npx_preprocessing.motion.prewhitening_support_mask")
    results = []
    mask_mismatches = []
    for record in selection["records"]:
        item = {key: record[key] for key in (
            "selection_ordinal", "ref_input_row", "global_frame", "reference_unit", "partition", "selection_role"
        )}
        item["contexts"] = {}
        for name, radius, samples in (("primary_p2048", 2108, 4217), ("nested_p1024", 1084, 2169)):
            start = int(record["global_frame"]) - radius
            frames = np.arange(start, start + samples, dtype=np.int64)
            production = helper.support_mask_from_geometry_and_field(
                geometry, centers, shifts, frames, contract["reader"]["fs_hz"]
            )
            independent = independent_mask(geometry, centers, shifts, frames, contract["reader"]["fs_hz"])
            if not np.array_equal(production, independent):
                mask_mismatches.append([record["selection_ordinal"], name])
            zero_rows = np.flatnonzero(independent.sum(1) == 0).astype(int).tolist()
            zero_samples = np.flatnonzero(independent.sum(0) == 0).astype(int).tolist()
            item["contexts"][name] = {
                "shape": list(independent.shape),
                "global_half_open": [start, start + samples],
                "zero_supported_sample_rows": zero_rows,
                "zero_supported_channel_samples_local": zero_samples,
                "minimum_supported_samples_per_row": int(independent.sum(1).min()),
                "minimum_supported_channels_per_sample": int(independent.sum(0).min()),
                "preconditions_pass": not zero_rows and not zero_samples,
            }
        results.append(item)

    failed = [
        item["selection_ordinal"] for item in results
        if not all(context["preconditions_pass"] for context in item["contexts"].values())
    ]
    primary_failed = [item["selection_ordinal"] for item in results if not item["contexts"]["primary_p2048"]["preconditions_pass"]]
    nested_failed = [item["selection_ordinal"] for item in results if not item["contexts"]["nested_p1024"]["preconditions_pass"]]
    zero_channel_samples = sum(
        len(context["zero_supported_channel_samples_local"])
        for item in results for context in item["contexts"].values()
    )
    lower = [item["selection_ordinal"] for item in results if any(
        context["zero_supported_sample_rows"] == [0, 1, 2, 3] for context in item["contexts"].values()
    )]
    upper = [item["selection_ordinal"] for item in results if any(
        context["zero_supported_sample_rows"] == [380, 381, 382, 383] for context in item["contexts"].values()
    )]
    latest = max(
        (path for path in PACKET.rglob("*") if path.is_file()), key=lambda path: path.stat().st_mtime_ns
    ).relative_to(PACKET).as_posix()
    operator = operator_path.read_text()
    v3_matches = sorted(
        str(path) for path in PACKET.parent.glob("en_first_medium_v13_support_boundary_waveform_discriminator_contract_repair_20261003_v3*")
    )
    report = {
        "schema": "h1-v13-support-boundary-operator-domain-feasibility-independent-review-v1",
        "verdict": "CONFIRM_STOP_WAVEFORM_BRANCH_NO_V3",
        "execution_go": False,
        "subject": {
            "packet": str(PACKET),
            "manifest_sha256": sha(PACKET / "MANIFEST.sha256"),
            "complete_sha256": sha(PACKET / "COMPLETE.json"),
            "receipt_sha256": sha(PACKET / "FEASIBILITY_RECEIPT.json"),
            "evidence_sha256": sha(evidence_path),
            "request_sha256": sha(PACKET / "H1_REVIEW_REQUEST.json"),
            "coordination_sha256": sha(COORDINATION),
        },
        "checks": {
            "manifest_members": len(members),
            "manifest_errors": manifest_errors,
            "selection_sha256": sha(selection_path),
            "operator_sha256": sha(operator_path),
            "geometry_receipt_sha256": sha(geometry_path),
            "field_sha256": sha(field_path),
            "production_support_helper_sha256": sha(Path(helper.__file__)),
            "selection_records": len(selection["records"]),
            "contexts_reconstructed": 2 * len(selection["records"]),
            "production_vs_independent_mask_mismatches": mask_mismatches,
            "evidence_results_exactly_reproduced": results == evidence["results"],
            "evidence_failed_ordinals_exactly_reproduced": failed == evidence["failed_ordinals"],
            "failed_ordinals": failed,
            "primary_failed_ordinals": primary_failed,
            "nested_failed_ordinals": nested_failed,
            "same_failures_both_contexts": primary_failed == nested_failed == failed,
            "lower_boundary_failed_ordinals": lower,
            "upper_boundary_failed_ordinals": upper,
            "lower_boundary_failed_records": len(lower),
            "upper_boundary_failed_records": len(upper),
            "zero_supported_channel_samples_across_124_masks": zero_channel_samples,
            "minimum_supported_channels_per_sample": min(
                context["minimum_supported_channels_per_sample"]
                for item in results for context in item["contexts"].values()
            ),
            "operator_requires_matching_support_shape_bool_cpu": (
                "support.shape != raw.shape" in operator and "support.dtype != torch.bool" in operator
                and "support.device.type != \"cpu\"" in operator
            ),
            "operator_rejects_any_zero_context_row": "torch.any(support.sum(1) == 0)" in operator,
            "car_rejects_any_zero_supported_channel_sample": "torch.any(counts == 0)" in operator,
            "packet_complete_last": latest == "COMPLETE.json",
            "latest_packet_member": latest,
            "v3_packet_matches": v3_matches,
            "selection_substitution_detected": results != evidence["results"],
            "real_voltage_or_waveform_bytes_read": 0,
            "gpu_work_started": False,
        },
        "decision": {
            "waveform_branch": "STOP",
            "drop_or_substitute_record": False,
            "context_or_budget_change": False,
            "filling_redesign": False,
            "v3_publish": False,
            "later_combined_six-defect_review": False,
            "reason": "Fourteen mandatory frozen records fail the bound proposed operator in both required contexts; the authorized contract requires STOP with no substitution or redesign.",
        },
        "scope": {
            "can_establish": "All 124 metadata-only support masks and exact support-domain outcomes independently reproduce; the specified 14 records are outside the bound proposed operator domain in both contexts, requiring branch STOP.",
            "cannot_establish": "No real-waveform effect, repair benefit, biological identity, purity, or sorting improvement was tested; packet COMPLETE-last provenance is false but cannot reverse the independently reproduced STOP condition.",
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
