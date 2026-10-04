#!/usr/bin/env python3
"""Independent read-only review of the execution-disabled H5 waveform contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


PACKET = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_v13_support_boundary_waveform_discriminator_contract_20261003_v1_h5")
LEDGER = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_v13_event_credit_ledger_result_20261003_v1_h5")
GEOMETRY_RECEIPT = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_arm_b_full_sort_result_20260930_v1/ARM_B_COMPLETE.json")
FIELD = Path("/mnt/NPX/Luke/20250804/shared_analysis/luke_improved_motion_two_machine_20260909_v1/estimation_huklaban1_v1/candidate_fields.npz")
SYNTHETIC = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/luke_v13_support_boundary_synthetic_falsifier_20261003_v1_h1")
SYNTHETIC_REVIEW = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/luke_v13_support_boundary_synthetic_falsifier_h5_independent_review_20261003_v1_h5")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def set_hash(values) -> str:
    return hashlib.sha256("".join(f"{int(value)}\n" for value in sorted(values)).encode()).hexdigest()


def array_hash(value) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def state_at(frames, centers, shifts, fs):
    dt = float(np.median(np.diff(centers)))
    positions = np.searchsorted(centers + dt / 2, np.asarray(frames, dtype=np.float64) / fs, side="right")
    return shifts[np.clip(positions, 0, shifts.size - 1)]


def support(geometry, shift):
    desired = geometry.copy()
    desired[:, 1] += float(shift)
    return np.all((desired >= geometry.min(axis=0)) & (desired <= geometry.max(axis=0)), axis=1)


def transitions(centers, shifts, fs, start, stop, geometry):
    dt = float(np.median(np.diff(centers)))
    result = []
    for index in np.flatnonzero(shifts[1:] != shifts[:-1]):
        frame = int(math.ceil(float(centers[index] + dt / 2) * fs))
        while state_at([frame - 1], centers, shifts, fs)[0] == state_at([frame], centers, shifts, fs)[0]:
            frame += 1
        before = float(state_at([frame - 1], centers, shifts, fs)[0])
        after = float(state_at([frame], centers, shifts, fs)[0])
        before_support = support(geometry, before)
        after_support = support(geometry, after)
        if start <= frame < stop and not np.array_equal(before_support, after_support):
            result.append({
                "first_new_state_global_frame": frame,
                "from_shift_um": before,
                "to_shift_um": after,
                "before_supported_rows": np.flatnonzero(before_support).tolist(),
                "after_supported_rows": np.flatnonzero(after_support).tolist(),
                "rows_losing_support": np.flatnonzero(before_support & ~after_support).tolist(),
                "rows_gaining_support": np.flatnonzero(~before_support & after_support).tolist(),
            })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    contract = json.loads((PACKET / "contract/support_boundary_waveform_discriminator.execution_disabled.v1.json").read_text())
    method = json.loads((PACKET / "method/METHOD_REUSE_AND_DELTA.json").read_text())
    exposure = json.loads((PACKET / "freeze/METADATA_SUPPORT_BOUNDARY_EXPOSURE.json").read_text())
    edge_freeze = json.loads((PACKET / "freeze/FROZEN_CASE_EDGES_AND_EVENT_SETS.json").read_text())

    manifest_errors = []
    manifest_members = []
    for line in (PACKET / "MANIFEST.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        manifest_members.append(name)
        observed = sha_file(PACKET / name)
        if observed != expected:
            manifest_errors.append({"name": name, "expected": expected, "observed": observed})

    ledger_rows = {}
    with (LEDGER / "output/EVENT_CREDIT_LEDGER.tsv").open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            ledger_rows.setdefault(row["edge_id"], []).append(row)
    edge_errors = []
    internal_edge_errors = []
    if edge_freeze["edge_ids"] != [item["edge"]["edge_id"] for item in edge_freeze["edge_event_sets"]] or len(set(edge_freeze["edge_ids"])) != 14:
        internal_edge_errors.append("edge_ids")
    for frozen in edge_freeze["edge_event_sets"]:
        count = int(frozen["matched_event_count"])
        arrays = [
            frozen["ref_input_rows"], frozen["candidate_input_rows"],
            frozen["ref_global_frames"], frozen["candidate_global_frames"], frozen["lags_frames"],
        ]
        if any(len(values) != count for values in arrays):
            internal_edge_errors.append(frozen["edge"]["edge_id"] + ":length")
        if any(candidate - ref != lag for ref, candidate, lag in zip(frozen["ref_global_frames"], frozen["candidate_global_frames"], frozen["lags_frames"])):
            internal_edge_errors.append(frozen["edge"]["edge_id"] + ":lag")
        if set_hash(frozen["ref_input_rows"]) != frozen["ref_set_sha256"] or set_hash(frozen["candidate_input_rows"]) != frozen["candidate_set_sha256"]:
            internal_edge_errors.append(frozen["edge"]["edge_id"] + ":hash")
    for frozen in edge_freeze["edge_event_sets"][:12]:
        edge_id = frozen["edge"]["edge_id"]
        rows = ledger_rows.get(edge_id, [])
        expected = {
            "matched_event_count": len(rows),
            "ref_input_rows": [int(row["ref_input_row"]) for row in rows],
            "candidate_input_rows": [int(row["candidate_input_row"]) for row in rows],
            "ref_global_frames": [int(row["ref_frame"]) for row in rows],
            "candidate_global_frames": [int(row["candidate_frame"]) for row in rows],
            "lags_frames": [int(row["lag_frames"]) for row in rows],
        }
        for field, value in expected.items():
            if frozen[field] != value:
                edge_errors.append(f"{edge_id}:{field}")
        if frozen["ref_set_sha256"] != set_hash(expected["ref_input_rows"]):
            edge_errors.append(f"{edge_id}:ref_hash")
        if frozen["candidate_set_sha256"] != set_hash(expected["candidate_input_rows"]):
            edge_errors.append(f"{edge_id}:candidate_hash")

    ref_partitions = {
        int(item["reference_unit"]): item
        for item in json.loads((LEDGER / "output/REF_ROW_PARTITIONS.json").read_text())
    }
    partition_errors = []
    selection_rows = []
    boundary_frames = np.asarray(
        [item["first_new_state_global_frame"] for item in exposure["support_changing_boundaries"]],
        dtype=np.int64,
    )
    for case in exposure["cases"]:
        unit = int(case["reference_unit"])
        if unit in ref_partitions:
            saved = ref_partitions[unit]["partitions"]
            for name, item in case["partitions"].items():
                expected = saved[name]
                if item["membership_row_ids"] != expected["row_ids"]:
                    partition_errors.append(f"{unit}:{name}:rows")
                if item["membership_count"] != expected["count"] or item["membership_sha256"] != expected["sha256"]:
                    partition_errors.append(f"{unit}:{name}:count_or_hash")
        for name, item in case["partitions"].items():
            membership = set(item["membership_row_ids"])
            for selected in item["frozen_waveform_selection"]:
                frame = int(selected["global_frame"])
                distance = int(np.min(np.abs(boundary_frames - frame)))
                role = "near_boundary" if distance <= 572 else "away_control"
                if int(selected["ref_input_row"]) not in membership or distance != int(selected["nearest_support_change_distance_frames"]) or role != selected["selection_role"]:
                    partition_errors.append(f"{unit}:{name}:selection")
                selection_rows.append((unit, name, int(selected["ref_input_row"]), frame, role))

    geometry = np.asarray(json.loads(GEOMETRY_RECEIPT.read_text())["recording"]["channel_locations_um"], dtype=np.float64)
    with np.load(FIELD, allow_pickle=False) as saved:
        centers = np.asarray(saved["time_s"], dtype=np.float64)
        displacement = np.asarray(saved["rigid_displacement_um"], dtype=np.float64)
    reference = float(np.median(displacement))
    scaled = (displacement - reference) / 40.0
    shifts = 40.0 * np.copysign(np.floor(np.abs(scaled) + 0.5), scaled)
    recomputed_boundaries = transitions(
        centers, shifts, float(contract["waveform_context"]["sampling_frequency_hz"]),
        *contract["waveform_context"]["crop_global_half_open"], geometry,
    )
    exposure_errors = []
    if recomputed_boundaries != exposure["support_changing_boundaries"]:
        exposure_errors.append("support_changing_boundaries")
    if array_hash(geometry) != contract["waveform_context"]["geometry_float64_sha256"]:
        exposure_errors.append("geometry_hash")
    for case in exposure["cases"]:
        for item in case["partitions"].values():
            for selected in item["frozen_waveform_selection"]:
                observed = float(state_at([selected["global_frame"]], centers, shifts, contract["waveform_context"]["sampling_frequency_hz"])[0])
                if observed.hex() != float(selected["shift_um"]).hex():
                    exposure_errors.append("selected_shift")

    selected_count = len(selection_rows)
    unique_records = len({row[2] for row in selection_rows})
    expected_bytes = selected_count * int(contract["resources"]["logical_bytes_per_record"])
    maximum_records = int(contract["resources"]["maximum_waveform_records"])
    maximum_bytes = int(contract["resources"]["logical_recording_read_bytes_max"])
    latest_name = max(
        (path for path in PACKET.rglob("*") if path.is_file()), key=lambda path: path.stat().st_mtime_ns
    ).relative_to(PACKET).as_posix()
    reused_source_errors = []
    for name, expected in method["reused_sources"].items():
        path = Path(name)
        if not path.is_file() or sha_file(path) != expected:
            reused_source_errors.append(name)
    accepted_review = method["accepted_h1_method_review"]
    accepted_review_errors = []
    accepted_path = Path(accepted_review["path"])
    if sha_file(accepted_path / "MANIFEST.sha256") != accepted_review["manifest_sha256"]:
        accepted_review_errors.append("manifest")
    if sha_file(accepted_path / "COMPLETE.json") != accepted_review["complete_sha256"]:
        accepted_review_errors.append("complete")
    synthetic_hashes = {
        "producer_manifest": sha_file(SYNTHETIC / "MANIFEST.sha256"),
        "producer_complete": sha_file(SYNTHETIC / "COMPLETE.json"),
        "independent_manifest": sha_file(SYNTHETIC_REVIEW / "MANIFEST.sha256"),
        "independent_complete": sha_file(SYNTHETIC_REVIEW / "COMPLETE.json"),
    }
    blockers = []
    if latest_name != "COMPLETE.json":
        blockers.append("COMPLETE_NOT_WRITTEN_LAST")
    if selected_count > maximum_records or expected_bytes > maximum_bytes:
        blockers.append("FROZEN_SELECTION_EXCEEDS_READ_BUDGET")
    blockers += [
        "NO_FROZEN_NUMERIC_REAL_WAVEFORM_METRICS_OR_TOLERANCES",
        "NO_BOUND_EXECUTABLE_FOR_CURRENT_VS_PREFILTER_REAL_DISCRIMINATOR",
        "NO_EXECUTED_KNOWN_ANSWER_FIXTURE_FOR_PROPOSED_PREFILTER_OPERATOR",
        "NO_FROZEN_OUTPUT_SCHEMA_OR_EXACT_RECONCILIATION_FOR_NEW_DISCRIMINATOR",
        "NO_FRESH_OUTPUT_NAMESPACE_OR_MANAGED_LAUNCH_BINDING",
        "SYNTHETIC_PRODUCER_AND_INDEPENDENT_REVIEW_NOT_HASH_BOUND_IN_CONTRACT",
    ]
    report = {
        "schema": "h1-v13-support-boundary-waveform-contract-independent-review-v1",
        "verdict": "BLOCK_REPAIR_AND_REREVIEW",
        "execution_go": False,
        "checks": {
            "manifest_errors": manifest_errors,
            "manifest_members": len(manifest_members),
            "execution_enabled_exact_false": contract.get("execution_enabled") is False,
            "ledger_edges_independently_reconciled": 12,
            "ledger_edge_errors": edge_errors,
            "all_14_internal_edge_errors": internal_edge_errors,
            "ref_partition_errors": partition_errors,
            "metadata_exposure_errors": exposure_errors,
            "support_changing_boundaries": len(recomputed_boundaries),
            "frozen_edges": len(edge_freeze["edge_event_sets"]),
            "frozen_selected_records": selected_count,
            "unique_selected_records": unique_records,
            "logical_bytes_required": expected_bytes,
            "maximum_waveform_records": maximum_records,
            "logical_bytes_allowed": maximum_bytes,
            "latest_packet_member": latest_name,
            "reused_source_errors": reused_source_errors,
            "accepted_predecessor_review_errors": accepted_review_errors,
            "synthetic_packet_hashes_observed": synthetic_hashes,
            "h5_local_ref190_arrays_opened_by_h1": False,
            "real_voltage_or_waveform_bytes_read": 0,
        },
        "blockers": blockers,
        "required_repair": [
            "republish in a fresh packet with COMPLETE written last",
            "make the frozen selected-record count and exact read budget agree without post-outcome substitution",
            "freeze explicit real-waveform measurements, formulas, aggregation, numerical tolerances, control gates, and deterministic three-way decision mapping",
            "bind an executable implementation for the current-vs-test-only-prefilter contrast, including support-aware CAR normalization and transition-discontinuity semantics",
            "bind and execute independent known-answer fixtures for that proposed operator before real voltage access",
            "freeze exact compact output schemas, per-record/read reconciliation, source/runtime/freshness checks, fresh output namespace, and one-start managed launch",
            "bind the accepted synthetic producer and H5 independent-review packet seals exactly",
        ],
        "scope": {
            "can_establish": "The 12 ledger-derived edges, four sealed REF partition families, geometry-derived 246 support transitions, selected-event exposure metadata, and execution-disabled state reconcile on H1.",
            "cannot_establish": "REF190 replay from H5-local arrays, readiness of a real-waveform discriminator, validity of a prefilter operator, or any repair/biological conclusion.",
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
