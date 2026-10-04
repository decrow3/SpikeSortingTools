#!/usr/bin/env python3
"""Independent H1 recomputation using REF-only saved evidence; never opens B or voltage."""
from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
from pathlib import Path


SHARED = Path("/mnt/NPX/Luke/DARTsort_motion_experiments")
RESULT = SHARED / "en_first_medium_saved_output_qualification_phase1_controls_20261002_v1_h5"
SERVICE = SHARED / "en_first_medium_saved_output_qualification_phase1_service_receipt_20261002_v1_h5"
SUBJECT = SHARED / "en_first_medium_saved_output_qualification_contract_targeted_repair_20261002_v2_h1"
AUTH = SHARED / "en_first_medium_saved_output_qualification_h5_rereview_20261002_v2_h5"
BASELINE = (
    SHARED / "en_first_medium_evaluator_post_qc_execution_result_20261002_v1_h5"
    / "results/pairs/REF384__existing_corrected_B384/unit_metrics_baseline.csv"
)
CLAIM = (
    SHARED / ".qualification_token_registry/en_first_medium_saved_output_v1"
    / "1be9f3b632b75afe87b1da6e01655c5943bc87304f78dc3dc02afe9f3af4deff.claim.json"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def verify_manifest(root: Path) -> int:
    lines = (root / "MANIFEST.sha256").read_text().splitlines()
    for line in lines:
        expected, name = line.split("  ", 1)
        member = root / name
        if not member.is_file() or sha(member) != expected:
            raise RuntimeError(f"manifest mismatch: {root.name}/{name}")
    return len(lines)


def cohort_digest(ids: list[int]) -> str:
    # Exact independent implementation of the frozen little-endian int64 digest.
    payload = b"".join(int(value).to_bytes(8, "little", signed=True) for value in ids)
    header = json.dumps({"dtype": "<i8", "shape": [len(ids)]}, sort_keys=True).encode()
    return hashlib.sha256(b"fixed-ref-cohort-v1\0" + header + payload).hexdigest()


def main() -> None:
    expected = {
        "result_manifest": "8b73c931155741f84300f3cd3a5d260ae8bf0bcf2e05460a6f717c114712bb04",
        "result_complete": "797ae2d395e25e35f59e15447df7435dfda73a0839ed092fbb25786cdd97077d",
        "service_manifest": "1f48c3c119adbd87bb9fddc275f51b7b5047ba6dd9bda5124aa9479bb0435ac5",
        "service_complete": "a742d3589c0e014792942017fe4de94704bbe2d9806f17478ca9af7df027b457",
    }
    observed = {
        "result_manifest": sha(RESULT / "MANIFEST.sha256"),
        "result_complete": sha(RESULT / "COMPLETE.json"),
        "service_manifest": sha(SERVICE / "MANIFEST.sha256"),
        "service_complete": sha(SERVICE / "COMPLETE.json"),
    }
    if observed != expected:
        raise RuntimeError("delegated result/service identities differ")
    manifest_members = {
        "result": verify_manifest(RESULT),
        "service": verify_manifest(SERVICE),
    }

    cohort_contract = load(SUBJECT / "cohorts/REF384_COHORT.v1.json")
    contract = load(AUTH / "phase1/en_first_medium_saved_output_qualification.phase1_enabled.one_start.v1.json")
    request = load(AUTH / "phase1/PHASE1_REQUEST.one_start.v1.json")
    report = load(RESULT / "PHASE1_CONTROL_REPORT.json")
    binding = load(RESULT / "ACTUAL_BINDING_RECEIPT.json")["REF384"]
    complete = load(RESULT / "COMPLETE.json")
    service = load(SERVICE / "SERVICE_RECEIPT.json")
    claim = load(CLAIM)
    source = SUBJECT / "source/testing/first_medium_saved_output_qualification.py"
    measurement = SUBJECT / "source/testing/first_medium_measurement_redesign.py"
    source_text = source.read_text()
    measurement_text = measurement.read_text()

    if request["phase"] != "phase1_controls" or [a["arm_id"] for a in request["arms"]] != ["REF384"]:
        raise RuntimeError("request is not exactly REF-only Phase 1")
    arm = request["arms"][0]
    if not (
        complete["contract_sha256"] == sha(AUTH / "phase1/en_first_medium_saved_output_qualification.phase1_enabled.one_start.v1.json")
        and complete["request_sha256"] == sha(AUTH / "phase1/PHASE1_REQUEST.one_start.v1.json")
        and binding["curated_output"] == arm["curated_output"]
        and binding["qc_dir"] == arm["qc_dir"]
        and binding["curated_identity_digest"] == arm["expected_identity_digest"]
        and binding["qc_request_digest"] == arm["expected_qc_request_digest"]
        and binding["clock_normalization"]["saved_frame"] == arm["clock_normalization"]["saved_frame"]
        and binding["clock_normalization"]["global_frames_half_open"] == arm["clock_normalization"]["global_frames_half_open"]
        and binding["clock_normalization"]["local_frames_half_open"] == arm["clock_normalization"]["local_frames_half_open"]
        and binding["spatial_identity"] == arm["provenance"]["spatial_identity"]
    ):
        raise RuntimeError("actual REF binding does not reproduce the exact request")
    if sha(BASELINE) != "890fe7c1301001a11b62308b7d2954fc204d49bb56e33c281b288f3bd8be1842":
        raise RuntimeError("REF-only baseline table identity differs")
    selected: list[int] = []
    cohort_events = 0
    with BASELINE.open(newline="") as stream:
        for row in csv.DictReader(stream):
            depth = float(row["median_depth_um"])
            if 1600.0 <= depth <= 2180.0:
                selected.append(int(row["cluster_id"]))
                cohort_events += int(row["spike_count"])
    selected.sort()
    frozen = [int(value) for value in cohort_contract["ordered_cluster_ids"]]
    digest = cohort_digest(selected)
    if selected != frozen or selected != request["reference_cohort"]:
        raise RuntimeError("independent REF cohort differs from frozen request")
    if digest != "3e28d8c3f4d7dec73427ed778d83562e75c51a0a15a91230994fe3f5647e32c0":
        raise RuntimeError("independent cohort digest differs")
    if len(selected) != 61 or cohort_events != 227895:
        raise RuntimeError("independent cohort size/event total differs")

    domain = contract["recording_domain"]
    fs = float(domain["sampling_frequency_hz"])
    stop = int(domain["local_frames_half_open"][1])
    edges = [min(math.ceil(k * 60.0 * fs), stop) for k in range(11)]
    if edges != domain["cell_edges_local_frames"]:
        raise RuntimeError("independently calculated cell edges differ")
    sentinels = [edges[0], *edges[1:-1], edges[-1] - 1]
    cells = [bisect.bisect_right(edges, value) - 1 for value in sentinels]
    if cells != report["controls"]["half_open_boundary_indices"] or edges[-1] != stop:
        raise RuntimeError("half-open boundary known answer differs")

    controls = report["controls"]
    if [controls[key] for key in (
        "ref_self_R_exact", "bijective_label_permutation_R_exact",
        "exclusive_duplicate_control_R_exact",
    )] != [1.0, 1.0, 1.0]:
        raise RuntimeError("reported R control differs")
    if controls["cohort_reference_events"] != cohort_events or controls["credited_candidate_events"] != cohort_events:
        raise RuntimeError("reported event credit differs from independent REF total")
    conditional = controls["conditional_ref_self"]
    if not (
        conditional["reference_cohort_units"] == 61
        and conditional["matched_units"] == 61
        and conditional["supported_units"] == 4
        and conditional["supported_unit_fraction_full_cohort"] == 4 / 61
        and conditional["reference_support_union_fraction_full_cohort_time"]
            == conditional["candidate_support_union_fraction_full_cohort_time"]
            == conditional["common_time_fraction_full_cohort_time"]
        and conditional["conditional_median_missingness_difference_pp"] == 0.0
    ):
        raise RuntimeError("conditional REF-self invariant differs")
    if not (
        report["provenance"]["reference_cohort_sha256"] == digest
        and report["provenance"]["reference_qc_request_digest"] == arm["expected_qc_request_digest"]
        and report["provenance"]["reference_spatial_identity"] == arm["provenance"]["spatial_identity"]
        and len(report["provenance"]["reference_amplitude_windows_sha256"]) == 64
    ):
        raise RuntimeError("reported REF provenance differs from the bound request/cohort")

    required_source_facts = [
        "ref.sort, ref.sort, cohort, rule",
        "permuted = {\"st\": np.asarray(ref.sort[\"st\"]",
        "np.repeat(np.asarray(ref.sort[\"st\"]",
        "credited_candidate_events: set[tuple[int, int]] = set()",
        "if key in credited_candidate_events:",
        "candidate event received double credit",
        "ref.qc[\"amplitude_windows\"], ref.qc[\"amplitude_windows\"]",
        "cross_arm_scalar_emitted\": False",
    ]
    joined = source_text + "\n" + measurement_text
    if not all(fact in joined for fact in required_source_facts):
        raise RuntimeError("load-bearing source construction differs")
    provenance = load(RESULT / "IMPLEMENTATION_PROVENANCE.json")
    for item in provenance.values():
        path = Path(item["path"])
        if not path.is_file() or sha(path) != item["sha256"]:
            raise RuntimeError(f"source provenance mismatch: {path}")

    if not (
        service["service"]["start_count_issued"] == 1
        and service["service"]["restarts"] == 0
        and service["service"]["automatic_retry"] is False
        and service["service"]["result"] == "success"
        and service["result_packet"]["manifest_members_verified"] is True
        and sha(CLAIM) == service["token_claim"]["sha256"]
        and claim["phase"] == "phase1_controls"
        and claim["input_identities"][0]["arm_id"] == "REF384"
    ):
        raise RuntimeError("one-start/service/claim invariant differs")
    if not (
        report["cross_arm_scalar_emitted"] is False
        and report["candidate_ranked_output_emitted"] is False
        and report["scientific_score"] is None
        and not (RESULT / "PHASE2_EXISTING_B_REPORT.json").exists()
    ):
        raise RuntimeError("forbidden cross-arm output exists")

    print(json.dumps({
        "schema": "phase1-ref-only-independent-recompute-v1",
        "verdict": "PASS",
        "input_reads": {
            "result_and_service_receipts": True,
            "exact_contract_request_and_source": True,
            "ref_only_baseline_table": {"path": str(BASELINE), "sha256": sha(BASELINE)},
            "b_arm_files": False,
            "voltage": False,
        },
        "identity_verification": {"observed": observed, "manifest_members": manifest_members},
        "one_start_no_retry": True,
        "cohort": {"units": len(selected), "sha256": digest, "cluster_ids": selected},
        "event_credit": {"independent_ref_events": cohort_events, "reported_credited_once": controls["credited_candidate_events"]},
        "retention_controls": {"ref_self": 1.0, "bijective_label_permutation": 1.0, "exclusive_duplicate": 1.0,
            "structural_basis": "exact timestamp identity/bijection/duplication plus exclusive one-to-one set credit in bound source"},
        "boundaries": {"edges": edges, "sentinel_cells": cells, "final_stop_rejected": controls["final_stop_rejected"]},
        "conditional_ref_self": conditional,
        "provenance_source_hashes_verified": True,
        "actual_ref_binding_matches_request": True,
        "reported_ref_provenance_matches_request": True,
        "cross_arm_scalar_emitted": False,
        "candidate_ranked_output_emitted": False,
        "scientific_score": None,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
