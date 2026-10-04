"""Thin, execution-gated composition of accepted saved-output qualification modules.

The module contains no sorter, voltage, RF, holdout, or parameter-search path.
It composes the accepted spatial loader with the accepted retention and
conditional-support measurements.  The published contract keeps real phases
disabled; saved-format synthetic fixtures may exercise the exact CLI path.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import resource
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

import pipeline.config as pipeline_config
import pipeline.truncation as pipeline_truncation
import testing.first_medium_evaluator as spatial_loader
import testing.first_medium_measurement_redesign as measurement
import testing.qualification_managed_launcher as managed_launcher
import testing.luke_amplitude_dropout_audit as amplitude_audit
import testing.sort_comparison as sort_comparison
from pipeline.config import fingerprint


CONTRACT_SCHEMA = "first-medium-saved-output-qualification-contract-v1"
REQUEST_SCHEMA = "first-medium-saved-output-qualification-request-v1"
PHASES = {"phase1_controls", "phase2_existing_b", "phase3_repaired_repeat"}
REAL_EVIDENCE_KIND = "real_saved"
FIXTURE_EVIDENCE_KIND = "packaged_fixture"
_OUTPUT_ROOT: Path | None = None
_OUTPUT_BYTES_MAX: int | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_text(path: Path, text: str) -> None:
    data = text.encode()
    if _OUTPUT_ROOT is not None and _OUTPUT_BYTES_MAX is not None:
        resolved = path.resolve()
        if resolved.parent != _OUTPUT_ROOT.resolve():
            raise RuntimeError("all runtime outputs must be direct members of the bound output namespace")
        current = sum(p.stat().st_size for p in _OUTPUT_ROOT.iterdir() if p.is_file())
        replaced = path.stat().st_size if path.exists() else 0
        if current - replaced + len(data) > _OUTPUT_BYTES_MAX:
            raise RuntimeError("persistent-output bound would be exceeded")
    path.write_bytes(data)


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _module_paths() -> dict[str, Path]:
    modules = {
        "testing.first_medium_saved_output_qualification": sys.modules[__name__],
        "testing.first_medium_measurement_redesign": measurement,
        "testing.first_medium_evaluator": spatial_loader,
        "testing.sort_comparison": sort_comparison,
        "testing.luke_amplitude_dropout_audit": amplitude_audit,
        "pipeline.config": pipeline_config,
        "pipeline.truncation": pipeline_truncation,
        "testing.qualification_managed_launcher": managed_launcher,
    }
    result = {}
    for name, module in modules.items():
        path = Path(module.__file__).resolve() if module.__file__ else None
        if path is None or not path.is_file():
            raise RuntimeError(f"cannot resolve implementation module {name}")
        result[name] = path
    return result


def _verify_implementation_boundary(contract: Mapping[str, Any]) -> dict[str, Any]:
    expected = contract["implementation_boundary"]["active_module_sha256"]
    paths = _module_paths()
    if set(paths) != set(expected):
        raise RuntimeError("active module boundary names differ from frozen contract")
    receipt = {}
    for name, path in paths.items():
        observed = _sha256(path)
        if observed != expected[name]:
            raise RuntimeError(f"implementation hash mismatch for {name}")
        receipt[name] = {"path": str(path), "sha256": observed}
    return receipt


def _cohort_digest(cohort: np.ndarray) -> str:
    payload = np.ascontiguousarray(cohort.astype("<i8", copy=False)).tobytes()
    header = json.dumps({"dtype": "<i8", "shape": list(cohort.shape)}, sort_keys=True).encode()
    return hashlib.sha256(b"fixed-ref-cohort-v1\0" + header + payload).hexdigest()


def _load_cohort(request: Mapping[str, Any], contract: Mapping[str, Any], reference_clusters: np.ndarray) -> tuple[np.ndarray, str]:
    frozen = contract["fixed_ref_cohort"]
    cohort = np.asarray(frozen.get("ordered_cluster_ids"))
    if cohort.ndim != 1 or not np.issubdtype(cohort.dtype, np.integer):
        raise ValueError("reference_cohort must be a one-dimensional integer array")
    cohort = cohort.astype(np.int64, copy=False)
    if not len(cohort) or len(np.unique(cohort)) != len(cohort):
        raise ValueError("reference_cohort must be nonempty and unique")
    if not set(cohort.tolist()) <= set(np.unique(reference_clusters).tolist()):
        raise ValueError("reference_cohort contains absent REF units")
    observed = _cohort_digest(cohort)
    if frozen.get("cohort_sha256") != observed:
        raise RuntimeError("reference cohort digest mismatch")
    if request.get("reference_cohort_sha256") != observed:
        raise RuntimeError("request does not bind the frozen REF cohort")
    return cohort, observed


def _match_rule(contract: Mapping[str, Any]) -> measurement.MatchRule:
    settings = contract["measurement_settings"]
    return measurement.MatchRule(
        tolerance_frames=int(settings["correspondence_tolerance_frames"]),
        minimum_overlap=float(settings["minimum_correspondence_overlap"]),
        primary_retention=float(settings["bilateral_primary_retention"]),
    )


def _load_arm(spec: Mapping[str, Any], contract: Mapping[str, Any]) -> spatial_loader.ArmInput:
    return spatial_loader.load_arm_from_spec(
        spec,
        sampling_frequency_hz=float(contract["recording_domain"]["sampling_frequency_hz"]),
        spatial_region=contract["recording_domain"]["spatial_region"],
    )


def _conditional_aggregate(result: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "reference_cohort_units", "matched_units", "supported_units",
        "supported_unit_fraction_full_cohort",
        "reference_support_union_fraction_full_cohort_time",
        "candidate_support_union_fraction_full_cohort_time",
        "common_time_fraction_full_cohort_time",
        "conditional_median_missingness_difference_pp",
    )
    return {key: result[key] for key in keys}


def _cell_aggregate(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compact accepted per-unit cell rows without publishing unit rankings."""
    cells = result["cell_diagnostics"].groupby(
        ["cell_index", "start_frame", "stop_frame"], as_index=False
    )[["reference_events", "matched_reference_events"]].sum()
    cells["retention"] = np.where(
        cells.reference_events > 0,
        cells.matched_reference_events / cells.reference_events,
        np.nan,
    )
    return _jsonable(cells.to_dict(orient="records"))


def _windows_sha256(frame: pd.DataFrame) -> str:
    required = ["cluster_id", "status", "start_s", "end_s", "missing_pct"]
    if not set(required) <= set(frame.columns):
        raise ValueError("amplitude window provenance columns are incomplete")
    ordered = frame[required].sort_values(required, kind="stable", na_position="last")
    return hashlib.sha256(ordered.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def _phase1_controls(
    request: Mapping[str, Any], contract: Mapping[str, Any], output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    arms = request.get("arms", [])
    if len(arms) != 1 or arms[0].get("arm_id") != "REF384":
        raise ValueError("phase1 accepts exactly one REF384 arm and no candidate arm")
    ref = _load_arm(arms[0], contract)
    cohort, cohort_sha = _load_cohort(request, contract, np.asarray(ref.sort["cl"]))
    rule = _match_rule(contract)
    edges = contract["recording_domain"]["cell_edges_local_frames"]

    self_result = measurement.reference_event_retention(
        ref.sort, ref.sort, cohort, rule, cell_edges_frames=edges
    )
    if self_result["R"] != 1.0:
        raise RuntimeError("REF self control failed")

    unique = np.unique(np.asarray(ref.sort["cl"], dtype=np.int64))
    mapping = {int(cluster): int(10_000_000 + index) for index, cluster in enumerate(unique)}
    permuted_clusters = np.asarray([mapping[int(value)] for value in ref.sort["cl"]], dtype=np.int64)
    permuted = {"st": np.asarray(ref.sort["st"], dtype=np.int64), "cl": permuted_clusters}
    permutation_result = measurement.reference_event_retention(
        ref.sort, permuted, cohort, rule, cell_edges_frames=edges
    )
    if permutation_result["R"] != 1.0:
        raise RuntimeError("bijective label-permutation control failed")

    duplicated = {
        "st": np.repeat(np.asarray(ref.sort["st"], dtype=np.int64), 2),
        "cl": np.repeat(permuted_clusters, 2),
    }
    duplicate_result = measurement.reference_event_retention(
        ref.sort, duplicated, cohort, rule, cell_edges_frames=edges
    )
    cohort_event_count = int(np.isin(ref.sort["cl"], cohort).sum())
    if duplicate_result["R"] != 1.0 or duplicate_result["credited_candidate_events"] != cohort_event_count:
        raise RuntimeError("exclusive/no-double-credit control failed")

    sentinels = np.asarray([edges[0], *edges[1:-1], edges[-1] - 1], dtype=np.int64)
    observed_cells = measurement.half_open_cell_indices(sentinels, edges)
    expected_cells = np.r_[
        np.arange(len(edges) - 1, dtype=np.int64), len(edges) - 2
    ]
    if not np.array_equal(observed_cells, expected_cells):
        raise RuntimeError("half-open boundary assignment control failed")
    try:
        measurement.half_open_cell_indices([edges[-1]], edges)
    except ValueError:
        stop_rejected = True
    else:
        raise RuntimeError("half-open final stop was accepted")

    primary = pd.DataFrame({"baseline_cluster": cohort, "candidate_cluster": cohort})
    conditional = measurement.conditional_amplitude_diagnostic(
        ref.qc["amplitude_windows"], ref.qc["amplitude_windows"], primary, cohort,
        duration_s=float(contract["recording_domain"]["duration_s"]),
        minimum_valid_windows=int(contract["measurement_settings"]["minimum_valid_windows"]),
        minimum_common_time_fraction=float(
            contract["measurement_settings"]["minimum_common_time_fraction"]
        ),
    )
    conditional_aggregate = _conditional_aggregate(conditional)
    if conditional_aggregate["reference_support_union_fraction_full_cohort_time"] != conditional_aggregate["candidate_support_union_fraction_full_cohort_time"]:
        raise RuntimeError("REF conditional self-support union is not invariant")
    if conditional_aggregate["conditional_median_missingness_difference_pp"] not in (0.0, None):
        raise RuntimeError("REF conditional self difference is not zero")

    report = {
        "schema": "first-medium-phase1-control-report-v1",
        "label": "DIAGNOSTIC_CONTROLS_ONLY_NOT_EFFICACY",
        "scientific_score": None,
        "cross_arm_scalar_emitted": False,
        "candidate_ranked_output_emitted": False,
        "controls": {
            "ref_self_R_exact": 1.0,
            "bijective_label_permutation_R_exact": 1.0,
            "exclusive_duplicate_control_R_exact": 1.0,
            "credited_candidate_events": duplicate_result["credited_candidate_events"],
            "cohort_reference_events": cohort_event_count,
            "half_open_boundary_indices": observed_cells.tolist(),
            "final_stop_rejected": stop_rejected,
            "conditional_ref_self": conditional_aggregate,
        },
        "provenance": {
            "reference_cohort_sha256": cohort_sha,
            "reference_qc_request_digest": ref.qc["request_digest"],
            "reference_amplitude_windows_sha256": _windows_sha256(ref.qc["amplitude_windows"]),
            "reference_spatial_identity": ref.provenance["spatial_identity"],
        },
    }
    bindings = {"REF384": dict(ref.provenance)}
    _write_json(output / "PHASE1_CONTROL_REPORT.json", report)
    return report, bindings


def _phase2_existing_b(
    request: Mapping[str, Any], contract: Mapping[str, Any], output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    arms = request.get("arms", [])
    if [row.get("arm_id") for row in arms] != ["REF384", "existing_corrected_B384"]:
        raise ValueError("phase2 accepts ordered REF384 and existing_corrected_B384 arms only")
    ref = _load_arm(arms[0], contract)
    expected_ref = request.get("expected_phase1_ref_binding", {})
    for key in ("curated_identity_digest", "qc_request_digest", "spatial_identity"):
        if not expected_ref.get(key) or ref.provenance.get(key) != expected_ref[key]:
            raise RuntimeError(f"Phase-2 REF differs from successful Phase-1 REF: {key}")
    # B is deliberately not opened until the exact Phase-2 REF has been loaded
    # and compared with the successful real-control Phase-1 receipt.
    existing = _load_arm(arms[1], contract)
    cohort, cohort_sha = _load_cohort(request, contract, np.asarray(ref.sort["cl"]))
    result = measurement.reference_event_retention(
        ref.sort, existing.sort, cohort, _match_rule(contract),
        cell_edges_frames=contract["recording_domain"]["cell_edges_local_frames"],
    )
    cells = result["cell_diagnostics"].groupby(
        ["cell_index", "start_frame", "stop_frame"], as_index=False
    )[["reference_events", "matched_reference_events"]].sum()
    cells["retention"] = np.where(
        cells.reference_events > 0,
        cells.matched_reference_events / cells.reference_events,
        np.nan,
    )
    conditional = measurement.conditional_amplitude_diagnostic(
        ref.qc["amplitude_windows"], existing.qc["amplitude_windows"],
        result["primary_matches"], cohort,
        duration_s=float(contract["recording_domain"]["duration_s"]),
        minimum_valid_windows=int(contract["measurement_settings"]["minimum_valid_windows"]),
        minimum_common_time_fraction=float(
            contract["measurement_settings"]["minimum_common_time_fraction"]
        ),
    )
    report = {
        "schema": "first-medium-phase2-existing-b-report-v1",
        "label": "ONE_SHOT_DESCRIPTIVE_EXISTING_B_NO_ADVANCEMENT",
        "R_existing_B": result["R"],
        "DeltaR": {"status": "UNAVAILABLE", "value": None, "reason": "no repaired arm is bound"},
        "conditional_support": _conditional_aggregate(conditional),
        "advancement_verdict": None,
        "candidate_ranked_output_emitted": False,
        "reference_cohort_sha256": cohort_sha,
    }
    _write_text(output / "PHASE2_FIXED_60S_DIAGNOSTICS.csv", cells.to_csv(index=False))
    _write_json(output / "PHASE2_EXISTING_B_REPORT.json", report)
    return report, {"REF384": dict(ref.provenance), "existing_corrected_B384": dict(existing.provenance)}


def _conditional_for_pair(
    reference: spatial_loader.ArmInput,
    candidate: spatial_loader.ArmInput,
    retention: Mapping[str, Any], cohort: np.ndarray,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an explicit measured/unmeasured conditional-QC diagnostic."""
    if not bool(reference.qc.get("available")) or not bool(candidate.qc.get("available")):
        return {
            "status": "UNMEASURED",
            "reason": "truncation QC absent in one or both arms; no empty default, refit, or imputation",
            "value": None,
            "passes": None,
        }
    result = measurement.conditional_amplitude_diagnostic(
        reference.qc["amplitude_windows"], candidate.qc["amplitude_windows"],
        retention["primary_matches"], cohort,
        duration_s=float(contract["recording_domain"]["duration_s"]),
        minimum_valid_windows=int(contract["measurement_settings"]["minimum_valid_windows"]),
        minimum_common_time_fraction=float(
            contract["measurement_settings"]["minimum_common_time_fraction"]
        ),
    )
    return {"status": "MEASURED", "passes": None, **_conditional_aggregate(result)}


def _phase3_repaired_repeat(
    request: Mapping[str, Any], contract: Mapping[str, Any], output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compose accepted four-arm loaders and R/DeltaR without changing estimators."""
    arms = request.get("arms", [])
    expected_order = ["REF384", "existing_corrected_B384", "repaired_B384", "REF384_repeat"]
    if [row.get("arm_id") for row in arms] != expected_order:
        raise ValueError(f"phase3 requires exact arm order {expected_order}")

    # Load and compare REF before any candidate path is opened.
    ref = _load_arm(arms[0], contract)
    expected_ref = request.get("expected_phase1_ref_binding", {})
    for key in ("curated_identity_digest", "qc_request_digest", "spatial_identity"):
        if not expected_ref.get(key) or ref.provenance.get(key) != expected_ref[key]:
            raise RuntimeError(f"Phase-3 REF differs from successful Phase-1 REF: {key}")
    existing = _load_arm(arms[1], contract)
    repaired = _load_arm(arms[2], contract)
    repeat = _load_arm(arms[3], contract)

    cohort, cohort_sha = _load_cohort(request, contract, np.asarray(ref.sort["cl"]))
    rule = _match_rule(contract)
    uncertainty = contract["phase3_measurement"]
    delta = measurement.retention_difference(
        ref.sort, existing.sort, repaired.sort, cohort, rule,
        bootstrap_replicates=int(uncertainty["bootstrap_replicates"]),
        bootstrap_seed=int(uncertainty["bootstrap_seed"]),
        minimum_effect=float(uncertainty["minimum_effect"]),
    )
    cell_edges = contract["recording_domain"]["cell_edges_local_frames"]
    existing_retention = measurement.reference_event_retention(
        ref.sort, existing.sort, cohort, rule, cell_edges_frames=cell_edges
    )
    repaired_retention = measurement.reference_event_retention(
        ref.sort, repaired.sort, cohort, rule, cell_edges_frames=cell_edges
    )
    repeat_retention = measurement.reference_event_retention(
        ref.sort, repeat.sort, cohort, rule,
        cell_edges_frames=cell_edges,
    )

    lower = float(delta["bootstrap"]["one_sided_lower_95"])
    minimum_effect = float(delta["frozen_replacement_rule"]["minimum_effect"])
    delta_pass = bool(delta["frozen_replacement_rule"]["passes"])
    repeat_minimum = float(uncertainty["repeat_minimum_R"])
    repeat_pass = bool(float(repeat_retention["R"]) >= repeat_minimum)
    conditional = {
        "existing_corrected_B384": _conditional_for_pair(ref, existing, existing_retention, cohort, contract),
        "repaired_B384": _conditional_for_pair(ref, repaired, repaired_retention, cohort, contract),
        "REF384_repeat": _conditional_for_pair(ref, repeat, repeat_retention, cohort, contract),
    }
    qc_complete = all(value["status"] == "MEASURED" for value in conditional.values())
    waveform = {
        "status": "UNMEASURED",
        "passes": None,
        "reason": "accepted frozen waveform-method receipt is not bound to all scored pairs",
        "method_manifest_sha256": contract["phase3_measurement"]["waveform_method_manifest_sha256"],
        "pair_coverage": {
            "REF384__existing_corrected_B384": "UNMEASURED",
            "REF384__repaired_B384": "UNMEASURED",
            "REF384__REF384_repeat": "UNMEASURED",
        },
    }
    reasons = []
    if not delta_pass:
        reasons.append("DELTA_LOWER95_NOT_ABOVE_0_05")
    if not repeat_pass:
        reasons.append("REF_REPEAT_RETENTION_BELOW_0_98")
    if not qc_complete:
        reasons.append("QC_UNMEASURED")
    reasons.append("WAVEFORM_UNMEASURED")
    decision_status = "reject" if not delta_pass or not repeat_pass else "inconclusive"
    report = {
        "schema": "first-medium-compact-postrun-report-v1",
        "execution": {"status": "COMPLETED_PHASE3_COMPOSITION", "one_start": True,
                      "both_arms_complete": True, "terminal_output_cap_within_limit": True},
        "bindings": {"status": "PASS", "reference_cohort_sha256": cohort_sha,
                     "clock_valid": True, "row_ancestry_valid": True, "spatial_valid": True,
                     "qc_status_by_arm": {arm.arm_id: ("MEASURED" if arm.qc.get("available") else "UNMEASURED")
                                          for arm in (ref, existing, repaired, repeat)}},
        "primary_retention": {
            "status": "MEASURED", "R_existing": float(delta["R_existing"]),
            "R_repaired": float(delta["R_repaired"]), "DeltaR": float(delta["DeltaR"]),
            "lower95": lower, "minimum_effect": minimum_effect, "passes": delta_pass,
            "rule": "one-sided paired-REF-unit bootstrap lower95 > 0.05",
            "bootstrap_replicates": int(delta["bootstrap"]["replicates"]),
            "bootstrap_seed": int(delta["bootstrap"]["seed"]),
            "scope": "continuity proxy; not biological identity, purity, or recovery",
        },
        "repeatability": {
            "status": "MEASURED", "R_repeat": float(repeat_retention["R"]),
            "criterion": "R_repeat >= 0.98", "minimum_R": repeat_minimum,
            "passes": repeat_pass, "same_window": True,
            "scope": "same-window engineering continuity diagnostic only",
            "criterion_source_sha256": uncertainty["repeat_criterion_source_sha256"],
        },
        "diagnostics": {"fixed_60s_cells": {
                            "existing_corrected_B384": _cell_aggregate(existing_retention),
                            "repaired_B384": _cell_aggregate(repaired_retention),
                            "REF384_repeat": _cell_aggregate(repeat_retention),
                        },
                        "conditional_amplitude": conditional,
                        "guardrails": {"status": "NOT_COMPOSED_BY_PHASE3_THIN_PATH"}},
        "waveform": waveform,
        "decision": {"status": decision_status, "advance": False, "reason_codes": reasons,
                     "language_scope": "execution success is not scientific advancement"},
        "provenance": {"request_sha256": request.get("_request_sha256"),
                       "source_sha256": {name: _sha256(path) for name, path in sorted(_module_paths().items())},
                       "input_receipts": [arm.provenance for arm in (ref, existing, repaired, repeat)],
                       "manifest_sha256": None},
    }
    _write_json(output / "PHASE3_COMPACT_REPORT.json", report)
    bindings = {arm.arm_id: dict(arm.provenance) for arm in (ref, existing, repaired, repeat)}
    return report, bindings


def _validate_gate(request: Mapping[str, Any], contract: Mapping[str, Any]) -> str:
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError(f"request schema must be {REQUEST_SCHEMA}")
    phase = request.get("phase")
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {sorted(PHASES)}")
    if "fixture_token" in request or request.get("data_scope") == "synthetic_saved_format_fixture":
        raise RuntimeError("the production CLI has no fixture mode")
    if any(arm.get("evidence_kind") != REAL_EVIDENCE_KIND for arm in request.get("arms", [])):
        raise RuntimeError("production requests require real_saved evidence for every arm")
    phase_contract = contract["phases"][phase]
    if not contract.get("execution_enabled") or not phase_contract.get("real_execution_enabled"):
        raise RuntimeError(f"{phase} real execution is disabled")
    required = phase_contract["required_tokens"]
    supplied = request.get("prerequisite_tokens", {})
    if supplied != required or any("PROSPECTIVE_" in str(value) for value in supplied.values()):
        raise RuntimeError(f"{phase} prerequisite tokens are absent or prospective")
    if supplied["fixed_ref_cohort_sha256"] != request.get("reference_cohort_sha256"):
        raise RuntimeError("fixed REF cohort token differs from the requested cohort")
    return phase


def _validate_fixture_request(request: Mapping[str, Any], contract: Mapping[str, Any], packet_root: Path) -> str:
    if request.get("schema") != REQUEST_SCHEMA or request.get("phase") != "phase1_controls":
        raise ValueError("packaged fixture supports only the frozen phase1 request")
    if any(arm.get("evidence_kind") != FIXTURE_EVIDENCE_KIND for arm in request.get("arms", [])):
        raise RuntimeError("fixture entry point requires packaged_fixture evidence")
    manifest = packet_root / "fixtures" / "FIXTURE_MANIFEST.sha256"
    for line in manifest.read_text().splitlines():
        digest, separator, relative = line.partition("  ")
        path = packet_root / relative
        if not separator or not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"packaged fixture member mismatch: {relative}")
    allowed = (packet_root / "fixtures" / "packaged_ref_v1").resolve()
    for arm in request["arms"]:
        for key in ("curated_output", "qc_dir"):
            path = Path(arm[key]).resolve()
            if allowed not in path.parents:
                raise RuntimeError("fixture path escaped the packet-owned fixture root")
    return "phase1_controls"


def _verify_phase2_receipts(request: Mapping[str, Any], tokens: Mapping[str, str]) -> None:
    phase1 = Path(request.get("phase1_receipt_dir", ""))
    if not phase1.is_absolute() or not phase1.is_dir():
        raise ValueError("phase2 requires an absolute existing phase1_receipt_dir")
    phase1_manifest = phase1 / "MANIFEST.sha256"
    phase1_complete = phase1 / "COMPLETE.json"
    if _sha256(phase1_manifest) != tokens["phase1_manifest_sha256"]:
        raise RuntimeError("phase1 MANIFEST token mismatch")
    if _sha256(phase1_complete) != tokens["phase1_complete_sha256"]:
        raise RuntimeError("phase1 COMPLETE token mismatch")
    complete = json.loads(phase1_complete.read_text())
    if (
        complete.get("phase") != "phase1_controls"
        or complete.get("status") != "COMPLETE_ONE_SHOT"
        or complete.get("receipt_type") != "REAL_CONTROL"
        or complete.get("fixture") is not False
        or complete.get("manifest_sha256") != _sha256(phase1_manifest)
        or complete.get("contract_sha256") != tokens["phase1_contract_sha256"]
        or complete.get("request_sha256") != tokens["phase1_request_sha256"]
    ):
        raise RuntimeError("phase1 completion receipt is not a successful bound phase1 result")
    for raw_line in phase1_manifest.read_text().splitlines():
        digest, separator, name = raw_line.partition("  ")
        if not separator or not name or Path(name).name != name:
            raise RuntimeError("phase1 manifest contains an invalid member")
        member = phase1 / name
        if not member.is_file() or _sha256(member) != digest:
            raise RuntimeError(f"phase1 manifest member mismatch: {name}")
    control = json.loads((phase1 / "PHASE1_CONTROL_REPORT.json").read_text())
    binding = json.loads((phase1 / "ACTUAL_BINDING_RECEIPT.json").read_text())
    if (
        control.get("label") != "DIAGNOSTIC_CONTROLS_ONLY_NOT_EFFICACY"
        or control.get("cross_arm_scalar_emitted") is not False
        or control.get("provenance", {}).get("reference_cohort_sha256")
        != tokens["fixed_ref_cohort_sha256"]
    ):
        raise RuntimeError("phase1 control report does not bind the frozen diagnostic cohort")
    ref = binding.get("REF384", {})
    expected_ref = request.get("expected_phase1_ref_binding", {})
    for key in (
        "curated_identity_digest", "qc_request_digest", "spatial_identity",
    ):
        if not expected_ref.get(key) or ref.get(key) != expected_ref[key]:
            raise RuntimeError(f"phase1 REF binding mismatch: {key}")
    freeze = Path(request.get("method_freeze_receipt", ""))
    if not freeze.is_absolute() or not freeze.is_file():
        raise ValueError("phase2 requires an absolute method_freeze_receipt")
    if _sha256(freeze) != tokens["method_freeze_token"]:
        raise RuntimeError("method-freeze token does not bind the supplied receipt")


def _verify_phase3_pair_receipts(request: Mapping[str, Any], tokens: Mapping[str, str]) -> dict[str, Any]:
    """Bind successful pair finalization before any Phase-3 arm is opened."""
    receipts = request.get("pair_receipts")
    if not isinstance(receipts, Mapping):
        raise ValueError("phase3 requires pair_receipts")
    expected = {
        "pair_complete": "pair_complete_sha256",
        "attempt_terminal": "pair_attempt_terminal_sha256",
    }
    loaded = {}
    for name, token_key in expected.items():
        path = Path(receipts.get(name, ""))
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise ValueError(f"phase3 {name} must be an absolute non-symlink file")
        if _sha256(path) != tokens[token_key]:
            raise RuntimeError(f"phase3 {name} hash mismatch")
        loaded[name] = json.loads(path.read_text())
    complete = loaded["pair_complete"]
    terminal = loaded["attempt_terminal"]
    if (
        complete.get("schema") != "en-first-medium-pair-authoritative-complete-v3"
        or complete.get("status") != "complete"
        or complete.get("arm_order") != ["REF384_repeat", "repaired_B384"]
        or complete.get("final_audit_read_only") is not True
        or complete.get("authoritative_success_published_by_no_growth_rename") is not True
        or complete.get("no_later_counted_write_permitted") is not True
        or int(complete.get("aggregate_persistent_output_bytes", -1))
        > int(complete.get("aggregate_persistent_output_limit_bytes", -2))
    ):
        raise RuntimeError("phase3 pair COMPLETE is not authoritative bounded v3 success")
    terminal_path = Path(receipts["attempt_terminal"])
    if (
        terminal.get("schema") != "en-first-medium-pair-attempt-terminal-pending-v3"
        or terminal.get("status") != "PENDING_FINAL_READ_ONLY_AUDIT"
        or terminal.get("both_arms_and_ordinary_writers_stopped") is not True
        or _sha256(terminal_path) != complete.get("terminal_attempt_receipt_sha256")
    ):
        raise RuntimeError("phase3 pair terminal receipt is not bound to authoritative success")
    return {"pair_complete": complete, "attempt_terminal_sha256": _sha256(terminal_path)}


def _resource_preflight(
    phase: str, contract: Mapping[str, Any], contract_path: Path,
    request_path: Path, output: Path,
) -> dict[str, Any]:
    bounds = contract["phases"][phase]["resources"]
    thread_keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")
    threads = {}
    for key in thread_keys:
        raw = os.environ.get(key)
        if raw is None or not raw.isdigit() or int(raw) <= 0:
            raise RuntimeError(f"explicit positive {key} is required")
        threads[key] = int(raw)
    if max(threads.values()) > int(bounds["cpu_threads_max"]):
        raise RuntimeError("CPU thread bound exceeded before input loading")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    nvidia_visible = os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
    if int(bounds["gpu_count"]) == 0 and (visible not in {"", "-1"} or nvidia_visible not in {"", "void", "none"}):
        raise RuntimeError("GPU visibility must be disabled before input loading")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) > int(bounds["cpu_threads_max"]):
        raise RuntimeError("managed CPU affinity exceeds the frozen bound")
    soft_as, hard_as = resource.getrlimit(resource.RLIMIT_AS)
    if soft_as == resource.RLIM_INFINITY or soft_as > int(bounds["peak_rss_bytes_max"]):
        raise RuntimeError("managed address-space limit is absent or exceeds the memory bound")
    envelope_path = Path(os.environ.get("QUALIFICATION_RESOURCE_ENVELOPE", ""))
    secret = os.environ.get("QUALIFICATION_RESOURCE_SECRET", "")
    if not envelope_path.is_absolute() or not envelope_path.is_file() or not secret:
        raise RuntimeError("reviewed managed resource envelope is required")
    envelope = json.loads(envelope_path.read_text())
    supplied_mac = envelope.pop("hmac_sha256", None)
    expected_mac = hmac.new(secret.encode(), json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode(), hashlib.sha256).hexdigest()
    expected = {
        "schema": "first-medium-managed-resource-envelope-v1",
        "phase": phase,
        "contract_sha256": _sha256(contract_path),
        "request_sha256": _sha256(request_path),
        "output": str(output.resolve()),
        "wall_seconds_max": bounds["wall_seconds_max"],
        "address_space_bytes_max": bounds["peak_rss_bytes_max"],
        "cpu_threads_max": bounds["cpu_threads_max"],
        "gpu_count": 0,
        "persistent_output_bytes_max": bounds["persistent_output_bytes_max"],
    }
    if not hmac.compare_digest(str(supplied_mac), expected_mac) or envelope != expected:
        raise RuntimeError("managed resource envelope mismatch")
    return {**expected, "cpu_affinity": affinity, "thread_environment": threads,
            "rlimit_as": [soft_as, hard_as], "gpu_visibility": {"CUDA_VISIBLE_DEVICES": visible, "NVIDIA_VISIBLE_DEVICES": nvidia_visible}}


def _identity_context(request: Mapping[str, Any], contract_path: Path, output: Path, phase: str) -> dict[str, Any]:
    paths = _module_paths()
    arms = []
    for arm in request.get("arms", []):
        arms.append({key: arm.get(key) for key in (
            "arm_id", "evidence_kind", "curated_output", "qc_dir",
            "expected_identity_digest", "expected_qc_request_digest",
        )})
    sanitized = json.loads(json.dumps(request))
    sanitized.pop("start_token_binding_sha256", None)
    return {
        "schema": "first-medium-single-use-token-binding-v1",
        "phase": phase,
        "contract_sha256": _sha256(contract_path),
        "request_identity_sha256": hashlib.sha256(json.dumps(sanitized, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "output_namespace": str(output.resolve()),
        "source_sha256": {name: _sha256(path) for name, path in sorted(paths.items())},
        "input_identities": arms,
    }


def _claim_start_token(
    request: Mapping[str, Any], contract: Mapping[str, Any], contract_path: Path,
    output: Path, phase: str,
) -> dict[str, Any]:
    token_key = {
        "phase1_controls": "phase1_start_token",
        "phase2_existing_b": "phase2_start_token",
        "phase3_repaired_repeat": "phase3_start_token",
    }[phase]
    token = str(request["prerequisite_tokens"][token_key])
    context = _identity_context(request, contract_path, output, phase)
    payload = json.dumps(context, sort_keys=True, separators=(",", ":")).encode()
    binding = hmac.new(token.encode(), payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(str(request.get("start_token_binding_sha256")), binding):
        raise RuntimeError("start token is not cryptographically bound to this exact run identity")
    registry = Path(contract["lifecycle"]["token_registry_root"])
    if not registry.is_absolute() or not registry.is_dir():
        raise RuntimeError("fixed start-token registry is absent")
    token_sha = hashlib.sha256(token.encode()).hexdigest()
    claim = registry / f"{token_sha}.claim.json"
    receipt = {**context, "token_sha256": token_sha, "binding_hmac_sha256": binding,
               "host": socket.gethostname(), "claim_path": str(claim)}
    fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        os.write(fd, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    return receipt


def _write_manifest_and_complete(
    output: Path, started: float, phase: str, fixture: bool, contract: Mapping[str, Any],
    contract_sha256: str, request_sha256: str, resource_preflight: Mapping[str, Any],
) -> None:
    bounds = contract["phases"][phase]["resources"]
    wall = time.monotonic() - started
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if wall > float(bounds["wall_seconds_max"]):
        raise RuntimeError("wall-time bound exceeded")
    if peak_rss_kib * 1024 > int(bounds["peak_rss_bytes_max"]):
        raise RuntimeError("peak-RSS bound exceeded")
    resources = {
        "wall_seconds": wall,
        "wall_seconds_max": bounds["wall_seconds_max"],
        "peak_rss_kib": peak_rss_kib,
        "peak_rss_bytes_max": bounds["peak_rss_bytes_max"],
        "cpu_thread_environment": resource_preflight["thread_environment"],
        "cpu_threads_max": bounds["cpu_threads_max"],
        "gpu_count": 0,
        "managed_external_limits": dict(resource_preflight),
    }
    _write_json(output / "RESOURCE_RECEIPT.json", resources)
    members = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name not in {"MANIFEST.sha256", "COMPLETE.json"}
    )
    manifest = "".join(f"{_sha256(path)}  {path.name}\n" for path in members)
    _write_text(output / "MANIFEST.sha256", manifest)
    current_bytes = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    if current_bytes + 4096 > int(bounds["persistent_output_bytes_max"]):
        raise RuntimeError("persistent-output bound would be exceeded")
    complete = {
        "schema": "first-medium-saved-output-qualification-complete-v1",
        "status": "COMPLETE_FIXTURE" if fixture else "COMPLETE_ONE_SHOT",
        "receipt_type": "PACKAGED_FIXTURE" if fixture else (
            "REAL_PHASE3_COMPOSITION" if phase == "phase3_repaired_repeat" else "REAL_CONTROL"
        ),
        "fixture": fixture,
        "phase": phase,
        "manifest_sha256": _sha256(output / "MANIFEST.sha256"),
        "contract_sha256": contract_sha256,
        "request_sha256": request_sha256,
        "scientific_advancement_authorized": False,
    }
    _write_json(output / "COMPLETE.json", complete)


def run(contract_path: Path, request_path: Path, output: Path, *, fixture_packet_root: Path | None = None) -> None:
    global _OUTPUT_ROOT, _OUTPUT_BYTES_MAX
    if output.exists():
        raise FileExistsError(f"fresh output namespace required: {output}")
    contract = json.loads(contract_path.read_text())
    request = json.loads(request_path.read_text())
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(f"contract schema must be {CONTRACT_SCHEMA}")
    fixture = fixture_packet_root is not None
    phase = (_validate_fixture_request(request, contract, fixture_packet_root)
             if fixture else _validate_gate(request, contract))
    if fixture:
        contract = json.loads(json.dumps(contract))
        request = json.loads(json.dumps(request))
        for arm in request["arms"]:
            arm["evidence_kind"] = REAL_EVIDENCE_KIND
        contract["fixed_ref_cohort"] = {
            "ordered_cluster_ids": request["reference_cohort"],
            "cohort_sha256": request["reference_cohort_sha256"],
        }
    resource_preflight = _resource_preflight(phase, contract, contract_path, request_path, output)
    if phase in {"phase2_existing_b", "phase3_repaired_repeat"}:
        _verify_phase2_receipts(request, request["prerequisite_tokens"])
    if phase == "phase3_repaired_repeat":
        _verify_phase3_pair_receipts(request, request["prerequisite_tokens"])
    token_claim = None if fixture else _claim_start_token(request, contract, contract_path, output, phase)
    # Keep this derived value out of the cryptographic request identity.  It is
    # inserted only after the single-use token is claimed and is report-only.
    request["_request_sha256"] = _sha256(request_path)
    output.mkdir(parents=True, exist_ok=False)
    _OUTPUT_ROOT = output
    _OUTPUT_BYTES_MAX = int(contract["phases"][phase]["resources"]["persistent_output_bytes_max"])
    started = time.monotonic()
    _write_json(output / "STARTED.json", {
        "schema": "first-medium-one-start-receipt-v1",
        "phase": phase,
        "fixture": fixture,
        "contract_sha256": _sha256(contract_path),
        "request_sha256": _sha256(request_path),
        "retry_allowed": False,
        "token_claim": token_claim,
        "resource_preflight": resource_preflight,
    })
    try:
        implementation = _verify_implementation_boundary(contract)
        _write_json(output / "IMPLEMENTATION_PROVENANCE.json", implementation)
        if phase == "phase1_controls":
            _, bindings = _phase1_controls(request, contract, output)
        elif phase == "phase2_existing_b":
            _, bindings = _phase2_existing_b(request, contract, output)
        else:
            _, bindings = _phase3_repaired_repeat(request, contract, output)
        _write_json(output / "ACTUAL_BINDING_RECEIPT.json", bindings)
        _write_manifest_and_complete(
            output, started, phase, fixture, contract,
            _sha256(contract_path), _sha256(request_path), resource_preflight,
        )
    except Exception as error:
        _write_json(output / "FAILURE.json", {
            "schema": "first-medium-qualification-failure-v1",
            "phase": phase,
            "error_type": type(error).__name__,
            "error": str(error),
            "retry_allowed": False,
        })
        raise
    finally:
        _OUTPUT_ROOT = None
        _OUTPUT_BYTES_MAX = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.contract, args.request, args.output)


if __name__ == "__main__":
    main()
