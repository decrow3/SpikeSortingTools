"""Thin, execution-gated composition of accepted saved-output qualification modules.

The module contains no sorter, voltage, RF, holdout, or parameter-search path.
It composes the accepted spatial loader with the accepted retention and
conditional-support measurements.  The published contract keeps real phases
disabled; saved-format synthetic fixtures may exercise the exact CLI path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
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
import testing.luke_amplitude_dropout_audit as amplitude_audit
import testing.sort_comparison as sort_comparison
from pipeline.config import fingerprint


CONTRACT_SCHEMA = "first-medium-saved-output-qualification-contract-v1"
REQUEST_SCHEMA = "first-medium-saved-output-qualification-request-v1"
FIXTURE_TOKEN = "PACKAGED_SAVED_FORMAT_FIXTURE_ONLY_V1"
PHASES = {"phase1_controls", "phase2_existing_b"}


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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _module_paths() -> dict[str, Path]:
    modules = {
        "testing.first_medium_saved_output_qualification": sys.modules[__name__],
        "testing.first_medium_measurement_redesign": measurement,
        "testing.first_medium_evaluator": spatial_loader,
        "testing.sort_comparison": sort_comparison,
        "testing.luke_amplitude_dropout_audit": amplitude_audit,
        "pipeline.config": pipeline_config,
        "pipeline.truncation": pipeline_truncation,
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


def _load_cohort(request: Mapping[str, Any], reference_clusters: np.ndarray) -> tuple[np.ndarray, str]:
    cohort = np.asarray(request.get("reference_cohort"))
    if cohort.ndim != 1 or not np.issubdtype(cohort.dtype, np.integer):
        raise ValueError("reference_cohort must be a one-dimensional integer array")
    cohort = cohort.astype(np.int64, copy=False)
    if not len(cohort) or len(np.unique(cohort)) != len(cohort):
        raise ValueError("reference_cohort must be nonempty and unique")
    if not set(cohort.tolist()) <= set(np.unique(reference_clusters).tolist()):
        raise ValueError("reference_cohort contains absent REF units")
    observed = _cohort_digest(cohort)
    if request.get("reference_cohort_sha256") != observed:
        raise RuntimeError("reference cohort digest mismatch")
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
    cohort, cohort_sha = _load_cohort(request, np.asarray(ref.sort["cl"]))
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
    ref, existing = (_load_arm(spec, contract) for spec in arms)
    cohort, cohort_sha = _load_cohort(request, np.asarray(ref.sort["cl"]))
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
    cells.to_csv(output / "PHASE2_FIXED_60S_DIAGNOSTICS.csv", index=False)
    _write_json(output / "PHASE2_EXISTING_B_REPORT.json", report)
    return report, {"REF384": dict(ref.provenance), "existing_corrected_B384": dict(existing.provenance)}


def _validate_gate(request: Mapping[str, Any], contract: Mapping[str, Any]) -> tuple[str, bool]:
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError(f"request schema must be {REQUEST_SCHEMA}")
    phase = request.get("phase")
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {sorted(PHASES)}")
    fixture = request.get("data_scope") == "synthetic_saved_format_fixture"
    phase_contract = contract["phases"][phase]
    if fixture:
        if request.get("fixture_token") != FIXTURE_TOKEN:
            raise RuntimeError("fixture token mismatch")
        if phase != "phase1_controls":
            raise RuntimeError("only phase1 is available to packaged fixtures")
    else:
        if not contract.get("execution_enabled") or not phase_contract.get("real_execution_enabled"):
            raise RuntimeError(f"{phase} real execution is disabled")
        required = phase_contract["required_tokens"]
        supplied = request.get("prerequisite_tokens", {})
        if supplied != required or any("PROSPECTIVE_" in str(value) for value in supplied.values()):
            raise RuntimeError(f"{phase} prerequisite tokens are absent or prospective")
        if supplied["fixed_ref_cohort_sha256"] != request.get("reference_cohort_sha256"):
            raise RuntimeError("fixed REF cohort token differs from the requested cohort")
    return phase, fixture


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
        or complete.get("status") not in {"COMPLETE_ONE_SHOT", "COMPLETE_FIXTURE"}
        or complete.get("manifest_sha256") != _sha256(phase1_manifest)
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
    if (
        control.get("label") != "DIAGNOSTIC_CONTROLS_ONLY_NOT_EFFICACY"
        or control.get("cross_arm_scalar_emitted") is not False
        or control.get("provenance", {}).get("reference_cohort_sha256")
        != tokens["fixed_ref_cohort_sha256"]
    ):
        raise RuntimeError("phase1 control report does not bind the frozen diagnostic cohort")
    freeze = Path(request.get("method_freeze_receipt", ""))
    if not freeze.is_absolute() or not freeze.is_file():
        raise ValueError("phase2 requires an absolute method_freeze_receipt")
    if _sha256(freeze) != tokens["method_freeze_token"]:
        raise RuntimeError("method-freeze token does not bind the supplied receipt")


def _write_manifest_and_complete(
    output: Path, started: float, phase: str, fixture: bool, contract: Mapping[str, Any],
) -> None:
    bounds = contract["phases"][phase]["resources"]
    wall = time.monotonic() - started
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    thread_keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")
    thread_values = {}
    for key in thread_keys:
        raw = os.environ.get(key)
        if raw is None or not raw.isdigit() or int(raw) <= 0:
            raise RuntimeError(f"explicit positive {key} is required")
        thread_values[key] = int(raw)
    if max(thread_values.values()) > int(bounds["cpu_threads_max"]):
        raise RuntimeError("CPU thread bound exceeded")
    if wall > float(bounds["wall_seconds_max"]):
        raise RuntimeError("wall-time bound exceeded")
    if peak_rss_kib * 1024 > int(bounds["peak_rss_bytes_max"]):
        raise RuntimeError("peak-RSS bound exceeded")
    resources = {
        "wall_seconds": wall,
        "wall_seconds_max": bounds["wall_seconds_max"],
        "peak_rss_kib": peak_rss_kib,
        "peak_rss_bytes_max": bounds["peak_rss_bytes_max"],
        "cpu_thread_environment": thread_values,
        "cpu_threads_max": bounds["cpu_threads_max"],
    }
    _write_json(output / "RESOURCE_RECEIPT.json", resources)
    members = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name not in {"MANIFEST.sha256", "COMPLETE.json"}
    )
    manifest = "".join(f"{_sha256(path)}  {path.name}\n" for path in members)
    (output / "MANIFEST.sha256").write_text(manifest)
    current_bytes = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    if current_bytes + 4096 > int(bounds["persistent_output_bytes_max"]):
        raise RuntimeError("persistent-output bound would be exceeded")
    complete = {
        "schema": "first-medium-saved-output-qualification-complete-v1",
        "status": "COMPLETE_FIXTURE" if fixture else "COMPLETE_ONE_SHOT",
        "phase": phase,
        "manifest_sha256": _sha256(output / "MANIFEST.sha256"),
        "scientific_advancement_authorized": False,
    }
    _write_json(output / "COMPLETE.json", complete)


def run(contract_path: Path, request_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"fresh output namespace required: {output}")
    contract = json.loads(contract_path.read_text())
    request = json.loads(request_path.read_text())
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(f"contract schema must be {CONTRACT_SCHEMA}")
    phase, fixture = _validate_gate(request, contract)
    if phase == "phase2_existing_b" and not fixture:
        _verify_phase2_receipts(request, request["prerequisite_tokens"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    _write_json(output / "STARTED.json", {
        "schema": "first-medium-one-start-receipt-v1",
        "phase": phase,
        "fixture": fixture,
        "contract_sha256": _sha256(contract_path),
        "request_sha256": _sha256(request_path),
        "retry_allowed": False,
    })
    try:
        implementation = _verify_implementation_boundary(contract)
        _write_json(output / "IMPLEMENTATION_PROVENANCE.json", implementation)
        if phase == "phase1_controls":
            _, bindings = _phase1_controls(request, contract, output)
        else:
            _, bindings = _phase2_existing_b(request, contract, output)
        _write_json(output / "ACTUAL_BINDING_RECEIPT.json", bindings)
        _write_manifest_and_complete(output, started, phase, fixture, contract)
    except Exception as error:
        _write_json(output / "FAILURE.json", {
            "schema": "first-medium-qualification-failure-v1",
            "phase": phase,
            "error_type": type(error).__name__,
            "error": str(error),
            "retry_allowed": False,
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.contract, args.request, args.output)


if __name__ == "__main__":
    main()
