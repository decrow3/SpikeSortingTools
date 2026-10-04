"""Four-arm, fail-closed adapter for the first medium trained comparison.

This module binds the already-qualified pairwise implementation in
``testing.sort_comparison`` to the four named arms.  Synthetic inputs are
supported only for known-answer qualification and are structurally excluded
from the scientific scorecard.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from pipeline.config import fingerprint
from testing.sort_comparison import compare_sorts


SCHEMA = "en-first-medium-four-arm-evaluator-v2"
ARM_ORDER = ("REF384", "existing_corrected_B384", "repaired_B384", "REF384_repeat")
PAIR_ORDER = (
    ("REF384", "existing_corrected_B384"),
    ("REF384", "repaired_B384"),
    ("existing_corrected_B384", "repaired_B384"),
    ("REF384", "REF384_repeat"),
)
VALID_EVIDENCE_KINDS = {"real_saved", "synthetic_fixture", "unavailable"}
MEASURED, UNMEASURED, UNRESOLVED = "MEASURED", "UNMEASURED", "UNRESOLVED"


@dataclass(frozen=True)
class ArmInput:
    arm_id: str
    evidence_kind: str
    sort: Mapping[str, Any] | None
    qc: Mapping[str, Any] | None
    provenance: Mapping[str, Any]
    waveform_evidence: Mapping[int, Mapping[str, np.ndarray]] | None = None

    def __post_init__(self) -> None:
        if self.arm_id not in ARM_ORDER:
            raise ValueError(f"unknown arm {self.arm_id!r}")
        if self.evidence_kind not in VALID_EVIDENCE_KINDS:
            raise ValueError(f"invalid evidence kind {self.evidence_kind!r}")
        has_arrays = self.sort is not None and self.qc is not None
        if self.evidence_kind == "unavailable" and has_arrays:
            raise ValueError("unavailable arms cannot carry arrays")
        if self.evidence_kind != "unavailable" and not has_arrays:
            raise ValueError("available arms require sort and QC arrays")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _endpoint(status: str, *, value: float | None = None, threshold: float | None = None,
              passes: bool | None = None, reason: str, **extra: Any) -> dict[str, Any]:
    if status not in {MEASURED, UNMEASURED, UNRESOLVED}:
        raise ValueError(status)
    if status != MEASURED and (value is not None or passes is not None):
        raise ValueError("unmeasured/unresolved endpoints cannot carry a numeric verdict")
    return {
        "status": status, "value": value, "threshold": threshold, "passes": passes,
        "reason": reason, **extra,
    }


def load_arm_from_spec(spec: Mapping[str, Any], *, sampling_frequency_hz: float) -> ArmInput:
    """Load one content-bound arm or retain an explicit unavailable row."""
    arm_id = str(spec["arm_id"])
    kind = str(spec["evidence_kind"])
    provenance = dict(spec.get("provenance", {}))
    if kind == "synthetic_fixture":
        raise ValueError("CLI specifications cannot deserialize synthetic arrays")
    if kind == "unavailable":
        return ArmInput(arm_id, kind, None, None, provenance)
    required_bindings = {
        "arm_contract_id", "input_identity", "clock_identity", "geometry_identity",
        "runtime_identity", "config_identity",
    }
    missing_bindings = sorted(required_bindings - set(provenance))
    if missing_bindings:
        raise ValueError(f"{arm_id}: unbound real-saved provenance {missing_bindings}")
    if provenance["arm_contract_id"] != arm_id:
        raise ValueError(f"{arm_id}: arm_contract_id mismatch")
    expected = spec.get("expected_identity_digest")
    expected_qc = spec.get("expected_qc_request_digest")
    if not isinstance(expected, str) or not expected or not isinstance(expected_qc, str) or not expected_qc:
        raise ValueError(f"{arm_id}: both saved-array and truncation-QC identity digests are required")
    curated = Path(spec["curated_output"])
    qc_dir = Path(spec["qc_dir"])
    sort, qc, clock_receipt = _load_normalized_comparison_inputs(
        arm_id, curated, qc_dir, spec.get("clock_normalization"),
        sampling_frequency_hz=sampling_frequency_hz,
    )
    if sort["identity_digest"] != expected:
        raise RuntimeError(f"{arm_id}: curated identity digest mismatch")
    if qc["request_digest"] != expected_qc:
        raise RuntimeError(f"{arm_id}: truncation-QC identity mismatch")
    provenance.update({
        "curated_output": str(curated.resolve()),
        "qc_dir": str(qc_dir.resolve()),
        "curated_identity_digest": sort["identity_digest"],
        "qc_request_digest": qc["request_digest"],
        "clock_normalization": clock_receipt,
    })
    waveform = None
    waveform_path = spec.get("waveform_evidence_npz")
    if waveform_path is not None:
        waveform = load_waveform_evidence(Path(waveform_path))
        provenance["waveform_evidence_sha256"] = _sha256(Path(waveform_path))
    return ArmInput(arm_id, kind, sort, qc, provenance, waveform)


def _load_normalized_comparison_inputs(
    arm_id: str,
    curated: Path,
    qc_dir: Path,
    clock: Any,
    *,
    sampling_frequency_hz: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Hash once, prove a declared clock transform, then validate saved arrays.

    A global-to-crop transform is accepted only when every exported time has
    the same exact integer offset from ``full_st[kept, 0]`` and both arrays
    obey their declared half-open domains.  No inferred or approximate offset
    is permitted.
    """
    from testing.luke_amplitude_dropout_audit import (
        build_windows_table,
        curated_arrays_from_raw,
        read_cached_truncation_qc,
        read_curated_arrays,
    )

    if not isinstance(clock, Mapping):
        raise ValueError(f"{arm_id}: explicit clock_normalization is required")
    saved_frame = clock.get("saved_frame")
    if saved_frame not in {"global_acquisition_samples", "crop_local_samples"}:
        raise ValueError(f"{arm_id}: unsupported saved_frame {saved_frame!r}")
    origin = int(clock.get("crop_origin_global_frame"))
    global_bounds = tuple(int(value) for value in clock.get("global_frames_half_open", ()))
    local_bounds = tuple(int(value) for value in clock.get("local_frames_half_open", ()))
    if len(global_bounds) != 2 or len(local_bounds) != 2:
        raise ValueError(f"{arm_id}: both clock domains must have two bounds")
    if global_bounds != (local_bounds[0] + origin, local_bounds[1] + origin):
        raise ValueError(f"{arm_id}: global/local clock domains do not share the declared origin")

    raw, hashes = read_curated_arrays(curated)
    original_times = np.asarray(raw["spike_times.npy"]).reshape(-1)
    if not np.issubdtype(original_times.dtype, np.integer):
        if not np.isfinite(original_times).all() or not np.equal(original_times, np.rint(original_times)).all():
            raise ValueError(f"{arm_id}: spike_times are not exact integer samples")
        original_times = original_times.astype(np.int64)
    else:
        original_times = original_times.astype(np.int64, copy=False)
    kept = np.asarray(raw["kept_spikes.npy"])
    full_st = np.asarray(raw["full_st.npy"])
    selected = full_st[kept]
    local_times = np.asarray(selected[:, 0])
    if not np.isfinite(local_times).all() or not np.equal(local_times, np.rint(local_times)).all():
        raise ValueError(f"{arm_id}: full_st kept times are not exact integer samples")
    local_times = local_times.astype(np.int64)
    if len(original_times) != len(local_times):
        raise ValueError(f"{arm_id}: clock arrays have unequal lengths")
    expected_offset = origin if saved_frame == "global_acquisition_samples" else 0
    if not np.array_equal(original_times - local_times, np.full(len(local_times), expected_offset, dtype=np.int64)):
        raise ValueError(f"{arm_id}: saved/full_st times do not have the exact declared offset")
    saved_bounds = global_bounds if saved_frame == "global_acquisition_samples" else local_bounds
    if len(original_times) and not (
        int(original_times.min()) >= saved_bounds[0] and int(original_times.max()) < saved_bounds[1]
    ):
        raise ValueError(f"{arm_id}: saved spike times fall outside the declared clock domain")
    if len(local_times) and not (int(local_times.min()) >= local_bounds[0] and int(local_times.max()) < local_bounds[1]):
        raise ValueError(f"{arm_id}: local full_st times fall outside the declared crop domain")

    normalized_raw = dict(raw)
    normalized_raw["spike_times.npy"] = local_times
    arrays = curated_arrays_from_raw(arm_id, normalized_raw)
    label_bytes = (curated / "cluster_KSLabel.tsv").read_bytes()
    labels_frame = pd.read_csv(io.BytesIO(label_bytes), sep="\t")
    label_column = next(column for column in labels_frame if column != "cluster_id")
    labels = dict(zip(labels_frame.cluster_id.astype(int), labels_frame[label_column].astype(str).str.lower()))
    identity = fingerprint({
        "curated": str(curated.resolve()), "files": hashes,
        "labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
    })
    sort = {"st": arrays.times, "cl": arrays.clusters, "amp": arrays.amplitudes,
            "labels": labels, "identity_digest": identity}
    positions_path = curated / "spike_positions.npy"
    if positions_path.is_file():
        positions = np.load(positions_path, mmap_mode="r")
        if positions.ndim != 2 or positions.shape[0] != len(arrays.times) or positions.shape[1] < 2:
            raise ValueError(f"invalid spike_positions.npy in {curated}")
        sort["depth"] = np.asarray(positions[:, 1])

    qc_path = qc_dir / "amp_truncation" / "truncation_qc.npz"
    if qc_path.is_file():
        cached, qc_hash = read_cached_truncation_qc(arm_id, qc_dir)
        windows = build_windows_table(arrays, cached, sampling_frequency_hz)
        qc = {"amplitude_windows": windows, "request_digest": qc_hash, "available": True}
    else:
        columns = ["cluster_id", "status", "start_s", "end_s", "missing_pct"]
        qc = {
            "amplitude_windows": pd.DataFrame(columns=columns),
            "request_digest": "UNMEASURED_ABSENT_TRUNCATION_QC",
            "available": False,
            "unmeasured_reason": "truncation_qc.npz absent; no value was regenerated or imputed",
        }
    normalized_contract = {
        "saved_frame": saved_frame,
        "crop_origin_global_frame": origin,
        "global_frames_half_open": list(global_bounds),
        "local_frames_half_open": list(local_bounds),
        "observed_exact_offset": expected_offset,
        "validated_rows": int(len(local_times)),
    }
    normalized_contract["digest"] = fingerprint(normalized_contract)
    return sort, qc, normalized_contract


def load_waveform_evidence(path: Path) -> dict[int, dict[str, np.ndarray]]:
    """Load compact per-unit epoch waveforms with explicit physical-channel axis.

    Required NPZ members are ``cluster_ids`` and three equally shaped arrays
    ``early``, ``middle`` and ``late`` of shape (unit, channel, sample).
    """
    with np.load(path, allow_pickle=False) as archive:
        required = {"cluster_ids", "early", "middle", "late"}
        if not required <= set(archive.files):
            raise ValueError(f"waveform evidence missing {sorted(required - set(archive.files))}")
        ids = np.asarray(archive["cluster_ids"])
        epochs = {name: np.asarray(archive[name], dtype=float) for name in ("early", "middle", "late")}
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("waveform cluster_ids must be unique and one-dimensional")
    shape = epochs["early"].shape
    if len(shape) != 3 or shape[0] != len(ids) or any(value.shape != shape for value in epochs.values()):
        raise ValueError("waveform epochs must share shape (unit, physical_channel, sample)")
    if not all(np.isfinite(value).all() for value in epochs.values()):
        raise ValueError("waveform evidence contains non-finite values")
    return {
        int(cluster): {name: epochs[name][index] for name in epochs}
        for index, cluster in enumerate(ids)
    }


def _exclusive_baseline_event_retention(report: Mapping[str, Any], baseline_sort: Mapping[str, Any]) -> dict[str, Any]:
    """Descriptive full-denominator retention; not the undefined improvement endpoint."""
    eligibility = report["baseline_eligibility"]
    eligible_ids = set(eligibility.baseline_cluster.astype(int))
    clusters = np.asarray(baseline_sort["cl"]).reshape(-1)
    denominator = int(np.isin(clusters, list(eligible_ids)).sum())
    primary = report["primary_matches"]
    numerator = int(primary[primary.baseline_cluster.isin(eligible_ids)].matched_events.sum()) if len(primary) else 0
    return {
        "status": MEASURED if denominator else UNMEASURED,
        "matched_baseline_events": numerator,
        "eligible_baseline_events": denominator,
        "value": numerator / denominator if denominator else None,
        "scope": "descriptive spike-train retention; not biological identity and not the frozen +0.05 improvement",
    }


def _split_merge_burden(report: Mapping[str, Any]) -> dict[str, Any]:
    """Fraction of eligible baseline units implicated in a split/merge ambiguity."""
    eligibility = report["baseline_eligibility"]
    eligible = set(eligibility.baseline_cluster.astype(int))
    if not eligible:
        return _endpoint(UNMEASURED, threshold=0.02, reason="no eligible baseline units")
    edges = report["edges"]
    implicated: set[int] = set()
    if len(edges):
        bdegree = edges.groupby("baseline_cluster").size()
        cdegree = edges.groupby("candidate_cluster").size()
        implicated.update(int(value) for value in bdegree[bdegree > 1].index if int(value) in eligible)
        merged_candidates = set(int(value) for value in cdegree[cdegree > 1].index)
        implicated.update(
            int(value) for value in edges.loc[edges.candidate_cluster.isin(merged_candidates), "baseline_cluster"]
            if int(value) in eligible
        )
        implicated.update(
            int(value) for value in edges.loc[~edges.primary_match.astype(bool), "baseline_cluster"]
            if int(value) in eligible
        )
    value = len(implicated) / len(eligible)
    return _endpoint(
        MEASURED, value=value, threshold=0.02, passes=value <= 0.02,
        reason="baseline-population normalized split/merge ambiguity burden",
        implicated_baseline_units=sorted(implicated), denominator_units=len(eligible),
        redesign_notice=(
            "Pre-outcome binding: the prior contract named a 0.02 regression margin but supplied no "
            "within-arm statistic. This adapter treats 0.02 as the maximum relational ambiguity burden."
        ),
    )


def _cosine_instability(a: np.ndarray, b: np.ndarray) -> float:
    av, bv = np.asarray(a, dtype=float).reshape(-1), np.asarray(b, dtype=float).reshape(-1)
    scale = np.linalg.norm(av) * np.linalg.norm(bv)
    if not scale:
        raise ValueError("zero-norm waveform cannot define cosine similarity")
    return float(1.0 - np.clip(np.dot(av, bv) / scale, -1.0, 1.0))


def _waveform_regression(baseline: ArmInput, candidate: ArmInput, report: Mapping[str, Any]) -> dict[str, Any]:
    if baseline.waveform_evidence is None or candidate.waveform_evidence is None:
        return _endpoint(
            UNMEASURED, threshold=0.05,
            reason="compact early/middle/late physical-channel waveform evidence is absent",
        )
    values = []
    primary = report["primary_matches"]
    for row in primary[primary.interior_primary].itertuples(index=False):
        be = baseline.waveform_evidence.get(int(row.baseline_cluster))
        ce = candidate.waveform_evidence.get(int(row.candidate_cluster))
        if be is None or ce is None:
            return _endpoint(UNMEASURED, threshold=0.05, reason="waveform evidence misses a primary pair")
        bvalue = max(_cosine_instability(be["early"], be["middle"]), _cosine_instability(be["middle"], be["late"]))
        cvalue = max(_cosine_instability(ce["early"], ce["middle"]), _cosine_instability(ce["middle"], ce["late"]))
        values.append(cvalue - bvalue)
    if not values:
        return _endpoint(UNMEASURED, threshold=0.05, reason="no interior primary pairs have waveform evidence")
    value = float(np.median(values))
    return _endpoint(MEASURED, value=value, threshold=0.05, passes=value <= 0.05,
                     reason="median candidate-minus-baseline maximum epoch cosine-instability")


def _guardrail_from_frame(report: Mapping[str, Any], metric: str, threshold: float) -> dict[str, Any]:
    rows = report["guardrail_summary"]
    selected = rows[rows.metric == metric]
    if len(selected) != 1:
        return _endpoint(UNMEASURED, threshold=threshold, reason=f"{metric} row absent")
    row = selected.iloc[0]
    value = row.candidate_minus_baseline
    if not bool(row.available) or not np.isfinite(value):
        return _endpoint(UNMEASURED, threshold=threshold, reason=f"{metric} inputs unavailable")
    value = float(value)
    return _endpoint(MEASURED, value=value, threshold=threshold, passes=value <= threshold,
                     reason="candidate minus comparator regression")


def bind_pair_endpoints(baseline: ArmInput, candidate: ArmInput,
                        report: Mapping[str, Any]) -> dict[str, Any]:
    """Bind every frozen endpoint to the pairwise report without imputation."""
    amplitude_qc_available = bool(baseline.qc.get("available", True) and candidate.qc.get("available", True))
    coverage = float(report["coverage_summary"]["amplitude_measurable_fraction"])
    missing = report["summary"]["median_missingness_improvement_pp"]
    missing_endpoint = _endpoint(
        UNMEASURED, threshold=5.0,
        reason="truncation QC absent in one or both arms; saved sorter arrays were not refit or imputed",
    ) if not amplitude_qc_available else (
        _endpoint(MEASURED, value=float(missing), threshold=5.0, passes=float(missing) >= 5.0,
                  reason="median baseline-minus-candidate missingness on measurable common-time primary pairs")
        if missing is not None and np.isfinite(missing)
        else _endpoint(UNMEASURED, threshold=5.0, reason="population/common-time support did not yield a finite effect")
    )
    boundary = (
        _guardrail_from_frame(report, "edge_spike_fraction", 0.01)
        if "depth" in baseline.sort and "depth" in candidate.sort
        else _endpoint(UNMEASURED, threshold=0.01, reason="spike_positions/depth evidence absent in one or both arms")
    )
    endpoints = {
        "population_measurable_fraction": _endpoint(
            UNMEASURED, threshold=0.5,
            reason="truncation QC absent in one or both arms; zero computed coverage is not a measured failure",
        ) if not amplitude_qc_available else _endpoint(
            MEASURED, value=coverage, threshold=0.5, passes=coverage >= 0.5,
            reason="full eligible baseline-unit denominator including lost/insufficient units",
        ),
        "median_common_time_missingness_improvement_pp": missing_endpoint,
        "exclusive_identity_continuity_improvement": _endpoint(
            UNRESOLVED, threshold=0.05,
            reason=(
                "the frozen contract supplies a name and margin but no mathematical statistic or null; "
                "Jaccard, rate correlation, unit count, and descriptive retention are not substitutes"
            ),
            descriptive_exclusive_baseline_event_retention=_exclusive_baseline_event_retention(report, baseline.sort),
        ),
        "refractory_contamination": _guardrail_from_frame(
            report, "median_refractory_violation_fraction_1_5ms", 0.01
        ),
        "chance_aware_duplicate_burden": _guardrail_from_frame(
            report, "chance_aware_near_coincident_excess", 0.01
        ),
        "split_merge_burden": _split_merge_burden(report),
        "waveform_instability": _waveform_regression(baseline, candidate, report),
        "boundary_burden": boundary,
    }
    return endpoints


def _pair_verdict(endpoints: Mapping[str, Mapping[str, Any]], *, scorecard_eligible: bool) -> tuple[str, list[str]]:
    if not scorecard_eligible:
        return "fixture_only", ["SYNTHETIC_OR_UNAVAILABLE_NOT_SCIENTIFIC"]
    reasons = []
    coverage = endpoints["population_measurable_fraction"]
    if coverage["status"] != MEASURED:
        return "inconclusive", ["COVERAGE_UNMEASURED"]
    if not coverage["passes"]:
        return "fail", ["POPULATION_COVERAGE_BELOW_0_5"]
    guardrail_names = (
        "refractory_contamination", "chance_aware_duplicate_burden", "split_merge_burden",
        "waveform_instability", "boundary_burden",
    )
    for name in guardrail_names:
        endpoint = endpoints[name]
        if endpoint["status"] != MEASURED:
            reasons.append(f"{name.upper()}_{endpoint['status']}")
        elif not endpoint["passes"]:
            reasons.append(f"{name.upper()}_BREACH")
    if any(reason.endswith("_BREACH") for reason in reasons):
        return "fail", reasons
    if reasons:
        return "inconclusive", reasons
    missingness = endpoints["median_common_time_missingness_improvement_pp"]
    if missingness["status"] == MEASURED and missingness["passes"]:
        return "pass", []
    identity = endpoints["exclusive_identity_continuity_improvement"]
    if identity["status"] != MEASURED:
        return "inconclusive", [f"IDENTITY_CONTINUITY_{identity['status']}"]
    return ("pass", []) if identity["passes"] else ("fail", ["NO_PRIMARY_AXIS_MET_MARGIN"])


def _repeat_verdict(endpoints: Mapping[str, Mapping[str, Any]], *, scorecard_eligible: bool) -> tuple[str, list[str]]:
    """Apply the separately frozen engineering repeatability rule."""
    if not scorecard_eligible:
        return "fixture_only", ["SYNTHETIC_OR_UNAVAILABLE_NOT_SCIENTIFIC"]
    required = (
        "population_measurable_fraction", "median_common_time_missingness_improvement_pp",
        "refractory_contamination", "chance_aware_duplicate_burden", "split_merge_burden",
        "waveform_instability", "boundary_burden",
    )
    missing = [name for name in required if endpoints[name]["status"] != MEASURED]
    retention = endpoints["exclusive_identity_continuity_improvement"][
        "descriptive_exclusive_baseline_event_retention"
    ]
    if retention["status"] != MEASURED:
        missing.append("descriptive_exclusive_baseline_event_retention")
    if missing:
        return "inconclusive", [f"REPEAT_{name.upper()}_UNMEASURED" for name in missing]
    failures = []
    if endpoints["population_measurable_fraction"]["value"] < 0.5:
        failures.append("REPEAT_COVERAGE")
    if abs(endpoints["median_common_time_missingness_improvement_pp"]["value"]) > 2.0:
        failures.append("REPEAT_MISSINGNESS_DIFFERENCE")
    if 1.0 - retention["value"] > 0.02:
        failures.append("REPEAT_EXCLUSIVE_RETENTION_SHORTFALL")
    for name in (
        "refractory_contamination", "chance_aware_duplicate_burden",
        "waveform_instability", "boundary_burden",
    ):
        if abs(endpoints[name]["value"]) > endpoints[name]["threshold"]:
            failures.append(f"REPEAT_{name.upper()}")
    if endpoints["split_merge_burden"]["value"] > 0.02:
        failures.append("REPEAT_SPLIT_MERGE_BURDEN")
    return ("fail", failures) if failures else ("pass", [])


def evaluate_pair(baseline: ArmInput, candidate: ArmInput, config: Mapping[str, Any],
                  spatial_region: Mapping[str, Any] | None, *, output_dir: Path | None = None) -> dict[str, Any]:
    key = f"{baseline.arm_id}__{candidate.arm_id}"
    if baseline.evidence_kind == "unavailable" or candidate.evidence_kind == "unavailable":
        return {
            "pair_id": key, "scorecard_eligible": False, "status": UNMEASURED,
            "verdict": "inconclusive", "reason_codes": ["ARM_UNAVAILABLE"], "endpoints": {},
        }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)
    report = compare_sorts(
        baseline.sort, candidate.sort, baseline.qc, candidate.qc, config, spatial_region,
        output_dir=output_dir,
    )
    eligible = baseline.evidence_kind == candidate.evidence_kind == "real_saved"
    endpoints = bind_pair_endpoints(baseline, candidate, report)
    if baseline.arm_id == "REF384" and candidate.arm_id == "REF384_repeat":
        verdict, reasons = _repeat_verdict(endpoints, scorecard_eligible=eligible)
    else:
        verdict, reasons = _pair_verdict(endpoints, scorecard_eligible=eligible)
    return {
        "pair_id": key,
        "scorecard_eligible": eligible,
        "status": MEASURED,
        "verdict": verdict,
        "reason_codes": reasons,
        "endpoints": endpoints,
        "ambiguity": {
            "ambiguous_edges": report["split_merge_summary"]["ambiguous_edges"],
            "unmatched_units": report["coverage_summary"]["baseline_cohort_status_counts"].get("unmatched", 0),
        },
        "pair_request_digest": report["candidate_manifest"]["request_digest"],
    }


def _overall_decision(pairs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = ("REF384__repaired_B384", "existing_corrected_B384__repaired_B384")
    repeat = pairs["REF384__REF384_repeat"]
    if any(not pairs[key]["scorecard_eligible"] for key in required) or not repeat["scorecard_eligible"]:
        return {"status": "inconclusive", "advance": False, "reason_codes": ["PROSPECTIVE_REAL_ARM_OUTPUTS_UNAVAILABLE"]}
    if repeat["verdict"] != "pass":
        return {"status": "inconclusive", "advance": False, "reason_codes": ["BASELINE_REPEAT_NOT_QUALIFIED"]}
    verdicts = [pairs[key]["verdict"] for key in required]
    if all(value == "pass" for value in verdicts):
        return {"status": "advance", "advance": True, "reason_codes": []}
    if any(value == "inconclusive" for value in verdicts):
        return {"status": "inconclusive", "advance": False, "reason_codes": ["REQUIRED_ENDPOINT_UNMEASURED_OR_UNRESOLVED"]}
    return {"status": "reject", "advance": False, "reason_codes": ["FROZEN_MARGIN_FAILURE"]}


def evaluate_four_arms(arms: Mapping[str, ArmInput], config: Mapping[str, Any],
                       spatial_region: Mapping[str, Any] | None = None,
                       *, output_dir: Path | None = None) -> dict[str, Any]:
    """Run the fixed pair order and return separated scientific/fixture rows."""
    if set(arms) != set(ARM_ORDER):
        raise ValueError(f"exact arms required: {ARM_ORDER}")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)
    pairs = {}
    for baseline_id, candidate_id in PAIR_ORDER:
        key = f"{baseline_id}__{candidate_id}"
        pair_dir = output_dir / "pairs" / key if output_dir is not None and (
            arms[baseline_id].evidence_kind != "unavailable" and arms[candidate_id].evidence_kind != "unavailable"
        ) else None
        pairs[key] = evaluate_pair(arms[baseline_id], arms[candidate_id], config, spatial_region, output_dir=pair_dir)
    scientific = {key: value for key, value in pairs.items() if value["scorecard_eligible"]}
    fixtures = {key: value for key, value in pairs.items() if value["status"] == MEASURED and not value["scorecard_eligible"]}
    result = {
        "schema": SCHEMA,
        "execution_enabled": False,
        "arm_order": list(ARM_ORDER),
        "pair_order": [f"{a}__{b}" for a, b in PAIR_ORDER],
        "arms": {
            arm_id: {
                "evidence_kind": arms[arm_id].evidence_kind,
                "scorecard_eligible": arms[arm_id].evidence_kind == "real_saved",
                "provenance": dict(arms[arm_id].provenance),
            }
            for arm_id in ARM_ORDER
        },
        "scientific_scorecard": scientific,
        "fixture_results_not_scientific": fixtures,
        "unavailable_pairs": {
            key: value for key, value in pairs.items() if value["status"] == UNMEASURED
        },
        "decision": _overall_decision(pairs),
        "implementation_provenance": _implementation_provenance(),
        "request_digest": fingerprint({
            "schema": SCHEMA, "config": dict(config), "spatial_region": spatial_region,
            "arms": {key: dict(value.provenance) | {"evidence_kind": value.evidence_kind} for key, value in arms.items()},
        }),
    }
    if output_dir is not None:
        (output_dir / "FOUR_ARM_REPORT.json").write_text(json.dumps(_json_value(result), indent=2, sort_keys=True) + "\n")
    return result


def _implementation_provenance() -> dict[str, Any]:
    names = (
        "testing.first_medium_evaluator", "testing.sort_comparison",
        "testing.luke_amplitude_dropout_audit", "pipeline.config", "pipeline.truncation",
    )
    result = {}
    for name in names:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve() if module is not None and module.__file__ else None
        if path is None or not path.is_file():
            raise RuntimeError(f"cannot attest imported implementation module {name}")
        result[name] = {"path": str(path), "sha256": _sha256(path)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if request.get("execution_enabled") is not False:
        raise ValueError("request must explicitly set execution_enabled=false")
    config = request["evaluation_config"]
    arms = {
        spec["arm_id"]: load_arm_from_spec(spec, sampling_frequency_hz=float(config["sampling_frequency_hz"]))
        for spec in request["arms"]
    }
    evaluate_four_arms(arms, config, request.get("spatial_region"), output_dir=args.output)


if __name__ == "__main__":
    main()
