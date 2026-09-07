"""Contracts and decision logic for longitudinal pipeline development comparisons.

This module deliberately does not run a sorter.  It is the thin, reusable layer
between existing arm-specific runners and their evaluator: it refuses unsuitable
development data, records the spatial halo contract, validates evaluator
controls, and makes Pareto-style advancement decisions without using unit count
as an efficacy endpoint.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pipeline.config import fingerprint
from testing.ladder_sorter import NAMED_CONFIGS


CONTRACT_SCHEMA = "longitudinal-development-comparison-v1"
RESULT_SCHEMA = "longitudinal-development-results-v1"
REQUIRED_EVALUATOR_INVARIANTS = {
    "exclusive_event_matching",
    "chance_aware_coincidence",
    "common_physical_time",
    "measurement_coverage_reported",
    "content_bound_inputs",
    "acquisition_and_selected_clocks_verified",
    "physical_probe_geometry_preserved",
}
REQUIRED_NEGATIVE_CONTROLS = {
    "pathological_claim_mask": "worse",
    "harmful_external_warp": "worse",
    "fake_improvement_by_dropping_difficult_spikes": "rejected",
}
SUPPORTED_CURATION_PROFILE = "legacy-compatible-cosine-0.90-ccg-0.5-v1"


class ContractError(ValueError):
    """The proposed experiment cannot support the intended decision."""


@dataclass(frozen=True)
class DevelopmentContract:
    source: Path
    raw: dict[str, Any]
    digest: str

    @property
    def candidates(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.raw["candidates"])

    @property
    def reference_name(self) -> str:
        return next(c["name"] for c in self.candidates if c["role"] == "reference")


def _require_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ContractError(f"{label} must be finite")
    if minimum is not None and value < minimum:
        raise ContractError(f"{label} must be >= {minimum}")
    return value


def _require_keys(value: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ContractError(f"{label} is missing {missing}")


def validate_contract(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a prospective comparison contract."""
    raw = dict(raw)
    _require_keys(
        raw,
        {
            "schema_version", "experiment_id", "decision_statement", "recording",
            "spatial_contract", "candidates", "curation_profile", "metrics",
            "evaluation", "validation_sequence",
        },
        "contract",
    )
    if raw["schema_version"] != CONTRACT_SCHEMA:
        raise ContractError(f"schema_version must be {CONTRACT_SCHEMA!r}")
    if not str(raw["experiment_id"]).strip():
        raise ContractError("experiment_id must be nonempty")
    decision = raw["decision_statement"]
    if not isinstance(decision, str) or "if" not in decision.lower() or "then" not in decision.lower():
        raise ContractError("decision_statement must state 'If result X, then decision Y'")

    recording = raw["recording"]
    _require_keys(
        recording,
        {
            "accepted_recording_path", "session_id", "recording_digest",
            "recording_content_sha256", "probe_geometry_hash", "stream_id",
            "total_duration_s", "start_s", "duration_s", "continuous", "full_session",
        },
        "recording",
    )
    duration_s = _require_number(recording["duration_s"], "recording.duration_s", minimum=1.0)
    total_duration_s = _require_number(
        recording["total_duration_s"], "recording.total_duration_s", minimum=duration_s
    )
    start_s = _require_number(recording["start_s"], "recording.start_s", minimum=0.0)
    if start_s + duration_s > total_duration_s:
        raise ContractError("selected recording duration exceeds total_duration_s")
    if duration_s < 3600.0 and recording["full_session"] is not True:
        raise ContractError("recording must be a full session or at least 3600 continuous seconds")
    if recording["full_session"] is True and (start_s != 0.0 or duration_s != total_duration_s):
        raise ContractError("full_session requires start_s=0 and duration_s=total_duration_s")
    if recording["continuous"] is not True:
        raise ContractError("development recording must be continuous")
    if not str(recording["recording_digest"]).strip():
        raise ContractError("recording.recording_digest must content-bind the accepted input")
    for name in ("recording_content_sha256", "probe_geometry_hash"):
        if not str(recording[name]).strip():
            raise ContractError(f"recording.{name} must be nonempty")

    spatial = raw["spatial_contract"]
    _require_keys(
        spatial,
        {
            "processing_depth_um", "scoring_depth_um", "halo_below_um",
            "halo_above_um", "required_support_um", "minimum_edge_exclusion_um",
            "boundary_qc",
        },
        "spatial_contract",
    )
    processing = spatial["processing_depth_um"]
    scoring = spatial["scoring_depth_um"]
    if not (isinstance(processing, list) and isinstance(scoring, list) and len(processing) == len(scoring) == 2):
        raise ContractError("processing_depth_um and scoring_depth_um must be [lower, upper]")
    p0, p1 = (_require_number(v, "processing_depth_um") for v in processing)
    s0, s1 = (_require_number(v, "scoring_depth_um") for v in scoring)
    support = _require_number(spatial["required_support_um"], "required_support_um", minimum=0.0)
    halo_below = _require_number(spatial["halo_below_um"], "halo_below_um", minimum=0.0)
    halo_above = _require_number(spatial["halo_above_um"], "halo_above_um", minimum=0.0)
    edge_exclusion = _require_number(
        spatial["minimum_edge_exclusion_um"], "minimum_edge_exclusion_um", minimum=0.0
    )
    if not p0 < s0 < s1 < p1:
        raise ContractError("scoring band must be strictly inside the processing band")
    if min(s0 - p0, p1 - s1) < support:
        raise ContractError("processing halo is smaller than required physical support")
    if not math.isclose(s0 - p0, halo_below) or not math.isclose(p1 - s1, halo_above):
        raise ContractError("declared halo distances do not match the physical depth ranges")
    if edge_exclusion > min(halo_below, halo_above):
        raise ContractError("minimum edge exclusion cannot exceed the available halo")
    if spatial["boundary_qc"] is not True:
        raise ContractError("boundary_qc must be enabled for a depth-reduced comparison")

    candidates = raw["candidates"]
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ContractError("at least one reference and one candidate are required")
    names = [candidate.get("name") for candidate in candidates]
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(names) != len(set(names)):
        raise ContractError("candidate names must be nonempty and unique")
    references = [candidate for candidate in candidates if candidate.get("role") == "reference"]
    if len(references) != 1:
        raise ContractError("exactly one candidate must have role='reference'")
    if raw["curation_profile"] != SUPPORTED_CURATION_PROFILE:
        raise ContractError(
            f"curation_profile must be the implemented frozen profile {SUPPORTED_CURATION_PROFILE!r}"
        )
    for candidate in candidates:
        _require_keys(
            candidate,
            {"name", "role", "sorter_config", "curation_profile"},
            f"candidate {candidate.get('name')!r}",
        )
        if candidate["curation_profile"] != raw["curation_profile"]:
            raise ContractError(f"candidate {candidate['name']}: curation must remain frozen")
        sorter_config = candidate["sorter_config"]
        if sorter_config not in NAMED_CONFIGS:
            raise ContractError(
                f"candidate {candidate['name']}: unknown standard SorterConfig {sorter_config!r}"
            )
        if candidate["role"] not in {"reference", "candidate"}:
            raise ContractError(f"candidate {candidate['name']}: invalid role")

    metrics = raw["metrics"]
    _require_keys(metrics, {"minimum_measurable_unit_fraction", "minimum_common_time_fraction", "maximum_full_probe_candidates", "primary", "guardrails"}, "metrics")
    for name in ("minimum_measurable_unit_fraction", "minimum_common_time_fraction"):
        value = _require_number(metrics[name], f"metrics.{name}", minimum=0.0)
        if value > 1:
            raise ContractError(f"metrics.{name} cannot exceed 1")
    if not isinstance(metrics["maximum_full_probe_candidates"], int) or not 1 <= metrics["maximum_full_probe_candidates"] <= 2:
        raise ContractError("metrics.maximum_full_probe_candidates must be 1 or 2")
    if not metrics["primary"]:
        raise ContractError("at least one primary efficacy metric is required")
    if "unit_count" in metrics["primary"] or "ks_good_unit_count" in metrics["primary"]:
        raise ContractError("unit count is descriptive, not a primary efficacy endpoint")
    for name, rule in metrics["primary"].items():
        _require_keys(rule, {"minimum_improvement", "repeatability"}, f"primary metric {name}")
        effect = _require_number(rule["minimum_improvement"], f"primary metric {name}.minimum_improvement", minimum=0.0)
        repeatability = _require_number(rule["repeatability"], f"primary metric {name}.repeatability", minimum=0.0)
        if effect < repeatability:
            raise ContractError(f"primary metric {name}: meaningful effect cannot be below repeatability")
    if not metrics["guardrails"]:
        raise ContractError("at least one guardrail is required")
    for name, rule in metrics["guardrails"].items():
        _require_keys(rule, {"maximum_regression"}, f"guardrail {name}")
        _require_number(rule["maximum_regression"], f"guardrail {name}.maximum_regression", minimum=0.0)

    evaluation = raw["evaluation"]
    _require_keys(
        evaluation,
        {
            "correspondence_tolerance_ms", "minimum_correspondence_overlap",
            "primary_retention", "amplitude_window_spikes", "longitudinal_bin_s",
            "refractory_period_ms", "coincidence_tolerance_ms",
            "coincidence_depth_um", "minimum_valid_amplitude_windows",
        },
        "evaluation",
    )
    if evaluation["amplitude_window_spikes"] != 1000:
        raise ContractError("the first ladder requires the nominal 1000-spike amplitude window")
    for name in (
        "correspondence_tolerance_ms", "minimum_correspondence_overlap",
        "primary_retention", "longitudinal_bin_s", "refractory_period_ms",
        "coincidence_tolerance_ms", "coincidence_depth_um",
    ):
        _require_number(evaluation[name], f"evaluation.{name}", minimum=0.0)
    for name in ("minimum_correspondence_overlap", "primary_retention"):
        if evaluation[name] > 1:
            raise ContractError(f"evaluation.{name} cannot exceed 1")
    if not isinstance(evaluation["minimum_valid_amplitude_windows"], int) or evaluation["minimum_valid_amplitude_windows"] < 1:
        raise ContractError("evaluation.minimum_valid_amplitude_windows must be a positive integer")

    sequence = raw["validation_sequence"]
    expected = ["smoke", "long_strip", "full_probe", "second_session"]
    if sequence != expected:
        raise ContractError(f"validation_sequence must be {expected}")
    raw["recording"] = {
        **recording,
        "total_duration_s": total_duration_s,
        "start_s": start_s,
        "duration_s": duration_s,
    }
    return raw


def load_contract(path: Path | str) -> DevelopmentContract:
    path = Path(path)
    raw = validate_contract(json.loads(path.read_text()))
    return DevelopmentContract(source=path, raw=raw, digest=fingerprint(raw))


def build_plan(contract: DevelopmentContract) -> dict[str, Any]:
    """Return a content-bound, operator-readable execution plan."""
    spatial = contract.raw["spatial_contract"]
    p0, p1 = spatial["processing_depth_um"]
    s0, s1 = spatial["scoring_depth_um"]
    return {
        "schema_version": CONTRACT_SCHEMA,
        "contract_digest": contract.digest,
        "experiment_id": contract.raw["experiment_id"],
        "reference": contract.reference_name,
        "recording": contract.raw["recording"],
        "processing_width_um": p1 - p0,
        "scoring_width_um": s1 - s0,
        "halo_below_um": s0 - p0,
        "halo_above_um": p1 - s1,
        "candidates": [
            {
                **candidate,
                "sorter_config_digest": NAMED_CONFIGS[candidate["sorter_config"]].digest,
                "sorter_overrides": NAMED_CONFIGS[candidate["sorter_config"]].overrides,
            }
            for candidate in contract.candidates
        ],
        "decision_statement": contract.raw["decision_statement"],
        "execution_order": contract.raw["validation_sequence"],
        "metrics": contract.raw["metrics"],
        "evaluation": contract.raw["evaluation"],
        "ranking_policy": "Pareto frontier after coverage, evaluator, primary, and guardrail gates; no composite score",
    }


def pin_plan(contract: DevelopmentContract, path: Path | str) -> dict[str, Any]:
    """Write the resolved contract before expensive work, or verify exact reuse."""
    path = Path(path)
    plan = build_plan(contract)
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise RuntimeError("existing resolved comparison plan belongs to another contract")
        return plan
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def _validate_evaluator_receipt(receipt: Mapping[str, Any]) -> list[str]:
    failures = []
    invariants = receipt.get("invariants", {})
    for name in sorted(REQUIRED_EVALUATOR_INVARIANTS):
        if invariants.get(name) is not True:
            failures.append(f"evaluator invariant not established: {name}")
    controls = receipt.get("negative_controls", {})
    for name, expected in REQUIRED_NEGATIVE_CONTROLS.items():
        observed = controls.get(name, {}).get("outcome")
        if observed != expected:
            failures.append(f"negative control {name}: expected {expected!r}, got {observed!r}")
    return failures


def _dominates(a: Mapping[str, Any], b: Mapping[str, Any], primary: set[str], guardrails: set[str]) -> bool:
    av = [a["primary_improvements"][name] for name in sorted(primary)]
    bv = [b["primary_improvements"][name] for name in sorted(primary)]
    # Guardrail values are explicitly normalized as regressions, so lower is better.
    ag = [a["guardrail_regressions"][name] for name in sorted(guardrails)]
    bg = [b["guardrail_regressions"][name] for name in sorted(guardrails)]
    no_worse = all(x >= y for x, y in zip(av, bv)) and all(x <= y for x, y in zip(ag, bg))
    strictly_better = any(x > y for x, y in zip(av, bv)) or any(x < y for x, y in zip(ag, bg))
    return no_worse and strictly_better


def evaluate_results(contract: DevelopmentContract, raw: Mapping[str, Any]) -> dict[str, Any]:
    """Gate candidates and return the non-dominated advancing set."""
    if raw.get("schema_version") != RESULT_SCHEMA:
        raise ContractError(f"results schema_version must be {RESULT_SCHEMA!r}")
    if raw.get("contract_digest") != contract.digest:
        raise ContractError("results belong to another comparison contract")
    evaluator_failures = _validate_evaluator_receipt(raw.get("evaluator", {}))
    rules = contract.raw["metrics"]
    primary = set(rules["primary"])
    guardrails = set(rules["guardrails"])
    expected_candidates = {candidate["name"] for candidate in contract.candidates}
    rows = raw.get("candidates")
    if not isinstance(rows, list):
        raise ContractError("results.candidates must be a list")
    names = [row.get("name") for row in rows]
    if len(names) != len(set(names)) or set(names) != expected_candidates:
        raise ContractError("results must contain each contracted candidate exactly once")

    decisions = []
    eligible = []
    for row in rows:
        reasons = list(evaluator_failures)
        try:
            measured = int(row.get("measurable_units", -1))
            total = int(row.get("eligible_units", -1))
        except (TypeError, ValueError) as error:
            raise ContractError(
                f"candidate {row['name']}: invalid measurable/eligible unit counts"
            ) from error
        if total <= 0 or measured < 0 or measured > total:
            raise ContractError(f"candidate {row['name']}: invalid measurable/eligible unit counts")
        measurable_fraction = measured / total
        common_time = _require_number(row.get("common_time_fraction"), f"candidate {row['name']}.common_time_fraction", minimum=0.0)
        if common_time > 1:
            raise ContractError(f"candidate {row['name']}: common_time_fraction cannot exceed 1")
        if measurable_fraction < rules["minimum_measurable_unit_fraction"]:
            reasons.append("inadequate measurable-unit coverage")
        if common_time < rules["minimum_common_time_fraction"]:
            reasons.append("inadequate common physical-time coverage")
        pvalues = row.get("primary_improvements", {})
        gvalues = row.get("guardrail_regressions", {})
        if set(pvalues) != primary or set(gvalues) != guardrails:
            raise ContractError(f"candidate {row['name']}: metric keys do not match the contract")
        primary_passes = []
        for name, rule in rules["primary"].items():
            value = _require_number(pvalues[name], f"candidate {row['name']}.{name}")
            primary_passes.append(value >= rule["minimum_improvement"])
        if row["name"] != contract.reference_name and not any(primary_passes):
            reasons.append("no primary efficacy axis has a meaningful improvement")
        for name, rule in rules["guardrails"].items():
            value = _require_number(gvalues[name], f"candidate {row['name']}.{name}")
            if value > rule["maximum_regression"]:
                reasons.append(f"guardrail {name} regressed")
        is_reference = row["name"] == contract.reference_name
        status = "reference" if is_reference else ("advance" if not reasons else "stop")
        decision = {
            "name": row["name"],
            "status": status,
            "measurable_unit_fraction": measurable_fraction,
            "common_time_fraction": common_time,
            "reasons": reasons,
        }
        decisions.append(decision)
        if status == "advance":
            eligible.append(row)

    frontier = [
        row for row in eligible
        if not any(_dominates(other, row, primary, guardrails) for other in eligible if other is not row)
    ]
    frontier_names = sorted(row["name"] for row in frontier)
    for decision in decisions:
        if decision["status"] == "advance" and decision["name"] not in frontier_names:
            decision["status"] = "stop_dominated"
            decision["reasons"].append("dominated by another passing candidate")
    return {
        "schema_version": RESULT_SCHEMA,
        "contract_digest": contract.digest,
        "experiment_id": contract.raw["experiment_id"],
        "evaluator_valid": not evaluator_failures,
        "decisions": decisions,
        "pareto_frontier": frontier_names,
        "selection_required": len(frontier_names) > 1,
        "maximum_full_probe_candidates": rules["maximum_full_probe_candidates"],
        "downselection_required": len(frontier_names) > rules["maximum_full_probe_candidates"],
        "policy": "Only the Pareto frontier advances; multiple incomparable survivors require an explicit decision, not a hidden composite score.",
    }
