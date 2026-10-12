"""Synthetic K7 candidate-union, frame-score, and abstention oracle.

This module consumes saved metadata and residuals. It deliberately does not
learn templates, detect matched-filter events, whiten voltage, calibrate
thresholds, or classify real units.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


SCHEMA = "joint-model-k7-frame-score-v1"
LATTICE_UM = 40
TRAINING_MIN_SEPARATION_UM = 80
HELD_MIN_INFORMATIVE_UM = 80
MIN_EVENTS_PER_STATE = 50


def _integer(name: str, value: Any) -> int:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise ValueError(f"{name} must be an exact integer")
    integer = int(array)
    if not np.iinfo(np.int64).min <= integer <= np.iinfo(np.int64).max:
        raise ValueError(f"{name} is outside int64 range")
    return integer


def _finite_real(name: str, value: Any) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf" or array.dtype.kind == "b":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def build_candidate_union(
    unit_events: Sequence[Mapping[str, Any]],
    matched_filter_events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return a stable metadata-checked union independent of residual outcomes."""
    required = {"event_id", "sample_index", "state_um", "block_id"}
    union: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for source, rows in (("unit", unit_events), ("matched_filter", matched_filter_events)):
        for row in rows:
            if not isinstance(row, Mapping) or not required <= set(row):
                raise ValueError("candidate row lacks required metadata")
            event_id = row["event_id"]
            if not isinstance(event_id, str) or not event_id:
                raise ValueError("event_id must be a nonempty string")
            normalized = {
                "event_id": event_id,
                "sample_index": _integer("sample_index", row["sample_index"]),
                "state_um": _integer("state_um", row["state_um"]),
                "block_id": _integer("block_id", row["block_id"]),
            }
            if normalized["state_um"] % LATTICE_UM:
                raise ValueError("candidate state_um must lie on the exact 40-um lattice")
            if event_id not in union:
                union[event_id] = {**normalized, "sources": [source]}
                order.append(event_id)
            else:
                prior = union[event_id]
                if any(prior[key] != normalized[key] for key in ("sample_index", "state_um", "block_id")):
                    raise ValueError(f"conflicting duplicate metadata for event {event_id}")
                if source not in prior["sources"]:
                    prior["sources"].append(source)
    return [union[event_id] for event_id in order]


def score_frame_residuals(
    training_states_um: Sequence[int],
    held_out_rows: Sequence[Mapping[str, Any]],
    *,
    probe_fixed_max_score: float,
    tissue_fixed_min_score: float,
    informative_abs_state_um: int = 80,
    minimum_events_per_state: int = 50,
) -> dict[str, Any]:
    """Score fixed residuals with state balance and fail-closed abstention."""
    train = np.asarray(training_states_um)
    if train.ndim != 1 or train.dtype.kind not in "iu" or train.dtype.kind == "b":
        raise ValueError("training states must be a one-dimensional integer array")
    if train.size and any(not np.iinfo(np.int64).min <= int(x) <= np.iinfo(np.int64).max for x in train):
        raise ValueError("training state is outside int64 range")
    train64 = train.astype(np.int64, copy=False)
    if np.any(train64 % LATTICE_UM):
        raise ValueError("training states must lie on the exact 40-um lattice")
    informative_abs_state_um = _integer("informative_abs_state_um", informative_abs_state_um)
    minimum_events_per_state = _integer("minimum_events_per_state", minimum_events_per_state)
    if informative_abs_state_um < HELD_MIN_INFORMATIVE_UM:
        raise ValueError(
            f"informative state threshold must be at least {HELD_MIN_INFORMATIVE_UM} um"
        )
    if minimum_events_per_state < MIN_EVENTS_PER_STATE:
        raise ValueError(
            f"minimum events per state must be at least {MIN_EVENTS_PER_STATE}"
        )
    probe_threshold = _finite_real("probe_fixed_max_score", probe_fixed_max_score)
    tissue_threshold = _finite_real("tissue_fixed_min_score", tissue_fixed_min_score)
    if not probe_threshold < tissue_threshold:
        raise ValueError("probe threshold must be below tissue threshold")

    train_unique = np.unique(train64)
    training_separation = (
        int(train_unique[-1]) - int(train_unique[0]) if train_unique.size else 0
    )
    training_eligible = bool(
        train_unique.size >= 2
        and training_separation >= TRAINING_MIN_SEPARATION_UM
    )

    scores_by_state: dict[int, list[float]] = {}
    for row in held_out_rows:
        if not isinstance(row, Mapping):
            raise ValueError("held-out rows must be mappings")
        state = _integer("held-out state_um", row.get("state_um"))
        if state % LATTICE_UM:
            raise ValueError("held-out state must lie on the exact 40-um lattice")
        if "residual_probe" not in row or "residual_tissue" not in row:
            raise ValueError("held-out row lacks residual fields")
        probe = _finite_real("residual_probe", row["residual_probe"])
        tissue = _finite_real("residual_tissue", row["residual_tissue"])
        if probe <= 0 or tissue <= 0:
            raise ValueError("held-out residuals must be finite and strictly positive")
        event_score = float(np.log(probe) - np.log(tissue))
        if not np.isfinite(event_score):
            raise ValueError("held-out residual log ratio must be finite")
        if abs(state) < informative_abs_state_um:
            continue
        scores_by_state.setdefault(state, []).append(event_score)

    state_counts = {state: len(values) for state, values in sorted(scores_by_state.items())}
    contributing = {
        state: float(np.median(values))
        for state, values in sorted(scores_by_state.items())
        if len(values) >= minimum_events_per_state
    }
    if not training_eligible or not contributing:
        label = "undetermined"
        score = None
        reason = "training_state_support" if not training_eligible else "held_out_state_support"
    else:
        score = float(np.median(list(contributing.values())))
        if score <= probe_threshold:
            label = "probe_fixed"
        elif score >= tissue_threshold:
            label = "tissue_fixed"
        else:
            label = "undetermined"
        reason = "scored"
    return {
        "schema": SCHEMA,
        "reported_label": label,
        "operational_frame": "probe" if label == "probe_fixed" else "tissue",
        "score": score,
        "reason": reason,
        "training_states_um": train_unique.tolist(),
        "training_max_separation_um": training_separation,
        "training_min_separation_um": TRAINING_MIN_SEPARATION_UM,
        "held_min_informative_um": informative_abs_state_um,
        "minimum_events_per_state": minimum_events_per_state,
        "state_event_counts": state_counts,
        "contributing_state_medians": contributing,
        "probe_fixed_max_score": probe_threshold,
        "tissue_fixed_min_score": tissue_threshold,
    }


def annotate_candidate_group(
    members: Sequence[Mapping[str, Any]],
    *,
    pulse_coincidence_pass_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Keep labels member-local; linking never becomes an automatic group merge."""
    pulse_ids = set(pulse_coincidence_pass_ids)
    output = []
    seen = set()
    for member in members:
        member_id = member.get("member_id")
        label = member.get("reported_label")
        if not isinstance(member_id, str) or not member_id or member_id in seen:
            raise ValueError("group member IDs must be unique nonempty strings")
        if label not in {"probe_fixed", "tissue_fixed", "undetermined"}:
            raise ValueError("group member has an invalid reported label")
        seen.add(member_id)
        evidence = label == "probe_fixed" or member_id in pulse_ids
        output.append({
            "member_id": member_id,
            "reported_label": label,
            "probe_fixed_evidence": evidence,
            "probe_fixed_candidate_group": not evidence,
        })
    unknown_pulse = pulse_ids - seen
    if unknown_pulse:
        raise ValueError("pulse evidence names an unknown group member")
    return {
        "schema": SCHEMA,
        "members": output,
        "any_probe_fixed_evidence": any(row["probe_fixed_evidence"] for row in output),
        "automatic_group_merge": False,
    }
