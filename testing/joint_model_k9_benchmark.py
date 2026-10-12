"""Input-independent K9 exact-lattice pre-CAR injection mechanics.

This module operates only on finite ``float32`` voltage arrays. It does not
choose or open a recording, bind a real window, run DARTsort, or write a
benchmark binary. A later frozen real-window contract must provide those
bindings. The functions here fail closed on fractional motion, clipped support,
ambiguous clocks, and duplicate truth identities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from testing.luke_au_cpu_preparation import exact_remap, exact_same_column_map
from testing.luke_injected_ground_truth_benchmark import validate_template


SCHEMA = "joint-model-k9-core-v1"
LATTICE_UM = 40
ALLOWED_FRAMES = {"tissue", "probe"}
ALLOWED_ROLES = {
    "moving_neural",
    "decoy_member",
    "field_error_neural",
    "probe_fixed_localized",
    "probe_fixed_broad",
    "probe_fixed_sparse",
    "switch_fixture",
    "designed_overlap",
}


@dataclass(frozen=True)
class K9Event:
    event_id: str
    biological_unit_id: str
    template_id: str
    sample_index: int
    frame: str
    role: str
    base_shift_um: int = 0
    true_state_um: int = 0
    supplied_state_um: int = 0
    amplitude_scale: float = 1.0
    decoy_group: str = ""


def _int64_scalar(name: str, value: Any) -> int:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise ValueError(f"{name} must be an exact int64-compatible integer")
    integer = int(array)
    if integer < np.iinfo(np.int64).min or integer > np.iinfo(np.int64).max:
        raise ValueError(f"{name} is outside int64 range")
    return integer


def _lattice_scalar(name: str, value: Any) -> int:
    integer = _int64_scalar(name, value)
    if integer % LATTICE_UM:
        raise ValueError(f"{name} must be an exact {LATTICE_UM}-um lattice value")
    return integer


def assign_half_open_states(
    samples: Sequence[int],
    starts: Sequence[int],
    stops: Sequence[int],
    states_um: Sequence[int],
) -> np.ndarray:
    """Assign exact lattice states under explicit contiguous ``[start, stop)`` bounds."""
    sample_values = np.asarray(samples)
    start_values = np.asarray(starts)
    stop_values = np.asarray(stops)
    state_values = np.asarray(states_um)
    for name, values in (
        ("samples", sample_values),
        ("starts", start_values),
        ("stops", stop_values),
        ("states_um", state_values),
    ):
        if values.ndim != 1 or values.dtype.kind not in "iu" or values.dtype.kind == "b":
            raise ValueError(f"{name} must be a one-dimensional integer array")
        if values.size and any(
            int(value) < np.iinfo(np.int64).min or int(value) > np.iinfo(np.int64).max
            for value in values
        ):
            raise ValueError(f"{name} contains an out-of-range value")
    if not (start_values.size == stop_values.size == state_values.size > 0):
        raise ValueError("state bounds and values must have equal positive length")
    starts64 = start_values.astype(np.int64, copy=False)
    stops64 = stop_values.astype(np.int64, copy=False)
    states64 = state_values.astype(np.int64, copy=False)
    samples64 = sample_values.astype(np.int64, copy=False)
    if np.any(starts64 >= stops64) or np.any(starts64[1:] != stops64[:-1]):
        raise ValueError("state intervals must be positive and contiguous")
    if np.any(states64 % LATTICE_UM):
        raise ValueError("states must lie on the exact 40-um lattice")
    indices = np.searchsorted(stops64, samples64, side="right")
    if np.any(samples64 < starts64[0]) or np.any(indices >= len(starts64)):
        raise ValueError("sample lies outside the explicit state bounds")
    if np.any(samples64 < starts64[indices]):
        raise ValueError("sample lies in an uncovered state interval")
    return states64[indices]


def _coerce_event(value: K9Event | Mapping[str, Any]) -> K9Event:
    if isinstance(value, K9Event):
        event = value
    elif isinstance(value, Mapping):
        try:
            event = K9Event(**value)
        except (TypeError, ValueError) as exc:
            raise ValueError("event fields do not match K9Event") from exc
    else:
        raise ValueError("events must be K9Event objects or mappings")
    if not event.event_id or not event.biological_unit_id or not event.template_id:
        raise ValueError("event, biological-unit and template identifiers must be nonempty")
    if event.frame not in ALLOWED_FRAMES:
        raise ValueError("event frame must be tissue or probe")
    if event.role not in ALLOWED_ROLES:
        raise ValueError("unknown K9 event role")
    return event


def _collision_groups(intervals: list[tuple[int, int, str]]) -> dict[str, int]:
    result: dict[str, int] = {}
    group = 0
    active_stop = -1
    for start, stop, event_id in sorted(intervals):
        if start >= active_stop:
            group += 1
            active_stop = stop
        else:
            active_stop = max(active_stop, stop)
        result[event_id] = group
    return result


def inject_k9_precar(
    background_uv: np.ndarray,
    geometry_um: np.ndarray,
    templates: Mapping[str, np.ndarray],
    events: Sequence[K9Event | Mapping[str, Any]],
    *,
    sampling_frequency_hz: float,
    session_start_frame: int,
    edge_guard_samples: int = 2,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Inject exact-lattice events before CAR and return voltage plus truth rows."""
    background = np.asarray(background_uv)
    if background.dtype != np.float32 or background.ndim != 2 or min(background.shape) == 0:
        raise TypeError("background must be a nonempty time-by-channel float32 array")
    if not np.all(np.isfinite(background)):
        raise ValueError("background contains non-finite voltage")
    geometry_raw = np.asarray(geometry_um)
    if (
        geometry_raw.shape != (background.shape[1], 2)
        or geometry_raw.dtype.kind not in "iuf"
        or geometry_raw.dtype.kind == "b"
    ):
        raise ValueError("geometry must be a real numeric array matching the background channels")
    geometry = geometry_raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(geometry)):
        raise ValueError("geometry must be finite")
    if len({tuple(row) for row in geometry}) != len(geometry):
        raise ValueError("geometry contains duplicate physical sites")
    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be finite and positive")
    session_origin = _int64_scalar("session_start_frame", session_start_frame)
    if not templates:
        raise ValueError("templates must be a nonempty mapping")
    checked_templates: dict[str, np.ndarray] = {}
    for template_id, value in templates.items():
        if not isinstance(template_id, str) or not template_id:
            raise ValueError("template identifiers must be nonempty strings")
        template = np.asarray(value)
        if template.dtype != np.float32 or template.ndim != 2:
            raise TypeError("templates must be sample-by-channel float32 arrays")
        if template.shape[1] != background.shape[1] or template.shape[0] % 2 != 1:
            raise ValueError("templates must span the full probe and have odd sample width")
        checked_templates[template_id] = validate_template(
            template, edge_guard_samples=edge_guard_samples
        )

    parsed = [_coerce_event(value) for value in events]
    identifiers = [event.event_id for event in parsed]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate event_id")
    output = background.copy()
    rows: list[dict[str, Any]] = []
    intervals: list[tuple[int, int, str]] = []
    for event in parsed:
        if event.template_id not in checked_templates:
            raise ValueError(f"unknown template_id {event.template_id}")
        sample = _int64_scalar("sample_index", event.sample_index)
        base = _lattice_scalar("base_shift_um", event.base_shift_um)
        true_state = _lattice_scalar("true_state_um", event.true_state_um)
        supplied_state = _lattice_scalar("supplied_state_um", event.supplied_state_um)
        if isinstance(event.amplitude_scale, (bool, np.bool_)):
            raise ValueError("amplitude_scale must be finite and positive")
        amplitude = float(event.amplitude_scale)
        if not np.isfinite(amplitude) or amplitude <= 0:
            raise ValueError("amplitude_scale must be finite and positive")
        placement_shift = base + (true_state if event.frame == "tissue" else 0)
        mapping = exact_same_column_map(geometry, placement_shift)
        template = checked_templates[event.template_id]
        active_source_channels = np.any(template != 0, axis=0)
        if np.any(active_source_channels & (mapping < 0)):
            raise ValueError(f"event {event.event_id} would truncate nonzero spatial support")
        moved = exact_remap(template, mapping)
        half = template.shape[0] // 2
        start = sample - half
        stop = start + template.shape[0]
        if start < 0 or stop > background.shape[0]:
            raise ValueError(f"event {event.event_id} crosses a time boundary")
        output[start:stop] += np.float32(amplitude) * moved
        intervals.append((start, stop, event.event_id))
        rows.append({
            "schema": SCHEMA,
            "event_id": event.event_id,
            "biological_unit_id": event.biological_unit_id,
            "template_id": event.template_id,
            "role": event.role,
            "frame": event.frame,
            "local_sample": sample,
            "session_sample": _int64_scalar("session_sample", session_origin + sample),
            "time_s_local": sample / float(sampling_frequency_hz),
            "base_shift_um": base,
            "true_state_um": true_state,
            "supplied_state_um": supplied_state,
            "field_error_um": supplied_state - true_state,
            "placement_shift_um": placement_shift,
            "amplitude_scale": amplitude,
            "sample_start": start,
            "sample_stop": stop,
            "decoy_group": event.decoy_group,
        })
    collision = _collision_groups(intervals)
    for row in rows:
        row["collision_group"] = collision[row["event_id"]]
    return output, rows


def validate_complementary_decoys(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Require each nonempty decoy group to contain two identities in disjoint states."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        group = str(row.get("decoy_group", ""))
        if group:
            groups.setdefault(group, []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for group, members in groups.items():
        identities = sorted({str(row["biological_unit_id"]) for row in members})
        if len(identities) != 2:
            raise ValueError(f"decoy group {group} must contain exactly two biological identities")
        state_sets = {
            identity: {int(row["true_state_um"]) for row in members if str(row["biological_unit_id"]) == identity}
            for identity in identities
        }
        if state_sets[identities[0]] & state_sets[identities[1]]:
            raise ValueError(f"decoy group {group} states are not complementary")
        summary[group] = {"biological_unit_ids": identities, "states_by_unit": state_sets}
    return summary
