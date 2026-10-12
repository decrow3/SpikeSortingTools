"""Fail-closed structural validator for a bounded joint-model N0/N1 contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import PurePath
from typing import Any

import numpy as np


SCHEMA = "joint-model-n0-n1-run-contract-v1"
EXACT_DARTSORT_COMMIT = "d40c516e47451e688dd4484813b562fa5fc0b4ef"
PLACEHOLDERS = {"", "TODO", "TBD", "UNKNOWN", "UNRESOLVED", "<path>", "<sha256>"}


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _exact_keys(name: str, value: Mapping[str, Any], keys: set[str]) -> None:
    if set(value) != keys:
        raise ValueError(f"{name} keys must be exactly {sorted(keys)}")


def _sha(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be hexadecimal SHA-256") from exc
    return value.lower()


def _string(name: str, value: Any) -> str:
    if not isinstance(value, str) or value.strip() in PLACEHOLDERS:
        raise ValueError(f"{name} must be a resolved non-placeholder string")
    return value


def _absolute_path(name: str, value: Any) -> str:
    path = _string(name, value)
    if not PurePath(path).is_absolute():
        raise ValueError(f"{name} must be absolute")
    return path


def _integer(name: str, value: Any, minimum: int | None = None) -> int:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise ValueError(f"{name} must be an exact integer")
    result = int(array)
    if not np.iinfo(np.int64).min <= result <= np.iinfo(np.int64).max:
        raise ValueError(f"{name} is outside int64 range")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _positive_real(name: str, value: Any) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf" or array.dtype.kind == "b":
        raise ValueError(f"{name} must be a finite positive real scalar")
    result = float(array)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive real scalar")
    return result


def validate_n0_n1_contract(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate structure and cross-arm invariants without opening referenced data."""
    contract = _mapping("contract", document)
    top_keys = {"schema", "source", "input", "arms", "evaluator", "resources", "outputs", "stop_conditions"}
    _exact_keys("contract", contract, top_keys)
    if contract["schema"] != SCHEMA:
        raise ValueError("unexpected N0-N1 contract schema")

    source = _mapping("source", contract["source"])
    _exact_keys("source", source, {"dartsort_commit", "patch_sha256", "import_closure_sha256", "environment_sha256"})
    if source["dartsort_commit"] != EXACT_DARTSORT_COMMIT:
        raise ValueError("DARTsort commit differs from the exact frozen baseline")
    for key in ("patch_sha256", "import_closure_sha256", "environment_sha256"):
        _sha(f"source.{key}", source[key])

    inputs = _mapping("input", contract["input"])
    _exact_keys(
        "input", inputs,
        {"recording", "channel_ids_sha256", "geometry_sha256", "motion_field", "cohort_sha256", "windows"},
    )
    recording = _mapping("input.recording", inputs["recording"])
    _exact_keys(
        "input.recording", recording,
        {"path", "content_sha256", "bytes", "dtype", "num_samples", "num_channels", "sampling_frequency_hz"},
    )
    _absolute_path("recording.path", recording["path"])
    _sha("recording.content_sha256", recording["content_sha256"])
    _integer("recording.bytes", recording["bytes"], 1)
    _string("recording.dtype", recording["dtype"])
    total_samples = _integer("recording.num_samples", recording["num_samples"], 1)
    _integer("recording.num_channels", recording["num_channels"], 1)
    _positive_real("recording.sampling_frequency_hz", recording["sampling_frequency_hz"])
    _sha("input.channel_ids_sha256", inputs["channel_ids_sha256"])
    _sha("input.geometry_sha256", inputs["geometry_sha256"])
    _sha("input.cohort_sha256", inputs["cohort_sha256"])

    motion = _mapping("input.motion_field", inputs["motion_field"])
    _exact_keys("input.motion_field", motion, {"path", "sha256", "clock", "sign", "lattice_um"})
    _absolute_path("motion_field.path", motion["path"])
    _sha("motion_field.sha256", motion["sha256"])
    _string("motion_field.clock", motion["clock"])
    _string("motion_field.sign", motion["sign"])
    if _integer("motion_field.lattice_um", motion["lattice_um"], 1) != 40:
        raise ValueError("motion-field lattice must be exactly 40 um")

    windows = inputs["windows"]
    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)) or not windows:
        raise ValueError("input.windows must be a nonempty sequence")
    window_ids = set()
    for index, raw_window in enumerate(windows):
        window = _mapping(f"window[{index}]", raw_window)
        _exact_keys(f"window[{index}]", window, {"window_id", "start_frame", "stop_frame", "padding_samples", "state_intervals"})
        window_id = _string("window_id", window["window_id"])
        if window_id in window_ids:
            raise ValueError("window IDs must be unique")
        window_ids.add(window_id)
        start = _integer("window.start_frame", window["start_frame"], 0)
        stop = _integer("window.stop_frame", window["stop_frame"], 1)
        padding = _integer("window.padding_samples", window["padding_samples"], 0)
        if not start < stop <= total_samples or start - padding < 0 or stop + padding > total_samples:
            raise ValueError("window or padding lies outside the recording")
        intervals = window["state_intervals"]
        if not isinstance(intervals, Sequence) or isinstance(intervals, (str, bytes)) or not intervals:
            raise ValueError("window state intervals must be nonempty")
        prior_stop = start
        for raw_interval in intervals:
            interval = _mapping("state interval", raw_interval)
            _exact_keys("state interval", interval, {"start_frame", "stop_frame", "state_um"})
            interval_start = _integer("state.start_frame", interval["start_frame"], 0)
            interval_stop = _integer("state.stop_frame", interval["stop_frame"], 1)
            state = _integer("state.state_um", interval["state_um"])
            if state % 40:
                raise ValueError("state must lie on the exact 40-um lattice")
            if interval_start != prior_stop or not interval_start < interval_stop <= stop:
                raise ValueError("state intervals must be contiguous, nonoverlapping and within the window")
            prior_stop = interval_stop
        if prior_stop != stop:
            raise ValueError("state intervals must cover the complete unpadded window")

    arms = contract["arms"]
    if not isinstance(arms, Sequence) or isinstance(arms, (str, bytes)) or len(arms) != 2:
        raise ValueError("arms must contain exactly N0 and N1")
    arm_by_name = {}
    arm_keys = {"name", "input_binding_sha256", "checkpoint_sha256", "common_config_sha256", "segment_owned_matching", "other_deltas"}
    for raw_arm in arms:
        arm = _mapping("arm", raw_arm)
        _exact_keys("arm", arm, arm_keys)
        name = arm["name"]
        if name not in {"N0", "N1"} or name in arm_by_name:
            raise ValueError("arms must contain unique N0 and N1 entries")
        for key in ("input_binding_sha256", "checkpoint_sha256", "common_config_sha256"):
            _sha(f"arm.{key}", arm[key])
        if type(arm["segment_owned_matching"]) is not bool:
            raise ValueError("segment_owned_matching must be boolean")
        if arm["other_deltas"] != []:
            raise ValueError("N0/N1 contract forbids unlisted arm deltas")
        arm_by_name[name] = arm
    if arm_by_name["N0"]["segment_owned_matching"] or not arm_by_name["N1"]["segment_owned_matching"]:
        raise ValueError("N0 must disable and N1 must enable segment ownership")
    for key in ("input_binding_sha256", "checkpoint_sha256", "common_config_sha256"):
        if arm_by_name["N0"][key] != arm_by_name["N1"][key]:
            raise ValueError(f"N0 and N1 must share {key}")

    evaluator = _mapping("evaluator", contract["evaluator"])
    _exact_keys("evaluator", evaluator, {"implementation_sha256", "review_complete_sha256", "endpoints", "inconclusive_is_equivalence"})
    _sha("evaluator.implementation_sha256", evaluator["implementation_sha256"])
    _sha("evaluator.review_complete_sha256", evaluator["review_complete_sha256"])
    if evaluator["inconclusive_is_equivalence"] is not False:
        raise ValueError("inconclusive may not be interpreted as equivalence")
    endpoints = _mapping("evaluator.endpoints", evaluator["endpoints"])
    if not endpoints:
        raise ValueError("at least one endpoint rule is required")
    for name, raw_rule in endpoints.items():
        _string("endpoint name", name)
        rule = _mapping(f"endpoint {name}", raw_rule)
        _exact_keys(f"endpoint {name}", rule, {"direction", "practical_loss_margin"})
        if rule["direction"] not in {"higher", "lower"}:
            raise ValueError("endpoint direction must be higher or lower")
        margin = np.asarray(rule["practical_loss_margin"])
        if margin.ndim != 0 or margin.dtype.kind not in "iuf" or margin.dtype.kind == "b":
            raise ValueError("practical loss margin must be a finite nonnegative real")
        margin_value = float(margin)
        if not np.isfinite(margin_value) or margin_value < 0:
            raise ValueError("practical loss margin must be a finite nonnegative real")

    resources = _mapping("resources", contract["resources"])
    _exact_keys("resources", resources, {"host", "runtime_limit_s", "memory_limit_bytes", "disk_free_gate_bytes", "gpu_required", "managed_service"})
    _string("resources.host", resources["host"])
    _integer("resources.runtime_limit_s", resources["runtime_limit_s"], 1)
    _integer("resources.memory_limit_bytes", resources["memory_limit_bytes"], 1)
    _integer("resources.disk_free_gate_bytes", resources["disk_free_gate_bytes"], 1)
    if resources["gpu_required"] is not True:
        raise ValueError("medium N0/N1 contract must require an actual GPU preflight")
    _string("resources.managed_service", resources["managed_service"])

    outputs = _mapping("outputs", contract["outputs"])
    _exact_keys("outputs", outputs, {"N0", "N1", "create_only", "preserve_failures"})
    n0_output = _absolute_path("outputs.N0", outputs["N0"])
    n1_output = _absolute_path("outputs.N1", outputs["N1"])
    if n0_output == n1_output or outputs["create_only"] is not True or outputs["preserve_failures"] is not True:
        raise ValueError("outputs must be distinct, create-only and failure-preserving")
    stop_conditions = contract["stop_conditions"]
    if not isinstance(stop_conditions, Sequence) or isinstance(stop_conditions, (str, bytes)) or not stop_conditions:
        raise ValueError("stop_conditions must be a nonempty sequence")
    for condition in stop_conditions:
        _string("stop condition", condition)

    return {
        "schema": SCHEMA,
        "status": "STRUCTURALLY_VALID_METADATA_ONLY",
        "window_count": len(windows),
        "window_ids": sorted(window_ids),
        "endpoint_names": sorted(endpoints),
        "does_not_verify_referenced_bytes": True,
        "does_not_release_execution": True,
    }
