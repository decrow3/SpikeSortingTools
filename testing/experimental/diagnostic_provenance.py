"""Default-disabled, standalone diagnostic provenance writer.

This module is experimental test tooling. It is not imported by DARTsort and
does not integrate with an active sorting path.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np


SCHEMA = "dartsort-diagnostic-provenance-v1"
ARRAY_NAMES = ("temporal_components", "spatial_components", "norm_discount")
DEFAULT_MAX_OUTPUT_BYTES = 100_000_000


class ProvenanceValidationError(ValueError):
    pass


def _canonical_json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _is_hash(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _array_bytes(array: np.ndarray) -> bytes:
    value = np.ascontiguousarray(array)
    if value.dtype.hasobject:
        raise ProvenanceValidationError("object arrays are forbidden")
    stream = io.BytesIO()
    np.lib.format.write_array(stream, value, allow_pickle=False)
    return stream.getvalue()


def _array_ref(name: str, array: np.ndarray, payload: bytes) -> dict:
    value = np.asarray(array)
    digest = _sha256_bytes(payload)
    return {
        "status": "available",
        "logical_name": name,
        "path": f"objects/{digest}.npy",
        "sha256": digest,
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "encoding": "npy-v1-content-addressed",
        "bytes": len(payload),
    }


def _require_fields(row: dict, names: set[str], label: str) -> None:
    missing = names.difference(row)
    if missing:
        raise ProvenanceValidationError(f"{label} missing {sorted(missing)}")


def _validate_representation_ref(name: str, ref: dict) -> None:
    _require_fields(ref, {"status", "path", "sha256", "shape", "dtype", "encoding", "bytes"},
                    f"representation {name}")
    valid_shape = (
        isinstance(ref["shape"], list)
        and all(_is_int(size) and size >= 0 for size in ref["shape"])
    )
    try:
        dtype = np.dtype(ref["dtype"])
    except (TypeError, ValueError):
        dtype = None
    if (ref["status"] != "available" or not _is_hash(ref["sha256"])
            or ref["encoding"] != "npy-v1-content-addressed"
            or not _is_int(ref["bytes"]) or ref["bytes"] <= 0
            or not valid_shape or dtype is None or dtype.hasobject):
        raise ProvenanceValidationError(f"invalid representation {name}")
    if Path(ref["path"]).is_absolute() or ".." in Path(ref["path"]).parts:
        raise ProvenanceValidationError("array references must stay inside the bundle")
    if ref["path"] != f"objects/{ref['sha256']}.npy":
        raise ProvenanceValidationError("array reference is not content-addressed")


def validate_manifest_semantics(manifest: dict) -> None:
    _require_fields(
        manifest,
        {"schema", "artifact_id", "parent_stage", "clock", "sources", "representations",
         "template_construction", "events", "candidates", "score_conventions"},
        "manifest",
    )
    if manifest["schema"] != SCHEMA:
        raise ProvenanceValidationError("unexpected schema")
    clock = manifest["clock"]
    _require_fields(clock, {"kind", "origin_sample", "origin_time_s", "sampling_frequency_hz",
                            "start_sample", "stop_sample", "bounds"}, "clock")
    for name in ("origin_sample", "start_sample", "stop_sample"):
        if not _is_int(clock[name]):
            raise ProvenanceValidationError(f"clock {name} must be an integer sample")
    if not _is_finite_number(clock["origin_time_s"]):
        raise ProvenanceValidationError("clock origin_time_s must be finite")
    if (clock["bounds"] != "half-open"
            or not _is_finite_number(clock["sampling_frequency_hz"])
            or clock["sampling_frequency_hz"] <= 0):
        raise ProvenanceValidationError("clock must have positive rate and half-open bounds")
    if not clock["start_sample"] < clock["stop_sample"]:
        raise ProvenanceValidationError("empty or reversed clock bounds")

    sources = manifest["sources"]
    _require_fields(sources, {"source", "config", "geometry", "coordinate_semantics"}, "sources")
    for name in ("source", "config", "geometry"):
        if not _is_hash(sources[name]["sha256"]):
            raise ProvenanceValidationError(f"invalid {name} hash")
    _require_fields(sources["coordinate_semantics"], {"time", "channel", "depth", "units"},
                    "coordinate semantics")

    reps = manifest["representations"]
    if set(reps) != set(ARRAY_NAMES):
        raise ProvenanceValidationError("representation keys must be exact")
    for name, ref in reps.items():
        if ref.get("status") == "unavailable":
            if not ref.get("reason"):
                raise ProvenanceValidationError(f"unavailable {name} needs a reason")
            if any(key in ref for key in ("path", "sha256", "shape", "dtype", "bytes")):
                raise ProvenanceValidationError(f"unavailable {name} must not manufacture a reference")
            continue
        _validate_representation_ref(name, ref)

    template_ids = set()
    for i, row in enumerate(manifest["template_construction"]):
        _require_fields(row, {"template_id", "membership_status", "member_event_ids",
                              "bank_unit_ids", "parent_stage"}, f"template row {i}")
        if row["template_id"] in template_ids:
            raise ProvenanceValidationError("duplicate template_id")
        template_ids.add(row["template_id"])
        status = row["membership_status"]
        if status == "exact":
            if not isinstance(row["member_event_ids"], list) or not row["member_event_ids"]:
                raise ProvenanceValidationError("exact membership needs event IDs")
            if len(row["bank_unit_ids"]) != 1:
                raise ProvenanceValidationError("exact membership needs one bank unit ID")
        elif status == "ambiguous":
            if not isinstance(row["bank_unit_ids"], list) or len(row["bank_unit_ids"]) < 2:
                raise ProvenanceValidationError("ambiguous membership needs candidate labels")
        elif status == "unknown":
            if row["member_event_ids"] is not None or row["bank_unit_ids"] is not None:
                raise ProvenanceValidationError("unknown membership must not manufacture records")
        else:
            raise ProvenanceValidationError("invalid membership status")

    events = {}
    for i, row in enumerate(manifest["events"]):
        _require_fields(row, {"event_id", "candidate_origin_id", "parent_stage", "sample_index",
                              "availability"}, f"event row {i}")
        if row["event_id"] in events:
            raise ProvenanceValidationError("duplicate event_id")
        if not _is_int(row["sample_index"]):
            raise ProvenanceValidationError("event sample_index must be an integer sample")
        if not clock["start_sample"] <= row["sample_index"] < clock["stop_sample"]:
            raise ProvenanceValidationError("event outside half-open clock bounds")
        if row["availability"] not in ("available", "unavailable"):
            raise ProvenanceValidationError("invalid event availability")
        events[row["event_id"]] = row

    candidate_ids = set()
    selected_by_snapshot = {}
    for i, row in enumerate(manifest["candidates"]):
        _require_fields(row, {"candidate_id", "event_id", "template_id", "parent_stage", "stage",
                              "iteration", "evaluation_status", "role", "ranking_scope", "score",
                              "scale", "threshold"}, f"candidate row {i}")
        if row["candidate_id"] in candidate_ids:
            raise ProvenanceValidationError("duplicate candidate_id")
        candidate_ids.add(row["candidate_id"])
        if row["event_id"] not in events:
            raise ProvenanceValidationError("candidate references unknown event")
        if row["template_id"] is not None and row["template_id"] not in template_ids:
            raise ProvenanceValidationError("candidate references unknown template")
        if not _is_int(row["iteration"]) or row["iteration"] < 0:
            raise ProvenanceValidationError("candidate iteration must be a nonnegative integer")
        status, role, scope = row["evaluation_status"], row["role"], row["ranking_scope"]
        if status == "not_evaluated":
            if row["score"] is not None or role != "not_evaluated" or scope != "none":
                raise ProvenanceValidationError("not-evaluated candidate must not have a score/rank")
        elif status == "evaluated_unmatched":
            if (role != "unmatched" or scope != "global_final"
                    or row["stage"] == "coarse" or row["template_id"] is not None
                    or row["score"] is not None):
                raise ProvenanceValidationError("evaluated-unmatched must remain template-free")
        elif status == "evaluated":
            if not _is_finite_number(row["score"]) or row["template_id"] is None:
                raise ProvenanceValidationError("evaluated candidate needs score and template")
            allowed_role_scopes = {
                ("coarse_selected", "coarse_local"),
                ("coarse_runner_up", "coarse_local"),
                ("selected", "global_final"),
                ("global_final_runner_up", "global_final"),
            }
            if (role, scope) not in allowed_role_scopes:
                raise ProvenanceValidationError("invalid evaluated role/ranking scope")
            if scope == "coarse_local" and row["stage"] != "coarse":
                raise ProvenanceValidationError("coarse-local candidate must use coarse stage")
            if scope == "global_final" and row["stage"] == "coarse":
                raise ProvenanceValidationError("coarse candidate cannot claim global-final rank")
            if role in ("coarse_selected", "selected"):
                snapshot = (row["event_id"], row["stage"], row["iteration"])
                selected_by_snapshot[snapshot] = selected_by_snapshot.get(snapshot, 0) + 1
        else:
            raise ProvenanceValidationError("invalid evaluation status")
        _require_fields(row["scale"], {"value", "convention"}, "scale convention")
        _require_fields(row["threshold"], {"value", "convention"}, "threshold convention")
        if not _is_finite_number(row["scale"]["value"]):
            raise ProvenanceValidationError("scale value must be finite")
        if not _is_finite_number(row["threshold"]["value"]):
            raise ProvenanceValidationError("threshold value must be finite")
    if any(count > 1 for count in selected_by_snapshot.values()):
        raise ProvenanceValidationError("multiple selected candidates in one event/stage/iteration snapshot")


class DiagnosticProvenanceWriter:
    def __init__(self, *, enabled: bool = False, max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES):
        self.enabled = bool(enabled)
        self.max_output_bytes = int(max_output_bytes)

    def write(self, output: Path, manifest: dict, *, arrays: dict[str, np.ndarray] | None = None) -> dict:
        """Write one validated bundle atomically.

        Available representation payloads must be supplied through ``arrays``.
        References already present in ``manifest`` are deliberately unsupported:
        accepting them would make their provenance and local availability
        ambiguous. Missing representations must be explicitly unavailable.
        """
        output = Path(output)
        if not self.enabled:
            return {"status": "disabled", "written": False, "path": str(output)}
        if output.exists():
            raise FileExistsError(output)
        partial = output.with_name(output.name + ".partial")
        if partial.exists():
            raise FileExistsError(partial)
        value = copy.deepcopy(manifest)
        value.setdefault("representations", {})
        for name, ref in value["representations"].items():
            if ref.get("status") == "available":
                _validate_representation_ref(name, ref)
                raise ProvenanceValidationError(
                    f"preexisting available reference unsupported; supply array payload: {name}"
                )
        payloads = {}
        for name, array in (arrays or {}).items():
            if name not in ARRAY_NAMES:
                raise ProvenanceValidationError(f"unknown array role {name}")
            payload = _array_bytes(np.asarray(array))
            ref = _array_ref(name, np.asarray(array), payload)
            value["representations"][name] = ref
            payloads[ref["sha256"]] = payload
        validate_manifest_semantics(value)
        manifest_bytes = _canonical_json(value)
        predicted = len(manifest_bytes) + sum(map(len, payloads.values())) + 1024
        if predicted > self.max_output_bytes:
            raise ProvenanceValidationError("bounded output limit exceeded")

        partial.mkdir(parents=True)
        (partial / "objects").mkdir()
        for digest, payload in sorted(payloads.items()):
            (partial / "objects" / f"{digest}.npy").write_bytes(payload)
        (partial / "manifest.json").write_bytes(manifest_bytes)
        complete = {
            "schema": SCHEMA,
            "status": "complete",
            "manifest_sha256": _sha256_bytes(manifest_bytes),
            "object_count": len(payloads),
            "total_bytes": sum(p.stat().st_size for p in partial.rglob("*") if p.is_file()),
        }
        (partial / "COMPLETE.json").write_bytes(_canonical_json(complete))
        validate_bundle(partial)
        os.replace(partial, output)
        return {"status": "complete", "written": True, "path": str(output), **complete}


def validate_bundle(output: Path) -> dict:
    output = Path(output)
    complete_path = output / "COMPLETE.json"
    manifest_path = output / "manifest.json"
    if not complete_path.is_file() or not manifest_path.is_file():
        raise ProvenanceValidationError("incomplete bundle: COMPLETE.json and manifest.json required")
    complete = json.loads(complete_path.read_text())
    if complete.get("status") != "complete" or complete.get("schema") != SCHEMA:
        raise ProvenanceValidationError("invalid completion marker")
    if _sha256_file(manifest_path) != complete.get("manifest_sha256"):
        raise ProvenanceValidationError("manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    validate_manifest_semantics(manifest)
    seen = set()
    for name, ref in manifest["representations"].items():
        if ref["status"] == "unavailable":
            continue
        path = output / ref["path"]
        if not path.is_file() or _sha256_file(path) != ref["sha256"]:
            raise ProvenanceValidationError(f"corrupt or missing array reference: {name}")
        if path.stat().st_size != ref["bytes"]:
            raise ProvenanceValidationError(f"array byte-count mismatch: {name}")
        array = np.load(path, allow_pickle=False)
        if list(array.shape) != ref["shape"] or array.dtype.str != ref["dtype"]:
            raise ProvenanceValidationError(f"array metadata mismatch: {name}")
        seen.add(ref["sha256"])
    if complete.get("object_count") != len(seen):
        raise ProvenanceValidationError("completion object count mismatch")
    return {"status": "valid", "artifact_id": manifest["artifact_id"], "object_count": len(seen)}
