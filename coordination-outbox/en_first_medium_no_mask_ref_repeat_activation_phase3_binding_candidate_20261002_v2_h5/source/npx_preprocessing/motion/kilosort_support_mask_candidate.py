"""Opt-in Kilosort 4 pointwise support mask at the single pre-W boundary.

This is an implementation candidate, not an authorization to read a recording
or launch a sort.  Disabled operation delegates exactly to Kilosort's native
``BinaryFiltered``.  Enabled operation runs the same native channel selection,
centering, CAR, high-pass and artifact logic, applies one geometry/time support
mask, then applies the unchanged whitening matrix.  No post-W mask is applied.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import numbers
from typing import Any

import numpy as np
import torch

from .prewhitening_support_mask import support_mask_from_geometry_and_field

EXACT_LATTICE_LINEAGE = (
    "npx_preprocessing.motion.lattice_remap_si.ExactLatticeRemapRecording"
)
REVIEWED_BASE_CONFIG_PATH = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "en_minimum_training_support_mask_execution_readiness_20261001_v1_h5/"
    "en_minimum_training_support_mask.infinity_hotfix.review_pending.v1.json"
)
REVIEWED_BASE_CONFIG_SHA256 = (
    "74f20c49418bbe4ad9a530da8a477b72093569fe2eec319682405fbd4431cb47"
)
SMOKE_ALLOWED_DELTAS = (
    (("key", "base_binding"), ("key", "path")),
    (("key", "base_binding"), ("key", "sha256")),
    (("key", "failure_reason"),),
    (("key", "native_invocation"), ("key", "results_dir")),
    (("key", "schema"),),
    (("key", "smoke"), ("key", "launch_evidence_dir")),
    (("key", "smoke"), ("key", "mode")),
    (("key", "smoke"), ("key", "run_root")),
    (("key", "smoke"), ("key", "stop_after_whitening")),
    (("key", "source_bindings"), ("index", 0), ("key", "sha256")),
    (("key", "source_bindings"), ("index", 2), ("key", "sha256")),
    (("key", "status"),),
)
SMOKE_RUN_ROOT = Path(
    "/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/"
    "en_minimum_training_support_mask_covariance_smoke_20261001_v2"
)
SMOKE_STATUS = "smoke_enabled_contract_review_pending_unlaunched"
SMOKE_FAILURE_REASON = (
    "Enabled contract is unlaunched pending H1 changed-delta review and normal "
    "tool approval"
)
SMOKE_SOURCE_HASH_INDICES = (0, 2)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


@dataclass(frozen=True)
class ValidatedSupportMaskContract:
    config_sha256: str
    contract_digest: str
    execution_enabled: bool
    geometry: np.ndarray = field(compare=False, repr=False)
    temporal_centers_s: np.ndarray = field(compare=False, repr=False)
    shifts_um: np.ndarray = field(compare=False, repr=False)
    fs: float
    imin: int
    imax: int
    n_channels: int
    recording_manifest_sha256: str
    recording_content_sha256: str
    field_sha256: str


@dataclass(frozen=True)
class CovarianceValidation:
    passed: bool
    reasons: tuple[str, ...]
    shape: tuple[int, ...]
    finite: bool
    rank: int | None
    condition_number: float | None
    maximum_condition_number: float
    covariance_sha256: str
    contract_digest: str


@dataclass
class WhiteningBoundaryAudit:
    computation_digest: str
    covariance_validation: CovarianceValidation | None = None
    whitening_sha256: str | None = None
    whitening_finite: bool | None = None
    sampled_batch_indices: tuple[int, ...] = ()


class SmokeStopAfterWhitening(RuntimeError):
    """Intentional bounded stop after durable covariance and W evidence."""


class RequiredWhiteningBoundaryMissing(RuntimeError):
    """Native runner returned without computing the required audited W."""


def validate_covariance(
    covariance: np.ndarray, *, expected_channels: int,
    contract_digest: str, maximum_condition_number: float = 1e8,
) -> CovarianceValidation:
    """Pure fail-closed covariance check suitable for synthetic fixtures."""
    value = np.asarray(covariance)
    reasons: list[str] = []
    expected_shape = (int(expected_channels), int(expected_channels))
    if value.shape != expected_shape:
        reasons.append(f"wrong_shape:{value.shape}!={expected_shape}")
    finite = bool(np.issubdtype(value.dtype, np.number) and np.isfinite(value).all())
    if not finite:
        reasons.append("nonfinite")
    rank: int | None = None
    condition: float | None = None
    if not reasons:
        rank = int(np.linalg.matrix_rank(value))
        if rank != expected_channels:
            reasons.append(f"rank_deficient:{rank}<{expected_channels}")
        condition = float(np.linalg.cond(value))
        if not np.isfinite(condition) or condition > float(maximum_condition_number):
            reasons.append(f"over_conditioned:{condition}")
    return CovarianceValidation(
        passed=not reasons, reasons=tuple(reasons), shape=tuple(value.shape),
        finite=finite, rank=rank, condition_number=condition,
        maximum_condition_number=float(maximum_condition_number),
        covariance_sha256=_array_digest(value), contract_digest=str(contract_digest),
    )


def _require_hash(path: str | Path, expected: str, label: str) -> None:
    observed = _sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} hash mismatch: {observed} != {expected}")


def _setting(ops: dict, dotted: str):
    value: Any = ops
    for part in dotted.split("."):
        value = value[part]
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    """Replace one small JSON receipt without exposing a partial file."""
    path.parent.mkdir(parents=False, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    """Replace one small NumPy intermediate without suffix rewriting."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _typed_structural_map(
    value: Any, path: tuple[tuple[str, Any], ...] = (),
) -> dict[tuple[tuple[str, Any], ...], tuple[Any, ...]]:
    """Map every JSON node using injective typed path components and values."""
    nodes: dict[tuple[tuple[str, Any], ...], tuple[Any, ...]] = {}
    if isinstance(value, dict):
        nodes[path] = ("dict", len(value))
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be exact strings")
            nodes.update(_typed_structural_map(child, path + (("key", key),)))
    elif isinstance(value, list):
        nodes[path] = ("list", len(value))
        for index, child in enumerate(value):
            nodes.update(_typed_structural_map(child, path + (("index", index),)))
    else:
        scalar_kind = (
            "null" if value is None else "bool" if type(value) is bool else
            "int" if type(value) is int else "float" if type(value) is float else
            "string" if type(value) is str else None
        )
        if scalar_kind is None:
            raise ValueError(f"unsupported JSON scalar type: {type(value).__name__}")
        nodes[path] = ("scalar", scalar_kind, value)
    return nodes


def _structural_diff(base: Any, candidate: Any) -> dict[tuple[tuple[str, Any], ...], tuple[Any, Any]]:
    """Return every changed node from the full injective structural maps."""
    missing = object()
    base_nodes = _typed_structural_map(base)
    candidate_nodes = _typed_structural_map(candidate)
    changed = {}
    for path in set(base_nodes) | set(candidate_nodes):
        old = base_nodes.get(path, missing)
        new = candidate_nodes.get(path, missing)
        if old is missing or new is missing or old != new:
            changed[path] = (None if old is missing else old, None if new is missing else new)
    return changed


def _json_pointer(path: tuple[tuple[str, Any], ...]) -> str:
    """Render diagnostics only; typed tuples above remain the identity keys."""
    tokens = []
    for kind, value in path:
        token = str(value) if kind == "index" else value.replace("~", "~0").replace("/", "~1")
        tokens.append(token)
    return "" if not tokens else "/" + "/".join(tokens)


def _path_sort_key(path: tuple[tuple[str, Any], ...]) -> tuple[Any, ...]:
    return tuple((0, value) if kind == "key" else (1, int(value)) for kind, value in path)


def _set_typed_path(root: Any, path: tuple[tuple[str, Any], ...], value: Any) -> None:
    current = root
    for offset, (kind, component) in enumerate(path):
        last = offset == len(path) - 1
        if kind == "key":
            if not isinstance(current, dict):
                raise ValueError("source allowed path expects a dictionary")
            if last:
                current[component] = value
            else:
                next_kind = path[offset + 1][0]
                current = current.setdefault(component, {} if next_kind == "key" else [])
        elif kind == "index":
            if not isinstance(current, list) or not 0 <= component < len(current):
                raise ValueError("source allowed path expects an existing list index")
            if last:
                current[component] = value
            else:
                current = current[component]
        else:
            raise ValueError("unknown typed path component")


def _expected_smoke_delta_values(
    config: dict[str, Any], base: dict[str, Any],
) -> dict[tuple[tuple[str, Any], ...], Any]:
    """Resolve the exact reviewed value and type for each allowed leaf delta."""
    expected = {
        (("key", "base_binding"), ("key", "path")): str(REVIEWED_BASE_CONFIG_PATH),
        (("key", "base_binding"), ("key", "sha256")): REVIEWED_BASE_CONFIG_SHA256,
        (("key", "failure_reason"),): SMOKE_FAILURE_REASON,
        (("key", "native_invocation"), ("key", "results_dir")): str(SMOKE_RUN_ROOT / "native_results"),
        (("key", "schema"),): "en-minimum-training-support-mask-v4-smoke",
        (("key", "smoke"), ("key", "launch_evidence_dir")): str(SMOKE_RUN_ROOT / "launch_evidence"),
        (("key", "smoke"), ("key", "mode")): "stop_after_covariance_whitening",
        (("key", "smoke"), ("key", "run_root")): str(SMOKE_RUN_ROOT),
        (("key", "smoke"), ("key", "stop_after_whitening")): True,
        (("key", "status"),): SMOKE_STATUS,
    }
    if base.get("execution_enabled") is not True and config.get("execution_enabled") is True:
        expected[(("key", "execution_enabled"),)] = True
    for index in SMOKE_SOURCE_HASH_INDICES:
        item = config["source_bindings"][index]
        path = (("key", "source_bindings"), ("index", index), ("key", "sha256"))
        expected[path] = _sha256_file(item["path"])
    return expected


def _validate_smoke_base_binding(config: dict[str, Any]) -> None:
    """Bind the narrow smoke config to the fixed reviewed v3 base."""
    binding = config.get("base_binding")
    if not isinstance(binding, dict):
        raise ValueError("smoke config requires base_binding")
    if binding.get("path") != str(REVIEWED_BASE_CONFIG_PATH):
        raise ValueError("smoke base path differs from reviewed base")
    if binding.get("sha256") != REVIEWED_BASE_CONFIG_SHA256:
        raise ValueError("smoke base hash differs from reviewed base")
    if set(binding) != {"path", "sha256"}:
        raise ValueError("smoke base_binding fields differ from reviewed policy")
    _require_hash(REVIEWED_BASE_CONFIG_PATH, REVIEWED_BASE_CONFIG_SHA256, "reviewed base config")
    base = json.loads(REVIEWED_BASE_CONFIG_PATH.read_text())
    allowed_paths = set(SMOKE_ALLOWED_DELTAS)
    if base.get("execution_enabled") is not True and config.get("execution_enabled") is True:
        allowed_paths.add((("key", "execution_enabled"),))
    expected_values = _expected_smoke_delta_values(config, base)
    if set(expected_values) != allowed_paths:
        raise ValueError("source smoke allowed paths and expected values differ")
    expected_candidate = copy.deepcopy(base)
    for path, expected in expected_values.items():
        _set_typed_path(expected_candidate, path, expected)
    base_nodes = _typed_structural_map(base)
    candidate_nodes = _typed_structural_map(config)
    expected_nodes = _typed_structural_map(expected_candidate)
    if candidate_nodes != expected_nodes:
        differences = _structural_diff(expected_candidate, config)
        rendered = [
            {"pointer": _json_pointer(path), "typed_path": repr(path)}
            for path in sorted(differences, key=_path_sort_key)
        ]
        raise ValueError(f"smoke full structural map differs from policy: {rendered}")
    scalar_changes = {
        path for path in set(base_nodes) | set(candidate_nodes)
        if base_nodes.get(path) != candidate_nodes.get(path)
        and (base_nodes.get(path, (None,))[0] == "scalar"
             or candidate_nodes.get(path, (None,))[0] == "scalar")
    }
    if scalar_changes != allowed_paths:
        undeclared = sorted(scalar_changes - allowed_paths, key=_path_sort_key)
        unchanged_declared = sorted(allowed_paths - scalar_changes, key=_path_sort_key)
        raise ValueError(
            "smoke scalar delta mismatch: "
            f"undeclared={[repr(path) for path in undeclared]}, "
            f"declared_but_unchanged={[repr(path) for path in unchanged_declared]}"
        )
    smoke = config.get("smoke")
    if not isinstance(smoke, dict) or smoke.get("mode") != "stop_after_covariance_whitening":
        raise ValueError("smoke mode must stop after covariance/whitening")
    if type(smoke.get("stop_after_whitening")) is not bool or not smoke["stop_after_whitening"]:
        raise ValueError("smoke stop_after_whitening must be true")
    run_root = Path(smoke.get("run_root", ""))
    evidence = Path(smoke.get("launch_evidence_dir", ""))
    results = Path(config["native_invocation"]["results_dir"])
    if not run_root.is_absolute() or run_root.name in {"", ".", ".."}:
        raise ValueError("smoke run_root must be a specific absolute path")
    if evidence != run_root / "launch_evidence" or results != run_root / "native_results":
        raise ValueError("smoke evidence/results paths must be distinct children of run_root")


def validate_support_mask_contract(
    config_path: str | Path, *, expected_config_sha256: str,
) -> ValidatedSupportMaskContract:
    """Validate reviewed metadata bindings without opening a voltage binary."""
    config_path = Path(config_path)
    _require_hash(config_path, expected_config_sha256, "config")
    config = json.loads(config_path.read_text())
    if config.get("schema") not in {"en-minimum-training-support-mask-v2", "en-minimum-training-support-mask-v3", "en-minimum-training-support-mask-v4-smoke"}:
        raise ValueError("wrong support-mask contract schema")
    if type(config.get("execution_enabled")) is not bool:
        raise ValueError("execution_enabled must be an exact JSON boolean")
    if config["schema"] == "en-minimum-training-support-mask-v4-smoke":
        _validate_smoke_base_binding(config)
    for item in config["source_bindings"]:
        _require_hash(item["path"], item["sha256"], f"source:{item['path']}")

    recording = config["recording"]
    manifest_path = Path(recording["manifest_path"])
    _require_hash(manifest_path, recording["manifest_sha256"], "recording manifest")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete"):
        raise ValueError("recording manifest is incomplete")
    if manifest.get("adapter") != recording["required_adapter"]:
        raise ValueError("recording manifest ExactLatticeRemap lineage mismatch")
    for key in ("recording_content_sha256", "num_samples", "num_channels",
                "sampling_frequency_hz", "dtype"):
        if manifest.get(key) != recording[key]:
            raise ValueError(f"recording manifest field mismatch: {key}")
    binary_entries = manifest.get("recording_binary_files", [])
    if len(binary_entries) != 1 or binary_entries[0].get("name") != recording["binary_name"]:
        raise ValueError("recording manifest binary identity mismatch")
    if binary_entries[0].get("sha256") != recording["manifest_recorded_binary_sha256"]:
        raise ValueError("recording manifest binary content hash mismatch")
    crop = config["crop"]
    if not (0 <= int(crop["imin"]) < int(crop["imax"]) <= int(manifest["num_samples"])):
        raise ValueError("crop lies outside manifest sample clock")

    field_spec = config["field"]
    _require_hash(field_spec["path"], field_spec["sha256"], "field")
    with np.load(field_spec["path"], allow_pickle=False) as saved:
        centers = np.asarray(saved[field_spec["time_key"]], dtype=np.float64)
        displacement = np.asarray(saved[field_spec["displacement_key"]], dtype=np.float64)
    reference = float(np.median(displacement))
    scaled = (displacement - reference) / float(field_spec["rounding_um"])
    shifts = float(field_spec["rounding_um"]) * np.copysign(
        np.floor(np.abs(scaled) + 0.5), scaled
    )
    geometry = np.asarray(manifest["channel_locations_um"], dtype=np.float64)
    if geometry.shape != (int(manifest["num_channels"]), 2):
        raise ValueError("recording manifest geometry shape mismatch")

    ops_spec = config["ops"]
    _require_hash(ops_spec["path"], ops_spec["sha256"], "ops")
    ops = np.load(ops_spec["path"], allow_pickle=True).item()
    for dotted, expected in config["frozen_kilosort_settings"].items():
        observed = _setting(ops, dotted)
        if expected == "Infinity":
            equal = bool(np.isposinf(observed))
        elif isinstance(expected, float):
            equal = bool(np.isclose(float(observed), expected, rtol=0, atol=1e-12))
        else:
            equal = observed == expected
        if not equal:
            raise ValueError(f"frozen Kilosort setting mismatch: {dotted}")
    if not np.array_equal(np.asarray(ops["chanMap"]), np.arange(len(geometry))):
        raise ValueError("ops chanMap is not full-probe identity")

    digest_payload = {
        "config_sha256": expected_config_sha256,
        "recording_manifest_sha256": recording["manifest_sha256"],
        "recording_content_sha256": recording["recording_content_sha256"],
        "field_sha256": field_spec["sha256"], "ops_sha256": ops_spec["sha256"],
        "crop": crop, "settings": config["frozen_kilosort_settings"],
    }
    contract_digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ValidatedSupportMaskContract(
        config_sha256=expected_config_sha256, contract_digest=contract_digest,
        execution_enabled=config["execution_enabled"], geometry=geometry,
        temporal_centers_s=centers, shifts_um=shifts,
        fs=float(manifest["sampling_frequency_hz"]), imin=int(crop["imin"]),
        imax=int(crop["imax"]), n_channels=int(manifest["num_channels"]),
        recording_manifest_sha256=recording["manifest_sha256"],
        recording_content_sha256=recording["recording_content_sha256"],
        field_sha256=field_spec["sha256"],
    )


@dataclass
class MaskAudit:
    batches: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SupportMaskPolicy:
    enabled: bool
    channel_locations_um: np.ndarray
    temporal_centers_s: np.ndarray
    shifts_um: np.ndarray
    contract: ValidatedSupportMaskContract | None
    direction: str = "y"
    audit: MaskAudit = field(default_factory=MaskAudit, compare=False)

    def validate_static(self) -> None:
        if not self.enabled:
            return
        if not isinstance(self.contract, ValidatedSupportMaskContract):
            raise ValueError("enabled candidate requires a validated metadata contract")
        geometry = np.asarray(self.channel_locations_um)
        if geometry.ndim != 2 or geometry.shape[1] < 2:
            raise ValueError("channel_locations_um must be channels by coordinates")
        if not np.isfinite(geometry).all():
            raise ValueError("channel geometry must be finite")
        if _array_digest(geometry) != _array_digest(self.contract.geometry):
            raise ValueError("policy geometry differs from validated contract")
        if _array_digest(np.asarray(self.temporal_centers_s, dtype=np.float64)) != _array_digest(self.contract.temporal_centers_s):
            raise ValueError("policy field times differ from validated contract")
        if _array_digest(np.asarray(self.shifts_um, dtype=np.float64)) != _array_digest(self.contract.shifts_um):
            raise ValueError("policy shifts differ from validated contract")


def policy_from_contract(contract: ValidatedSupportMaskContract) -> SupportMaskPolicy:
    return SupportMaskPolicy(
        enabled=True, channel_locations_um=contract.geometry,
        temporal_centers_s=contract.temporal_centers_s, shifts_um=contract.shifts_um,
        contract=contract,
    )


def metadata_covariance_support_audit(
    policy: SupportMaskPolicy, *, fs: float, imin: int, imax: int,
    batch_size: int, nt: int, nskip: int = 25,
) -> dict[str, Any]:
    """Inspect mask coverage on Kilosort's actual covariance batch schedule.

    This reads only supplied metadata.  Transition columns remain eligible and
    are counted separately; no row is dropped or geometry changed.
    """
    policy.validate_static()
    n_batches = int(np.ceil((int(imax) - int(imin)) / int(batch_size)))
    last_start = int(imin) + (n_batches - 1) * int(batch_size) - int(nt)
    last_end = min(int(imax), last_start + int(batch_size) + 2 * int(nt))
    if last_end - last_start - int(nt) < int(nt):
        n_batches -= 1
    indices = list(range(0, n_batches - 1, int(nskip)))
    counts = np.zeros(len(policy.channel_locations_um), dtype=np.int64)
    transition_columns = 0
    total_columns = 0
    for ibatch in indices:
        start = int(imin) + ibatch * int(batch_size)
        frames = np.arange(start, start + int(batch_size), dtype=np.int64)
        support = support_mask_from_geometry_and_field(
            policy.channel_locations_um, policy.temporal_centers_s,
            policy.shifts_um, frames, float(fs), direction=policy.direction,
        )
        counts += support.sum(axis=1)
        states = _states(policy, frames, fs)
        transition_columns += int(np.count_nonzero(states[1:] != states[:-1]))
        total_columns += frames.size
    return {
        "covariance_nskip": int(nskip), "n_batches": n_batches,
        "sampled_batch_indices": indices, "total_columns": total_columns,
        "valid_samples_per_row": counts.tolist(),
        "all_rows_have_valid_samples": bool(np.all(counts > 0)),
        "transition_boundaries_in_sampled_interiors": transition_columns,
        "action": "retain_all_rows" if np.all(counts > 0) else "fail_closed_no_geometry_change",
    }


def _states(policy: SupportMaskPolicy, frames: np.ndarray, fs: float) -> np.ndarray:
    centers = np.asarray(policy.temporal_centers_s, dtype=np.float64)
    edges = centers + np.median(np.diff(centers)) / 2
    return np.asarray(policy.shifts_um)[np.clip(
        np.searchsorted(edges, frames.astype(np.float64) / float(fs), side="right"),
        0, centers.size - 1,
    )]


def make_support_masked_binary_class(native_io, policy: SupportMaskPolicy):
    """Return a standard-signature BinaryFiltered subclass bound to a policy."""
    policy.validate_static()
    native_binary = native_io.BinaryFiltered
    native_rw = native_io.BinaryRWFile

    class SupportMaskedBinaryFiltered(native_binary):
        support_mask_policy = policy

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if policy.enabled:
                if self.dshift is not None:
                    raise ValueError("candidate forbids drift matrix / post-W spatial mixing")
                if self.chan_map is None or not np.array_equal(
                    np.asarray(self.chan_map), np.arange(len(policy.channel_locations_um))
                ):
                    raise ValueError("candidate requires identity full-probe chanMap")
                if self.n_chan_bin != len(policy.channel_locations_um):
                    raise ValueError("binary and support geometry row counts differ")

        def padded_batch_to_torch(self, ibatch, ops=None, return_inds=False):
            if not policy.enabled:
                return native_binary.padded_batch_to_torch(
                    self, ibatch, ops=ops, return_inds=return_inds
                )
            raw, inds = native_rw.padded_batch_to_torch(self, ibatch, return_inds=True)
            whitening = self.whiten_mat
            self.whiten_mat = None
            try:
                prewhite = native_binary.filter(self, raw, ops=ops, ibatch=ibatch)
            finally:
                self.whiten_mat = whitening
            frames = np.arange(int(inds[0]), int(inds[0]) + prewhite.shape[1], dtype=np.int64)
            support_np = support_mask_from_geometry_and_field(
                policy.channel_locations_um, policy.temporal_centers_s,
                policy.shifts_um, frames, float(self.fs), direction=policy.direction,
            )
            support = torch.from_numpy(support_np).to(prewhite.device)
            candidate_prewhite = prewhite.masked_fill(~support, 0)
            output = candidate_prewhite if whitening is None else whitening @ candidate_prewhite
            states = _states(policy, frames, float(self.fs))
            policy.audit.batches.append({
                "consumer_batch": int(ibatch), "global_start_frame": int(inds[0]),
                "global_stop_frame_exclusive": int(inds[0]) + int(prewhite.shape[1]),
                "unsupported_values": int((~support_np).sum()),
                "transition_boundaries": int(np.count_nonzero(states[1:] != states[:-1])),
                "transition_samples_retained": True,
            })
            return (output, inds) if return_inds else output

        def filter(self, X, ops=None, ibatch=None):
            if policy.enabled:
                raise RuntimeError("enabled mask requires padded_batch_to_torch global coordinates")
            return native_binary.filter(self, X, ops=ops, ibatch=ibatch)

    SupportMaskedBinaryFiltered.__name__ = "SupportMaskedBinaryFiltered"
    return SupportMaskedBinaryFiltered


@contextmanager
def installed_support_mask(native_io, policy: SupportMaskPolicy):
    """Temporarily route every native Kilosort BinaryFiltered constructor."""
    if not policy.enabled:
        yield
        return
    original = native_io.BinaryFiltered
    native_io.BinaryFiltered = make_support_masked_binary_class(native_io, policy)
    try:
        yield
    finally:
        native_io.BinaryFiltered = original


def execute_with_validated_support_mask(
    *, config_path: str | Path, expected_config_sha256: str,
    covariance_validation: CovarianceValidation | None, runner, native_io,
    runner_kwargs: dict[str, Any] | None = None,
):
    """Fail-closed execution boundary shared by synthetic and native runners."""
    contract = validate_support_mask_contract(
        config_path, expected_config_sha256=expected_config_sha256
    )
    if not contract.execution_enabled:
        raise PermissionError("reviewed support-mask contract has execution disabled")
    if not isinstance(covariance_validation, CovarianceValidation):
        raise PermissionError("no covariance validation receipt is bound")
    if covariance_validation.contract_digest != contract.contract_digest:
        raise PermissionError("covariance receipt belongs to another contract")
    if covariance_validation.shape != (contract.n_channels, contract.n_channels):
        raise PermissionError("covariance receipt channel shape mismatch")
    if not covariance_validation.passed:
        raise PermissionError(f"covariance validation failed: {covariance_validation.reasons}")
    policy = policy_from_contract(contract)
    with installed_support_mask(native_io, policy):
        return runner(**(runner_kwargs or {}))


def _native_invocation(config_path: str | Path, contract: ValidatedSupportMaskContract) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text())
    invocation = config["native_invocation"]
    settings = dict(invocation["settings"])
    artifact = settings.get("artifact_threshold")
    if artifact == "Infinity":
        settings["artifact_threshold"] = float("inf")
    elif isinstance(artifact, (str, bool)) or not isinstance(artifact, numbers.Real):
        raise ValueError("native artifact_threshold must be numeric or canonical 'Infinity'")
    elif not np.isfinite(float(artifact)) and not np.isposinf(float(artifact)):
        raise ValueError("native artifact_threshold must be finite or positive infinity")
    if int(np.int64(float(settings["tmin"]) * contract.fs)) != contract.imin:
        raise ValueError("native tmin does not equal global crop origin/fs")
    if int(np.int64(float(settings["tmax"]) * contract.fs)) != contract.imax:
        raise ValueError("native tmax does not equal global crop end/fs")
    if settings["fs"] != contract.fs or settings["n_chan_bin"] != contract.n_channels:
        raise ValueError("native fs/channel count differs from contract")
    if invocation["filename"] != config["recording"]["binary_path"]:
        raise ValueError("native filename is not the manifest-bound binary path")
    probe = {"chanMap": np.arange(contract.n_channels),
             "xc": contract.geometry[:, 0].astype(np.float32),
             "yc": contract.geometry[:, 1].astype(np.float32),
             "kcoords": np.zeros(contract.n_channels, dtype=np.float32),
             "n_chan": contract.n_channels}
    return {"settings": settings, "probe": probe, "filename": invocation["filename"],
            "results_dir": invocation["results_dir"], "data_dtype": invocation["data_dtype"],
            "do_CAR": invocation["do_CAR"], "invert_sign": invocation["invert_sign"],
            "device": invocation["device"], "save_extra_vars": invocation["save_extra_vars"],
            "clear_cache": invocation["clear_cache"],
            "save_preprocessed_copy": invocation["save_preprocessed_copy"],
            "bad_channels": None, "verbose_console": False, "verbose_log": True}


@contextmanager
def installed_validated_whitening(
    native_preprocessing, contract: ValidatedSupportMaskContract,
    audit: WhiteningBoundaryAudit, *, policy: SupportMaskPolicy | None = None,
    evidence_dir: Path | None = None, stop_after_whitening: bool = False,
):
    """Replace only native covariance->W construction with an exact checked copy."""
    original = native_preprocessing.get_whitening_matrix

    def checked(f, xc, yc, nskip=25, nrange=32):
        if int(nskip) != 25 or int(nrange) != 32:
            raise ValueError("whitening schedule/range differs from reviewed contract")
        n_chan = len(f.chan_map)
        covariance = torch.zeros((n_chan, n_chan), device=f.device)
        indices = tuple(range(0, int(f.n_batches) - 1, int(nskip)))
        if not indices:
            if evidence_dir is not None:
                _atomic_json(evidence_dir / "covariance_failure.json", {
                    "stage": "schedule", "reason": "empty covariance schedule",
                    "contract_digest": contract.contract_digest,
                })
            raise RuntimeError("actual covariance schedule is empty")
        for j in indices:
            batch = f.padded_batch_to_torch(j)
            batch = batch[:, f.nt:-f.nt]
            covariance = covariance + (batch @ batch.T) / batch.shape[1]
            if evidence_dir is not None and policy is not None:
                _atomic_json(evidence_dir / "covariance_batch_journal.json", {
                    "contract_digest": contract.contract_digest,
                    "completed_batch_indices": list(indices[:len(policy.audit.batches)]),
                    "batches": policy.audit.batches,
                })
        covariance = covariance / len(indices)
        covariance_np = covariance.detach().cpu().numpy()
        if evidence_dir is not None:
            _atomic_npy(evidence_dir / "actual_covariance.npy", covariance_np)
        validation = validate_covariance(
            covariance_np, expected_channels=contract.n_channels,
            contract_digest=contract.contract_digest, maximum_condition_number=1e8)
        audit.covariance_validation = validation
        audit.sampled_batch_indices = indices
        if evidence_dir is not None:
            _atomic_json(evidence_dir / "covariance_validation.json", asdict(validation))
        if not validation.passed:
            if evidence_dir is not None:
                _atomic_json(evidence_dir / "covariance_failure.json", {
                    "stage": "validation", "validation": asdict(validation),
                })
            raise RuntimeError(f"actual pre-W covariance failed validation: {validation.reasons}")
        whitening = native_preprocessing.whitening_local(
            covariance, xc, yc, nrange=nrange, device=f.device)
        whitening_np = whitening.detach().cpu().numpy()
        audit.whitening_finite = bool(np.isfinite(whitening_np).all())
        audit.whitening_sha256 = _array_digest(whitening_np)
        if evidence_dir is not None:
            _atomic_npy(evidence_dir / "actual_whitening.npy", whitening_np)
            _atomic_json(evidence_dir / "whitening_validation.json", {
                "contract_digest": contract.contract_digest,
                "finite": audit.whitening_finite,
                "shape": list(whitening_np.shape),
                "whitening_sha256": audit.whitening_sha256,
                "sampled_batch_indices": list(indices),
            })
        if not audit.whitening_finite:
            if evidence_dir is not None:
                _atomic_json(evidence_dir / "whitening_failure.json", {
                    "reason": "nonfinite", "whitening_sha256": audit.whitening_sha256,
                })
            raise RuntimeError("actual whitening matrix is nonfinite")
        if stop_after_whitening:
            if evidence_dir is not None:
                _atomic_json(evidence_dir / "smoke_stop.json", {
                    "status": "passed_covariance_and_whitening",
                    "before_training_detection": True,
                    "contract_digest": contract.contract_digest,
                    "whitening_sha256": audit.whitening_sha256,
                })
            raise SmokeStopAfterWhitening("intentional smoke stop after covariance/W")
        return whitening

    native_preprocessing.get_whitening_matrix = checked
    try:
        yield
    finally:
        native_preprocessing.get_whitening_matrix = original


def _reserve_fresh_output_namespace(config: dict[str, Any]) -> tuple[Path, Path]:
    """Atomically claim a never-used run root before the native runner can read."""
    smoke = config["smoke"]
    run_root = Path(smoke["run_root"])
    evidence_dir = Path(smoke["launch_evidence_dir"])
    results_dir = Path(config["native_invocation"]["results_dir"])
    run_root.mkdir(parents=False, exist_ok=False)
    evidence_dir.mkdir(exist_ok=False)
    results_dir.mkdir(exist_ok=False)
    _atomic_json(evidence_dir / "namespace_reservation.json", {
        "run_root": str(run_root), "launch_evidence_dir": str(evidence_dir),
        "native_results_dir": str(results_dir), "fresh": True,
    })
    return evidence_dir, results_dir


def _planned_covariance_read(
    config: dict[str, Any], contract: ValidatedSupportMaskContract,
    policy: SupportMaskPolicy,
) -> dict[str, Any]:
    settings = config["native_invocation"]["settings"]
    report = metadata_covariance_support_audit(
        policy, fs=contract.fs, imin=contract.imin, imax=contract.imax,
        batch_size=int(settings["batch_size"]), nt=int(settings["nt"]),
        nskip=int(settings["nskip"]),
    )
    sampled = report["sampled_batch_indices"]
    dtype_bytes = int(np.dtype(config["native_invocation"]["data_dtype"]).itemsize)
    padded_frames = int(settings["batch_size"]) + 2 * int(settings["nt"])
    report.update({
        "sampled_interior_seconds": report["total_columns"] / contract.fs,
        "planned_padded_frame_columns": len(sampled) * padded_frames,
        "planned_read_bytes_upper_bound": len(sampled) * padded_frames * contract.n_channels * dtype_bytes,
        "dtype_bytes": dtype_bytes,
        "global_crop_frames": [contract.imin, contract.imax],
        "global_sampled_interior_ranges": [
            [contract.imin + int(j) * int(settings["batch_size"]),
             contract.imin + (int(j) + 1) * int(settings["batch_size"])]
            for j in sampled
        ],
    })
    return report


def execute_native_with_runtime_covariance(
    *, config_path: str | Path, expected_config_sha256: str,
    native_runner, native_io, native_preprocessing,
):
    """Defined native path: contract-derived invocation and checked actual covariance."""
    config = json.loads(Path(config_path).read_text())
    contract = validate_support_mask_contract(config_path, expected_config_sha256=expected_config_sha256)
    if not contract.execution_enabled:
        raise PermissionError("reviewed support-mask contract has execution disabled")
    if config.get("schema") != "en-minimum-training-support-mask-v4-smoke":
        raise PermissionError("native runtime requires the frozen v4 smoke contract")
    kwargs = _native_invocation(config_path, contract)
    evidence_dir, reserved_results = _reserve_fresh_output_namespace(config)
    audit = WhiteningBoundaryAudit(computation_digest=contract.contract_digest)
    policy = policy_from_contract(contract)
    stage = "verify_reserved_paths"
    try:
        if Path(kwargs["results_dir"]) != reserved_results:
            raise RuntimeError("reserved native results path mismatch")
        stage = "plan_covariance_read"
        plan = _planned_covariance_read(config, contract, policy)
        _atomic_json(evidence_dir / "planned_covariance_read.json", plan)
        stage = "native_runner"
        with installed_support_mask(native_io, policy), installed_validated_whitening(
            native_preprocessing, contract, audit, policy=policy,
            evidence_dir=evidence_dir,
            stop_after_whitening=bool(config["smoke"]["stop_after_whitening"]),
        ):
            result = native_runner(**kwargs)
        stage = "verify_whitening_boundary"
        if audit.covariance_validation is None or audit.whitening_sha256 is None:
            raise RequiredWhiteningBoundaryMissing(
                "native runner returned without the required validated whitening boundary"
            )
    except SmokeStopAfterWhitening:
        _atomic_json(evidence_dir / "run_outcome.json", {
            "status": "smoke_stop_after_covariance_whitening",
            "before_training_detection": True,
            "audit": asdict(audit),
        })
        return {"status": "smoke_stop_after_covariance_whitening"}, audit
    except Exception as exc:
        crossed_whitening = bool(
            audit.covariance_validation is not None and audit.whitening_sha256 is not None
        )
        if isinstance(exc, RequiredWhiteningBoundaryMissing):
            reason = "normal_return_without_required_whitening_boundary"
        elif crossed_whitening:
            reason = "exception_after_required_whitening_boundary"
        else:
            reason = "exception_before_required_whitening_boundary"
        _atomic_json(evidence_dir / "run_failure.json", {
            "status": "failed", "stage": stage, "reason": reason,
            "exception_type": type(exc).__name__, "exception_message": str(exc),
            "crossed_whitening_boundary": crossed_whitening,
            "mask_audit_batch_count": len(policy.audit.batches),
            "audit": asdict(audit),
        })
        raise
    _atomic_json(evidence_dir / "run_outcome.json", {
        "status": "native_runner_returned", "audit": asdict(audit),
    })
    return result, audit


def run_kilosort_with_support_mask(
    *, config_path: str | Path, expected_config_sha256: str,
    covariance_validation: CovarianceValidation | None, **kwargs,
):
    """Native wrapper; refuses disabled/unbound/unvalidated real execution."""
    from kilosort import io, run_kilosort
    return execute_with_validated_support_mask(
        config_path=config_path, expected_config_sha256=expected_config_sha256,
        covariance_validation=covariance_validation, runner=run_kilosort,
        native_io=io, runner_kwargs=kwargs,
    )


def run_kilosort_with_runtime_covariance(
    *, config_path: str | Path, expected_config_sha256: str,
):
    """Native-signature-bound v3 entry point; actual reviewed config is disabled."""
    from kilosort import io, preprocessing, run_kilosort
    return execute_native_with_runtime_covariance(
        config_path=config_path, expected_config_sha256=expected_config_sha256,
        native_runner=run_kilosort, native_io=io,
        native_preprocessing=preprocessing,
    )
