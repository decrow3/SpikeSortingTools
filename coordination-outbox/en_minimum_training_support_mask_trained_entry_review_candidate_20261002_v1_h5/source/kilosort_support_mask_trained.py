"""Execution-disabled trained-mode orchestration for the reviewed support hook.

The support-mask consumer implementation remains in
``kilosort_support_mask_candidate`` and is imported unchanged.  This module
adds the trained-only contract, phase audit, adaptive-bank snapshot, failure
receipts, and completion validation.  Synthetic orchestration is allowed only
when explicitly requested by the caller and every substituted path is beneath
one caller-supplied fixture root.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np
import torch

from . import kilosort_support_mask_candidate as reviewed


TRAINED_SCHEMA = "en-minimum-training-support-mask-v5-trained"
RECOMPUTED_STATE = (
    "covariance_C", "whitening_W", "wPCA", "wTEMP",
    "universal_detections_and_features", "Wall3", "iU", "iCC",
    "iCC_mask", "learned_detections_and_features", "final_clusters",
    "final_templates",
)
REQUIRED_NATIVE_FILES = (
    "ops.npy", "channel_map.npy", "channel_positions.npy",
    "channel_shanks.npy", "spike_times.npy", "spike_clusters.npy",
    "spike_templates.npy", "spike_detection_templates.npy",
    "spike_positions.npy", "amplitudes.npy", "kept_spikes.npy",
    "templates.npy", "templates_ind.npy", "similar_templates.npy",
    "whitening_mat.npy", "whitening_mat_dat.npy",
    "whitening_mat_inv.npy", "pc_features.npy", "pc_feature_ind.npy",
    "cluster_Amplitude.tsv", "cluster_ContamPct.tsv",
    "cluster_KSLabel.tsv", "cluster_group.tsv", "params.py",
    "kilosort4.log", "tF.npy", "Wall.npy", "full_st.npy",
    "full_clu.npy", "full_amp.npy",
)
REQUIRED_SNAPSHOT_FILES = (
    "Wall3.npy", "wPCA.npy", "wTEMP.npy", "chanMap.npy", "iU.npy",
    "iCC.npy", "iCC_mask.npy", "iU_physical_binary_channel.npy",
    "RECEIPT.json",
)
REQUIRED_PATHS = (
    ("schema",), ("status",), ("execution_enabled",),
    ("approval", "h1_review_manifest_sha256"),
    ("trained", "mode"), ("trained", "stop_after_whitening"),
    ("trained", "run_root"), ("trained", "launch_evidence_dir"),
    ("trained", "pre_extraction_snapshot_dir"),
    ("trained", "allow_saved_smoke_bank"),
    ("trained", "recompute_adaptive_state"),
    ("trained", "required_native_files"),
    ("trained", "required_snapshot_files"),
    ("trained", "completion_written_last"),
    ("trained", "failure_stage_preserved"),
    ("resources", "wall_seconds_max"), ("resources", "ram_bytes_max"),
    ("resources", "gpu_count_max"), ("resources", "cpu_threads_max"),
    ("resources", "persistent_output_bytes_max"),
    ("recording", "manifest_path"), ("recording", "manifest_sha256"),
    ("recording", "required_adapter"),
    ("recording", "recording_content_sha256"),
    ("recording", "num_samples"), ("recording", "num_channels"),
    ("recording", "sampling_frequency_hz"), ("recording", "dtype"),
    ("recording", "binary_path"), ("recording", "binary_name"),
    ("recording", "manifest_recorded_binary_sha256"),
    ("crop", "imin"), ("crop", "imax"),
    ("field", "path"), ("field", "sha256"),
    ("field", "time_key"), ("field", "displacement_key"),
    ("field", "rounding_um"), ("ops", "path"), ("ops", "sha256"),
    ("frozen_kilosort_settings",), ("native_invocation", "filename"),
    ("native_invocation", "results_dir"),
    ("native_invocation", "data_dtype"),
    ("native_invocation", "do_CAR"),
    ("native_invocation", "invert_sign"),
    ("native_invocation", "device"),
    ("native_invocation", "save_extra_vars"),
    ("native_invocation", "save_preprocessed_copy"),
    ("native_invocation", "settings"), ("source_bindings",),
    ("comparators", "REF384"), ("comparators", "B384"),
    ("orchestration_only", "synthetic"),
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=False, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npy(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _at(config: Any, path: tuple[str, ...]) -> Any:
    current = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise ValueError("missing required config key before data access: " + ".".join(path))
        current = current[key]
    return current


def prevalidate_trained_structure(config: dict[str, Any]) -> None:
    """Validate all required structure before hashing/opening bound data paths."""
    for path in REQUIRED_PATHS:
        _at(config, path)
    if config["schema"] != TRAINED_SCHEMA:
        raise ValueError("wrong trained contract schema")
    if type(config["execution_enabled"]) is not bool:
        raise ValueError("execution_enabled must be an exact JSON boolean")
    trained = config["trained"]
    if trained["mode"] != "full_native_trained" or trained["stop_after_whitening"] is not False:
        raise ValueError("trained mode must run beyond whitening")
    if trained["allow_saved_smoke_bank"] is not False:
        raise ValueError("saved smoke bank injection is forbidden")
    if tuple(trained["recompute_adaptive_state"]) != RECOMPUTED_STATE:
        raise ValueError("trained adaptive-state recomputation list differs")
    if tuple(trained["required_native_files"]) != REQUIRED_NATIVE_FILES:
        raise ValueError("required native artifact list differs")
    if tuple(trained["required_snapshot_files"]) != REQUIRED_SNAPSHOT_FILES:
        raise ValueError("required snapshot artifact list differs")
    if trained["completion_written_last"] is not True or trained["failure_stage_preserved"] is not True:
        raise ValueError("completion/failure policy differs")
    invocation = config["native_invocation"]
    if invocation["save_extra_vars"] is not True or invocation["save_preprocessed_copy"] is not False:
        raise ValueError("native save policy differs")
    if invocation["filename"] != config["recording"]["binary_path"]:
        raise ValueError("native filename is not the manifest-bound binary")
    if Path(invocation["results_dir"]) != Path(trained["run_root"]) / "native_results":
        raise ValueError("native results path differs from trained namespace")
    if Path(trained["launch_evidence_dir"]) != Path(trained["run_root"]) / "launch_evidence":
        raise ValueError("launch evidence path differs from trained namespace")
    if Path(trained["pre_extraction_snapshot_dir"]) != Path(trained["run_root"]) / "pre_extraction_snapshot":
        raise ValueError("snapshot path differs from trained namespace")
    if not isinstance(config["source_bindings"], list) or not config["source_bindings"]:
        raise ValueError("source_bindings must be nonempty")
    for index, item in enumerate(config["source_bindings"]):
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError(f"source binding {index} structure differs")
    if config["execution_enabled"]:
        review = config["approval"]["h1_review_manifest_sha256"]
        if not isinstance(review, str) or len(review) != 64:
            raise PermissionError("enabled trained execution requires an H1 review manifest")


def _require_beneath(path: str | Path, root: Path, label: str) -> None:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise PermissionError(f"synthetic {label} escapes fixture root") from exc


def validate_trained_contract(
    config_path: str | Path, *, expected_config_sha256: str,
    allow_synthetic_orchestration: bool = False, fixture_root: str | Path | None = None,
) -> tuple[dict[str, Any], reviewed.ValidatedSupportMaskContract]:
    """Validate trained structure first, then use the reviewed metadata adapter."""
    config_path = Path(config_path)
    observed = sha256_file(config_path)
    if observed != expected_config_sha256:
        raise ValueError(f"config hash mismatch: {observed} != {expected_config_sha256}")
    config = json.loads(config_path.read_text())
    prevalidate_trained_structure(config)
    synthetic = config["orchestration_only"]["synthetic"]
    if type(synthetic) is not bool:
        raise ValueError("orchestration_only.synthetic must be Boolean")
    if synthetic:
        if not allow_synthetic_orchestration or fixture_root is None:
            raise PermissionError("synthetic orchestration requires explicit caller isolation")
        root = Path(fixture_root).resolve()
        for label, path in (
            ("manifest", config["recording"]["manifest_path"]),
            ("binary", config["recording"]["binary_path"]),
            ("field", config["field"]["path"]), ("ops", config["ops"]["path"]),
            ("run_root", config["trained"]["run_root"]),
        ):
            _require_beneath(path, root, label)
        for item in config["source_bindings"]:
            _require_beneath(item["path"], root, "source binding")
        if config["native_invocation"]["device"] != "cpu":
            raise PermissionError("synthetic orchestration must use CPU")
    elif config["orchestration_only"] != {
        "synthetic": False,
        "label": "none_real_future_execution_requires_separate_h1_go",
    }:
        raise ValueError("real trained orchestration label differs")

    # Reuse the reviewed real metadata/ExactLattice validator through a schema
    # adapter.  No structural field is removed; only the schema token changes.
    adapted = json.loads(json.dumps(config))
    adapted["schema"] = "en-minimum-training-support-mask-v3"
    with tempfile.TemporaryDirectory(prefix="trained-contract-adapter-") as temporary:
        adapted_path = Path(temporary) / "adapted.json"
        adapted_path.write_text(json.dumps(adapted, sort_keys=True, separators=(",", ":")))
        adapted_sha = sha256_file(adapted_path)
        contract = reviewed.validate_support_mask_contract(
            adapted_path, expected_config_sha256=adapted_sha
        )
    trained_digest = hashlib.sha256(json.dumps({
        "trained_config_sha256": observed,
        "reviewed_adapter_contract_digest": contract.contract_digest,
        "mode": "full_native_trained",
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    contract = replace(contract, config_sha256=observed, contract_digest=trained_digest)
    return config, contract


@dataclass
class PhaseState:
    name: str = "native_initialization_or_drift"


class StageStampedList(list):
    def __init__(self, phase: PhaseState):
        super().__init__()
        self.phase = phase

    def append(self, value):
        row = dict(value)
        row["pipeline_phase"] = self.phase.name
        super().append(row)


@contextmanager
def phase(phase_state: PhaseState, name: str):
    previous = phase_state.name
    phase_state.name = name
    try:
        yield
    finally:
        phase_state.name = previous


def _cpu_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _snapshot_adaptive_bank(snapshot_dir: Path, ops: dict, wall3: Any,
                            prepared: tuple[Any, Any, Any, Any]) -> None:
    iCC, iCC_mask, iU, _ = prepared
    values = {
        "Wall3.npy": wall3, "wPCA.npy": ops["wPCA"], "wTEMP.npy": ops["wTEMP"],
        "chanMap.npy": ops["chanMap"], "iU.npy": iU, "iCC.npy": iCC,
        "iCC_mask.npy": iCC_mask,
        "iU_physical_binary_channel.npy": _cpu_array(ops["chanMap"])[_cpu_array(iU).astype(int)],
    }
    for name, value in values.items():
        atomic_npy(snapshot_dir / name, _cpu_array(value))
    hashes = {name: sha256_file(snapshot_dir / name) for name in values}
    atomic_json(snapshot_dir / "RECEIPT.json", {
        "status": "saved_before_learned_batch_extraction",
        "wall3_axis0_binding": "spike_detection_templates.npy",
        "orchestration_only": False,
        "files": hashes,
    })


@contextmanager
def installed_phase_and_snapshot_hooks(
    *, native_preprocessing, native_spikedetect, native_template_matching,
    phase_state: PhaseState, snapshot_dir: Path,
):
    originals = {
        "whitening": native_preprocessing.get_whitening_matrix,
        "detect_run": native_spikedetect.run,
        "wpca": native_spikedetect.extract_wPCA_wTEMP,
        "extract": native_template_matching.extract,
        "prepare_extract": native_template_matching.prepare_extract,
    }
    snapshot_context: dict[str, Any] = {}

    def whitening(*args, **kwargs):
        with phase(phase_state, "covariance_and_whitening"):
            return originals["whitening"](*args, **kwargs)

    def wpca(*args, **kwargs):
        with phase(phase_state, "pca_and_universal_template_learning"):
            return originals["wpca"](*args, **kwargs)

    def detect_run(*args, **kwargs):
        with phase(phase_state, "universal_detection_and_features"):
            return originals["detect_run"](*args, **kwargs)

    def prepare_extract(*args, **kwargs):
        prepared = originals["prepare_extract"](*args, **kwargs)
        if snapshot_context:
            _snapshot_adaptive_bank(
                snapshot_dir, snapshot_context["ops"], snapshot_context["wall3"], prepared
            )
            snapshot_context["saved"] = True
        return prepared

    def extract(ops, bfile, wall3, *args, **kwargs):
        snapshot_context.update(ops=ops, wall3=wall3, saved=False)
        with phase(phase_state, "learned_template_extraction"):
            result = originals["extract"](ops, bfile, wall3, *args, **kwargs)
        if not snapshot_context.get("saved"):
            raise RuntimeError("learned extraction returned without adaptive-bank snapshot")
        return result

    native_preprocessing.get_whitening_matrix = whitening
    native_spikedetect.run = detect_run
    native_spikedetect.extract_wPCA_wTEMP = wpca
    native_template_matching.extract = extract
    native_template_matching.prepare_extract = prepare_extract
    try:
        yield
    finally:
        native_preprocessing.get_whitening_matrix = originals["whitening"]
        native_spikedetect.run = originals["detect_run"]
        native_spikedetect.extract_wPCA_wTEMP = originals["wpca"]
        native_template_matching.extract = originals["extract"]
        native_template_matching.prepare_extract = originals["prepare_extract"]


def reserve_namespace(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    trained = config["trained"]
    run_root = Path(trained["run_root"])
    evidence = Path(trained["launch_evidence_dir"])
    snapshot = Path(trained["pre_extraction_snapshot_dir"])
    results = Path(config["native_invocation"]["results_dir"])
    run_root.mkdir(parents=False, exist_ok=False)
    evidence.mkdir(exist_ok=False)
    snapshot.mkdir(exist_ok=False)
    results.mkdir(exist_ok=False)
    atomic_json(evidence / "namespace_reservation.json", {
        "fresh": True, "run_root": str(run_root), "launch_evidence_dir": str(evidence),
        "pre_extraction_snapshot_dir": str(snapshot), "native_results_dir": str(results),
    })
    return evidence, snapshot, results


def _inventory(root: Path, names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    result = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"required trained artifact missing: {path}")
        result[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def _native_kwargs(config: dict[str, Any], contract: reviewed.ValidatedSupportMaskContract) -> dict[str, Any]:
    # The reviewed helper preserves the exact native signature and Infinity handling.
    with tempfile.TemporaryDirectory(prefix="trained-native-adapter-") as temporary:
        adapted = json.loads(json.dumps(config))
        adapted_path = Path(temporary) / "adapted.json"
        adapted_path.write_text(json.dumps(adapted))
        return reviewed._native_invocation(adapted_path, contract)


def execute_trained(
    *, config_path: str | Path, expected_config_sha256: str,
    native_runner: Callable[..., Any], native_io, native_preprocessing,
    native_spikedetect, native_template_matching,
    allow_synthetic_orchestration: bool = False,
    fixture_root: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    config, contract = validate_trained_contract(
        config_path, expected_config_sha256=expected_config_sha256,
        allow_synthetic_orchestration=allow_synthetic_orchestration,
        fixture_root=fixture_root,
    )
    if not contract.execution_enabled:
        raise PermissionError("trained contract is execution-disabled pending H1 review")
    evidence, snapshot, results = reserve_namespace(config)
    phase_state = PhaseState()
    stamped = StageStampedList(phase_state)
    policy = reviewed.policy_from_contract(contract)
    policy.audit.batches = stamped
    whitening_audit = reviewed.WhiteningBoundaryAudit(contract.contract_digest)
    stage = "reserved_namespace"
    atomic_json(evidence / "run_status.json", {"status": "running", "stage": stage})
    try:
        stage = "construct_native_invocation"
        kwargs = _native_kwargs(config, contract)
        if Path(kwargs["results_dir"]) != results:
            raise RuntimeError("reserved results path differs from native invocation")
        stage = "native_trained_pipeline"
        atomic_json(evidence / "run_status.json", {"status": "running", "stage": stage})
        with ExitStack() as stack:
            stack.enter_context(reviewed.installed_support_mask(native_io, policy))
            stack.enter_context(reviewed.installed_validated_whitening(
                native_preprocessing, contract, whitening_audit, policy=policy,
                evidence_dir=evidence, stop_after_whitening=False,
            ))
            stack.enter_context(installed_phase_and_snapshot_hooks(
                native_preprocessing=native_preprocessing,
                native_spikedetect=native_spikedetect,
                native_template_matching=native_template_matching,
                phase_state=phase_state, snapshot_dir=snapshot,
            ))
            result = native_runner(**kwargs)
        stage = "validate_whitening"
        if whitening_audit.covariance_validation is None or not whitening_audit.covariance_validation.passed:
            raise RuntimeError("trained runner lacks passing covariance validation")
        if whitening_audit.whitening_finite is not True or whitening_audit.whitening_sha256 is None:
            raise RuntimeError("trained runner lacks finite audited whitening")
        stage = "validate_and_inventory_saved_artifacts"
        native_inventory = _inventory(results, REQUIRED_NATIVE_FILES)
        snapshot_inventory = _inventory(snapshot, REQUIRED_SNAPSHOT_FILES)
        atomic_json(evidence / "support_reads_by_phase.json", {
            "contract_digest": contract.contract_digest,
            "rows": list(stamped),
            "phase_counts": {
                name: sum(row["pipeline_phase"] == name for row in stamped)
                for name in sorted({row["pipeline_phase"] for row in stamped})
            },
        })
        atomic_json(evidence / "artifact_inventory.json", {
            "native": native_inventory, "pre_extraction_snapshot": snapshot_inventory,
        })
        stage = "write_completion_last"
        complete = {
            "status": "complete",
            "contract_digest": contract.contract_digest,
            "config_sha256": expected_config_sha256,
            "orchestration_only_synthetic": config["orchestration_only"]["synthetic"],
            "recomputed_adaptive_state": list(RECOMPUTED_STATE),
            "smoke_bank_injected": False,
            "required_artifacts_validated": True,
            "completion_written_last": True,
        }
        atomic_json(Path(config["trained"]["run_root"]) / "COMPLETE.json", complete)
        return result, complete
    except Exception as exc:
        atomic_json(evidence / "run_failure.json", {
            "status": "failed", "stage": stage,
            "exception_type": type(exc).__name__, "exception_message": str(exc),
            "contract_digest": contract.contract_digest,
            "mask_read_count": len(stamped), "phase_at_failure": phase_state.name,
            "whitening_audit": asdict(whitening_audit),
            "completion_written": False,
        })
        raise


def run_native_trained(
    *, config_path: str | Path, expected_config_sha256: str,
    allow_synthetic_orchestration: bool = False,
    fixture_root: str | Path | None = None,
    native_runner: Callable[..., Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    from kilosort import io, preprocessing, spikedetect, template_matching
    if native_runner is None:
        from kilosort import run_kilosort as native_runner
    return execute_trained(
        config_path=config_path, expected_config_sha256=expected_config_sha256,
        native_runner=native_runner, native_io=io,
        native_preprocessing=preprocessing, native_spikedetect=spikedetect,
        native_template_matching=template_matching,
        allow_synthetic_orchestration=allow_synthetic_orchestration,
        fixture_root=fixture_root,
    )
