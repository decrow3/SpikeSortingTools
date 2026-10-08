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


def _load_array(path: Path, *, allow_pickle: bool = False) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=allow_pickle, mmap_mode=None if allow_pickle else "r")
    except Exception as exc:
        raise ValueError(f"unparseable required array {path}: {exc}") from exc
    return value


def _require_array(name: str, value: np.ndarray, *, ndim: int | None = None,
                   integer: bool = False, finite: bool = True) -> None:
    if not isinstance(value, np.ndarray) or value.size == 0:
        raise ValueError(f"{name} must be a nonempty ndarray")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} rank {value.ndim} != {ndim}")
    if integer and not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{name} must have integer dtype")
    if finite and np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")


def validate_saved_artifacts(results: Path, snapshot: Path) -> dict[str, Any]:
    """Parse and semantically cross-check every required trained artifact.

    Native Kilosort exports and ``full_*`` arrays share the crop-local sample
    frame.  ``kept_spikes`` is resolved to a unique, ordered index vector and
    every exported spike row must equal the corresponding full-array row.
    """
    arrays = {name: _load_array(results / name, allow_pickle=name == "ops.npy")
              for name in REQUIRED_NATIVE_FILES if name.endswith(".npy")}
    one_dim_int = ("spike_times.npy", "spike_clusters.npy", "spike_templates.npy",
                   "spike_detection_templates.npy", "channel_map.npy", "channel_shanks.npy",
                   "full_clu.npy")
    for name in one_dim_int:
        _require_array(name, arrays[name], ndim=1, integer=True)
    _require_array("kept_spikes.npy", arrays["kept_spikes.npy"], ndim=1, finite=False)
    if arrays["kept_spikes.npy"].dtype != np.bool_ and not np.issubdtype(arrays["kept_spikes.npy"].dtype, np.integer):
        raise ValueError("kept_spikes.npy must be Boolean or integer")
    for name, rank in {
        "spike_positions.npy": 2, "pc_features.npy": 3, "templates.npy": 3,
        "templates_ind.npy": 2, "similar_templates.npy": 2,
        "whitening_mat.npy": 2, "whitening_mat_dat.npy": 2,
        "whitening_mat_inv.npy": 2, "pc_feature_ind.npy": 2,
        "tF.npy": 3, "Wall.npy": 3, "full_st.npy": 2,
        "full_amp.npy": 1, "amplitudes.npy": 1,
    }.items():
        _require_array(name, arrays[name], ndim=rank)
    n_spikes = len(arrays["spike_times.npy"])
    for name in ("spike_clusters.npy", "spike_templates.npy", "spike_detection_templates.npy",
                 "spike_positions.npy", "amplitudes.npy", "pc_features.npy", "tF.npy"):
        if arrays[name].shape[0] != n_spikes:
            raise ValueError(f"{name} is not aligned to spike_times.npy")
    n_full = arrays["full_st.npy"].shape[0]
    if arrays["full_st.npy"].shape[1] < 3:
        raise ValueError("full_st.npy requires at least three columns")
    if len(arrays["full_clu.npy"]) != n_full or len(arrays["full_amp.npy"]) != n_full:
        raise ValueError("full_st/full_clu/full_amp are misaligned")
    kept = arrays["kept_spikes.npy"]
    if kept.dtype == np.bool_ and len(kept) != n_full:
        raise ValueError("Boolean kept_spikes length differs from full_st")
    kept_indices = np.flatnonzero(kept) if kept.dtype == np.bool_ else kept.astype(np.int64, copy=False)
    if kept.dtype != np.bool_:
        if kept_indices.min() < 0 or kept_indices.max() >= n_full:
            raise ValueError("integer kept_spikes index is outside full_st")
        if len(np.unique(kept_indices)) != len(kept_indices):
            raise ValueError("integer kept_spikes indices must be unique")
        if len(kept_indices) > 1 and np.any(np.diff(kept_indices) <= 0):
            raise ValueError("integer kept_spikes indices must be strictly increasing in native full-array order")
    if len(kept_indices) != n_spikes:
        raise ValueError("kept_spikes selected count differs from exported spike count")
    selected_st = arrays["full_st.npy"][kept_indices]
    selected_times = selected_st[:, 0]
    if not np.equal(selected_times, np.rint(selected_times)).all():
        raise ValueError("selected full_st times are not exact integer crop-local samples")
    if not np.array_equal(arrays["spike_times.npy"], selected_times.astype(arrays["spike_times.npy"].dtype)):
        raise ValueError("spike_times row ancestry differs from full_st[kept_spikes, 0]")
    if len(selected_times) > 1 and np.any(np.diff(selected_times) < 0):
        raise ValueError("exported and selected crop-local spike times must be nondecreasing")
    if not np.array_equal(arrays["spike_clusters.npy"], arrays["full_clu.npy"][kept_indices]):
        raise ValueError("spike_clusters row ancestry differs from full_clu[kept_spikes]")
    if not np.array_equal(arrays["amplitudes.npy"], arrays["full_amp.npy"][kept_indices]):
        raise ValueError("amplitudes row ancestry differs from full_amp[kept_spikes]")
    chan_map = arrays["channel_map.npy"].reshape(-1)
    n_channels = len(chan_map)
    if len(np.unique(chan_map)) != n_channels:
        raise ValueError("channel_map.npy is not unique")
    if arrays["channel_positions.npy"].shape != (n_channels, 2) or len(arrays["channel_shanks.npy"]) != n_channels:
        raise ValueError("channel metadata is not aligned")
    for name in ("whitening_mat.npy", "whitening_mat_dat.npy", "whitening_mat_inv.npy"):
        if arrays[name].shape != (n_channels, n_channels):
            raise ValueError(f"{name} does not match channel count")
    n_templates = arrays["templates.npy"].shape[0]
    if arrays["templates.npy"].shape[2] != n_channels:
        raise ValueError("templates.npy physical-channel axis differs")
    if arrays["templates_ind.npy"].shape[0] != n_templates or arrays["similar_templates.npy"].shape != (n_templates, n_templates):
        raise ValueError("template arrays are not aligned")
    template_ids = arrays["spike_templates.npy"]
    if template_ids.min() < 0 or template_ids.max() >= n_templates:
        raise ValueError("spike_templates is outside templates axis 0")
    ops_value = arrays["ops.npy"]
    if ops_value.shape != () or not isinstance(ops_value.item(), dict):
        raise ValueError("ops.npy must contain one dictionary")
    ops = ops_value.item()
    if "chanMap" not in ops or not np.array_equal(np.asarray(ops["chanMap"]).reshape(-1), chan_map):
        raise ValueError("ops chanMap ancestry differs from channel_map.npy")
    snapshot_arrays = {name: _load_array(snapshot / name)
                       for name in REQUIRED_SNAPSHOT_FILES if name.endswith(".npy")}
    for name, rank in {"Wall3.npy": 3, "wPCA.npy": 2, "wTEMP.npy": 2,
                       "chanMap.npy": 1, "iU.npy": 1, "iCC.npy": 2,
                       "iCC_mask.npy": 2, "iU_physical_binary_channel.npy": 1}.items():
        _require_array("snapshot/" + name, snapshot_arrays[name], ndim=rank,
                       integer=name in {"chanMap.npy", "iU.npy", "iCC.npy", "iU_physical_binary_channel.npy"})
    wall3 = snapshot_arrays["Wall3.npy"]
    if not np.array_equal(snapshot_arrays["chanMap.npy"], chan_map):
        raise ValueError("snapshot chanMap differs from native channel_map")
    if wall3.shape[1] != snapshot_arrays["wPCA.npy"].shape[0] or snapshot_arrays["wPCA.npy"].shape != snapshot_arrays["wTEMP.npy"].shape:
        raise ValueError("Wall3/wPCA/wTEMP basis axes differ")
    if wall3.shape[2] != n_channels or len(snapshot_arrays["iU.npy"]) != wall3.shape[0]:
        raise ValueError("Wall3/iU physical ancestry differs")
    iCC, iCC_mask = snapshot_arrays["iCC.npy"], snapshot_arrays["iCC_mask.npy"]
    if iCC.shape != iCC_mask.shape or iCC.shape[1] != n_channels:
        raise ValueError("iCC/iCC_mask/channel axes differ")
    if iCC.min() < 0 or iCC.max() >= n_channels:
        raise ValueError("iCC contains invalid channel indices")
    iU = snapshot_arrays["iU.npy"].astype(int)
    if iU.min() < 0 or iU.max() >= n_channels:
        raise ValueError("iU contains invalid channel indices")
    if not np.array_equal(snapshot_arrays["iU_physical_binary_channel.npy"], chan_map[iU]):
        raise ValueError("iU physical-channel binding differs from chanMap[iU]")
    detection_ids = arrays["spike_detection_templates.npy"]
    if detection_ids.min() < 0 or detection_ids.max() >= wall3.shape[0]:
        raise ValueError("spike_detection_templates is outside Wall3 axis 0")
    receipt = json.loads((snapshot / "RECEIPT.json").read_text())
    if receipt.get("wall3_axis0_binding") != "spike_detection_templates.npy":
        raise ValueError("snapshot receipt lacks Wall3 detection-template ancestry")
    recorded = receipt.get("files")
    if not isinstance(recorded, dict):
        raise ValueError("snapshot receipt lacks per-file hashes")
    for name in snapshot_arrays:
        if recorded.get(name) != sha256_file(snapshot / name):
            raise ValueError(f"snapshot receipt hash differs: {name}")
    exported_cluster_ids = set(int(value) for value in arrays["spike_clusters.npy"])
    for name in ("cluster_Amplitude.tsv", "cluster_ContamPct.tsv", "cluster_KSLabel.tsv", "cluster_group.tsv"):
        text = (results / name).read_text()
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines or "cluster_id" not in lines[0].split("\t"):
            raise ValueError(f"{name} is not a parseable nonempty cluster table")
        header = lines[0].split("\t")
        cluster_column = header.index("cluster_id")
        try:
            table_ids = [int(line.split("\t")[cluster_column]) for line in lines[1:]]
        except (IndexError, ValueError) as error:
            raise ValueError(f"{name} has an invalid cluster_id column") from error
        if len(table_ids) != len(set(table_ids)):
            raise ValueError(f"{name} has duplicate cluster_id rows")
        if set(table_ids) != exported_cluster_ids:
            raise ValueError(f"{name} cluster IDs differ from exported spike_clusters")
    for name in ("params.py", "kilosort4.log"):
        if not (results / name).read_text().strip():
            raise ValueError(f"{name} is empty")
    return {"status": "semantic_validation_passed", "spikes": n_spikes,
            "full_rows": n_full, "channels": n_channels, "templates": n_templates,
            "wall3_templates": wall3.shape[0]}


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
        semantic_validation = validate_saved_artifacts(results, snapshot)
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
            "semantic_validation": semantic_validation,
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
