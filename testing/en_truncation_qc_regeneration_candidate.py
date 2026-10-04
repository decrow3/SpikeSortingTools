#!/usr/bin/env python3
"""One-shot, resource-bounded truncation-QC regeneration candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "en-truncation-qc-regeneration-v2"
PURPOSE = "One-shot regeneration of production truncation QC from hash-bound saved sorter arrays only; no recording or voltage access"
ESTIMATOR = {
    "function": "pipeline.qc.truncation_qc", "max_isi_s": 10.0, "spikes_per_window": 1000,
    "fit_x_min_rule": "minimum observed amplitude in each historical 999-value half-open fit slice",
    "amplitude_source": "full_st[kept_spikes,2] sorter-native units",
    "time_source": "full_st[kept_spikes,0] crop-local samples divided by sampling_frequency_hz",
}
OUTPUT_SCHEMA = {
    "truncation_qc.npz": {"cid": ["N"], "window_blocks": ["N", 2], "popts": ["N", 3], "mpcts": ["N"]},
    "present_qc.npz": {"cid": ["M"], "valid_blocks": ["M", 2]},
    "truncation_qc.pdf": "nonempty PDF",
}
RESOURCE_KEYS = {"wall_seconds_max", "cpu_time_seconds_max", "ram_bytes_max", "cpu_threads_max", "persistent_output_bytes_max"}
STOP_CONDITIONS = [
    "any config/source/runtime/input hash mismatch", "any clock, dtype or row-semantics mismatch",
    "global authorization token already consumed", "caller output differs from frozen real destination",
    "any existing output namespace", "any resource bound exceeded", "any nonfinite production amplitude",
    "any output-schema or synthetic-known-answer mismatch", "first exception", "no automatic retry",
]
PROHIBITIONS = [
    "recording or voltage access", "sorting, training, detection or clustering", "RF or sealed holdout access",
    "outcome-derived threshold changes", "substitute estimator", "overwrite prior QC or evidence", "automatic retry",
    "caller-selected real output destination",
]
REQUIRED_TOP = {
    "schema", "status", "execution_enabled", "purpose", "source_bindings", "runtime", "authorization",
    "estimator", "clock", "inputs", "output_schema", "resources", "synthetic_known_answer",
    "stop_conditions", "prohibitions", "activation",
}
REQUIRED_INPUT = {"arm_id", "curated_output", "files", "output_dir"}
REQUIRED_FILES = ("spike_times.npy", "spike_clusters.npy", "full_st.npy", "kept_spikes.npy")
RUNTIME_MODULES = {"numpy", "scipy", "matplotlib", "sklearn", "torch", "tqdm"}
SOURCE_NAMES = {"en_truncation_qc_regeneration_candidate.py", "qc.py", "truncation.py", "refractory.py"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _hex_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def prevalidate_structure(config: dict[str, Any], *, phase: str = "review_pending") -> None:
    """Reject every missing or changed semantic before bound-path access."""
    if set(config) != REQUIRED_TOP:
        raise ValueError(f"top-level config keys differ before file access: {sorted(set(config) ^ REQUIRED_TOP)}")
    if config["schema"] != SCHEMA:
        raise ValueError("schema/status changed before file access")
    expected_state = {
        "review_pending": ("review_pending_execution_disabled", False),
        "synthetic": ("review_pending_execution_disabled", False),
        "real": ("reviewed_one_execution_enabled", True),
    }
    if phase not in expected_state or (config["status"], config["execution_enabled"]) != expected_state[phase]:
        raise ValueError(f"config lifecycle state is invalid for requested phase {phase}")
    if config["purpose"] != PURPOSE or config["estimator"] != ESTIMATOR:
        raise ValueError("purpose or production estimator semantics changed before file access")
    if config["output_schema"] != OUTPUT_SCHEMA:
        raise ValueError("output schema changed before file access")
    if config["stop_conditions"] != STOP_CONDITIONS or config["prohibitions"] != PROHIBITIONS:
        raise ValueError("stop conditions or prohibitions changed before file access")

    clock = config["clock"]
    if set(clock) != {"sampling_frequency_hz", "crop_origin_global_frame", "local_frames_half_open", "global_frames_half_open", "saved_spike_times_frame", "full_st_frame"}:
        raise ValueError("clock keys changed before file access")
    if clock != {
        "sampling_frequency_hz": 29999.835983263598, "crop_origin_global_frame": 208498882,
        "local_frames_half_open": [0, 17999901], "global_frames_half_open": [208498882, 226498783],
        "saved_spike_times_frame": "global_acquisition_samples", "full_st_frame": "crop_local_samples",
    }:
        raise ValueError("clock values or frame semantics changed before file access")

    resources = config["resources"]
    if set(resources) != RESOURCE_KEYS:
        raise ValueError("resource keys changed before file access")
    if any(isinstance(resources[k], bool) or not isinstance(resources[k], int) or resources[k] <= 0 for k in RESOURCE_KEYS):
        raise ValueError("resource limits must be positive integer base units before file access")
    if resources["cpu_threads_max"] > 64 or resources["ram_bytes_max"] > 137438953472:
        raise ValueError("resource limits exceed reviewed ceilings before file access")

    authorization = config["authorization"]
    if set(authorization) != {"one_shot_token", "consume_on_invocation", "automatic_retry", "real_execution_review_required"}:
        raise ValueError("authorization keys changed before file access")
    if not isinstance(authorization["one_shot_token"], str) or not authorization["one_shot_token"]:
        raise ValueError("one-shot token missing before file access")
    if authorization["consume_on_invocation"] is not True or authorization["automatic_retry"] is not False or authorization["real_execution_review_required"] is not True:
        raise ValueError("one-shot authorization semantics changed before file access")

    activation = config["activation"]
    if set(activation) != {
        "gate_schema", "required_verdict", "enabled_status", "enabled_execution_value",
        "allowed_config_mutations", "global_ledger_root", "frozen_real_output_root",
    }:
        raise ValueError("activation keys changed before file access")
    if activation["gate_schema"] != "truncation-qc-one-execution-gate-v1" or activation["required_verdict"] != "ONE_EXECUTION_GO":
        raise ValueError("activation gate semantics changed before file access")
    if activation["enabled_status"] != "reviewed_one_execution_enabled" or activation["enabled_execution_value"] is not True:
        raise ValueError("enabled lifecycle state changed before file access")
    if activation["allowed_config_mutations"] != ["status", "execution_enabled"]:
        raise ValueError("gate transition permits non-lifecycle mutation before file access")
    if not Path(activation["global_ledger_root"]).is_absolute() or not Path(activation["frozen_real_output_root"]).is_absolute():
        raise ValueError("activation ledger/output roots must be absolute before file access")

    bindings = config["source_bindings"]
    if not isinstance(bindings, list) or {Path(row.get("path", "")).name for row in bindings} != SOURCE_NAMES:
        raise ValueError("source binding closure changed before file access")
    for row in bindings:
        if set(row) != {"path", "sha256"} or not _hex_digest(row["sha256"]):
            raise ValueError("malformed source binding before file access")

    runtime = config["runtime"]
    if set(runtime) != {"executable", "executable_sha256", "python_version", "modules"}:
        raise ValueError("runtime keys changed before file access")
    if not isinstance(runtime["executable"], str) or not _hex_digest(runtime["executable_sha256"]) or not isinstance(runtime["python_version"], str):
        raise ValueError("malformed interpreter binding before file access")
    if set(runtime["modules"]) != RUNTIME_MODULES:
        raise ValueError("runtime dependency set changed before file access")
    for name, row in runtime["modules"].items():
        if set(row) != {"version", "origin", "sha256"} or not all(isinstance(row[k], str) for k in ("version", "origin")) or not _hex_digest(row["sha256"]):
            raise ValueError(f"malformed runtime module binding for {name} before file access")

    arms = config["inputs"]
    if not isinstance(arms, list) or [row.get("arm_id") for row in arms] != ["REF384", "existing_corrected_B384"]:
        raise ValueError("input arm order changed before file access")
    for arm in arms:
        if set(arm) != REQUIRED_INPUT or set(arm["files"]) != set(REQUIRED_FILES):
            raise ValueError("input binding keys changed before file access")
        if any(not _hex_digest(arm["files"][name]) for name in REQUIRED_FILES):
            raise ValueError("malformed input hash before file access")
        if Path(arm["output_dir"]).name != arm["arm_id"]:
            raise ValueError("arm output namespace changed before file access")
    if len({str(Path(arm["output_dir"]).parent) for arm in arms}) != 1:
        raise ValueError("arm outputs do not share one frozen root before file access")
    if str(Path(arms[0]["output_dir"]).parent) != activation["frozen_real_output_root"]:
        raise ValueError("input outputs differ from the frozen real root before file access")

    known = config["synthetic_known_answer"]
    if set(known) != {"input_rows", "window_blocks", "valid_blocks", "popts", "mpcts", "absolute_tolerance"}:
        raise ValueError("synthetic known-answer keys changed before file access")
    if known["input_rows"] != 2200 or known["window_blocks"] != [[100, 1099], [1100, 2099]] or known["valid_blocks"] != [[100, 2100]]:
        raise ValueError("synthetic combinatorial known answer changed before file access")
    if not isinstance(known["absolute_tolerance"], (int, float)) or known["absolute_tolerance"] <= 0:
        raise ValueError("synthetic numeric tolerance invalid before file access")


def _validate_sources(config: dict[str, Any]) -> dict[str, str]:
    observed = {}
    for binding in config["source_bindings"]:
        path = Path(binding["path"])
        digest = sha256_file(path)
        if digest != binding["sha256"]:
            raise RuntimeError(f"source binding moved: {path}")
        observed[str(path.resolve())] = digest
    return observed


def _validate_runtime(config: dict[str, Any]) -> dict[str, Any]:
    frozen = config["runtime"]
    executable = Path(sys.executable).resolve()
    if str(executable) != frozen["executable"] or sha256_file(executable) != frozen["executable_sha256"]:
        raise RuntimeError("reviewed interpreter identity mismatch")
    if sys.version != frozen["python_version"]:
        raise RuntimeError("reviewed Python version mismatch")
    observed = {"executable": str(executable), "executable_sha256": sha256_file(executable), "python_version": sys.version, "modules": {}}
    for name in sorted(RUNTIME_MODULES):
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        row = {"version": str(module.__version__), "origin": str(origin), "sha256": sha256_file(origin)}
        if row != frozen["modules"][name]:
            raise RuntimeError(f"reviewed runtime module identity mismatch: {name}")
        observed["modules"][name] = row
    return observed


def _load_one(arm: dict[str, Any], clock: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import numpy as np
    curated = Path(arm["curated_output"])
    arrays, observed = {}, {}
    for name in REQUIRED_FILES:
        blob = (curated / name).read_bytes()
        observed[name] = hashlib.sha256(blob).hexdigest()
        if observed[name] != arm["files"][name]:
            raise RuntimeError(f"{arm['arm_id']}: saved input hash mismatch for {name}")
        arrays[name] = np.load(io.BytesIO(blob), allow_pickle=False)
    full_st, kept = np.asarray(arrays["full_st.npy"]), np.asarray(arrays["kept_spikes.npy"])
    if full_st.ndim != 2 or full_st.shape[1] < 3 or not np.issubdtype(full_st.dtype, np.number) or kept.ndim != 1:
        raise ValueError(f"{arm['arm_id']}: invalid full_st/kept_spikes shape or dtype")
    if kept.dtype == np.bool_:
        if len(kept) != len(full_st):
            raise ValueError(f"{arm['arm_id']}: Boolean kept_spikes length mismatch")
        indices = np.flatnonzero(kept)
    elif np.issubdtype(kept.dtype, np.integer):
        indices = kept.astype(np.int64, copy=False)
        if indices.size and (indices.min() < 0 or indices.max() >= len(full_st)):
            raise ValueError(f"{arm['arm_id']}: kept_spikes index out of bounds")
        if len(np.unique(indices)) != len(indices) or (len(indices) > 1 and np.any(np.diff(indices) <= 0)):
            raise ValueError(f"{arm['arm_id']}: kept_spikes indices must be unique and increasing")
    else:
        raise ValueError(f"{arm['arm_id']}: unsupported kept_spikes dtype")
    selected = full_st[indices]
    saved_times, clusters = np.asarray(arrays["spike_times.npy"]), np.asarray(arrays["spike_clusters.npy"])
    if saved_times.ndim != 1 or clusters.ndim != 1 or not np.issubdtype(saved_times.dtype, np.integer) or not np.issubdtype(clusters.dtype, np.integer):
        raise ValueError(f"{arm['arm_id']}: spike times/clusters must be one-dimensional integer arrays")
    if clusters.size and clusters.min() < 0:
        raise ValueError(f"{arm['arm_id']}: cluster IDs must be nonnegative")
    if not (len(saved_times) == len(clusters) == len(selected)):
        raise ValueError(f"{arm['arm_id']}: saved arrays are not row-aligned")
    local_times = selected[:, 0]
    if not np.isfinite(local_times).all() or not np.equal(local_times, np.rint(local_times)).all():
        raise ValueError(f"{arm['arm_id']}: full_st times are not exact samples")
    local_times = local_times.astype(np.int64)
    origin = int(clock["crop_origin_global_frame"])
    if not np.array_equal(saved_times.astype(np.int64) - local_times, np.full(len(local_times), origin, dtype=np.int64)):
        raise ValueError(f"{arm['arm_id']}: exact declared clock transform failed")
    bounds = tuple(clock["local_frames_half_open"])
    if len(local_times) and (local_times.min() < bounds[0] or local_times.max() >= bounds[1]):
        raise ValueError(f"{arm['arm_id']}: local times outside frozen crop")
    if len(local_times) > 1 and np.any(np.diff(local_times) < 0):
        raise ValueError(f"{arm['arm_id']}: saved rows are not time ordered")
    amplitudes = selected[:, 2].astype(np.float64, copy=False)
    if not np.isfinite(amplitudes).all():
        raise ValueError(f"{arm['arm_id']}: production amplitudes contain nonfinite values")
    seconds = local_times.astype(np.float64) / float(clock["sampling_frequency_hz"])
    return seconds, clusters.astype(np.int64, copy=False), amplitudes, {"files": observed, "rows": len(seconds), "observed_exact_offset": origin, "amplitude_source": "full_st[kept_spikes,2]"}


def _validate_outputs(cache_dir: Path, known: dict[str, Any] | None = None) -> dict[str, Any]:
    import numpy as np
    paths = {name: cache_dir / name for name in OUTPUT_SCHEMA}
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
        raise RuntimeError("production truncation QC did not create all frozen outputs")
    with np.load(paths["truncation_qc.npz"], allow_pickle=False) as data:
        if set(data.files) != {"cid", "window_blocks", "popts", "mpcts"}:
            raise ValueError("truncation_qc.npz keys differ from frozen schema")
        n = data["cid"].reshape(-1).size
        if data["window_blocks"].shape != (n, 2) or data["popts"].shape != (n, 3) or data["mpcts"].shape != (n,):
            raise ValueError("truncation_qc.npz shapes differ from frozen schema")
        trunc = {name: np.asarray(data[name]) for name in data.files}
    with np.load(paths["present_qc.npz"], allow_pickle=False) as data:
        if set(data.files) != {"cid", "valid_blocks"}:
            raise ValueError("present_qc.npz keys differ from frozen schema")
        m = data["cid"].reshape(-1).size
        if data["valid_blocks"].shape != (m, 2):
            raise ValueError("present_qc.npz shapes differ from frozen schema")
        present = {name: np.asarray(data[name]) for name in data.files}
    if known is not None:
        tolerance = float(known["absolute_tolerance"])
        if trunc["window_blocks"].tolist() != known["window_blocks"] or present["valid_blocks"].tolist() != known["valid_blocks"]:
            raise ValueError("synthetic combinatorial known answer mismatch")
        if not np.allclose(trunc["popts"], np.asarray(known["popts"]), rtol=0.0, atol=tolerance) or not np.allclose(trunc["mpcts"], np.asarray(known["mpcts"]), rtol=0.0, atol=tolerance):
            raise ValueError("synthetic numeric known answer mismatch")
    return {"window_rows": n, "present_rows": m, "files": {name: sha256_file(path) for name, path in paths.items()}}


def _run_production(times: np.ndarray, clusters: np.ndarray, amplitudes: np.ndarray, cache_dir: Path,
                    known: dict[str, Any] | None = None, *, corrupt_output: bool = False) -> dict[str, Any]:
    from pipeline.qc import truncation_qc
    truncation_qc(times, clusters, amplitudes, cache_dir, recalc=True)
    if corrupt_output:
        (cache_dir / "truncation_qc.npz").write_bytes(b"injected-corrupt-output")
    return _validate_outputs(cache_dir, known)


def _synthetic_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np
    n = 2200
    return np.arange(n, dtype=float) * 0.01, np.zeros(n, dtype=np.int64), 8.0 + np.linspace(0.01, 4.0, n)


def _apply_process_limits(resources: dict[str, int]) -> dict[str, Any]:
    resource.setrlimit(resource.RLIMIT_AS, (resources["ram_bytes_max"], resources["ram_bytes_max"]))
    resource.setrlimit(resource.RLIMIT_CPU, (resources["cpu_time_seconds_max"], resources["cpu_time_seconds_max"]))
    affinity = sorted(os.sched_getaffinity(0))[:resources["cpu_threads_max"]]
    os.sched_setaffinity(0, affinity)
    return {"address_space_bytes": resources["ram_bytes_max"], "cpu_time_seconds": resources["cpu_time_seconds_max"], "cpu_affinity": affinity,
            "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}


def _worker(config: dict[str, Any], output: Path, mode: str, outcome: str,
            bootstrap: dict[str, Any]) -> dict[str, Any]:
    if outcome == "injected-failure":
        raise RuntimeError("injected synthetic worker failure")
    if mode == "synthetic":
        times, clusters, amplitudes = _synthetic_arrays()
        _atomic_json(output / "evidence/INPUTS.json", {
            "mode": "synthetic", "rows": len(times),
            "identity": "frozen-generated-known-answer", "saved_input_files_opened": 0,
        })
        result = _run_production(times, clusters, amplitudes, output / "arms/synthetic/amp_truncation",
                                 config["synthetic_known_answer"], corrupt_output=outcome == "corrupt-output")
        return {"mode": mode, "not_scientific": True, "input": {"rows": len(times), "identity": "frozen-generated-known-answer"}, "output": result, "bootstrap": bootstrap}
    if config["execution_enabled"] is not True:
        raise PermissionError("real regeneration is execution-disabled pending a distinct reviewed decision")
    arms = {}
    for arm in config["inputs"]:
        times, clusters, amplitudes, input_receipt = _load_one(arm, config["clock"])
        arms[arm["arm_id"]] = {"input": input_receipt, "output": None}
        _atomic_json(output / "evidence/INPUTS.json", {"mode": "real", "arms": arms})
        result = _run_production(times, clusters, amplitudes, Path(arm["output_dir"]) / "amp_truncation")
        arms[arm["arm_id"]] = {"input": input_receipt, "output": result}
        _atomic_json(output / "evidence/INPUTS.json", {"mode": "real", "arms": arms})
    return {"mode": mode, "arms": arms, "bootstrap": bootstrap}


def _inventory(root: Path) -> dict[str, Any]:
    files, total = {}, 0
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"COMPLETE.json", "OUTPUT_INVENTORY.json"}:
            size = path.stat().st_size
            total += size
            files[path.relative_to(root).as_posix()] = {"bytes": size, "sha256": sha256_file(path)}
    return {"files": files, "total_bytes": total}


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _validate_gate(gate: dict[str, Any], config: dict[str, Any], config_hash: str,
                   *, allow_synthetic: bool) -> tuple[dict[str, Any], str]:
    required = {
        "schema", "verdict", "review_manifest_sha256", "disabled_config_sha256",
        "source_sha256", "frozen_real_output_root", "one_shot_token", "synthetic_fixture",
    }
    if set(gate) != required:
        raise ValueError("gate keys differ from the frozen activation schema")
    if gate["schema"] != config["activation"]["gate_schema"] or gate["verdict"] != config["activation"]["required_verdict"]:
        raise ValueError("wrong gate schema or verdict")
    if not _hex_digest(gate["review_manifest_sha256"]) or gate["disabled_config_sha256"] != config_hash:
        raise ValueError("gate is not bound to this config/review")
    wrapper_hash = next(row["sha256"] for row in config["source_bindings"]
                        if Path(row["path"]).name == "en_truncation_qc_regeneration_candidate.py")
    if gate["source_sha256"] != wrapper_hash:
        raise ValueError("gate source binding mismatch")
    if gate["frozen_real_output_root"] != config["activation"]["frozen_real_output_root"]:
        raise ValueError("gate destination binding mismatch")
    if not isinstance(gate["one_shot_token"], str) or not gate["one_shot_token"]:
        raise ValueError("gate one-shot token is missing")
    if gate["synthetic_fixture"] is not False and not (allow_synthetic and gate["synthetic_fixture"] is True):
        raise ValueError("synthetic gate cannot enable real execution")
    effective = json.loads(json.dumps(config))
    effective["status"] = config["activation"]["enabled_status"]
    effective["execution_enabled"] = config["activation"]["enabled_execution_value"]
    prevalidate_structure(effective, phase="real")
    identity = hashlib.sha256(json.dumps({
        "disabled_config_sha256": config_hash,
        "review_manifest_sha256": gate["review_manifest_sha256"],
        "source_sha256": gate["source_sha256"],
        "frozen_real_output_root": gate["frozen_real_output_root"],
        "one_shot_token": gate["one_shot_token"],
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return effective, identity


def _reserve_global_token(ledger_root: Path, token_identity: str, payload: dict[str, Any]) -> tuple[Path, str]:
    ledger_root.mkdir(parents=True, exist_ok=True)
    lock = ledger_root / f"{token_identity}.consumed.json"
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return lock, hashlib.sha256(encoded).hexdigest()


def _managed_run(config_path: Path, config_hash: str, config: dict[str, Any], output: Path,
                 mode: str, outcome: str, *, ledger_root: Path, token_identity: str,
                 review_manifest_sha256: str) -> tuple[bool, dict[str, Any]]:
    if mode == "real" and config["execution_enabled"] is not True:
        raise PermissionError("real regeneration is execution-disabled before namespace reservation or saved-input access")
    source_receipt = _validate_sources(config)
    lock_payload = {
        "schema": "truncation-qc-global-consumption-v1", "token_identity": token_identity,
        "disabled_config_sha256": config_hash, "review_manifest_sha256": review_manifest_sha256,
        "source_hashes": source_receipt, "frozen_output": str(output), "mode": mode,
        "consumed_unix_s": time.time(), "automatic_retry": False,
    }
    global_lock, global_lock_sha = _reserve_global_token(ledger_root, token_identity, lock_payload)
    output.mkdir(parents=False, exist_ok=False)
    evidence = output / "evidence"
    evidence.mkdir()
    effective_config_path = evidence / "EFFECTIVE_CONFIG.json"
    _atomic_json(effective_config_path, config)
    effective_config_hash = sha256_file(effective_config_path)
    started = time.time()
    worker_command = [sys.executable, str(Path(__file__).resolve()), "--config", str(effective_config_path), "--expected-config-sha256", effective_config_hash,
                      "--worker-mode", mode, "--output", str(output), "--synthetic-outcome", outcome]
    launch = {"schema": "truncation-qc-managed-launch-v1", "mode": mode, "command": worker_command,
              "config_path": str(config_path), "config_sha256": config_hash,
              "effective_config_path": str(effective_config_path), "effective_config_sha256": effective_config_hash,
              "source_hashes": source_receipt,
              "runtime": config["runtime"], "resources": config["resources"], "started_unix_s": started,
              "global_token_identity": token_identity, "global_lock_path": str(global_lock),
              "global_lock_sha256": global_lock_sha}
    _atomic_json(evidence / "LAUNCH.json", launch)
    _atomic_json(evidence / "AUTHORIZATION_CONSUMED.json", {
        "schema": "one-shot-authorization-consumption-v1", "consumed": True, "consumed_on_invocation": True,
        "automatic_retry": False, "global_token_identity": token_identity,
        "global_lock_path": str(global_lock), "global_lock_sha256": global_lock_sha,
        "config_sha256": config_hash, "review_manifest_sha256": review_manifest_sha256,
        "mode": mode, "consumed_unix_s": time.time(),
    })
    _atomic_json(evidence / "STATUS.json", {"stage": "worker_starting", "final": False})
    env = os.environ.copy()
    threads = str(config["resources"]["cpu_threads_max"])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = threads
    stdout_path, stderr_path = evidence / "worker.stdout", evidence / "worker.stderr"
    reason, exit_code = None, None
    try:
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            process = subprocess.Popen(worker_command, stdout=stdout, stderr=stderr, env=env, cwd=Path("/tmp"))
            deadline = time.monotonic() + config["resources"]["wall_seconds_max"]
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    reason = "wall_seconds_max exceeded"
                    process.kill()
                    break
                if _tree_bytes(output) > config["resources"]["persistent_output_bytes_max"]:
                    reason = "persistent_output_bytes_max exceeded"
                    process.kill()
                    break
                time.sleep(0.05)
            exit_code = process.wait()
        if reason is None and exit_code != 0:
            reason = "worker exited nonzero"
        if reason is None and _tree_bytes(output) > config["resources"]["persistent_output_bytes_max"]:
            reason = "persistent_output_bytes_max exceeded"
        if reason is not None:
            raise RuntimeError(reason)
        worker_result = json.loads((evidence / "WORKER_RESULT.json").read_text())
        _atomic_json(evidence / "STATUS.json", {"stage": "succeeded", "final": True, "exit_code": exit_code})
        _atomic_json(evidence / "OUTPUT_INVENTORY.json", _inventory(output))
        run_receipt = {"schema": "truncation-qc-run-receipt-v1", "status": "success", "stage": "complete", "exit_code": exit_code,
                       "config_sha256": config_hash, "command": worker_command, "worker_result": worker_result,
                       "stdout_sha256": sha256_file(stdout_path), "stderr_sha256": sha256_file(stderr_path),
                       "output_inventory_sha256": sha256_file(evidence / "OUTPUT_INVENTORY.json"), "elapsed_seconds": time.time() - started}
        _atomic_json(evidence / "RUN_RECEIPT.json", run_receipt)
        if _tree_bytes(output) > config["resources"]["persistent_output_bytes_max"]:
            raise RuntimeError("persistent_output_bytes_max exceeded after durable receipts")
        _atomic_json(output / "COMPLETE.json", {"schema": "truncation-qc-managed-completion-v1", "status": "complete",
                     "completion_written_last": True, "config_sha256": config_hash,
                     "run_receipt_sha256": sha256_file(evidence / "RUN_RECEIPT.json"),
                     "authorization_consumed_sha256": sha256_file(evidence / "AUTHORIZATION_CONSUMED.json")})
        return True, run_receipt
    except Exception as error:
        failure = {"schema": "truncation-qc-failure-v1", "status": "failed", "stage": "managed_worker", "exit_code": exit_code,
                   "reason": reason or str(error), "exception_type": type(error).__name__, "config_sha256": config_hash,
                   "command": worker_command, "elapsed_seconds": time.time() - started, "completion_written": False, "automatic_retry": False}
        if stdout_path.is_file():
            failure["stdout_sha256"] = sha256_file(stdout_path)
        if stderr_path.is_file():
            failure["stderr_sha256"] = sha256_file(stderr_path)
        _atomic_json(evidence / "STATUS.json", {"stage": "failed", "final": True, "exit_code": exit_code})
        _atomic_json(evidence / "FAILURE.json", failure)
        _atomic_json(evidence / "OUTPUT_INVENTORY.json", _inventory(output))
        return False, failure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-config-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--synthetic", action="store_true")
    mode.add_argument("--execute-real", action="store_true")
    mode.add_argument("--worker-mode", choices=("synthetic", "real"))
    mode.add_argument("--gate-fixture-check", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--fixture-ledger-root", type=Path)
    parser.add_argument("--synthetic-outcome", choices=("success", "injected-failure", "corrupt-output"), default="success")
    args = parser.parse_args()
    blob = args.config.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != args.expected_config_sha256:
        raise RuntimeError("config hash mismatch")
    config = json.loads(blob)
    phase = args.worker_mode if args.worker_mode else "review_pending"
    prevalidate_structure(config, phase=phase)
    if args.validate_only:
        print(json.dumps({"status": "validated", "execution_enabled": False,
                          "sources": _validate_sources(config),
                          "runtime": "frozen_identity_checked_inside_preimport_bounded_worker"}, sort_keys=True))
        return 0
    if args.gate_fixture_check:
        if args.gate is None:
            raise ValueError("--gate is required for gate fixture check")
        gate = json.loads(args.gate.read_text())
        effective, identity = _validate_gate(gate, config, digest, allow_synthetic=True)
        print(json.dumps({"status": "gate_transition_valid", "token_identity": identity,
                          "effective_status": effective["status"],
                          "effective_execution_enabled": effective["execution_enabled"],
                          "synthetic_fixture": gate["synthetic_fixture"]}, sort_keys=True))
        return 0
    if args.worker_mode:
        if args.output is None:
            raise ValueError("--output is required for worker mode")
        output = args.output.resolve()
        heavy_before = sorted(name for name in RUNTIME_MODULES if name in sys.modules)
        limits = _apply_process_limits(config["resources"])
        bootstrap = {"heavy_modules_present_before_limits": heavy_before,
                     "limits_applied_before_runtime_imports": not heavy_before,
                     "enforced_limits": limits}
        _atomic_json(output / "evidence/BOOTSTRAP.json", bootstrap)
        _validate_sources(config)
        runtime = _validate_runtime(config)
        _atomic_json(output / "evidence/RUNTIME_ATTESTATION.json", runtime)
        result = _worker(config, output, args.worker_mode, args.synthetic_outcome, bootstrap)
        _atomic_json(output / "evidence/WORKER_RESULT.json", result)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.synthetic:
        if args.output is None or args.fixture_ledger_root is None:
            raise ValueError("synthetic mode requires --output and --fixture-ledger-root")
        output = args.output.resolve()
        ledger_root = args.fixture_ledger_root.resolve()
        token_identity = hashlib.sha256(json.dumps({
            "config_sha256": digest, "token": config["authorization"]["one_shot_token"],
            "fixture_ledger_root": str(ledger_root),
        }, sort_keys=True).encode()).hexdigest()
        effective = config
        review_manifest = "synthetic_fixture"
    else:
        expected_output = Path(config["activation"]["frozen_real_output_root"]).resolve()
        if args.output is not None and args.output.resolve() != expected_output:
            raise ValueError("caller output override differs from frozen real destination")
        output = expected_output
        if args.gate is None:
            raise ValueError("real execution requires a reviewed one-execution gate")
        gate = json.loads(args.gate.read_text())
        effective, token_identity = _validate_gate(gate, config, digest, allow_synthetic=False)
        ledger_root = Path(config["activation"]["global_ledger_root"]).resolve()
        review_manifest = gate["review_manifest_sha256"]
    selected_mode = "synthetic" if args.synthetic else "real"
    success, receipt = _managed_run(args.config.resolve(), digest, effective, output,
                                    selected_mode, args.synthetic_outcome,
                                    ledger_root=ledger_root, token_identity=token_identity,
                                    review_manifest_sha256=review_manifest)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
