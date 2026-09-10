"""Wait for the shared Luke0804 field contract, then correct and sort once.

This service-side controller deliberately refuses the estimator-only package.
The producer must separately publish a hash-bound, correction-ready application
contract after scientific review.  No estimation or parameter fitting occurs
on this host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.config import fingerprint
from pipeline.preprocess import validate_accepted_recording
from pipeline.runtime import validate_production_environment
from pipeline.sorting import _json_safe_params, build_kilosort4_params, run_kilosort4
from testing.luke_external_warp_pipeline import _materialize_arm


SCHEMA = "luke-improved-nonrigid-queue-v1"
CONTRACT_SCHEMA = "luke-improved-motion-application-contract-v1"
EXPECTED_RECORDING_SHA256 = "2d6fac755db3182841bf121d822754f963ebada3bcb8cd9e96e0b63e6f9b7372"
EXPECTED_SAMPLES = 314_204_894
EXPECTED_CHANNELS = 384
MIN_FREE_BYTES = 600 * 1024**3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_status(local_path: Path, shared_path: Path | None, **updates: Any) -> dict[str, Any]:
    current = json.loads(local_path.read_text()) if local_path.exists() else {}
    payload = {
        "schema_version": SCHEMA,
        "owner": "huklaban5",
        "host": socket.gethostname(),
        **current,
        **updates,
        "updated_at": _now(),
    }
    _atomic_json(local_path, payload)
    if shared_path is not None:
        try:
            _atomic_json(shared_path, payload)
        except OSError as error:
            payload["shared_status_error"] = repr(error)
            _atomic_json(local_path, payload)
    return payload


def _validate_manifest(shared: Path) -> tuple[dict[str, Any], str]:
    path = shared / "field_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "luke-shared-medicine-field-v1":
        raise ValueError("unsupported field manifest schema")
    if manifest.get("status") != "estimation_artifacts_complete":
        raise ValueError("field artifacts are not complete")
    if manifest.get("recording_content_sha256") != EXPECTED_RECORDING_SHA256:
        raise ValueError("field manifest names the wrong accepted recording")
    package = shared / manifest["field_package"]
    if package.name.endswith(".partial") or not package.is_dir():
        raise ValueError("field package is absent or partial")
    for relative, expected in manifest.get("files", {}).items():
        candidate = package / relative
        if not candidate.is_file() or _sha256(candidate) != expected:
            raise ValueError(f"field package digest mismatch: {relative}")
    if not manifest.get("files"):
        raise ValueError("field manifest has no file digests")
    return manifest, _sha256(path)


def _contract_path(shared: Path) -> Path:
    authorized = shared / "nonrigid_run_contract_authorized_v1.json"
    return authorized if authorized.is_file() else shared / "run_contract.json"


def _validate_contract(shared: Path, manifest: dict[str, Any], manifest_sha: str) -> dict[str, Any]:
    contract = json.loads(_contract_path(shared).read_text())
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("unsupported application contract schema")
    if contract.get("status") != "frozen" or contract.get("correction_ready") is not True:
        raise ValueError("application contract is not frozen and correction-ready")
    explicitly_released = (
        contract.get("development_only") is True
        and contract.get("scientific_status") == "requires_review"
        and contract.get("release_policy")
        == "Explicit experimental authorization replaces scientific-validation gate; preserve original field manifest and its requires_review status. Consumer must support this release policy."
        and "User explicitly directs nonrigid comparison to start" in contract.get("user_authorization", "")
    )
    if manifest.get("correction_ready") is not True and not explicitly_released:
        raise ValueError("field manifest is not promoted and no explicit experimental release applies")
    if contract.get("field_manifest_sha256") != manifest_sha:
        raise ValueError("application contract does not bind the current field manifest")
    if contract.get("recording_content_sha256") != EXPECTED_RECORDING_SHA256:
        raise ValueError("application contract names the wrong accepted recording")
    arm = contract.get("nonrigid", {})
    required = {
        "field_file",
        "time_array",
        "depth_array",
        "displacement_array",
        "polarity_convention",
        "border_mode",
        "spatial_interpolation_method",
        "sigma_um",
        "p",
        "output_channel_ids",
        "leading_nearest_margin_max_s",
        "trailing_nearest_margin_max_s",
        "dtype",
    }
    missing = sorted(required - set(arm))
    if missing:
        raise ValueError(f"nonrigid contract is incomplete: {missing}")
    if arm["polarity_convention"] != "output_depth = source_depth - displacement":
        raise ValueError("unvalidated displacement sign convention")
    if arm["dtype"] != "int16":
        raise ValueError("only the agreed int16 output contract is supported")
    if contract.get("sorter", {}).get("do_correction") is not False:
        raise ValueError("Kilosort internal motion correction must be disabled")
    if [contract["sorter"].get("Th_universal"), contract["sorter"].get("Th_learned")] != [12, 9]:
        raise ValueError("Kilosort thresholds must remain frozen at 12/9")
    if _json_safe_params(build_kilosort4_params()) != contract["sorter"]:
        raise ValueError("resolved locked Kilosort settings differ from the frozen run contract")
    return contract


def ready_package(shared: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    manifest, manifest_sha = _validate_manifest(shared)
    contract = _validate_contract(shared, manifest, manifest_sha)
    return manifest, contract, manifest_sha


def _load_field(shared: Path, manifest: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    arm = contract["nonrigid"]
    package = shared / manifest["field_package"]
    field_path = package / arm["field_file"]
    relative = str(Path(arm["field_file"]))
    if relative not in manifest["files"] or _sha256(field_path) != manifest["files"][relative]:
        raise ValueError("contract field file is not digest-bound by the field manifest")
    with np.load(field_path, allow_pickle=False) as data:
        times = np.asarray(data[arm["time_array"]], dtype=np.float64)
        depths = np.asarray(data[arm["depth_array"]], dtype=np.float64)
        displacement = np.asarray(data[arm["displacement_array"]], dtype=np.float64)
    if displacement.shape != (times.size, depths.size):
        raise ValueError("nonrigid field must be time-by-depth")
    if not (np.all(np.isfinite(displacement)) and np.all(np.diff(times) > 0) and np.all(np.diff(depths) > 0)):
        raise ValueError("nonrigid field or axes are non-finite/non-monotonic")
    return {"time_s": times, "depth_um": depths, "displacement_um": displacement, "file_sha256": _sha256(field_path)}


def _gpu_busy() -> bool:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if query.returncode:
        raise RuntimeError(f"nvidia-smi failed: {query.stderr.strip()}")
    # ``validate_production_environment(require_cuda=True)`` may initialize a
    # CUDA context in this worker.  That is not a competing job and must not
    # make the queue wait on itself forever.  The parent is the receipt wrapper.
    ignored_pids = {os.getpid(), os.getppid()}
    desktop_contexts = {"Xorg", "gnome-shell", "snapd-desktop-integration"}
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2 or not fields[0].isdigit():
            continue
        pid = int(fields[0])
        process_name = Path(fields[1]).name
        if pid in ignored_pids or process_name in desktop_contexts:
            continue
        return True
    return False


def _corrected_recording(source: Any, field: dict[str, Any], arm: dict[str, Any]) -> Any:
    from spikeinterface.core.motion import Motion
    from spikeinterface.preprocessing import astype
    from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording
    from testing.checked_int16_recording import CheckedInt16Recording

    duration = source.get_num_samples() / source.get_sampling_frequency()
    leading = float(field["time_s"][0])
    trailing = float(duration - field["time_s"][-1])
    if leading < 0 or trailing < 0:
        raise ValueError("field time grid extends outside the recording")
    if leading > float(arm["leading_nearest_margin_max_s"]) + 1e-9:
        raise ValueError("leading nearest-field margin exceeds frozen policy")
    if trailing > float(arm["trailing_nearest_margin_max_s"]) + 1.0 / source.get_sampling_frequency():
        raise ValueError("trailing nearest-field margin exceeds frozen policy")
    motion = Motion(
        displacement=[field["displacement_um"]],
        temporal_bins_s=[field["time_s"] + float(source.sample_index_to_time(0))],
        spatial_bins_um=field["depth_um"],
        direction="y",
        interpolation_method="linear",
    )
    corrected = InterpolateMotionRecording(
        astype(source, "float32"),
        motion,
        border_mode=arm["border_mode"],
        spatial_interpolation_method=arm["spatial_interpolation_method"],
        sigma_um=float(arm["sigma_um"]),
        p=int(arm["p"]),
    )
    selected = corrected.select_channels(channel_ids=arm["output_channel_ids"])
    if [str(value) for value in selected.get_channel_ids()] != [
        str(value) for value in arm["output_channel_ids"]
    ]:
        raise ValueError("selected correction channels differ from the frozen order")
    return CheckedInt16Recording(selected)


def execute(shared: Path, source_dir: Path, output_root: Path, local_status: Path, shared_status: Path) -> None:
    from spikeinterface.core import load

    manifest, contract, manifest_sha = ready_package(shared)
    source_manifest = validate_accepted_recording(source_dir)
    if source_manifest.get("recording_content_sha256") != EXPECTED_RECORDING_SHA256:
        raise ValueError("accepted source content identity differs")
    if source_manifest.get("num_samples") != EXPECTED_SAMPLES or source_manifest.get("num_channels") != EXPECTED_CHANNELS:
        raise ValueError("accepted source extent differs")
    if shutil.disk_usage(output_root.parent).free < MIN_FREE_BYTES:
        raise RuntimeError("less than 600 GiB free for corrected voltage and sorter scratch")
    environment = validate_production_environment(require_cuda=True)
    field = _load_field(shared, manifest, contract)
    source = load(source_dir)
    corrected = _corrected_recording(source, field, contract["nonrigid"])
    expected_ids = [str(value) for value in contract["nonrigid"]["output_channel_ids"]]
    observed_ids = [str(value) for value in corrected.get_channel_ids()]
    if observed_ids != expected_ids:
        raise ValueError("operator output channels/order differ from frozen common channel set")
    request = {
        "schema": SCHEMA,
        "arm": "nonrigid",
        "source_content_sha256": EXPECTED_RECORDING_SHA256,
        "source_request_digest": source_manifest["request_digest"],
        "field_manifest_sha256": manifest_sha,
        "field_file_sha256": field["file_sha256"],
        "run_contract_sha256": _sha256(_contract_path(shared)),
        "application": contract["nonrigid"],
        "sorter": contract["sorter"],
        "correction_once": True,
    }
    _publish_status(local_status, shared_status, phase="correcting", job_state="running", environment=environment)
    corrected_dir = output_root / "corrected_recording"
    corrected_manifest = _materialize_arm(
        corrected,
        corrected_dir,
        source_manifest=source_manifest,
        request=request,
        n_jobs=int(contract["nonrigid"].get("n_jobs", 8)),
    )
    if _gpu_busy():
        _publish_status(local_status, shared_status, phase="ready_waiting_for_gpu", job_state="running")
        while _gpu_busy():
            time.sleep(60)
    _publish_status(local_status, shared_status, phase="sorting", job_state="running", corrected_recording_manifest=corrected_manifest)
    sort_manifest = run_kilosort4(corrected_dir, output_root / "kilosort4")
    _publish_status(
        local_status,
        shared_status,
        phase="complete",
        job_state="complete",
        corrected_recording_manifest=corrected_manifest,
        sort_manifest=sort_manifest,
        completed_at=_now(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--reuse-completed-correction", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        completed = args.output / "corrected_recording" / "rescue_recording_manifest.json"
        if not args.reuse_completed_correction or not completed.is_file():
            raise RuntimeError("existing output is not an explicitly reusable completed correction")
    else:
        args.output.mkdir(parents=True)
    local_status = args.output / "queue_status.json"
    shared_status = args.shared / "huklaban5_status.json"
    _publish_status(
        local_status,
        shared_status,
        phase="ready_waiting_for_field",
        job_state="running",
        source=str(args.source),
        local_output_root=str(args.output),
        required_contract_schema=CONTRACT_SCHEMA,
        no_automatic_restart=True,
        interruption_policy="Correction can reuse only a fully validated materialization; Kilosort interruption requires investigation and whole-sort restart.",
    )
    while True:
        try:
            ready_package(args.shared)
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(args.poll_seconds)
            continue
        except ValueError as error:
            _publish_status(local_status, shared_status, phase="ready_waiting_for_field", job_state="running", waiting_reason=str(error))
            time.sleep(args.poll_seconds)
            continue
        break
    try:
        execute(args.shared, args.source, args.output, local_status, shared_status)
    except BaseException as error:
        _publish_status(
            local_status,
            shared_status,
            phase="blocked",
            job_state="failed",
            failure={"type": type(error).__name__, "message": str(error)},
            failed_at=_now(),
        )
        raise


if __name__ == "__main__":
    main()
