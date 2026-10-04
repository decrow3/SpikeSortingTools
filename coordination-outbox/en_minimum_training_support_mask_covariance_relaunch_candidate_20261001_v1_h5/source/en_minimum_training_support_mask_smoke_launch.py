#!/usr/bin/env python3
"""Exact one-shot launcher/preflight for the reviewed support-mask smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from npx_preprocessing.motion import kilosort_support_mask_candidate as candidate


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight(config_path: Path, expected_sha256: str) -> dict:
    observed = sha256(config_path)
    if observed != expected_sha256:
        raise ValueError(f"config hash mismatch: {observed} != {expected_sha256}")
    contract = candidate.validate_support_mask_contract(
        config_path, expected_config_sha256=expected_sha256
    )
    config = json.loads(config_path.read_text())
    if contract.execution_enabled is not True:
        raise PermissionError("exact enabled smoke contract is required")
    binary = Path(config["recording"]["binary_path"])
    binary_stat = binary.stat()
    run_root = Path(config["smoke"]["run_root"])
    if run_root.exists():
        raise FileExistsError(f"planned run root already exists: {run_root}")
    packages = {}
    for name in ("kilosort", "numpy", "torch", "spikeinterface"):
        module = __import__(name)
        packages[name] = {
            "version": importlib.metadata.version(name),
            "origin": str(Path(inspect.getfile(module)).resolve()),
        }
    return {
        "status": "zero_voltage_preflight_passed",
        "python_executable": sys.executable,
        "python_version": sys.version,
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_build": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "config_path": str(config_path),
        "config_sha256": observed,
        "contract_digest": contract.contract_digest,
        "execution_enabled": contract.execution_enabled,
        "recording_path": str(binary),
        "recording_stat_size": binary_stat.st_size,
        "recording_stat_mode": oct(binary_stat.st_mode & 0o777),
        "recording_metadata_readable": os.access(binary, os.R_OK),
        "recording_content_opened": False,
        "run_root": str(run_root),
        "run_root_absent": True,
        "kilosort_resumable_within_sort": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-config-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute-smoke", action="store_true")
    args = parser.parse_args()
    report = preflight(args.config.resolve(), args.expected_config_sha256)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.preflight_only:
        return 0
    result, audit = candidate.run_kilosort_with_runtime_covariance(
        config_path=args.config.resolve(),
        expected_config_sha256=args.expected_config_sha256,
    )
    if result.get("status") != "smoke_stop_after_covariance_whitening":
        raise RuntimeError("native smoke did not stop at the required boundary")
    if audit.covariance_validation is None or not audit.covariance_validation.passed:
        raise RuntimeError("native smoke lacks a passing covariance validation")
    if audit.whitening_finite is not True or audit.whitening_sha256 is None:
        raise RuntimeError("native smoke lacks a finite audited whitening matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
