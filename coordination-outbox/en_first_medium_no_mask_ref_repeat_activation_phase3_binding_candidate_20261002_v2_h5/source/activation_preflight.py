#!/usr/bin/env python3
"""Read-only activation gate. It never opens recording voltage or outcomes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_file(path: str | Path, expected: str, label: str) -> Path:
    member = Path(path)
    if not member.is_file() or member.is_symlink():
        raise ValueError(f"{label} is not a non-symlink regular file")
    observed = sha256_file(member)
    if observed != expected:
        raise ValueError(f"{label} hash differs: {observed} != {expected}")
    return member


def verify_packet(packet: Path, manifest_sha: str, complete_sha: str) -> None:
    manifest = exact_file(packet / "MANIFEST.sha256", manifest_sha, "packet manifest")
    exact_file(packet / "COMPLETE.json", complete_sha, "packet complete")
    for line in manifest.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        exact_file(packet / relative, expected, f"packet member {relative}")


def load_reviewed_launcher(contract: dict):
    contract_dir = Path(contract["_contract_dir"])
    binding = contract["sources"]["pair_launcher"]
    launcher = (contract_dir / binding["path"]).resolve()
    exact_file(launcher, binding["sha256"], "reviewed pair launcher")
    spec = importlib.util.spec_from_file_location("reviewed_pair_launcher", launcher)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load reviewed pair launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_runtime(binding_path: Path, expected_sha: str) -> dict:
    binding_path = exact_file(binding_path, expected_sha, "runtime binding")
    binding = json.loads(binding_path.read_text())
    executable = Path(binding["python_executable"])
    if not executable.is_symlink():
        raise ValueError("runtime executable is no longer the bound symlink")
    resolved = executable.resolve()
    if str(resolved) != binding["python_resolved_path"]:
        raise ValueError("runtime executable target differs")
    if sha256_file(resolved) != binding["python_resolved_sha256"]:
        raise ValueError("runtime executable hash differs")
    if sha256_file(binding["pyvenv_cfg_path"]) != binding["pyvenv_cfg_sha256"]:
        raise ValueError("pyvenv configuration hash differs")
    for package, expected in binding["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"runtime package version differs: {package}")
    return binding


def inspect(contract_path: Path, expected_contract_sha256: str) -> dict:
    exact_file(contract_path, expected_contract_sha256, "activation candidate contract")
    raw = json.loads(contract_path.read_text())
    raw["_contract_dir"] = str(contract_path.resolve().parent)
    activation = raw["activation"]
    contract_dir = contract_path.resolve().parent

    parent = activation["reviewed_launcher_packet"]
    verify_packet(Path(parent["path"]), parent["manifest_sha256"], parent["complete_sha256"])
    parent_contract_path = exact_file(parent["contract_path"], parent["contract_sha256"], "reviewed parent contract")
    parent_contract = json.loads(parent_contract_path.read_text())

    review = activation["h1_launcher_go_review"]
    verify_packet(Path(review["path"]), review["manifest_sha256"], review["complete_sha256"])
    independent = json.loads((Path(review["path"]) / "INDEPENDENT_REVIEW.json").read_text())
    if independent.get("verdict") != "GO_R2_FINALIZATION_REPAIR_ONLY_EXECUTION_REMAINS_DISABLED":
        raise PermissionError("exact H1 launcher GO verdict is absent")
    if independent.get("subject", {}).get("manifest_sha256") != parent["manifest_sha256"]:
        raise PermissionError("H1 GO subject differs from reviewed launcher packet")
    if raw.get("approval", {}).get("h1_review_manifest_sha256") != review["manifest_sha256"]:
        raise PermissionError("contract H1 approval binding differs from exact GO review")
    if raw.get("gates", {}).get("launcher_r2_h1_go_manifest_sha256") != review["manifest_sha256"]:
        raise PermissionError("contract launcher GO gate differs")
    if parent["launcher_sha256"] != "238c9c997bad67415a13671deaa5569ee121f3a27aa07188fef51c424f2029f8":
        raise PermissionError("reviewed launcher-v3 parent identity differs")

    launcher = load_reviewed_launcher(raw)
    validated = launcher.validate_contract(contract_path, expected_contract_sha256)
    if validated["resources"]["persistent_output_bytes_max"] != 17179869184:
        raise ValueError("pair-wide total cap differs")
    if validated["resources"]["terminal_control_reserve_bytes"] != 1048576:
        raise ValueError("terminal reserve differs")
    if validated["resources"]["live_output_bytes_max"] != 17178820608:
        raise ValueError("live threshold differs")
    if validated["voltage_identity"] != parent_contract["voltage_identity"]:
        raise ValueError("input/voltage identity differs from reviewed parent")

    runtime = activation["runtime_binding"]
    runtime_path = Path(runtime["path"])
    if not runtime_path.is_absolute():
        runtime_path = contract_dir / runtime_path
    verify_runtime(runtime_path, runtime["sha256"])
    preflight = activation["activation_preflight"]
    preflight_path = Path(preflight["path"])
    if not preflight_path.is_absolute():
        preflight_path = contract_dir / preflight_path
    exact_file(preflight_path, preflight["sha256"], "activation preflight")

    planned = validated["planned_paths"]
    fresh = {name: not Path(planned[name]).exists() for name in ("root", "numba_cache", "logs", "attempt_evidence")}
    phase3 = activation["required_phase3_review"]
    candidate = phase3["candidate"]
    review_binding = phase3["independent_go_review"]
    verify_packet(
        Path(candidate["path"]), candidate["manifest_sha256"], candidate["complete_sha256"]
    )
    producer_delta = json.loads((Path(candidate["path"]) / "PRODUCER_DELTA.json").read_text())
    if producer_delta.get("base_launcher_sha256") != parent["launcher_sha256"]:
        raise PermissionError("Phase3 producer delta does not descend from the reviewed launcher")
    if producer_delta.get("candidate_launcher_sha256") != raw["sources"]["pair_launcher"]["sha256"]:
        raise PermissionError("active proposed launcher differs from the accepted Phase3 producer delta")
    verify_packet(
        Path(review_binding["path"]),
        review_binding["manifest_sha256"],
        review_binding["complete_sha256"],
    )
    independent = json.loads((Path(review_binding["path"]) / "REVIEW.json").read_text())
    if independent.get("verdict") != "GO_PROVENANCE_REPAIR_DELTA":
        raise PermissionError("exact Phase3 independent GO verdict is absent")
    if (
        independent.get("candidate_manifest_sha256") != candidate["manifest_sha256"]
        or independent.get("candidate_complete_sha256") != candidate["complete_sha256"]
    ):
        raise PermissionError("Phase3 independent GO subject differs from the bound candidate")
    status_path = exact_file(
        review_binding["status_path"], review_binding["status_sha256"],
        "Phase3 independent GO coordination status",
    )
    status = json.loads(status_path.read_text())
    if (
        status.get("verdict") != "GO_PROVENANCE_REPAIR_DELTA"
        or status.get("manifest_sha256") != review_binding["manifest_sha256"]
        or status.get("complete_sha256") != review_binding["complete_sha256"]
    ):
        raise PermissionError("Phase3 coordination status differs from the bound GO review")
    phase3_bound = (
        phase3.get("status") == "BOUND_GO_EXECUTION_STILL_DISABLED"
        and raw.get("approval", {}).get("phase3_review_manifest_sha256")
        == review_binding["manifest_sha256"]
        and raw.get("gates", {}).get("phase3_review_manifest_sha256")
        == review_binding["manifest_sha256"]
    )
    report = {
        "status": "INSPECTED_FAIL_CLOSED",
        "execution_enabled": validated["execution_enabled"],
        "exact_h1_launcher_go_bound": True,
        "phase3_review_bound": phase3_bound,
        "exact_phase3_candidate_bound": True,
        "exact_phase3_independent_go_bound": True,
        "fresh_namespaces": fresh,
        "all_namespaces_fresh": all(fresh.values()),
        "recording_or_voltage_opened": False,
        "outcomes_opened": False,
        "activation_eligible": bool(validated["execution_enabled"] and phase3_bound and all(fresh.values())),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--expected-contract-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect-disabled", action="store_true")
    mode.add_argument("--require-activation", action="store_true")
    args = parser.parse_args()
    report = inspect(args.contract, args.expected_contract_sha256)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_activation and not report["activation_eligible"]:
        raise PermissionError("activation remains fail-closed: an enabled independently reviewed derivative is required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
