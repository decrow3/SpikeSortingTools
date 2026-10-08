from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


EXPECTED_INTERPRETER = Path(
    "/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/"
    "environments/rescue-production/.venv/bin/python"
)
DEFAULT_ROOT = Path(__file__).resolve().parents[1]
STATIC_PROPERTIES = (
    "FragmentPath", "Environment", "TimeoutStartUSec", "OOMPolicy", "Restart",
    "MemoryHigh", "MemoryMax", "CPUQuotaPerSecUSec", "RuntimeMaxUSec",
    "Requires", "BindsTo", "After",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, text=True, capture_output=True, check=False, **kwargs)


def systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    completed = run(["systemctl", "--user", *args])
    if check and completed.returncode != 0:
        raise RuntimeError(f"systemctl {' '.join(args)} failed: {completed.stderr}")
    return completed


def loaded_properties(unit: str) -> dict[str, str]:
    completed = systemctl("show", unit, *[f"--property={key}" for key in STATIC_PROPERTIES])
    return dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def invoke_preflight(root: Path, contract: Path, manifest: Path, authorization: Path,
                     receipt: Path, target_fragment: Path, monitor_fragment: Path,
                     binary: Path) -> subprocess.CompletedProcess:
    code = (
        "import sys; sys.dont_write_bytecode=True; root=sys.argv.pop(1); "
        "sys.path.insert(0,root); "
        "sys.argv[0]='testing.candidate2_cache_managed_preflight'; "
        "from testing.candidate2_cache_managed_preflight import main; main()"
    )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["KILOSORT_BOUNDED_PREAD_PATH"] = str(binary)
    return run([
        str(EXPECTED_INTERPRETER), "-I", "-u", "-c", code, str(root / "source"),
        "--contract", str(contract), "--contract-sha256", sha256(contract),
        "--source-root", str(root / "source"), "--source-manifest", str(manifest),
        "--source-manifest-sha256", sha256(manifest), "--receipt", str(receipt),
        "--installed-unit", str(target_fragment), "--monitor-unit", str(monitor_fragment),
        "--authorization", str(authorization),
    ], env=environment)


def tree_identity(source_root: Path) -> dict[str, str]:
    return {str(path.relative_to(source_root)): sha256(path)
            for path in sorted(source_root.rglob("*")) if path.is_file()}


def invoke_preflight_bootstrap(root: Path, source_root: Path, contract: Path,
                               manifest: Path, authorization: Path, receipt: Path,
                               target_fragment: Path, monitor_fragment: Path,
                               binary: Path) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment["KILOSORT_BOUNDED_PREAD_PATH"] = str(binary)
    return run([
        str(EXPECTED_INTERPRETER), "-I", "-B", "-S", "-u", str(root / "preflight_bootstrap.py"),
        "--self-sha256", sha256(root / "preflight_bootstrap.py"),
        "--source-root", str(source_root), "--source-manifest", str(manifest),
        "--source-manifest-sha256", sha256(manifest),
        "--contract", str(contract), "--contract-sha256", sha256(contract),
        "--receipt", str(receipt), "--installed-unit", str(target_fragment),
        "--monitor-unit", str(monitor_fragment), "--authorization", str(authorization),
    ], env=environment)


def invoke_main_import_only(root: Path, source_root: Path, contract: Path,
                            manifest: Path, authorization: Path, receipt: Path,
                            target_fragment: Path, monitor_fragment: Path) -> subprocess.CompletedProcess:
    return run([
        str(EXPECTED_INTERPRETER), "-I", "-B", "-S", "-u", str(root / "bootstrap.py"),
        "--self-sha256", sha256(root / "bootstrap.py"),
        "--source-root", str(source_root), "--source-manifest", str(manifest),
        "--source-manifest-sha256", sha256(manifest),
        "--contract", str(contract), "--contract-sha256", sha256(contract),
        "--receipt", str(receipt), "--authorization", str(authorization),
        "--installed-unit", str(target_fragment), "--monitor-unit", str(monitor_fragment),
        "--sequential-acceptance-import-only",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    evidence_root = args.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / "source"))
    from testing.candidate2_cache_managed_preflight import capture_typed_exec_identity

    fixture = Path(tempfile.mkdtemp(prefix="candidate2-v8-full-preflight-"))
    suffix = str(os.getpid())
    target = f"candidate2-v8-preflight-target-{suffix}.service"
    monitor = f"candidate2-v8-preflight-monitor-{suffix}.service"
    unit_root = Path.home() / ".config/systemd/user"
    target_unit = unit_root / target
    monitor_unit = unit_root / monitor
    try:
        binary = fixture / "synthetic.raw"
        binary.write_bytes(b"\x00\x01" * 64)
        recording_manifest = fixture / "recording_manifest.json"
        write_json(recording_manifest, {"fixture": True})
        candidate_request = fixture / "candidate_request.json"
        write_json(candidate_request, {"sorter_params": {"fixture": True}})
        output_root = fixture / "output"
        monitor_output = fixture / "monitor"
        monitor_output.mkdir()
        runtime_source = fixture / "runtime" / "source"
        shutil.copytree(root / "source", runtime_source)

        target_script = fixture / "candidate2_cache_managed_full.py"
        monitor_script = fixture / "candidate2_cache_managed_monitor.py"
        target_script.write_text("import time\ntime.sleep(120)\n")
        monitor_script.write_text("import time\ntime.sleep(120)\n")
        unit_root.mkdir(parents=True, exist_ok=True)
        target_unit.write_text(
            "[Service]\nType=simple\nRestart=no\nOOMPolicy=stop\n"
            "TimeoutStartSec=infinity\nRuntimeMaxSec=infinity\n"
            "Environment=PYTHONDONTWRITEBYTECODE=1\n"
            f"Environment=KILOSORT_BOUNDED_PREAD_PATH={binary}\n"
            f"ExecStart=/usr/bin/python3 -I -S -u {target_script}\n"
            "[Install]\nWantedBy=default.target\n"
        )
        monitor_unit.write_text(
            "[Service]\nType=simple\nRestart=no\nOOMPolicy=stop\n"
            "TimeoutStartSec=infinity\nRuntimeMaxSec=infinity\n"
            f"ExecStart=/usr/bin/python3 -I -S -u {monitor_script}\n"
            "[Install]\nWantedBy=default.target\n"
        )
        systemctl("daemon-reload")
        systemctl("enable", target, monitor)
        systemctl("start", target, monitor)
        systemctl("stop", target, monitor)

        target_values = loaded_properties(target)
        monitor_values = loaded_properties(monitor)
        target_fragment = Path(target_values["FragmentPath"])
        monitor_fragment = Path(monitor_values["FragmentPath"])
        assert target_fragment.is_file() and monitor_fragment.is_file()
        unit_names = {"target": target, "monitor": monitor}
        inactive_typed = capture_typed_exec_identity(unit_names)

        source_files = tree_identity(runtime_source)
        frozen_tree = dict(source_files)
        manifest = fixture / "SOURCE_MANIFEST.json"
        write_json(manifest, {"schema": "candidate2-v8-fixture-source-v1", "files": source_files})
        installed_io = Path(
            "/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/"
            "environments/rescue-production/.venv/lib/python3.12/site-packages/kilosort/io.py"
        )
        contract = fixture / "CONTRACT.json"
        contract_value = {
            "schema": "candidate2-v8-full-preflight-fixture-v1",
            "execution_enabled": True,
            "attempt": {
                "service_name": target,
                "monitor_service_name": monitor,
                "output_root": str(output_root),
                "results_dir": str(output_root / "kilosort4"),
                "partial_results_dir": str(output_root / "kilosort4.partial"),
                "monitor_output": str(monitor_output),
            },
            "input": {
                "recording_binary": str(binary), "recording_binary_bytes": binary.stat().st_size,
                "recording_manifest": str(recording_manifest),
                "recording_manifest_sha256": sha256(recording_manifest),
            },
            "science_binding": {
                "candidate2_saved_request": str(candidate_request),
                "candidate2_saved_request_sha256": sha256(candidate_request),
            },
            "resource_conditions": {
                "minimum_output_free_bytes": 1,
                "minimum_mem_available_bytes": 1,
                "maximum_memory_psi_some_avg10": 100.0,
                "minimum_gpu_free_bytes": 1,
            },
            "cache_managed_mode": {"installed_kilosort_io_sha256": sha256(installed_io)},
            "source_and_environment": json.loads((root / "CONTRACT.json").read_text())["source_and_environment"],
        }
        contract_value["preflight_receipt"] = str(fixture / "positive" / "PREFLIGHT.json")
        write_json(contract, contract_value)

        authorization = fixture / "AUTHORIZATION.json"
        authorization_value = {
            "status": "AUTHORIZED_ONE_FRESH_CANDIDATE2_CACHE_MANAGED_RUN",
            "contract_sha256": sha256(contract),
            "source_manifest_sha256": sha256(manifest),
            "target_service_sha256": sha256(target_fragment),
            "monitor_service_sha256": sha256(monitor_fragment),
            "loaded_units": {"target": target_values, "monitor": monitor_values},
            "typed_exec_identity": inactive_typed,
        }
        write_json(authorization, authorization_value)

        systemctl("start", monitor, target)
        active_typed = capture_typed_exec_identity(unit_names)
        assert active_typed == inactive_typed
        target_show = systemctl("show", target, "--property=InvocationID", "--property=ControlGroup")
        target_runtime = dict(line.split("=", 1) for line in target_show.stdout.splitlines() if "=" in line)
        ready = {
            "schema": "candidate2-external-monitor-ready-v1",
            "contract_sha256": sha256(contract),
            "target_service_sha256": sha256(target_fragment),
            "monitor_service_sha256": sha256(monitor_fragment),
            "first_sample_complete": True,
            "first_sample_unix": time.time(),
            "ready_failures": [],
            "systemctl_returncode": "0",
            "target_load_state": "loaded",
            "target_active_state": "active",
            "target_invocation_id": target_runtime["InvocationID"],
            "target_control_group": target_runtime["ControlGroup"],
            "gpu_returncode": 0,
            "gpu_compute_returncode": 0,
        }
        write_json(monitor_output / "READY.json", ready)

        positive_receipt = fixture / "positive" / "PREFLIGHT.json"
        positive = invoke_preflight_bootstrap(root, runtime_source, contract, manifest, authorization,
                                              positive_receipt, target_fragment, monitor_fragment, binary)
        (evidence_root / "PREFLIGHT.stdout.txt").write_text(positive.stdout)
        (evidence_root / "PREFLIGHT.stderr.txt").write_text(positive.stderr)
        if positive_receipt.is_file():
            shutil.copy2(positive_receipt, evidence_root / "PREFLIGHT.json")
        write_json(evidence_root / "TREE_AFTER_PREFLIGHT.json", tree_identity(runtime_source))
        assert positive.returncode == 0, positive.stderr + "\nREADY=" + json.dumps(ready, sort_keys=True)
        receipt_value = json.loads(positive_receipt.read_text())
        assert receipt_value["pass"] is True
        assert receipt_value["recording_opened"] is False
        assert receipt_value["recording_bytes_read"] == 0
        assert receipt_value["typed_exec_identity"] == active_typed
        assert not output_root.exists()
        assert tree_identity(runtime_source) == frozen_tree
        job_receipt = fixture / "positive" / "JOB.json"
        imported = invoke_main_import_only(root, runtime_source, contract, manifest, authorization,
                                           job_receipt, target_fragment, monitor_fragment)
        (evidence_root / "MAIN_IMPORT.stdout.txt").write_text(imported.stdout)
        (evidence_root / "MAIN_IMPORT.stderr.txt").write_text(imported.stderr)
        if job_receipt.is_file():
            shutil.copy2(job_receipt, evidence_root / "JOB.json")
        write_json(evidence_root / "TREE_AFTER_MAIN_IMPORT.json", tree_identity(runtime_source))
        assert imported.returncode == 0, imported.stderr
        job = json.loads(job_receipt.read_text())
        assert job["state"] == "complete" and job["returncode"] == 0
        assert job["mode"] == "sequential_acceptance_import_only"
        assert tree_identity(runtime_source) == frozen_tree
        mutation = runtime_source / "SEQUENTIAL_ACCEPTANCE_MUTATION.txt"
        mutation.write_text("reject me\n")
        mutation_receipt = fixture / "mutation" / "JOB.json"
        mutation_receipt.parent.mkdir(parents=True, exist_ok=False)
        rejected = invoke_main_import_only(root, runtime_source, contract, manifest, authorization,
                                           mutation_receipt,
                                           target_fragment, monitor_fragment)
        (evidence_root / "MUTATION.stdout.txt").write_text(rejected.stdout)
        (evidence_root / "MUTATION.stderr.txt").write_text(rejected.stderr)
        if mutation_receipt.is_file():
            shutil.copy2(mutation_receipt, evidence_root / "MUTATION_JOB.json")
        write_json(evidence_root / "TREE_DURING_MUTATION.json", tree_identity(runtime_source))
        assert rejected.returncode != 0
        assert "runtime source inventory mismatch" in rejected.stderr
        mutation.unlink()
        assert tree_identity(runtime_source) == frozen_tree
        write_json(evidence_root / "TREE_FINAL.json", tree_identity(runtime_source))

        corruptions = {
            "path": lambda value: value["typed_exec_identity"]["target"]["ExecStart"][0].__setitem__("executable_path", "/wrong/python"),
            "argv": lambda value: value["typed_exec_identity"]["target"]["ExecStart"][0]["argv"].append("--extra-boundary"),
            "ignore_errors": lambda value: value["typed_exec_identity"]["target"]["ExecStart"][0].__setitem__("ignore_errors", True),
        }
        for name, corrupt in corruptions.items():
            changed = json.loads(json.dumps(authorization_value))
            corrupt(changed)
            changed_path = fixture / f"AUTHORIZATION.{name}.json"
            write_json(changed_path, changed)
            completed = invoke_preflight(
                root, contract, manifest, changed_path, fixture / name / "PREFLIGHT.json",
                target_fragment, monitor_fragment, binary,
            )
            (evidence_root / f"AUTH_{name}.stdout.txt").write_text(completed.stdout)
            (evidence_root / f"AUTH_{name}.stderr.txt").write_text(completed.stderr)
            negative_receipt = fixture / name / "PREFLIGHT.json"
            if negative_receipt.is_file():
                shutil.copy2(negative_receipt, evidence_root / f"AUTH_{name}_PREFLIGHT.json")
            assert completed.returncode != 0
            assert "typed execution identity differs" in completed.stderr

        summary = {
            "pass": True,
            "positive_receipt": str(positive_receipt),
            "inactive_equals_active_typed_identity": True,
            "negative_cases": sorted(corruptions),
            "actual_preflight_bootstrap": True,
            "actual_main_bootstrap_runner_import": True,
            "source_unchanged_at_each_boundary": True,
            "source_mutation_rejected": True,
            "recording_opened": False,
            "recording_bytes_read": 0,
            "sort_gpu_work_started": False,
        }
        write_json(evidence_root / "SUMMARY.json", summary)
        print(json.dumps(summary, sort_keys=True))
    finally:
        systemctl("stop", target, monitor, check=False)
        systemctl("disable", target, monitor, check=False)
        target_unit.unlink(missing_ok=True)
        monitor_unit.unlink(missing_ok=True)
        systemctl("daemon-reload", check=False)
        systemctl("reset-failed", target, monitor, check=False)
        shutil.rmtree(fixture, ignore_errors=True)


if __name__ == "__main__":
    main()
