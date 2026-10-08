"""Stdlib-only, fail-closed launcher for the candidate-2 managed service."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


ENV_ROOT = Path("/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/environments/rescue-production")
INSTALLED = {
    "spikeinterface/sorters/external/kilosort4.py": ENV_ROOT / ".venv/lib/python3.12/site-packages/spikeinterface/sorters/external/kilosort4.py",
    "kilosort/run_kilosort.py": ENV_ROOT / ".venv/lib/python3.12/site-packages/kilosort/run_kilosort.py",
    "kilosort/io.py": ENV_ROOT / ".venv/lib/python3.12/site-packages/kilosort/io.py",
    "kilosort/parameters.py": ENV_ROOT / ".venv/lib/python3.12/site-packages/kilosort/parameters.py",
    "kilosort/datashift.py": ENV_ROOT / ".venv/lib/python3.12/site-packages/kilosort/datashift.py",
    "environment_uv_lock": ENV_ROOT / "uv.lock",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_receipt(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def verify_source(root: Path, manifest_path: Path, manifest_sha256: str) -> dict:
    if sha256(manifest_path) != manifest_sha256:
        raise RuntimeError("runtime source manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("runtime source manifest has no files")
    observed_paths = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"runtime source contains symlink: {path}")
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            observed_paths.add(rel)
            if path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts:
                raise RuntimeError(f"runtime source contains bytecode: {rel}")
    if observed_paths != set(expected):
        raise RuntimeError(f"runtime source inventory mismatch: missing={sorted(set(expected)-observed_paths)}, extra={sorted(observed_paths-set(expected))}")
    mismatches = {}
    for rel, digest in expected.items():
        observed = sha256(root / rel)
        if observed != digest:
            mismatches[rel] = {"expected": digest, "observed": observed}
    if mismatches:
        raise RuntimeError(f"runtime source content mismatch: {mismatches}")
    return {"manifest_sha256": manifest_sha256, "members": len(expected)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--installed-unit", type=Path, required=True)
    parser.add_argument("--monitor-unit", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--sequential-acceptance-import-only", action="store_true")
    args = parser.parse_args()

    receipt = {"schema": "candidate2-verified-managed-job-v1", "state": "starting", "started_at": now(), "bootstrap_pid": os.getpid()}
    write_receipt(args.receipt, receipt)
    try:
        if sha256(Path(__file__).resolve()) != args.self_sha256:
            raise RuntimeError("bootstrap self hash mismatch")
        if sha256(args.contract) != args.contract_sha256:
            raise RuntimeError("contract hash mismatch")
        contract = json.loads(args.contract.read_text())
        authorization = json.loads(args.authorization.read_text())
        expected_authorization = {
            "status": "AUTHORIZED_ONE_FRESH_CANDIDATE2_CACHE_MANAGED_RUN",
            "contract_sha256": args.contract_sha256,
            "source_manifest_sha256": args.source_manifest_sha256,
            "target_service_sha256": sha256(args.installed_unit),
            "monitor_service_sha256": sha256(args.monitor_unit),
        }
        if any(authorization.get(key) != value for key, value in expected_authorization.items()):
            raise RuntimeError("installed service/contract/source authorization differs")
        if contract.get("execution_enabled") is not True:
            raise RuntimeError("contract is execution-disabled")
        source_proof = verify_source(args.source_root.resolve(), args.source_manifest.resolve(), args.source_manifest_sha256)
        installed_expected = contract.get("source_and_environment", {}).get("installed_source_sha256", {})
        if set(installed_expected) != set(INSTALLED):
            raise RuntimeError("installed-source inventory differs from bootstrap inventory")
        installed_observed = {name: sha256(path) for name, path in INSTALLED.items()}
        if installed_observed != installed_expected:
            raise RuntimeError("installed sorter/environment source hash mismatch")
        if args.verify_only:
            receipt.update({"state": "complete", "mode": "verify_only", "returncode": 0,
                            "verified_at": now(), "finished_at": now(), "source": source_proof,
                            "installed_source_sha256": installed_observed})
            write_receipt(args.receipt, receipt)
            return 0

        child_env = dict(os.environ)
        child_env.pop("PYTHONPATH", None)
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        isolated_import = (
            "import sys; sys.dont_write_bytecode=True; root=sys.argv.pop(1); sys.path.insert(0, root); "
            "sys.argv[0]='testing.candidate2_cache_managed_full'; "
            "from testing.candidate2_cache_managed_full import main; "
            "main() if sys.argv.pop(1)=='run' else None"
        )
        child_mode = "import-only" if args.sequential_acceptance_import_only else "run"
        command = [
            sys.executable, "-I", "-B", "-u", "-c", isolated_import, str(args.source_root.resolve()), child_mode,
            "--contract", str(args.contract), "--contract-sha256", args.contract_sha256,
            "--source-manifest-sha256", args.source_manifest_sha256,
            "--preflight", str(contract["preflight_receipt"]),
        ]
        receipt.update({"state": "running", "verified_at": now(), "source": source_proof,
                        "installed_source_sha256": installed_observed, "command": command})
        write_receipt(args.receipt, receipt)
        process = subprocess.Popen(command, cwd=args.source_root, env=child_env)
        receipt["child_pid"] = process.pid
        write_receipt(args.receipt, receipt)
        returncode = process.wait()
        receipt.update({"state": "complete" if returncode == 0 else "failed", "returncode": returncode, "finished_at": now(),
                        "mode": "sequential_acceptance_import_only" if args.sequential_acceptance_import_only else "production"})
        write_receipt(args.receipt, receipt)
        return returncode
    except BaseException as exc:
        receipt.update({"state": "failed", "returncode": None, "finished_at": now(),
                        "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        write_receipt(args.receipt, receipt)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
