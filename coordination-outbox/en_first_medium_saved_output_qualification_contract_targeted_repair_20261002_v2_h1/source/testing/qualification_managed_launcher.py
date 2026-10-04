"""Externally enforce the frozen resource envelope before qualification starts."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import resource
import secrets
import subprocess
import sys
import tempfile
from typing import Sequence


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def launch(contract_path: Path, request_path: Path, output: Path, worker_module: str,
           extra_args: Sequence[str] = ()) -> subprocess.CompletedProcess[str]:
    contract = json.loads(contract_path.read_text())
    request = json.loads(request_path.read_text())
    phase = request["phase"]
    bounds = contract["phases"][phase]["resources"]
    envelope = {
        "schema": "first-medium-managed-resource-envelope-v1",
        "phase": phase,
        "contract_sha256": _sha256(contract_path),
        "request_sha256": _sha256(request_path),
        "output": str(output.resolve()),
        "wall_seconds_max": bounds["wall_seconds_max"],
        "address_space_bytes_max": bounds["peak_rss_bytes_max"],
        "cpu_threads_max": bounds["cpu_threads_max"],
        "gpu_count": 0,
        "persistent_output_bytes_max": bounds["persistent_output_bytes_max"],
    }
    secret = secrets.token_hex(32)
    envelope["hmac_sha256"] = hmac.new(
        secret.encode(), json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode(), hashlib.sha256
    ).hexdigest()
    fd, raw_path = tempfile.mkstemp(prefix="qualification-resource-", suffix=".json")
    envelope_path = Path(raw_path)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(envelope, stream, sort_keys=True)
        os.chmod(envelope_path, 0o400)
        env = os.environ.copy()
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
            env[key] = str(min(int(env.get(key, bounds["cpu_threads_max"])), int(bounds["cpu_threads_max"])))
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["NVIDIA_VISIBLE_DEVICES"] = "void"
        env["QUALIFICATION_RESOURCE_ENVELOPE"] = str(envelope_path)
        env["QUALIFICATION_RESOURCE_SECRET"] = secret
        cpus = sorted(os.sched_getaffinity(0))[:int(bounds["cpu_threads_max"])]

        def limits() -> None:
            resource.setrlimit(resource.RLIMIT_AS, (
                int(bounds["peak_rss_bytes_max"]), int(bounds["peak_rss_bytes_max"])
            ))
            os.sched_setaffinity(0, cpus)

        command = [sys.executable, "-m", worker_module, "--contract", str(contract_path),
                   "--request", str(request_path), "--output", str(output), *extra_args]
        try:
            return subprocess.run(
                command, env=env, text=True, capture_output=True, preexec_fn=limits,
                timeout=float(bounds["wall_seconds_max"]),
            )
        except subprocess.TimeoutExpired as error:
            if output.is_dir() and not (output / "COMPLETE.json").exists():
                (output / "MANAGED_TERMINATION.json").write_text(json.dumps({
                    "schema": "first-medium-managed-termination-v1",
                    "reason": "external_wall_timeout",
                    "wall_seconds_max": bounds["wall_seconds_max"],
                    "retry_allowed": False,
                }, indent=2, sort_keys=True) + "\n")
            raise RuntimeError("managed external wall timeout") from error
    finally:
        envelope_path.unlink(missing_ok=True)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = launch(
        args.contract, args.request, args.output,
        "testing.first_medium_saved_output_qualification",
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

