"""Independent verifier for the actual H5 CPU waveform executable closure."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time


EXPECTED_CLOSURE_SHA256 = "f5db3034ab11ac65e17fda77e455300be93bfd82ef30f851da60ae2777269f8e"
IMPULSE_SHA256 = "984639dee1d75eb82c170c1a96ba3e96dc196eaa24383b598266a804729e52b5"
ADAPTER_SHA256 = "e9abe2a9a1c22db118a8c652e2a6a083db2fc9f84e8c77a71cec992447bcbe57"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_counter(value: dict) -> Counter:
    return Counter(
        (row["role"], row["basename"], int(row["bytes"]), row["sha256"])
        for row in value["members"]
    )


def exercise(impulse_path: Path, adapter_path: Path) -> dict:
    import numpy as np
    import torch
    from waveform_preprocessing import preprocess_padded_record

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("CPU-only verifier requires CUDA_VISIBLE_DEVICES empty or -1")
    if sha256(impulse_path) != IMPULSE_SHA256 or sha256(adapter_path) != ADAPTER_SHA256:
        raise RuntimeError("accepted numerical-probe dependency changed")
    impulse = np.load(impulse_path, allow_pickle=False)
    if impulse.shape != (30122,) or impulse.dtype != np.float32:
        raise RuntimeError("accepted impulse shape/dtype changed")
    hp_filter = torch.from_numpy(impulse.copy())
    values = (torch.arange(384 * 1145, dtype=torch.float32).reshape(384, 1145) % 997) - 498
    observed = preprocess_padded_record(values, hp_filter)
    if observed.shape != (384, 1145) or observed.dtype != torch.float32 or observed.device.type != "cpu":
        raise RuntimeError("numerical probe returned the wrong shape/dtype/device")
    probe_sha = hashlib.sha256(observed.numpy().tobytes()).hexdigest()
    return {
        "torch_version": torch.__version__,
        "torch_config_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "num_threads": int(torch.get_num_threads()),
        "num_interop_threads": int(torch.get_num_interop_threads()),
        "mkl_available": bool(torch.backends.mkl.is_available()),
        "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "openmp_available": bool(torch.backends.openmp.is_available()),
        "probe_output_sha256": probe_sha,
    }


def mapped_elf_members() -> list[dict]:
    paths = {Path(sys.executable).resolve()}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[-1].startswith("/"):
            continue
        path = Path(fields[-1].removesuffix(" (deleted)")).resolve()
        if path.is_file():
            paths.add(path)
    members = []
    for path in sorted(paths, key=str):
        with path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                continue
        members.append({
            "role": "python_executable" if path == Path(sys.executable).resolve() else "mapped_elf",
            "basename": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "observed_path": str(path),
        })
    return members


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--impulse", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    if sha256(args.expected) != EXPECTED_CLOSURE_SHA256:
        raise RuntimeError("expected closure file hash changed")
    expected = json.loads(args.expected.read_text())
    backend = exercise(args.impulse, args.adapter)
    members = mapped_elf_members()
    identities = [
        {key: row[key] for key in ("role", "basename", "bytes", "sha256")}
        for row in members
    ]
    identities.sort(key=lambda row: (row["role"], row["basename"], row["bytes"], row["sha256"]))
    aggregate = hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    observed = {
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "backend": backend,
        "member_count": len(members),
        "member_bytes": sum(row["bytes"] for row in members),
        "aggregate_sha256": aggregate,
        "members": members,
    }
    differences = [
        key for key in (
            "platform_system", "platform_machine", "python_version", "python_implementation",
            "backend", "member_count", "member_bytes", "aggregate_sha256",
        ) if observed[key] != expected[key]
    ]
    if identity_counter(observed) != identity_counter(expected):
        differences.append("member_identity_multiset")
    if differences:
        raise RuntimeError(f"actual H5 runtime closure mismatch: {sorted(set(differences))}")
    receipt = {
        "schema": "independent-h5-waveform-runtime-verification-receipt-v1",
        "status": "PASS",
        "scope": "actual H5 deployment exact relocatable-path identity; no general cross-host portability claim",
        "verifier_path": str(Path(__file__).resolve()),
        "verifier_sha256": sha256(Path(__file__).resolve()),
        "command_argv": sys.argv,
        "cwd": str(Path.cwd()),
        "environment": {key: os.environ.get(key) for key in (
            "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS",
        )},
        "host": platform.node(),
        "uname": list(platform.uname()),
        "python_executable": sys.executable,
        "python_executable_resolved": str(Path(sys.executable).resolve()),
        "expected_closure_path": str(args.expected.resolve()),
        "expected_closure_sha256": sha256(args.expected),
        "impulse_path": str(args.impulse.resolve()),
        "impulse_sha256": sha256(args.impulse),
        "adapter_path": str(args.adapter.resolve()),
        "adapter_sha256": sha256(args.adapter),
        "platform_system": observed["platform_system"],
        "platform_machine": observed["platform_machine"],
        "python_version": observed["python_version"],
        "python_implementation": observed["python_implementation"],
        "backend": backend,
        "member_count": observed["member_count"],
        "member_bytes_hashed": observed["member_bytes"],
        "aggregate_sha256": aggregate,
        "members": members,
        "differences": [],
        "wall_seconds": time.monotonic() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "vm_hwm_kib": int(next(
            line.split()[1] for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith("VmHWM:")
        )),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
