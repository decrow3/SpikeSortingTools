"""Capture and fail-closed verify the exact executable CPU waveform runtime."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any


SCHEMA = "waveform-executable-runtime-closure-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _exercise_cpu_path(impulse_path: Path) -> tuple[Any, Any]:
    import numpy as np
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty or -1 for the CPU-only runtime")
    impulse = np.load(impulse_path, allow_pickle=False)
    if impulse.shape != (30122,) or impulse.dtype != np.float32:
        raise RuntimeError("frozen high-pass impulse shape or dtype changed")
    hp_filter = torch.from_numpy(impulse.copy())
    x = (torch.arange(384 * 1145, dtype=torch.float32).reshape(384, 1145) % 997) - 498
    from waveform_preprocessing import preprocess_padded_record

    result = preprocess_padded_record(x, hp_filter)
    if result.device.type != "cpu" or result.dtype != torch.float32 or result.shape != (384, 1145):
        raise RuntimeError("CPU waveform probe returned an unexpected tensor")
    # Materialize reductions so median, forward/inverse FFT, and fftshift have completed.
    probe_digest = hashlib.sha256(result.numpy().tobytes()).hexdigest()
    backend = {
        "torch_version": torch.__version__,
        "torch_config_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "num_threads": int(torch.get_num_threads()),
        "num_interop_threads": int(torch.get_num_interop_threads()),
        "mkl_available": bool(torch.backends.mkl.is_available()),
        "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "openmp_available": bool(torch.backends.openmp.is_available()),
        "probe_output_sha256": probe_digest,
    }
    return torch, backend


def _mapped_elf_paths() -> list[Path]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        raise RuntimeError("Linux /proc/self/maps is required for runtime closure verification")
    paths = {Path(sys.executable).resolve()}
    for line in maps.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[-1].startswith("/"):
            continue
        raw = fields[-1].removesuffix(" (deleted)")
        path = Path(raw).resolve()
        if path.is_file():
            paths.add(path)
    elf = []
    for path in sorted(paths, key=str):
        with path.open("rb") as stream:
            if stream.read(4) == b"\x7fELF":
                elf.append(path)
    return elf


def capture(impulse_path: Path) -> dict[str, Any]:
    _, backend = _exercise_cpu_path(impulse_path)
    paths = _mapped_elf_paths()
    members = []
    for path in paths:
        members.append({
            "basename": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "observed_path": str(path),
            "role": "python_executable" if path == Path(sys.executable).resolve() else "mapped_elf",
        })
    identities = [
        {key: member[key] for key in ("role", "basename", "bytes", "sha256")}
        for member in members
    ]
    identities.sort(key=lambda row: (row["role"], row["basename"], row["bytes"], row["sha256"]))
    aggregate = hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": SCHEMA,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "backend": backend,
        "member_count": len(members),
        "member_bytes": sum(member["bytes"] for member in members),
        "aggregate_sha256": aggregate,
        "members": members,
        "portable_identity_rule": "Paths are diagnostics only. A path may differ if and only if the exact multiset of (role, basename, bytes, sha256), platform fields, backend fields, member count, member bytes, and aggregate all match.",
    }


def _identity_counter(closure: dict[str, Any]) -> Counter:
    return Counter(
        (member["role"], member["basename"], int(member["bytes"]), member["sha256"])
        for member in closure["members"]
    )


def compare_closures(expected: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    """Return deterministic mismatch fields while intentionally ignoring paths."""
    scalar_keys = (
        "platform_system", "platform_machine", "python_version", "python_implementation",
        "backend", "member_count", "member_bytes", "aggregate_sha256",
    )
    differences = [key for key in scalar_keys if observed.get(key) != expected.get(key)]
    if _identity_counter(observed) != _identity_counter(expected):
        differences.append("member_identity_multiset")
    return sorted(set(differences))


def verify(expected_path: Path, impulse_path: Path) -> dict[str, Any]:
    expected = json.loads(expected_path.read_text())
    if expected.get("schema") != SCHEMA:
        raise RuntimeError("runtime closure schema mismatch")
    observed = capture(impulse_path)
    differences = compare_closures(expected, observed)
    if differences:
        raise RuntimeError(f"runtime closure mismatch: {sorted(set(differences))}")
    return {
        "schema": "waveform-executable-runtime-verification-receipt-v1",
        "status": "PASS",
        "expected_closure_sha256": _sha256(expected_path),
        "aggregate_sha256": observed["aggregate_sha256"],
        "member_count": observed["member_count"],
        "member_bytes_hashed": observed["member_bytes"],
        "portable_rule": "verified by exact identity multiset; observed paths may differ",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impulse", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit", action="store_true")
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()
    value = capture(args.impulse) if args.emit else verify(args.verify, args.impulse)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
