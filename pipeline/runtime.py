"""Frozen runtime contract for the conservative production pipeline."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any


PRODUCTION_PYTHON = (3, 12, 4)
PRODUCTION_PACKAGES = {
    "dredge-ephys": "0.3.0",
    "h5py": "3.13.0",
    "kilosort": "4.0.27",
    "matplotlib": "3.9.1",
    "neo": "0.14.0",
    "numba": "0.61.0",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "probeinterface": "0.2.25",
    "scikit-learn": "1.6.1",
    "scipy": "1.15.2",
    "spikeinterface": "0.102.1",
    "torch": "2.6.0+cu124",
}
PRODUCTION_UV_PROJECT = "environments/rescue-production"
PRODUCTION_UV_SETUP = (
    "uv sync --project environments/rescue-production --frozen --no-group test",
    "source environments/rescue-production/.venv/bin/activate",
)
PRODUCTION_LOCKFILE = (
    Path(__file__).resolve().parents[1]
    / "environments"
    / "rescue-production"
    / "uv.lock"
)


def production_lock_sha256() -> str | None:
    """Return the committed production lock identity, or ``None`` if absent."""
    if not PRODUCTION_LOCKFILE.is_file():
        return None
    digest = hashlib.sha256()
    with PRODUCTION_LOCKFILE.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def production_environment_contract() -> dict[str, Any]:
    """Return the expected environment identity used in cache requests."""
    return {
        "uv_project": PRODUCTION_UV_PROJECT,
        "python": ".".join(map(str, PRODUCTION_PYTHON)),
        "packages": dict(PRODUCTION_PACKAGES),
        "uv_lock_sha256": production_lock_sha256(),
    }


def production_environment_receipt(*, check_cuda: bool = False) -> dict[str, Any]:
    """Return installed versions without importing heavy runtime packages."""
    installed: dict[str, str | None] = {}
    for distribution in PRODUCTION_PACKAGES:
        try:
            installed[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            installed[distribution] = None
    receipt: dict[str, Any] = {
        "uv_project": PRODUCTION_UV_PROJECT,
        "canonical_setup": list(PRODUCTION_UV_SETUP),
        "python_required": ".".join(map(str, PRODUCTION_PYTHON)),
        "python_installed": platform.python_version(),
        "packages_required": dict(PRODUCTION_PACKAGES),
        "packages_installed": installed,
        "uv_lockfile": str(PRODUCTION_LOCKFILE),
        "uv_lock_sha256": production_lock_sha256(),
    }
    if check_cuda:
        try:
            import torch

            available = bool(torch.cuda.is_available())
            receipt["cuda_available"] = available
            receipt["cuda_device"] = torch.cuda.get_device_name(0) if available else None
        except Exception as error:
            receipt["cuda_available"] = False
            receipt["cuda_device"] = None
            receipt["cuda_check_error"] = f"{type(error).__name__}: {error}"
    return receipt


def validate_production_environment(*, require_cuda: bool = False) -> dict[str, Any]:
    """Refuse data-changing work outside the exact locked production runtime."""
    receipt = production_environment_receipt(check_cuda=require_cuda)
    problems = []
    if sys.version_info[:3] != PRODUCTION_PYTHON:
        problems.append(
            f"Python {receipt['python_installed']} != {receipt['python_required']}"
        )
    installed = receipt["packages_installed"]
    if receipt["uv_lock_sha256"] is None:
        problems.append(f"production lockfile is missing at {PRODUCTION_LOCKFILE}")
    for distribution, expected in PRODUCTION_PACKAGES.items():
        observed = installed[distribution]
        if observed != expected:
            problems.append(f"{distribution} {observed or 'missing'} != {expected}")
    if require_cuda and not receipt.get("cuda_available"):
        problems.append("CUDA is not available to PyTorch")
    if problems:
        details = "; ".join(problems)
        raise RuntimeError(
            "Production environment validation failed: "
            f"{details}. From the repository root, run `{PRODUCTION_UV_SETUP[0]}`, "
            f"then `{PRODUCTION_UV_SETUP[1]}`."
        )
    return receipt
