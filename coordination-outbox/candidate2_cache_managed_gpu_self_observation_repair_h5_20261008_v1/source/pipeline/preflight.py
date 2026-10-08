"""Host-level preflight checks for running the pipeline on a new machine.

``pipeline.runtime`` validates the *software* contract: interpreter, pinned
package versions, lockfile identity, CUDA. That contract is portable, so it
says nothing about the parts that actually differ between machines: whether the
recording server is mounted, whether the NVMe scratch directory exists and has
room, whether the output directory is writable, and whether the requested
SpikeGLX stream is discoverable in the source folder.

Those are the checks here. They are deliberately cheap and read-only: nothing in
this module writes to the recording tree or allocates scratch space, apart from
a single temporary probe file used to prove a directory is writable.

Use :func:`preflight_report` for a machine-readable result and
:func:`format_preflight` for an operator-readable one.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

# A staged AP stream plus its preprocessed copy and sorter working files needs
# considerably more room than the source. This is a floor, not an estimate.
DEFAULT_MIN_FREE_GB = 250.0

# Multiplier applied to the discovered source size when estimating how much
# scratch a run will need. Staging writes one copy, preprocessing writes
# another, and Kilosort keeps its own temporary files alongside them.
SCRATCH_SIZE_FACTOR = 3.0


def _describe_mount(path: Path) -> dict[str, Any]:
    """Report the mount point backing ``path``, and whether it is a real mount."""
    resolved = path.resolve()
    probe = resolved
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    mount = probe
    while not os.path.ismount(mount) and mount != mount.parent:
        mount = mount.parent
    return {
        "nearest_existing": str(probe),
        "mount_point": str(mount),
        "is_mount_point": os.path.ismount(mount),
    }


def _disk_free_gb(path: Path) -> float | None:
    probe = path.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free / 1024**3
    except OSError:
        return None


def _is_writable(path: Path) -> tuple[bool, str | None]:
    """Prove writability by creating and removing a probe file."""
    target = path if path.is_dir() else path.parent
    if not target.is_dir():
        return False, f"{target} is not an existing directory"
    probe = target / f".preflight_write_probe_{os.getpid()}"
    try:
        probe.touch()
        probe.unlink()
        return True, None
    except OSError as error:
        return False, f"{type(error).__name__}: {error}"


def discover_spikeglx_stream(data_dir: Path, stream_id: str) -> dict[str, Any]:
    """Locate the binary/meta pair for ``stream_id`` beneath ``data_dir``.

    ``stream_id`` follows the SpikeInterface convention, e.g. ``"imec0.ap"``.
    """
    data_dir = Path(data_dir)
    result: dict[str, Any] = {
        "stream_id": stream_id,
        "data_dir": str(data_dir),
        "data_dir_exists": data_dir.is_dir(),
        "binary": None,
        "meta": None,
        "size_bytes": None,
        "ok": False,
    }
    if not data_dir.is_dir():
        result["error"] = f"source folder does not exist: {data_dir}"
        return result

    binaries = sorted(data_dir.glob(f"**/*.{stream_id}.bin"))
    metas = sorted(data_dir.glob(f"**/*.{stream_id}.meta"))
    if not binaries:
        available = sorted(
            {p.name.split(".", 1)[1].rsplit(".", 1)[0] for p in data_dir.glob("**/*.bin") if "." in p.name}
        )
        result["error"] = f"no *.{stream_id}.bin under {data_dir}"
        result["streams_present"] = available
        return result
    if len(binaries) > 1:
        result["error"] = f"{len(binaries)} candidate binaries for {stream_id}; expected one"
        result["candidates"] = [str(p) for p in binaries]
        return result

    binary = binaries[0]
    result["binary"] = str(binary)
    result["size_bytes"] = binary.stat().st_size
    result["size_gb"] = round(binary.stat().st_size / 1024**3, 2)
    if not metas:
        result["error"] = f"found {binary.name} but no matching .meta sidecar"
        return result
    result["meta"] = str(metas[0])
    result["ok"] = True
    return result


def preflight_report(
    *,
    data_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    local_work_dir: Path | str | None = None,
    stream_id: str | None = None,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
) -> dict[str, Any]:
    """Check the host-level preconditions for a run. Never raises for a failed check."""
    report: dict[str, Any] = {"checks": [], "ok": True}

    def add(name: str, ok: bool, detail: str, **extra: Any) -> None:
        report["checks"].append({"name": name, "ok": ok, "detail": detail, **extra})
        if not ok:
            report["ok"] = False

    source = None
    if data_dir is not None:
        data_dir = Path(data_dir)
        mount = _describe_mount(data_dir)
        if data_dir.is_dir():
            add("source_mounted", True, f"{data_dir} present on {mount['mount_point']}", **mount)
        else:
            add(
                "source_mounted",
                False,
                f"{data_dir} is not present; nearest existing path is "
                f"{mount['nearest_existing']}. If this is a server path, the mount is missing.",
                **mount,
            )
        if stream_id is not None:
            source = discover_spikeglx_stream(data_dir, stream_id)
            add(
                "source_stream",
                source["ok"],
                (
                    f"{stream_id}: {source['binary']} ({source.get('size_gb')} GB)"
                    if source["ok"]
                    else source.get("error", "stream not found")
                ),
                **{k: v for k, v in source.items() if k not in {"ok", "detail"}},
            )

    if output_dir is not None:
        output_dir = Path(output_dir)
        writable, error = _is_writable(output_dir)
        free = _disk_free_gb(output_dir)
        if not output_dir.exists():
            parent_writable, parent_error = _is_writable(output_dir.parent)
            add(
                "output_writable",
                parent_writable,
                (
                    f"{output_dir} does not exist yet but its parent is writable"
                    if parent_writable
                    else f"cannot create {output_dir}: {parent_error}"
                ),
                free_gb=round(free, 1) if free is not None else None,
            )
        else:
            add(
                "output_writable",
                writable,
                f"{output_dir} writable" if writable else f"{output_dir} not writable: {error}",
                free_gb=round(free, 1) if free is not None else None,
            )

    if local_work_dir is not None:
        local_work_dir = Path(local_work_dir)
        writable, error = _is_writable(
            local_work_dir if local_work_dir.exists() else local_work_dir.parent
        )
        free = _disk_free_gb(local_work_dir)
        add(
            "scratch_writable",
            writable,
            f"{local_work_dir} writable" if writable else f"{local_work_dir} not writable: {error}",
        )
        needed = min_free_gb
        if source and source.get("ok") and source.get("size_bytes"):
            needed = max(needed, source["size_bytes"] / 1024**3 * SCRATCH_SIZE_FACTOR)
        if free is None:
            add("scratch_free_space", False, f"could not determine free space at {local_work_dir}")
        else:
            add(
                "scratch_free_space",
                free >= needed,
                f"{free:.0f} GB free, {needed:.0f} GB required "
                f"({SCRATCH_SIZE_FACTOR:g}x source, floor {min_free_gb:g} GB)",
                free_gb=round(free, 1),
                required_gb=round(needed, 1),
            )

    return report


def format_preflight(report: dict[str, Any]) -> str:
    """Render :func:`preflight_report` output for an operator."""
    lines = []
    for check in report["checks"]:
        mark = "PASS" if check["ok"] else "FAIL"
        lines.append(f"  [{mark}] {check['name']}: {check['detail']}")
    lines.append("")
    lines.append("Preflight OK" if report["ok"] else "Preflight FAILED")
    return "\n".join(lines)
