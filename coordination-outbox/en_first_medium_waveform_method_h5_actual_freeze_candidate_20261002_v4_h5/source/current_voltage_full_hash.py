"""One-shot current full-file SHA-256 verification of the immutable recording."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import stat as stat_module
import sys
import time


BINARY = Path("/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0/recording/traces_cached_seg0.raw")
EXPECTED_BYTES = 241_309_358_592
EXPECTED_SHA256 = "152f8d43a9360a95be1e9fc6bd66bce6e0fe0c091836787b37d50941f8682482"
OUTPUT = Path("/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/coordination-outbox/en_first_medium_voltage_current_full_hash_20261002_v1_h5")
BUFFER_BYTES = 16 * 1024 * 1024
PROGRESS_BYTES = 16 * 1024 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stat_tuple(value: os.stat_result) -> dict:
    return {
        "st_dev": int(value.st_dev),
        "st_ino": int(value.st_ino),
        "st_mode": int(value.st_mode),
        "st_size": int(value.st_size),
        "st_mtime_ns": int(value.st_mtime_ns),
        "st_ctime_ns": int(value.st_ctime_ns),
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def proc_io() -> dict:
    result = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        key, value = line.split(":", 1)
        result[key] = int(value.strip())
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"fresh one-shot output required: {OUTPUT}")
    before = os.lstat(BINARY)
    if not stat_module.S_ISREG(before.st_mode) or before.st_size != EXPECTED_BYTES:
        raise RuntimeError("exact non-symlink regular recording path/size required")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started_wall_epoch_ns = time.time_ns()
    started_monotonic = time.monotonic()
    started_cpu = time.process_time()
    source = Path(__file__).resolve()
    started = {
        "schema": "current-voltage-full-hash-started-v1",
        "status": "STARTED_ONE_SHOT_NO_RETRY",
        "host": platform.node(),
        "python": sys.executable,
        "source_path": str(source),
        "source_sha256": sha256(source),
        "binary_path": str(BINARY),
        "expected_bytes": EXPECTED_BYTES,
        "expected_sha256": EXPECTED_SHA256,
        "buffer_bytes": BUFFER_BYTES,
        "initial_lstat": stat_tuple(before),
        "started_epoch_ns": started_wall_epoch_ns,
    }
    write_json(OUTPUT / "STARTED.json", started)
    io_before = proc_io()
    digest = hashlib.sha256()
    total = 0
    calls = 0
    next_progress = PROGRESS_BYTES
    fd = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(BINARY, flags)
        opened = os.fstat(fd)
        if stat_tuple(opened) != stat_tuple(before):
            raise RuntimeError("recording changed between lstat and no-follow open")
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_SEQUENTIAL"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        while True:
            block = os.read(fd, BUFFER_BYTES)
            calls += 1
            if not block:
                break
            digest.update(block)
            total += len(block)
            if total >= next_progress:
                with (OUTPUT / "PROGRESS.jsonl").open("a") as stream:
                    stream.write(json.dumps({
                        "bytes_read": total,
                        "fraction": total / EXPECTED_BYTES,
                        "wall_seconds": time.monotonic() - started_monotonic,
                    }, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                next_progress += PROGRESS_BYTES
        final_fd = os.fstat(fd)
    except Exception as error:
        write_json(OUTPUT / "FAILURE.json", {
            "schema": "current-voltage-full-hash-failure-v1",
            "status": "FAILED_NO_RETRY",
            "error_type": type(error).__name__,
            "error": str(error),
            "bytes_read_before_failure": total,
        })
        raise
    finally:
        if fd is not None:
            os.close(fd)
    after = os.lstat(BINARY)
    actual = digest.hexdigest()
    wall = time.monotonic() - started_monotonic
    cpu = time.process_time() - started_cpu
    if stat_tuple(before) != stat_tuple(final_fd) or stat_tuple(before) != stat_tuple(after):
        write_json(OUTPUT / "FAILURE.json", {
            "schema": "current-voltage-full-hash-failure-v1",
            "status": "FAILED_NO_RETRY",
            "error_type": "RuntimeError",
            "error": "recording stat identity changed during full read",
            "bytes_read_before_failure": total,
        })
        raise RuntimeError("recording stat identity changed during full read")
    if total != EXPECTED_BYTES or actual != EXPECTED_SHA256:
        write_json(OUTPUT / "FAILURE.json", {
            "schema": "current-voltage-full-hash-failure-v1",
            "status": "FAILED_NO_RETRY",
            "error_type": "ContentMismatch",
            "expected_bytes": EXPECTED_BYTES,
            "actual_bytes": total,
            "expected_sha256": EXPECTED_SHA256,
            "actual_sha256": actual,
        })
        raise RuntimeError("current recording content mismatch; stop without retry")
    io_after = proc_io()
    receipt = {
        "schema": "current-voltage-full-hash-receipt-v1",
        "status": "PASS_EXACT_MATCH",
        "host": platform.node(),
        "binary_path": str(BINARY),
        "binary_is_symlink": False,
        "expected_bytes": EXPECTED_BYTES,
        "bytes_read": total,
        "expected_sha256": EXPECTED_SHA256,
        "actual_sha256": actual,
        "buffer_bytes": BUFFER_BYTES,
        "read_calls_including_final_eof": calls,
        "initial_lstat": stat_tuple(before),
        "final_fstat": stat_tuple(final_fd),
        "final_lstat": stat_tuple(after),
        "stat_unchanged": True,
        "started_epoch_ns": started_wall_epoch_ns,
        "finished_epoch_ns": time.time_ns(),
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "application_throughput_bytes_per_second": total / wall,
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "proc_io_before": io_before,
        "proc_io_after": io_after,
        "proc_io_delta": {key: io_after[key] - io_before.get(key, 0) for key in io_after},
        "source_sha256": sha256(source),
        "no_retry": True,
    }
    write_json(OUTPUT / "CURRENT_VOLTAGE_FULL_HASH_RECEIPT.json", receipt)
    members = sorted(path for path in OUTPUT.iterdir() if path.is_file())
    manifest = "".join(f"{sha256(path)}  {path.name}\n" for path in members)
    (OUTPUT / "MANIFEST.sha256").write_text(manifest)
    complete = {
        "schema": "immutable-evidence-packet-complete-v1",
        "packet": OUTPUT.name,
        "producer": "H5",
        "completed_at_local_date": "2026-10-02",
        "manifest": "MANIFEST.sha256",
        "manifest_sha256": sha256(OUTPUT / "MANIFEST.sha256"),
        "status": "PASS_EXACT_MATCH",
        "bytes_read": total,
        "actual_sha256": actual,
        "retry_count": 0,
    }
    write_json(OUTPUT / "COMPLETE.json", complete)


if __name__ == "__main__":
    main()
