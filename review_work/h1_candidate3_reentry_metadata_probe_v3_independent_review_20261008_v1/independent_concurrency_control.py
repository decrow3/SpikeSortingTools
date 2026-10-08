#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "h1_candidate3_reentry_metadata_probe_v3",
    ROOT / "testing/h1_candidate3_reentry_metadata_probe_v3.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

with tempfile.TemporaryDirectory(prefix="h1_probe_v3_concurrency_") as td:
    root = Path(td)
    receipt = root / "snapshot_1" / "RECEIPT.json"
    contract = {
        "snapshots": ["snapshot_1", "snapshot_2"],
        "receipt_paths": {
            "snapshot_1": str(receipt),
            "snapshot_2": str(root / "snapshot_2" / "RECEIPT.json"),
        },
        "live_target": "/not-accessed-by-mocked-worker",
        "timeout_seconds": 5.0,
        "post_timeout_reap_seconds": 1.0,
        "outcome_interpretation": {
            MODULE.PASS: "pass",
            MODULE.BLOCKED_TIMEOUT: "timeout",
            MODULE.BLOCKED_OPERATION: "operation",
        },
    }
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    worker_calls = 0

    def fake_load(*_args):
        return contract, "contract-sha"

    def fake_sha(*_args):
        return "runner-sha"

    def fake_worker(*_args, **_kwargs):
        global worker_calls
        with lock:
            worker_calls += 1
        barrier.wait(timeout=2)
        return {
            "timed_out": False,
            "worker_pid": 1,
            "worker_exit_code": 0,
            "worker_stdout": json.dumps({"status": MODULE.PASS, "result": {}}),
            "worker_stderr": "",
            "worker_reaped": True,
            "elapsed_seconds": 0.01,
        }

    MODULE.load_bound_contract = fake_load
    MODULE.sha256_path = fake_sha
    MODULE.run_worker_bounded = fake_worker
    results = []

    def invoke():
        try:
            results.append(MODULE.live_probe(Path("contract.json"), "contract-sha", "snapshot_1"))
        except Exception as exc:
            results.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=invoke) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(json.dumps({
        "mocked_worker_calls": worker_calls,
        "invocation_results": sorted(results, key=str),
        "final_receipt_exists": receipt.exists(),
        "live_target_accessed": False,
    }, indent=2, sort_keys=True))
