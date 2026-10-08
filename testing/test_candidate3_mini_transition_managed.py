import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from testing.candidate3_mini_transition_managed import (
    finalize,
    run_child,
    sha256,
    tree_file_bytes,
    verify_preflight,
)


def contract_fixture(tmp_path: Path) -> tuple[Path, dict]:
    packet = tmp_path / "packet"
    packet.mkdir()
    runner = packet / "runner.py"
    runner.write_text("pass\n")
    input_path = tmp_path / "input.npy"
    input_path.write_bytes(b"input")
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    contract = {
        "schema": "candidate3-mini-transition-managed-v2",
        "packet_hashes": {"runner.py": sha256(runner)},
        "input_hashes": {"transition": {"path": str(input_path), "sha256": sha256(input_path)}},
        "dartsort_repo": "/home/huklab/Documents/DARTsort",
        "dartsort_commit": subprocess.check_output(
            ["git", "-C", "/home/huklab/Documents/DARTsort", "rev-parse", "HEAD"], text=True
        ).strip(),
        "attempt_parent": str(tmp_path),
        "attempt_root": str(tmp_path / "attempt"),
        "preflight_failure_receipt": str(tmp_path / "failure.json"),
        "preflight_manager_receipt": str(tmp_path / "manager.json"),
        "ordinary_root": str(ordinary),
        "minimum_free_bytes": 1,
    }
    path = packet / "RUN_CONTRACT.json"
    path.write_text(json.dumps(contract))
    return path, contract


def test_preflight_verifies_hashes_freshness_and_commit(tmp_path):
    path, contract = contract_fixture(tmp_path)
    result = verify_preflight(path, contract)
    assert result["dartsort_commit"] == contract["dartsort_commit"]
    assert result["free_bytes"] > 0


def test_preflight_rejects_symlink_attempt(tmp_path):
    path, contract = contract_fixture(tmp_path)
    (tmp_path / "other").mkdir()
    (tmp_path / "attempt").symlink_to(tmp_path / "other", target_is_directory=True)
    with pytest.raises(RuntimeError, match="freshness failure"):
        verify_preflight(path, contract)


def test_tree_bytes_excludes_symlink_target(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "owned").write_bytes(b"12345")
    outside = tmp_path / "outside"
    outside.write_bytes(b"x" * 100)
    (root / "link").symlink_to(outside)
    assert tree_file_bytes(root) == 5


def test_run_child_enforces_storage_cap(tmp_path):
    root = tmp_path / "attempt"
    root.mkdir()
    code = "from pathlib import Path; import time; Path(r'%s').write_bytes(b'x'*1000); time.sleep(20)" % (root / "large")
    returncode, reason = run_child(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=os.environ.copy(),
        stdout=tmp_path / "stdout",
        stderr=tmp_path / "stderr",
        root=root,
        baseline_bytes=0,
        cap_bytes=10,
    )
    assert returncode != 0
    assert reason and reason.startswith("storage_cap_exceeded:")


def test_finalize_writes_separate_manager_receipt(tmp_path, monkeypatch):
    path, contract = contract_fixture(tmp_path)
    digest = sha256(path)
    monkeypatch.setenv("SERVICE_RESULT", "exit-code")
    monkeypatch.setenv("EXIT_CODE", "exited")
    monkeypatch.setenv("EXIT_STATUS", "7")
    assert finalize(path, digest) == 0
    saved = json.loads(Path(contract["preflight_manager_receipt"]).read_text())
    assert saved["service_result"] == "exit-code"
    assert saved["exit_status"] == "7"
