import json
import sys

import pytest

from testing.managed_job import run_managed_command


def test_managed_job_persists_command_and_success(tmp_path):
    receipt = tmp_path / "job.json"
    returncode = run_managed_command(
        [sys.executable, "-c", "print('dummy survived')"],
        receipt_path=receipt,
        cwd=tmp_path,
    )
    saved = json.loads(receipt.read_text())
    assert returncode == 0
    assert saved["state"] == "complete"
    assert saved["returncode"] == 0
    assert saved["command"][:2] == [sys.executable, "-c"]


def test_managed_job_persists_failure_and_refuses_receipt_reuse(tmp_path):
    receipt = tmp_path / "job.json"
    assert run_managed_command(
        [sys.executable, "-c", "raise SystemExit(7)"],
        receipt_path=receipt,
        cwd=tmp_path,
    ) == 7
    assert json.loads(receipt.read_text())["state"] == "failed"
    with pytest.raises(RuntimeError, match="receipt already exists"):
        run_managed_command(
            [sys.executable, "-c", "pass"], receipt_path=receipt, cwd=tmp_path
        )
