import hashlib
import json
from pathlib import Path

import pytest

from testing.em2b_prepare_arms_manifest import attest_runner_sort, parse_stage


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "sort").mkdir(parents=True)
    config = run / "config.json"
    sorting = run / "sort/dartsort_sorting.npz"
    config.write_text("{}\n")
    sorting.write_bytes(b"sorting")
    (run / "receipt.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "exit_status": 0,
                "config_sha256": digest(config),
                "stages": {
                    "sort": {
                        "artifacts": [
                            {
                                "resolved_path": str(sorting.resolve()),
                                "sha256": digest(sorting),
                            }
                        ]
                    }
                },
            }
        )
    )
    return run


def test_attest_runner_sort_requires_final_receipt_and_exact_sort_hash(tmp_path):
    run = complete_run(tmp_path)
    result = attest_runner_sort(run)
    assert result["sha256"] == digest(run / "sort/dartsort_sorting.npz")
    assert result["run_receipt_sha256"] == digest(run / "receipt.json")


def test_attest_runner_sort_rejects_changed_sort(tmp_path):
    run = complete_run(tmp_path)
    (run / "sort/dartsort_sorting.npz").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="does not attest"):
        attest_runner_sort(run)


def test_attest_runner_sort_rejects_nonzero_final_status(tmp_path):
    run = complete_run(tmp_path)
    value = json.loads((run / "receipt.json").read_text())
    value["exit_status"] = 1
    (run / "receipt.json").write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="did not complete"):
        attest_runner_sort(run)


def test_parse_stage():
    assert parse_stage("unrounded_kriging=/tmp/run") == (
        "unrounded_kriging",
        Path("/tmp/run"),
    )
    with pytest.raises(Exception, match="NAME=RUN_DIRECTORY"):
        parse_stage("missing-separator")
