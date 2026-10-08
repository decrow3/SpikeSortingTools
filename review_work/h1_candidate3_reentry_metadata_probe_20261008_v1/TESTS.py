import json
from pathlib import Path
import subprocess
import sys

import pytest

from testing import h1_candidate3_reentry_metadata_probe as probe


EXPECTED = {
    "COMPLETE.json": 203,
    "MANIFEST.json": 541,
    "OBSERVATIONS.json": 2855,
    "PROVENANCE.json": 887,
    "RECOMMENDATION.md": 2956,
}


def make_fixture(root: Path) -> None:
    root.mkdir()
    for name, size in EXPECTED.items():
        (root / name).write_bytes(b"x" * size)


def test_metadata_probe_accepts_exact_fixture_without_reading_members(tmp_path, monkeypatch):
    target = tmp_path / "packet"
    make_fixture(target)

    def forbidden_open(*args, **kwargs):
        raise AssertionError("member content open is forbidden")

    monkeypatch.setattr(Path, "open", forbidden_open)
    result = probe.probe_directory_metadata(target, EXPECTED)
    assert sorted(result["observed_members"]) == sorted(EXPECTED)


def test_metadata_probe_rejects_extra_member(tmp_path):
    target = tmp_path / "packet"
    make_fixture(target)
    (target / "EXTRA").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="membership mismatch"):
        probe.probe_directory_metadata(target, EXPECTED)


def test_metadata_probe_rejects_wrong_size(tmp_path):
    target = tmp_path / "packet"
    make_fixture(target)
    (target / "MANIFEST.json").write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="size mismatch"):
        probe.probe_directory_metadata(target, EXPECTED)


def test_metadata_probe_rejects_symlink(tmp_path):
    target = tmp_path / "packet"
    make_fixture(target)
    (target / "MANIFEST.json").unlink()
    (target / "MANIFEST.json").symlink_to(target / "COMPLETE.json")
    with pytest.raises(RuntimeError, match="not a regular file"):
        probe.probe_directory_metadata(target, EXPECTED)


def test_bounded_worker_passes_and_is_reaped():
    result = probe.run_worker_bounded(
        [sys.executable, "-c", "print('ok')"], timeout_seconds=1.0
    )
    assert result["timed_out"] is False
    assert result["worker_reaped"] is True
    assert result["worker_exit_code"] == 0


def test_bounded_worker_times_out_and_is_reaped():
    result = probe.run_worker_bounded(
        [sys.executable, "-c", "import time; time.sleep(5)"], timeout_seconds=0.05
    )
    assert result["timed_out"] is True
    assert result["worker_reaped"] is True
    assert result["worker_exit_code"] is not None


def test_atomic_receipt_refuses_no_data_and_round_trips(tmp_path):
    receipt_path = tmp_path / "receipt" / "RECEIPT.json"
    payload = {"status": probe.PASS, "value": 1}
    probe.write_receipt_atomic(receipt_path, payload)
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == payload


def test_source_has_no_member_content_read_in_probe_function():
    source = Path(probe.__file__).read_text(encoding="utf-8")
    function = source.split("def probe_directory_metadata", 1)[1].split(
        "def write_receipt_atomic", 1
    )[0]
    assert ".read(" not in function
    assert ".read_bytes(" not in function
    assert "os.stat(" in function
    assert "os.listdir(" in function
