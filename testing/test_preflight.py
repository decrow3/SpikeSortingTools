"""Host-level preflight checks."""

import os

import pytest

from pipeline.preflight import (
    DEFAULT_MIN_FREE_GB,
    discover_spikeglx_stream,
    format_preflight,
    preflight_report,
)


def _make_stream(root, run="Luke0804_V2V1_g0", stream="imec0.ap", size=2048):
    folder = root / run / f"{run}_imec0"
    folder.mkdir(parents=True)
    binary = folder / f"{run}_t0.{stream}.bin"
    binary.write_bytes(b"\0" * size)
    (folder / f"{run}_t0.{stream}.meta").write_text("fileSizeBytes=%d\n" % size)
    return binary


def test_discovers_stream_and_meta(tmp_path):
    binary = _make_stream(tmp_path)
    found = discover_spikeglx_stream(tmp_path, "imec0.ap")
    assert found["ok"]
    assert found["binary"] == str(binary)
    assert found["meta"].endswith(".imec0.ap.meta")
    assert found["size_bytes"] == 2048


def test_missing_stream_reports_what_is_present(tmp_path):
    _make_stream(tmp_path, stream="imec0.ap")
    found = discover_spikeglx_stream(tmp_path, "imec1.ap")
    assert not found["ok"]
    assert "no *.imec1.ap.bin" in found["error"]
    assert "imec0.ap" in found["streams_present"]


def test_missing_meta_is_a_failure(tmp_path):
    binary = _make_stream(tmp_path)
    binary.with_suffix(".meta").unlink()
    found = discover_spikeglx_stream(tmp_path, "imec0.ap")
    assert not found["ok"]
    assert "no matching .meta" in found["error"]


def test_ambiguous_stream_is_a_failure(tmp_path):
    _make_stream(tmp_path, run="run_a")
    _make_stream(tmp_path, run="run_b")
    found = discover_spikeglx_stream(tmp_path, "imec0.ap")
    assert not found["ok"]
    assert "expected one" in found["error"]


def test_absent_source_dir_is_a_failure(tmp_path):
    found = discover_spikeglx_stream(tmp_path / "nope", "imec0.ap")
    assert not found["ok"]
    assert "does not exist" in found["error"]


def test_unmounted_source_is_reported_clearly(tmp_path):
    report = preflight_report(data_dir=tmp_path / "mnt" / "NPX" / "missing")
    check = next(c for c in report["checks"] if c["name"] == "source_mounted")
    assert not check["ok"]
    assert not report["ok"]
    assert "mount is missing" in check["detail"]


def test_output_dir_that_does_not_exist_yet_passes_on_writable_parent(tmp_path):
    report = preflight_report(output_dir=tmp_path / "results")
    check = next(c for c in report["checks"] if c["name"] == "output_writable")
    assert check["ok"]
    assert "does not exist yet" in check["detail"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_unwritable_output_dir_fails(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        report = preflight_report(output_dir=locked)
        check = next(c for c in report["checks"] if c["name"] == "output_writable")
        assert not check["ok"]
        assert not report["ok"]
    finally:
        locked.chmod(0o700)


def test_scratch_space_floor_is_enforced(tmp_path):
    scratch = tmp_path / "nvme"
    scratch.mkdir()
    report = preflight_report(local_work_dir=scratch, min_free_gb=10**9)
    check = next(c for c in report["checks"] if c["name"] == "scratch_free_space")
    assert not check["ok"]
    assert check["required_gb"] >= 10**9
    assert not report["ok"]


def test_scratch_requirement_scales_with_source_size(tmp_path):
    """A large source must raise the scratch requirement above the floor."""
    data = tmp_path / "data"
    data.mkdir()
    binary = _make_stream(data)
    # Sparse file: reports a realistic AP-stream size without consuming disk.
    with open(binary, "r+b") as handle:
        handle.truncate(400 * 1024**3)
    scratch = tmp_path / "nvme"
    scratch.mkdir()
    report = preflight_report(
        data_dir=data,
        stream_id="imec0.ap",
        local_work_dir=scratch,
        min_free_gb=10.0,
    )
    check = next(c for c in report["checks"] if c["name"] == "scratch_free_space")
    # 400 GB source * 3 dominates the 10 GB floor.
    assert check["required_gb"] == pytest.approx(1200.0, abs=0.1)
    assert not check["ok"], "1200 GB should exceed scratch on any test host"


def test_floor_applies_when_source_is_small(tmp_path):
    """With a small source the floor, not the 3x scaling, sets the requirement."""
    data = tmp_path / "data"
    data.mkdir()
    _make_stream(data, size=4096)
    scratch = tmp_path / "nvme"
    scratch.mkdir()
    report = preflight_report(
        data_dir=data,
        stream_id="imec0.ap",
        local_work_dir=scratch,
        min_free_gb=12.5,
    )
    check = next(c for c in report["checks"] if c["name"] == "scratch_free_space")
    assert check["required_gb"] == pytest.approx(12.5, abs=0.1)


def test_happy_path_reports_ok(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _make_stream(data)
    report = preflight_report(
        data_dir=data,
        stream_id="imec0.ap",
        output_dir=tmp_path / "out",
        local_work_dir=tmp_path / "nvme",
        min_free_gb=0.0,
    )
    assert report["ok"], format_preflight(report)
    assert {c["name"] for c in report["checks"]} == {
        "source_mounted",
        "source_stream",
        "output_writable",
        "scratch_writable",
        "scratch_free_space",
    }


def test_format_is_operator_readable(tmp_path):
    report = preflight_report(data_dir=tmp_path / "missing")
    text = format_preflight(report)
    assert "[FAIL] source_mounted" in text
    assert text.strip().endswith("Preflight FAILED")


def test_no_arguments_is_trivially_ok():
    report = preflight_report()
    assert report["ok"]
    assert report["checks"] == []
    assert DEFAULT_MIN_FREE_GB > 0
