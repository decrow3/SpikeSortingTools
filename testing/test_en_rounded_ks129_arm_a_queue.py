import json

import numpy as np
import pytest

from testing.en_rounded_ks129_arm_a_queue import materialization_request, rigid_projection, wait_for_field
from testing import en_rounded_ks129_queue
from testing import luke_external_warp_pipeline


def test_rigid_projection_accepts_rigid_or_time_by_depth() -> None:
    time = np.array([0.0, 1.0, 2.0])
    rigid = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(rigid_projection(time, rigid), rigid)
    field = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
    assert np.array_equal(rigid_projection(time, field), [1.0, 3.0, 5.0])


def test_rigid_projection_rejects_depth_by_time() -> None:
    with pytest.raises(ValueError, match="time-by-depth"):
        rigid_projection(np.arange(4.0), np.ones((2, 4)))


def test_arm_a_materialization_request_preserves_manifest_schema() -> None:
    contract = {"digest": "contract", "q0_receipt_sha256": "q0", "adapter": "exact"}
    request = materialization_request(contract, "field", "full_session", [0, 10])
    assert request["schema"] == "en-rounded-field-ks129-materialization-v1"
    assert request["arm"] == "A"
    assert "schema_version" not in request


def test_wait_for_field_returns_when_payload_exists(tmp_path) -> None:
    field = tmp_path / "field.npz"
    field.write_bytes(b"ready")
    wait_for_field(field, tmp_path / "status.json", poll_seconds=0.001)
    assert not (tmp_path / "status.json").exists()


def test_materialization_reports_verification_after_write(monkeypatch, tmp_path) -> None:
    events = []

    class Recording:
        def save(self, *, folder, dtype, n_jobs, progress_bar):
            folder.mkdir(parents=True)
            events.append(("save", dtype, n_jobs, progress_bar))

    def accepted(folder, source_manifest, *, request):
        events.append(("accept", folder.name, source_manifest, request))
        return {"complete": True, "request_digest": "test"}

    monkeypatch.setattr(luke_external_warp_pipeline, "_accepted_manifest", accepted)
    output = tmp_path / "recording"
    result = luke_external_warp_pipeline._materialize_arm(
        Recording(),
        output,
        source_manifest={"source": "accepted"},
        request={"schema": "test"},
        n_jobs=3,
        before_accept=lambda: events.append(("verifying",)),
    )

    assert result["complete"] is True
    assert output.is_dir()
    assert events == [
        ("save", "int16", 3, True),
        ("verifying",),
        ("accept", "recording.partial", {"source": "accepted"}, {"schema": "test"}),
    ]


def test_run_sort_distinguishes_input_validation_from_sorting(monkeypatch, tmp_path) -> None:
    status = tmp_path / "STATUS.json"
    stages = []

    monkeypatch.setattr(en_rounded_ks129_queue, "gpu_busy", lambda: False)

    def fake_sort(recording_dir, sort_dir, *, before_sort):
        stages.append(json.loads(status.read_text())["stage"])
        before_sort()
        stages.append(json.loads(status.read_text())["stage"])
        return {"complete": True}

    monkeypatch.setattr(en_rounded_ks129_queue, "run_kilosort4", fake_sort)
    manifest, elapsed = en_rounded_ks129_queue.run_sort(
        tmp_path / "recording", tmp_path / "sort", status, "full"
    )

    assert manifest == {"complete": True}
    assert elapsed >= 0.0
    assert stages == ["validating_sort_input_full", "sorting_full"]
