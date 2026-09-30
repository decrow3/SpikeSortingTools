import numpy as np
import pytest

from testing.en_rounded_ks129_arm_a_queue import materialization_request, rigid_projection, wait_for_field


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
