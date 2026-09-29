import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_full_only_positive_state_is_cross_version_identical():
    result = json.loads(
        (ROOT / "testing/outputs/em2f_full_kernel_version_20260929/RESULT.json").read_text()
    )
    assert result["status"] == "pass"
    assert result["voltage_read"] is False
    assert result["sort_launched"] is False
    assert result["state_um"] == 80.0
    assert result["versions"] == ["0.104.7", "0.104.8"]
    assert result["byte_identical"] is True
