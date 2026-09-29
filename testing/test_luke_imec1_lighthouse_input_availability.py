import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_missing_lighthouse_sources_are_explicitly_bounded():
    audit = json.loads(
        (ROOT / "testing/outputs/luke_imec1_lighthouse_input_availability_20260929/AUDIT.json").read_text()
    )
    assert audit["status"] == "blocked_missing_sources"
    assert audit["voltage_read"] is False
    assert audit["sort_launched"] is False
    assert len(audit["inputs"]) == 6
    assert not any(row["hash_matches"] for row in audit["inputs"])
    assert audit["archive"]["exists"] is True
    assert audit["archive"]["matching_source_members"] == []
    assert audit["git_object_name_matches"] == []
