import json

import pytest

from testing.em2f_resume_worker import validate_existing_link_inputs


def test_existing_materialized_inputs_must_match_receipt_and_source(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "subtraction.h5").write_bytes(b"abc")
    (target / "subtraction.h5").write_bytes(b"abc")
    receipt = {
        "status": "complete",
        "source": str(source),
        "link_step": "detection",
        "files": [{"path": "subtraction.h5", "bytes": 3, "kind": "file"}],
    }
    (target / "link-input-copy.json").write_text(json.dumps(receipt))
    assert validate_existing_link_inputs(source, target, "detection") == receipt

    (target / "subtraction.h5").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="size changed"):
        validate_existing_link_inputs(source, target, "detection")
