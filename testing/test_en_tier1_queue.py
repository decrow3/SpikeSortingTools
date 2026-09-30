import json

from testing.en_tier1_queue import wait_for_inputs


def test_wait_for_inputs_returns_immediately_when_all_arms_ready(tmp_path, monkeypatch) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"sentinel": True}) + "\n")
    expected = {"all_ready": True, "arms": {}}
    monkeypatch.setattr("testing.en_tier1_queue.audit_inputs", lambda value: expected)
    assert wait_for_inputs(config, tmp_path / "status.json", poll_seconds=0.001) == expected
    assert not (tmp_path / "status.json").exists()
