from types import SimpleNamespace

import pytest

from testing.development_runner import run_development_arms


def fake_contract():
    candidates = (
        {"name": "baseline", "role": "reference", "sorter_config": "rescue", "curation_profile": "legacy-compatible-cosine-0.90-ccg-0.5-v1"},
        {"name": "rigid", "role": "candidate", "sorter_config": "rescue_rigid", "curation_profile": "legacy-compatible-cosine-0.90-ccg-0.5-v1"},
    )
    return SimpleNamespace(
        digest="contract-digest",
        candidates=candidates,
        raw={"recording": {"recording_digest": "source-request"}},
    )


def install_mocks(monkeypatch, *, accepted_digest="strip-request"):
    calls = {"sort": [], "curation": [], "qc": [], "export": []}
    monkeypatch.setattr(
        "testing.development_runner.validate_accepted_recording",
        lambda path: {
            "request_digest": accepted_digest,
            "source_recording_request_digest": "source-request",
        },
    )
    monkeypatch.setattr("testing.development_runner.validate_production_environment", lambda **kwargs: {"ok": True})
    monkeypatch.setattr("testing.development_runner.repository_receipt", lambda: {"git_commit": "abc"})

    def sort(recording, output, config):
        calls["sort"].append(config.label)
        return {"recording_request_digest": accepted_digest, "summary": {"unit_count": 2}}
    monkeypatch.setattr("testing.development_runner.run_sorter_config", sort)
    monkeypatch.setattr("testing.development_runner.check_effective_settings", lambda label, manifest: {"label": label})
    monkeypatch.setattr(
        "testing.development_runner.pin_sort_identity",
        lambda sort_dir, identity_path: {"identity_digest": f"identity-{sort_dir.name}"},
    )

    def curation(sorter, output, identity):
        calls["curation"].append((output, identity["identity_digest"]))
        return {"request_digest": "cur", "summary": {"unit_count": 2}}
    def qc(recording, curated, output, identity, **kwargs):
        calls["qc"].append((output, identity["identity_digest"], kwargs))
        return {"request_digest": "qc", "summary": {"unit_count": 2}}
    monkeypatch.setattr("testing.development_runner.run_curation_stage", curation)
    monkeypatch.setattr("testing.development_runner.run_qc_stage", qc)
    monkeypatch.setattr(
        "testing.development_runner.run_matlab_export_stage",
        lambda curated, qc, identity: calls["export"].append((curated, qc)),
    )
    return calls


def test_all_arms_use_shared_identity_bound_downstream(monkeypatch, tmp_path):
    calls = install_mocks(monkeypatch)
    report = run_development_arms(
        fake_contract(), recording_dir=tmp_path / "recording",
        output_root=tmp_path / "outputs", require_cuda=False,
    )
    assert calls["sort"] == ["rescue", "rescue_rigid"]
    assert len(calls["curation"]) == len(calls["qc"]) == len(calls["export"]) == 2
    assert all(call[2]["waveform_seed"] == 0 for call in calls["qc"])
    assert set(report["arms"]) == {"baseline", "rigid"}
    assert report["arms"]["rigid"]["pre_curation_summary"] == {"unit_count": 2}
    assert report["arms"]["rigid"]["post_curation_summary"] == {"unit_count": 2}


def test_sort_reuse_from_another_recording_fails_closed(monkeypatch, tmp_path):
    install_mocks(monkeypatch)
    monkeypatch.setattr(
        "testing.development_runner.run_sorter_config",
        lambda *args, **kwargs: {"recording_request_digest": "wrong", "summary": {}},
    )
    with pytest.raises(RuntimeError, match="another recording"):
        run_development_arms(
            fake_contract(), recording_dir=tmp_path / "recording",
            output_root=tmp_path / "outputs", require_cuda=False,
        )


def test_outputs_cannot_overlap_the_recording(tmp_path):
    with pytest.raises(ValueError, match="disjoint"):
        run_development_arms(
            fake_contract(), recording_dir=tmp_path / "recording",
            output_root=tmp_path / "recording/outputs", require_cuda=False,
        )
