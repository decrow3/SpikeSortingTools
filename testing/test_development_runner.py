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
        raw={"recording": {"recording_digest": "source-request"}, "spatial_contract": {}},
    )


def install_mocks(monkeypatch, *, accepted_digest="strip-request"):
    calls = {"sort": [], "curation": [], "qc": [], "export": []}
    monkeypatch.setattr("testing.development_runner.validate_development_selection", lambda *args: None)
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
    assert (tmp_path / "outputs/arms_summary.json").is_file()
    assert (tmp_path / "outputs/group_receipts/all-arms.json").is_file()


def test_selected_group_runs_only_named_arms_and_does_not_finalize(monkeypatch, tmp_path):
    calls = install_mocks(monkeypatch)
    report = run_development_arms(
        fake_contract(), recording_dir=tmp_path / "recording",
        output_root=tmp_path / "outputs", require_cuda=False,
        candidate_names=["rigid"], group_id="motion-axis",
    )
    assert calls["sort"] == ["rescue_rigid"]
    assert set(report["arms"]) == {"rigid"}
    assert report["group_receipt"]["candidate_names"] == ["rigid"]
    assert (tmp_path / "outputs/group_receipts/motion-axis.json").is_file()
    assert not (tmp_path / "outputs/arms_summary.json").exists()


def test_candidate_selection_refuses_unknown_duplicate_or_unsafe_group(monkeypatch, tmp_path):
    install_mocks(monkeypatch)
    common = dict(contract=fake_contract(), recording_dir=tmp_path / "recording",
                  output_root=tmp_path / "outputs", require_cuda=False)
    with pytest.raises(ValueError, match="unknown candidate"):
        run_development_arms(**common, candidate_names=["missing"])
    with pytest.raises(ValueError, match="repeated"):
        run_development_arms(**common, candidate_names=["rigid", "rigid"])
    with pytest.raises(ValueError, match="group_id"):
        run_development_arms(**common, candidate_names=["rigid"], group_id="../unsafe")


def test_finalize_requires_every_requested_compatible_manifest(monkeypatch, tmp_path):
    from testing.development_runner import finalize_development_arms
    install_mocks(monkeypatch)
    root = tmp_path / "outputs"
    run_development_arms(
        fake_contract(), recording_dir=tmp_path / "recording", output_root=root,
        require_cuda=False, candidate_names=["baseline"], group_id="reference",
    )
    with pytest.raises(RuntimeError, match="candidate is not complete: rigid"):
        finalize_development_arms(
            fake_contract(), recording_dir=tmp_path / "recording", output_root=root,
        )
    summary = finalize_development_arms(
        fake_contract(), recording_dir=tmp_path / "recording", output_root=root,
        candidate_names=["baseline"],
    )
    assert set(summary["arms"]) == {"baseline"}


def test_arm_lock_refuses_concurrent_owner(tmp_path):
    from testing.development_runner import arm_execution_lock
    with arm_execution_lock(tmp_path / "arm"):
        with pytest.raises(RuntimeError, match="already owned"):
            with arm_execution_lock(tmp_path / "arm"):
                pass


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


@pytest.mark.parametrize("damage", ["params", "digest"])
def test_existing_sort_requires_complete_frozen_config(damage):
    from types import SimpleNamespace
    from testing.development_runner import _validate_existing_sort_config
    config = SimpleNamespace(params=lambda: {"do_CAR": True, "artifact_threshold": 1000}, digest="config-digest")
    manifest = {
        "sorter_params": {"do_CAR": True, "artifact_threshold": 1000},
        "config_digest": "config-digest",
    }
    _validate_existing_sort_config(manifest, config)
    if damage == "params":
        manifest["sorter_params"]["artifact_threshold"] = 2000
    else:
        manifest["config_digest"] = "other-config"
    with pytest.raises(RuntimeError, match="frozen candidate configuration"):
        _validate_existing_sort_config(manifest, config)


def test_outputs_cannot_overlap_the_recording(tmp_path):
    with pytest.raises(ValueError, match="inside the recording"):
        run_development_arms(
            fake_contract(), recording_dir=tmp_path / "recording",
            output_root=tmp_path / "recording/outputs", require_cuda=False,
        )


def test_output_root_may_own_recording_and_sibling_arms(monkeypatch, tmp_path):
    calls = install_mocks(monkeypatch)
    root = tmp_path / "experiment"
    run_development_arms(
        fake_contract(), recording_dir=root / "recording",
        output_root=root, require_cuda=False, candidate_names=["baseline"],
    )
    assert calls["sort"] == ["rescue"]


@pytest.mark.parametrize("damage", [None,"curation_settings","qc_settings","incomplete","missing_file"])
def test_existing_downstream_validates_requests_and_outputs(tmp_path,damage):
    import json
    from pipeline.downstream import ensure_stage_request,write_stage_receipt
    from pipeline.config import fingerprint
    from testing.development_runner import _validated_existing_downstream
    root=tmp_path/"existing"; curated=root/"cur/cur_output"; qc=root/"qc"
    sorter=tmp_path/"sorter_output"; recording=tmp_path/"recording"
    identity={"identity_digest":"id"}
    stages=[(root/"cur","curation","legacy_compatible_curation",
        dict(sorter_output=str(sorter),strategy="run_cur_final_cosine",cosine_threshold=.9,ccg_threshold=.5,automatic_artifact_pair_merging=False),
        [curated/n for n in ("spike_times.npy","spike_clusters.npy","cluster_KSLabel.tsv","ops.npy")]),
        (qc,"qc","legacy_compatible_qc",
        dict(recording_dir=str(recording),curated_output=str(curated),waveform_seed=0,
             waveform_extractor="ordered_chunked_local_memmap_v1",waveform_read_chunk_duration_s=1.,waveforms_per_unit=512,waveform_samples=82),
        [qc/n for n in ("waveforms/waveforms.npz","refractory/refractory_qc.npz","refractory/refractory_qc.pdf",
                        "amp_truncation/truncation_qc.npz","amp_truncation/present_qc.npz","amp_truncation/truncation_qc.pdf")])]
    for folder,name,stage,settings,files in stages:
        for f in files:
            f.parent.mkdir(parents=True,exist_ok=True);f.touch()
        request=ensure_stage_request(folder/f"{name}_request.json",stage=stage,sort_identity=identity,settings=settings)
        write_stage_receipt(folder/f"{name}_receipt.json",request=request,required_files=files,summary={})
    if damage in {"curation_settings","qc_settings"}:
        folder,name,_,_,_=stages[0 if damage=="curation_settings" else 1]
        path=folder/f"{name}_request.json"; data=json.loads(path.read_text())
        data["settings"]["cosine_threshold" if name=="curation" else "waveform_seed"]=.8 if name=="curation" else 2
        data["request_digest"]=fingerprint({k:v for k,v in data.items() if k not in {"request_digest","created_at"}})
        path.write_text(json.dumps(data))
    elif damage=="incomplete":
        path=qc/"qc_receipt.json";data=json.loads(path.read_text());data["complete"]=False;path.write_text(json.dumps(data))
    elif damage=="missing_file":
        (curated/"ops.npy").unlink()
    if damage:
        with pytest.raises(RuntimeError):
            _validated_existing_downstream(root,"id",sorter_output=sorter,recording_dir=recording)
    else:
        result=_validated_existing_downstream(root,"id",sorter_output=sorter,recording_dir=recording)
        assert result[:2]==(curated,qc)
