import numpy as np
import pytest

from pipeline.config import fingerprint
from testing.development_strip import classify_unit_depths, select_depth_channels


def test_repository_receipt_hashes_real_git_bytes(tmp_path):
    import subprocess
    import hashlib
    from testing.development_strip import repository_receipt
    def git(*args):
        return subprocess.run(["git","-C",str(tmp_path),*args],check=True,capture_output=True).stdout
    git("init")
    git("-c","user.name=Test","-c","user.email=test@example.test","commit","--allow-empty","-m","initial")
    assert repository_receipt(tmp_path)["tracked_diff_sha256"] == hashlib.sha256(b"").hexdigest()
    (tmp_path/"tracked").write_text("change\n")
    git("add","tracked")
    result = repository_receipt(tmp_path)
    assert result["tracked_dirty"]
    assert result["tracked_diff_sha256"] == hashlib.sha256(git("diff","HEAD","--binary","--no-ext-diff")).hexdigest()


@pytest.fixture
def prepared_strip(tmp_path):
    import json
    from spikeinterface.core import NumpyRecording
    from pipeline.preprocess import recording_geometry_receipt, recording_binary_receipt, MANIFEST_NAME, RECORDING_MANIFEST_SCHEMA
    from testing.development_strip import materialize_development_strip
    source = tmp_path/"source"
    recording = NumpyRecording(np.random.default_rng(5).integers(-100,100,(1000,6),dtype=np.int16), sampling_frequency=1000.)
    recording.set_channel_locations(np.c_[np.zeros(6),np.arange(6)*100.])
    recording.save(folder=source, n_jobs=1, progress_bar=False)
    manifest = dict(schema_version=RECORDING_MANIFEST_SCHEMA, request_digest="source-request",complete=True,
                    expected_binary_bytes=12000, **recording_geometry_receipt(recording), **recording_binary_receipt(source))
    (source/MANIFEST_NAME).write_text(json.dumps(manifest))
    rec = dict(accepted_recording_path=str(source),recording_digest="source-request",
               recording_content_sha256=manifest["recording_content_sha256"],probe_geometry_hash=manifest["probe_geometry_hash"],
               start_s=.1,duration_s=.8)
    spatial = dict(processing_depth_um=[50,450],scoring_depth_um=[150,350],minimum_edge_exclusion_um=50)
    strip = tmp_path/"strip"
    _,saved = materialize_development_strip(source,strip,recording_spec=rec,spatial_spec=spatial)
    return source,strip,rec,spatial,saved


def test_materialized_strip_validates_reuses_and_dispatches(prepared_strip, monkeypatch, tmp_path):
    from pipeline.preprocess import validate_accepted_recording, RECORDING_MANIFEST_SCHEMA
    from testing.development_strip import materialize_development_strip, validate_development_selection
    from testing.test_development_runner import install_mocks, fake_contract
    from testing.development_runner import run_development_arms
    source,strip,rec,spatial,saved = prepared_strip
    assert saved["schema_version"] == RECORDING_MANIFEST_SCHEMA
    accepted = validate_accepted_recording(strip)
    validate_development_selection(strip,accepted,rec,spatial)
    _, reused = materialize_development_strip(source,strip,recording_spec=rec,spatial_spec=spatial)
    assert reused == saved
    calls = install_mocks(monkeypatch,accepted_digest=saved["request_digest"])
    monkeypatch.setattr("testing.development_runner.validate_accepted_recording",validate_accepted_recording)
    monkeypatch.setattr("testing.development_runner.validate_development_selection",validate_development_selection)
    contract = fake_contract()
    contract.raw = dict(recording=rec,spatial_contract=spatial)
    run_development_arms(contract,recording_dir=strip,output_root=tmp_path/"arms",require_cuda=False)
    assert calls["sort"] == ["rescue","rescue_rigid"]


def test_wrong_slice_refused_before_sort(prepared_strip,monkeypatch,tmp_path):
    from copy import deepcopy
    from pipeline.preprocess import validate_accepted_recording
    from testing.development_strip import validate_development_selection
    from testing.test_development_runner import install_mocks,fake_contract
    from testing.development_runner import run_development_arms
    source,strip,rec,spatial,saved = prepared_strip
    calls=install_mocks(monkeypatch)
    monkeypatch.setattr("testing.development_runner.validate_accepted_recording",validate_accepted_recording)
    monkeypatch.setattr("testing.development_runner.validate_development_selection",validate_development_selection)
    for folder,r,s in [(source,rec,spatial),(strip,{**rec,"start_s":0.},spatial),
                       (strip,rec,{**spatial,"processing_depth_um":[0,450]})]:
        contract=fake_contract(); contract.raw=dict(recording=deepcopy(r),spatial_contract=deepcopy(s))
        with pytest.raises(RuntimeError,match="contracted slice"):
            run_development_arms(contract,recording_dir=folder,output_root=tmp_path/"arms",require_cuda=False)
    assert not calls["sort"]


class FakeRecording:
    def __init__(self):
        self.ids = np.array(["c7", "c2", "c9", "c1", "c4"], dtype=object)
        self.locations = np.array(
            [[0, 0], [32, 100], [0, 200], [32, 300], [0, 400]], dtype=float
        )

    def get_channel_ids(self):
        return self.ids

    def get_channel_locations(self):
        return self.locations


def test_physical_selection_preserves_ids_geometry_and_halo():
    selected = select_depth_channels(
        FakeRecording(), processing_depth_um=[50, 350], scoring_depth_um=[150, 250]
    )
    assert selected["processing_channel_ids"] == ["c2", "c9", "c1"]
    assert selected["interior_channel_ids"] == ["c9"]
    assert selected["halo_channel_ids"] == ["c2", "c1"]
    assert selected["processing_channel_locations_um"] == [[32.0, 100.0], [0.0, 200.0], [32.0, 300.0]]


@pytest.mark.parametrize(
    "processing, scoring, match",
    [([500, 600], [525, 575], "no channels"), ([0, 400], [410, 420], "strictly inside")],
)
def test_invalid_or_out_of_range_selection_is_refused(processing, scoring, match):
    with pytest.raises(ValueError, match=match):
        select_depth_channels(
            FakeRecording(), processing_depth_um=processing, scoring_depth_um=scoring
        )


def test_interior_scoring_keeps_edges_as_diagnostics():
    labels = classify_unit_depths(
        np.array([0, 25, 100, 200, 300, 375, 500]),
        processing_depth_um=[0, 400],
        scoring_depth_um=[100, 300],
        minimum_edge_exclusion_um=50,
    )
    assert labels.tolist() == ["edge", "edge", "interior", "interior", "interior", "edge", "outside"]


def test_strip_request_digest_changes_with_source_or_geometry():
    base = {"source_request": "a", "geometry": [[0, 100]], "depth": [0, 200]}
    assert fingerprint(base) != fingerprint({**base, "source_request": "b"})
    assert fingerprint(base) != fingerprint({**base, "geometry": [[0, 101]]})
