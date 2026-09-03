import json

import numpy as np
import pytest

from testing import ladder_l1
from testing.ladder_l1 import CurationConfig, l1_root, l1_run


def test_curation_config_digest_is_stable_and_sensitive():
    a = CurationConfig()
    assert a.digest == CurationConfig().digest
    assert a.digest != CurationConfig(ccg_threshold=0.6).digest


def test_l1_root_prefers_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LADDER_L1_ROOT", str(tmp_path / "here"))
    assert l1_root() == tmp_path / "here"


class _FakeSnippet:
    def __init__(self, d):
        self.dir = d
        self.manifest = {"spec_digest": "deadbeef" * 8, "axes": {"snr": "high"}}
        self.fs = 30_000.0
        self.duration_s = 120.0


@pytest.fixture
def patched(monkeypatch, tmp_path):
    snip_dir = tmp_path / "snip"
    snip_dir.mkdir()
    monkeypatch.setattr(ladder_l1, "load_snippet", lambda p: _FakeSnippet(snip_dir))

    calls = {"sort": 0, "cur": 0}

    def fake_sort(recording_dir, sort_dir):
        calls["sort"] += 1
        so = sort_dir / "sorter_output"
        so.mkdir(parents=True)
        np.save(so / "amplitudes.npy", np.array([10.0, 20.0, 30.0, 40.0]))
        np.save(so / "spike_clusters.npy", np.array([0, 0, 1, 1]))
        (sort_dir / "rescue_sort_manifest.json").write_text(
            json.dumps({"complete": True, "summary": {"unit_count": 2}})
        )

    def fake_identity(sort_dir):
        return {"identity_digest": "id"}

    def fake_curation(sorter_output, cur_dir, identity, **kw):
        cur = cur_dir / "cur_output"
        if (cur / "cluster_KSLabel.tsv").exists():
            return  # idempotent, like the real stage
        calls["cur"] += 1
        cur.mkdir(parents=True)
        np.save(cur / "spike_clusters.npy", np.array([0, 0, 1]))
        (cur / "cluster_KSLabel.tsv").write_text("cluster_id\tKSLabel\n0\tgood\n1\tmua\n")

    def fake_score(curated, **kw):
        return {"schema": "x", "headline": 1, "runtime": kw.get("runtime_s")}

    monkeypatch.setattr(ladder_l1, "run_kilosort4", fake_sort)
    monkeypatch.setattr(ladder_l1, "build_sort_identity", fake_identity)
    monkeypatch.setattr(ladder_l1, "run_curation_stage", fake_curation)
    monkeypatch.setattr(ladder_l1, "score_sort", fake_score)
    return calls, tmp_path


def test_l1_run_orchestrates_and_writes_result(patched):
    calls, tmp_path = patched
    out = tmp_path / "l1"
    result = l1_run("ignored", out_root=out)

    assert calls == {"sort": 1, "cur": 1}
    assert result["schema"] == ladder_l1.L1_SCHEMA
    assert result["score"]["headline"] == 1
    assert result["stage_observables"]["amplitude_p50"] == 25.0
    assert result["stage_observables"]["curated_unit_count"] == 2
    assert result["wall_clock"]["sort_was_cached"] is False

    work = out / ("deadbeef" * 8)[:16]
    written = json.loads(
        (work / f"cur-{CurationConfig().digest[:12]}" / "l1_result.json").read_text()
    )
    assert written["curation_digest"] == CurationConfig().digest


def test_l1_run_reuses_sort_across_curation_variants(patched):
    calls, tmp_path = patched
    out = tmp_path / "l1"
    l1_run("x", out_root=out, curation=CurationConfig())
    l1_run("x", out_root=out, curation=CurationConfig(ccg_threshold=0.6))

    assert calls["sort"] == 1  # sort computed once
    assert calls["cur"] == 2  # curation re-run per config


def test_l1_run_marks_sort_cached_on_second_call(patched):
    calls, tmp_path = patched
    out = tmp_path / "l1"
    first = l1_run("x", out_root=out)
    second = l1_run("x", out_root=out)
    assert first["wall_clock"]["sort_was_cached"] is False
    assert second["wall_clock"]["sort_was_cached"] is True
    assert calls["sort"] == 1


def test_l1_run_refuses_mnt(patched):
    with pytest.raises(ValueError, match="/mnt"):
        l1_run("x", out_root="/mnt/somewhere")


def test_l1_run_routes_non_rescue_sorter_to_its_own_cache_leaf(patched, monkeypatch):
    calls, tmp_path = patched
    from testing.ladder_sorter import LEGACY_STYLE

    routed = {"n": 0}

    def fake_config_sort(recording_dir, sort_dir, config):
        routed["n"] += 1
        so = sort_dir / "sorter_output"
        so.mkdir(parents=True)
        np.save(so / "amplitudes.npy", np.array([10.0, 20.0]))
        np.save(so / "spike_clusters.npy", np.array([0, 1]))
        (sort_dir / "rescue_sort_manifest.json").write_text(
            json.dumps({"complete": True, "summary": {"unit_count": 2}})
        )

    monkeypatch.setattr(
        "testing.ladder_sorter.run_sorter_config", fake_config_sort, raising=False
    )
    out = tmp_path / "l1"
    result = l1_run("x", out_root=out, sorter=LEGACY_STYLE)

    assert routed["n"] == 1 and calls["sort"] == 0  # rescue path not taken
    assert result["sorter_config"] == "legacy_style"
    assert (out / ("deadbeef" * 8)[:16] / f"sort-{LEGACY_STYLE.digest[:12]}").exists()
