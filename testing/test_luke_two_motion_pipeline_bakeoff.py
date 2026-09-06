"""Unit tests for testing/luke_two_motion_pipeline_bakeoff.py."""

import json
import numpy as np
from pathlib import Path

import pytest

from testing.first_pipeline_candidate_contract import freeze_acceptance, DEFAULT_CONTRACT
from testing.luke_two_motion_pipeline_bakeoff import run_bakeoff


def test_bakeoff_verify_mode(tmp_path):
    out = tmp_path / "out"
    manifest = run_bakeoff(
        option="control",
        mode="verify",
        out_root=out,
    )
    assert manifest["option"] == "control"
    assert manifest["mode"] == "verify"
    assert manifest["status"] == "completed"
    assert (out / "bakeoff_manifest.json").exists()


def test_bakeoff_unwarped_identity_smoke_with_real_inputs(tmp_path):
    out = tmp_path / "out"
    freeze_acceptance(DEFAULT_CONTRACT, out)

    snippet_dir = tmp_path / "snippet"
    snippet_dir.mkdir()
    np.save(snippet_dir / "spike_times.npy", np.array([30000, 60000, 90000, 120000], dtype=np.int64))
    np.save(snippet_dir / "spike_clusters.npy", np.array([0, 0, 0, 0], dtype=np.int64))
    np.save(snippet_dir / "spike_depths.npy", np.array([100.0, 100.0, 105.0, 105.0]))
    np.save(snippet_dir / "amplitudes.npy", np.array([10.0, 10.0, 10.0, 10.0]))

    manifest = run_bakeoff(
        option="unwarped_identity",
        mode="smoke",
        out_root=out,
        snippet_dir=snippet_dir,
    )
    assert manifest["option"] == "unwarped_identity"
    assert manifest["mode"] == "smoke"
    assert (out / "unwarped_identity" / "unwarped_identity_manifest.json").exists()
    assert (out / "candidate_export" / "spike_tracks.npy").exists()
    assert (out / "candidate_score.json").exists()


def test_bakeoff_refuses_execution_without_real_inputs(tmp_path):
    out = tmp_path / "out"
    freeze_acceptance(DEFAULT_CONTRACT, out)
    with pytest.raises(FileNotFoundError, match="Required real spike inputs"):
        run_bakeoff(
            option="unwarped_identity",
            mode="smoke",
            out_root=out,
            snippet_dir=tmp_path / "nonexistent_dir",
        )


def test_bakeoff_refuses_output_under_mnt(tmp_path):
    with pytest.raises(ValueError, match="/mnt"):
        run_bakeoff(
            option="control",
            mode="verify",
            out_root=Path("/mnt/bad_output_root"),
        )
