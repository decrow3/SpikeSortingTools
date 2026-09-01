from __future__ import annotations

import argparse

import pytest

from pipeline.runtime import (
    PRODUCTION_PACKAGES,
    PRODUCTION_PYTHON,
    PRODUCTION_UV_PREFIX,
    production_environment_contract,
    production_environment_receipt,
    validate_production_environment,
)
from SpikeGLX_ext_ref_rescue_testing import plan_payload
from pipeline import RescueConfig


def test_production_contract_is_exact_and_uses_frozen_uv():
    assert PRODUCTION_PYTHON == (3, 12, 4)
    assert PRODUCTION_PACKAGES["spikeinterface"] == "0.102.1"
    assert PRODUCTION_PACKAGES["kilosort"] == "4.0.27"
    assert PRODUCTION_PACKAGES["dredge-ephys"] == "0.3.0"
    assert PRODUCTION_PACKAGES["neo"] == "0.14.0"
    assert PRODUCTION_PACKAGES["torch"] == "2.6.0+cu124"
    assert "--frozen" in PRODUCTION_UV_PREFIX
    assert "environments/rescue-production" in PRODUCTION_UV_PREFIX
    contract = production_environment_contract()
    assert contract["python"] == "3.12.4"
    assert len(contract["uv_lock_sha256"]) == 64


def test_runtime_validation_refuses_mismatched_environment(monkeypatch):
    receipt = production_environment_receipt()
    installed = dict(receipt["packages_installed"])
    installed["spikeinterface"] = "999.0"
    receipt["packages_installed"] = installed
    monkeypatch.setattr(
        "pipeline.runtime.production_environment_receipt", lambda **kwargs: receipt
    )
    with pytest.raises(RuntimeError, match="Production environment validation failed"):
        validate_production_environment()


def test_plan_reports_production_environment(tmp_path):
    args = argparse.Namespace(
        data_dir=tmp_path,
        stream_id="imec0.ap",
        start_s=0.0,
        duration_s=60.0,
        bad_channel=None,
        no_motion_sidecar=False,
        motion_split_half=False,
        n_jobs=2,
        motion_chunk_duration="2s",
    )
    plan = plan_payload(args, tmp_path / "out", RescueConfig())
    environment = plan["production_environment"]
    assert environment["lock_required"] is True
    assert environment["packages"] == PRODUCTION_PACKAGES
    assert environment["canonical_prefix"] == PRODUCTION_UV_PREFIX
