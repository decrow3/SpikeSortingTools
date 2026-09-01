from __future__ import annotations

import pytest

from pipeline.runtime import (
    PRODUCTION_PACKAGES,
    PRODUCTION_PYTHON,
    PRODUCTION_UV_SETUP,
    production_environment_contract,
    production_environment_receipt,
    validate_production_environment,
)
from SpikeGLX_ext_ref_rescue import build_run_plan


def test_production_contract_is_exact_and_uses_frozen_uv():
    assert PRODUCTION_PYTHON == (3, 12, 4)
    assert PRODUCTION_PACKAGES["spikeinterface"] == "0.102.1"
    assert PRODUCTION_PACKAGES["kilosort"] == "4.0.27"
    assert PRODUCTION_PACKAGES["dredge-ephys"] == "0.3.0"
    assert PRODUCTION_PACKAGES["neo"] == "0.14.0"
    assert PRODUCTION_PACKAGES["torch"] == "2.6.0+cu124"
    assert "--frozen" in PRODUCTION_UV_SETUP[0]
    assert "environments/rescue-production" in PRODUCTION_UV_SETUP[0]
    assert ".venv/bin/activate" in PRODUCTION_UV_SETUP[1]
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


def test_plan_reports_production_environment():
    plan = build_run_plan()
    environment = plan["production_environment"]
    assert environment["lock_required"] is True
    assert environment["packages"] == PRODUCTION_PACKAGES
    assert environment["canonical_setup"] == list(PRODUCTION_UV_SETUP)
