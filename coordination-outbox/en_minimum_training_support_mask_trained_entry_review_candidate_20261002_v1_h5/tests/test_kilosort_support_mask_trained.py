import json
from pathlib import Path
import tempfile

import pytest

from npx_preprocessing.motion import kilosort_support_mask_trained as trained
from testing.en_minimum_training_support_mask_trained_fixture import build_fixture, invoke


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = ROOT / "configs/en_minimum_training_support_mask.trained.review_pending.v1.json"
SMOKE_BASE = ROOT / "configs/en_minimum_training_support_mask.covariance_smoke.enabled.review_pending.v2.json"


def test_production_candidate_preserves_full_frozen_structure_and_is_disabled():
    config = json.loads(PRODUCTION_CONFIG.read_text())
    base = json.loads(SMOKE_BASE.read_text())
    trained.prevalidate_trained_structure(config)
    for key in (
        "recording", "crop", "field", "ops", "frozen_kilosort_settings",
        "covariance_gate", "native_invocation", "source_bindings", "prohibitions",
    ):
        assert key in config
        assert key in base
    assert config["execution_enabled"] is False
    assert config["trained"]["allow_saved_smoke_bank"] is False
    assert config["native_invocation"]["save_preprocessed_copy"] is False


def test_missing_required_key_fails_before_any_bound_source_or_manifest_access():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "fixture"
        path, digest = build_fixture(root, "run", missing_key=True)
        with pytest.raises(ValueError, match="missing required config key before data access"):
            trained.validate_trained_contract(
                path, expected_config_sha256=digest,
                allow_synthetic_orchestration=True, fixture_root=root,
            )
        assert not (root / "must_not_open").exists()
        assert not (root / "must_not_open_manifest").exists()


def test_actual_cli_success_reaches_completion_and_required_artifact_inventory():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "fixture"
        path, digest = build_fixture(root, "run")
        result = invoke(path, digest, root, "success")
        assert result.returncode == 0, result.stderr
        run = root / "run"
        complete = json.loads((run / "COMPLETE.json").read_text())
        inventory = json.loads((run / "launch_evidence/artifact_inventory.json").read_text())
        phases = json.loads((run / "launch_evidence/support_reads_by_phase.json").read_text())
        assert complete["required_artifacts_validated"] is True
        assert complete["smoke_bank_injected"] is False
        assert set(inventory["native"]) == set(trained.REQUIRED_NATIVE_FILES)
        assert set(inventory["pre_extraction_snapshot"]) == set(trained.REQUIRED_SNAPSHOT_FILES)
        assert set(phases["phase_counts"]) == {
            "covariance_and_whitening", "pca_and_universal_template_learning",
            "universal_detection_and_features", "learned_template_extraction",
        }


def test_actual_cli_injected_failure_preserves_stage_and_never_completes():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "fixture"
        path, digest = build_fixture(root, "run")
        result = invoke(path, digest, root, "fail-after-whitening")
        assert result.returncode != 0
        run = root / "run"
        failure = json.loads((run / "launch_evidence/run_failure.json").read_text())
        assert failure["stage"] == "native_trained_pipeline"
        assert failure["exception_message"] == "injected synthetic downstream failure"
        assert failure["whitening_audit"]["covariance_validation"]["passed"] is True
        assert failure["completion_written"] is False
        assert not (run / "COMPLETE.json").exists()
