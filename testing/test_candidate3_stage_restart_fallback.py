from pathlib import Path

import pytest

from testing.candidate3_stage_restart_fallback import (
    StageAttemptExistsError,
    run_fresh_stage_attempt,
)


def test_failure_is_preserved_and_attempt_id_cannot_be_reused(tmp_path: Path):
    def fail(stage_dir):
        (stage_dir / "partial.bin").write_bytes(b"partial")
        raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        run_fresh_stage_attempt(
            attempts_directory=tmp_path,
            attempt_id="a",
            input_bindings={"input": "sha256:x"},
            run_stage=fail,
            validate_stage=lambda _: {"valid": True},
        )
    assert (tmp_path / "a.failed" / "partial.bin").read_bytes() == b"partial"
    with pytest.raises(StageAttemptExistsError):
        run_fresh_stage_attempt(
            attempts_directory=tmp_path,
            attempt_id="a",
            input_bindings={"input": "sha256:x"},
            run_stage=fail,
            validate_stage=lambda _: {"valid": True},
        )


def test_only_validated_attempt_is_published(tmp_path: Path):
    def succeed(stage_dir):
        output = stage_dir / "out.bin"
        output.write_bytes(b"complete")
        return output

    output = run_fresh_stage_attempt(
        attempts_directory=tmp_path,
        attempt_id="b",
        input_bindings={"input": "sha256:x"},
        run_stage=succeed,
        validate_stage=lambda p: {"valid": p.read_bytes() == b"complete"},
    )
    assert output == tmp_path / "b.complete" / "out.bin"
    assert not (tmp_path / "b.partial").exists()
    assert (tmp_path / "b.complete" / "COMPLETE.json").exists()
