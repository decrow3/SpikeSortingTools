"""Candidate-3 whole-stage restart guard.

This deliberately does not implement within-stage resume.  Each attempt gets a
fresh directory.  A failed or interrupted attempt is preserved, and only a
validated attempt is atomically renamed to ``*.complete`` for downstream use.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Mapping


class StageAttemptExistsError(RuntimeError):
    pass


def _write_json(path: Path, value: Mapping) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with path.open("rb") as f:
        os.fsync(f.fileno())


def run_fresh_stage_attempt(
    *,
    attempts_directory: str | Path,
    attempt_id: str,
    input_bindings: Mapping,
    run_stage: Callable[[Path], Path],
    validate_stage: Callable[[Path], Mapping],
) -> Path:
    """Run one non-resuming stage attempt and publish only after validation.

    ``run_stage`` must create outputs under its provided fresh directory.  The
    caller is responsible for binding unchanged upstream inputs in
    ``input_bindings``.  Existing partial, failed, or complete attempts are
    never reused or overwritten.
    """

    # DARTsort normalizes its returned output path to absolute.  Normalize our
    # containment root the same way before comparing paths.
    attempts_directory = Path(attempts_directory).resolve()
    attempts_directory.mkdir(parents=True, exist_ok=True)
    partial = attempts_directory / f"{attempt_id}.partial"
    failed = attempts_directory / f"{attempt_id}.failed"
    complete = attempts_directory / f"{attempt_id}.complete"
    conflicts = [p for p in (partial, failed, complete) if p.exists()]
    if conflicts:
        raise StageAttemptExistsError(
            "Refusing to reuse an existing stage attempt: "
            + ", ".join(map(str, conflicts))
        )

    partial.mkdir()
    _write_json(
        partial / "ATTEMPT.json",
        {
            "attempt_id": attempt_id,
            "input_bindings": dict(input_bindings),
            "resume_policy": "fresh_whole_stage_only",
            "status": "running",
        },
    )
    try:
        stage_output = Path(run_stage(partial))
        try:
            stage_output.relative_to(partial)
        except ValueError as exc:
            raise ValueError("Stage output escaped the fresh attempt directory") from exc
        validation = dict(validate_stage(stage_output))
        if not validation.get("valid", False):
            raise RuntimeError(f"Stage validation failed: {validation}")
        _write_json(
            partial / "COMPLETE.json",
            {
                "attempt_id": attempt_id,
                "input_bindings": dict(input_bindings),
                "stage_output": str(stage_output.relative_to(partial)),
                "status": "complete",
                "validation": validation,
            },
        )
        partial.rename(complete)
        return complete / stage_output.relative_to(partial)
    except BaseException as exc:
        _write_json(
            partial / "FAILED.json",
            {
                "attempt_id": attempt_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "input_bindings": dict(input_bindings),
                "status": "failed",
            },
        )
        partial.rename(failed)
        raise
