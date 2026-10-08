#!/usr/bin/env python3
"""Zero-voltage known-answer test for the covariance launcher's atomic reservation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from npx_preprocessing.motion.kilosort_support_mask_candidate import (
    _reserve_fresh_output_namespace,
)


ROOT = Path(
    "/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/"
    "testing/outputs/en_minimum_training_support_mask_covariance_freshness_dummy_20261001_v1"
)


def atomic_json(path: Path, value: dict) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def main() -> None:
    config = {
        "smoke": {
            "run_root": str(ROOT),
            "launch_evidence_dir": str(ROOT / "launch_evidence"),
        },
        "native_invocation": {"results_dir": str(ROOT / "native_results")},
    }
    evidence, results = _reserve_fresh_output_namespace(config)
    reservation = json.loads((evidence / "namespace_reservation.json").read_text())
    refused = False
    refusal_type = None
    try:
        _reserve_fresh_output_namespace(config)
    except FileExistsError as exc:
        refused = True
        refusal_type = type(exc).__name__
    if not refused:
        raise RuntimeError("second reservation did not fail closed")
    atomic_json(
        evidence / "DUMMY_TEST.json",
        {
            "status": "pass",
            "recording_paths_present_in_config": False,
            "recording_reads": 0,
            "first_atomic_reservation": reservation,
            "first_results_directory_exists": results.is_dir(),
            "second_reservation_refused": refused,
            "second_refusal_type": refusal_type,
            "authoritative_guard": "run_root.mkdir(parents=False, exist_ok=False)",
        },
    )


if __name__ == "__main__":
    main()
