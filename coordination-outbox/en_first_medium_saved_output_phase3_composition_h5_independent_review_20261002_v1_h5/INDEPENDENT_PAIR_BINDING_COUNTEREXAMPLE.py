#!/usr/bin/env python3
"""Known-answer: unrelated saved outputs pass with a structurally thin pair receipt."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


PACKET = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "en_first_medium_saved_output_phase3_composition_candidate_20261002_v2_h1"
)
sys.path.insert(0, str(PACKET / "source"))
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/h5_phase3_binding_counterexample_numba")


def load_candidate_tests():
    path = PACKET / "tests/test_phase3_composition.py"
    spec = importlib.util.spec_from_file_location("candidate_phase3_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    tests = load_candidate_tests()
    with tempfile.TemporaryDirectory(prefix="h5-phase3-binding-") as temporary:
        root = Path(temporary)
        contract, request_path, output = tests.build_case(root, 20)
        request = json.loads(request_path.read_text())
        pair_complete_path = Path(request["pair_receipts"]["pair_complete"])
        pair_complete = json.loads(pair_complete_path.read_text())
        candidate_paths = {
            arm["arm_id"]: str(Path(arm["curated_output"]).resolve())
            for arm in request["arms"]
            if arm["arm_id"] in {"repaired_B384", "REF384_repeat"}
        }
        pair_namespace = pair_complete_path.parent.resolve()
        candidates_outside_pair_namespace = all(
            pair_namespace not in Path(path).parents for path in candidate_paths.values()
        )
        result = tests.managed(contract, request_path, output)
        report = {
            "candidate_manifest_sha256": "48b32b79f161b94d9c3a09d1bb8be1c5b2366e2e63cdf398333b4c27a7f090a7",
            "managed_returncode": result.returncode,
            "pair_complete_has_execution_states": "execution_states" in pair_complete,
            "candidate_paths": candidate_paths,
            "pair_namespace": str(pair_namespace),
            "candidates_outside_pair_namespace": candidates_outside_pair_namespace,
            "authoritative_complete_written": (output / "COMPLETE.json").is_file(),
            "phase3_report_written": (output / "PHASE3_COMPACT_REPORT.json").is_file(),
            "finding_reproduced": bool(
                result.returncode == 0
                and "execution_states" not in pair_complete
                and candidates_outside_pair_namespace
                and (output / "COMPLETE.json").is_file()
            ),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["finding_reproduced"]:
            raise AssertionError(result.stderr or "pair-binding counterexample did not reproduce")


if __name__ == "__main__":
    main()
