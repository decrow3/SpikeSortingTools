#!/usr/bin/env python3
"""Read-only audit of the accepted post-run composition boundary."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path("/mnt/NPX/Luke/DARTsort_motion_experiments")
PACKET = ROOT / "en_first_medium_saved_output_qualification_contract_targeted_repair_20261002_v2_h1"
EXPECTED = {
    "source/testing/first_medium_saved_output_qualification.py": "35ab0e5886ed1ba39dbde475c37794e9b053c10175f97d0d88e25c35f954bd6e",
    "source/testing/first_medium_measurement_redesign.py": "779b3b82e4328f92f6d50cb0a411153e03d755172d9493ccda937a686f42d0d9",
    "source/testing/first_medium_evaluator.py": "da3d5d5af8227f6ef70747dca32b3e2c52206230b5494d7bc9015cba24fcdf35",
    "source/testing/sort_comparison.py": "174a4bb6d621489260927dc0bbc06558e408564158a44657ecdd306fa07db5da",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    observed = {name: sha(PACKET / name) for name in EXPECTED}
    if observed != EXPECTED:
        raise SystemExit("accepted implementation hashes changed")
    source = (PACKET / "source/testing/first_medium_saved_output_qualification.py").read_text()
    tree = ast.parse(source)
    phases = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PHASES" for t in node.targets):
            phases = ast.literal_eval(node.value)
    if phases != {"phase1_controls", "phase2_existing_b"}:
        raise SystemExit(f"unexpected accepted phases: {phases!r}")
    if 'DeltaR": {"status": "UNAVAILABLE"' not in source:
        raise SystemExit("accepted Phase-2 unavailable-DeltaR guard changed")
    template = json.loads(Path("POST_RUN_REQUEST.template.json").read_text())
    if [row["arm_id"] for row in template["arms"]] != ["REF384", "existing_corrected_B384", "repaired_B384", "REF384_repeat"]:
        raise SystemExit("prepared arm order changed")
    if not any("PROSPECTIVE" in json.dumps(row) for row in template["arms"]):
        raise SystemExit("template no longer fails closed on prospective bindings")
    print(json.dumps({"status": "PASS", "accepted_phases": sorted(phases), "phase3_present": False, "hashes": observed}, sort_keys=True))


if __name__ == "__main__":
    main()
