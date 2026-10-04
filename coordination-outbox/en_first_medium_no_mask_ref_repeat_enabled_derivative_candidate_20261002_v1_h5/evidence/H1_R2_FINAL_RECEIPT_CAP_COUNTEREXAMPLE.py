#!/usr/bin/env python3
"""Independent known-answer fixture for terminal receipt cap accounting."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile

import en_first_medium_trained_pair_launch as pair


PACKET = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "en_first_medium_no_mask_ref_repeat_launcher_targeted_repair_20261002_v2_h5"
)


def load_candidate_tests():
    spec = importlib.util.spec_from_file_location(
        "candidate_tests", PACKET / "tests/test_pair_launcher.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def one_run(test_helpers, cap: int):
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    contract_path, _, service_sha = test_helpers.make_contract(root)
    contract = pair.validate_contract(contract_path, test_helpers.sha(contract_path))
    contract["resources"] = dict(contract["resources"])
    contract["resources"]["persistent_output_bytes_max"] = cap
    os.environ[pair.PERSISTENT_MARKER] = "1"
    pair.consume_start_claim(
        contract["planned_paths"]["attempt_evidence"],
        contract_path=contract_path,
        expected_contract_sha256=test_helpers.sha(contract_path),
        expected_service_sha256=service_sha,
    )
    pair_root = Path(contract["planned_paths"]["root"])

    def ref(_):
        return test_helpers.write_complete(pair_root / "REF384_repeat/COMPLETE.json")

    def repaired(_):
        return test_helpers.write_complete(pair_root / "repaired_B384/COMPLETE.json")

    result = pair.execute_pair(
        contract,
        ref_executor=ref,
        repaired_executor=repaired,
        resource_probe=lambda _: None,
        voltage_probe=test_helpers.pass_voltage,
    )
    actual = pair.aggregate_output_bytes(contract)
    return temporary, int(result["aggregate_persistent_output_bytes"]), actual


def main() -> None:
    helpers = load_candidate_tests()
    probe, measured, actual = one_run(helpers, 100_000)
    cap = measured + 64
    replay, measured_replay, actual_replay = one_run(helpers, cap)
    try:
        print(
            {
                "configured_cap": cap,
                "reported_before_final_receipts": measured_replay,
                "actual_after_successful_return": actual_replay,
                "successful_return_above_cap": actual_replay > cap,
                "probe_unaccounted_terminal_bytes": actual - measured,
            }
        )
        if actual_replay <= cap:
            raise AssertionError("fixture did not reproduce terminal receipt cap escape")
    finally:
        probe.cleanup()
        replay.cleanup()


if __name__ == "__main__":
    main()
