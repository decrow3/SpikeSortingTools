#!/usr/bin/env python3
"""Independent outer-path byte sums for the H5 v3 finalizer."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile

import pytest


PACKET = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_no_mask_ref_repeat_launcher_r2_finalization_repair_20261002_v3_h5")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def independent_sum(contract) -> int:
    total = 0
    for key in ("root", "logs", "attempt_evidence"):
        root = Path(contract["planned_paths"][key])
        if not root.exists():
            continue
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            base = Path(directory)
            if any((base / name).is_symlink() for name in dirnames):
                raise AssertionError("symlinked directory in counted tree")
            for name in filenames:
                path = base / name
                if path.is_symlink() or not path.is_file():
                    raise AssertionError("non-regular counted member")
                total += os.lstat(path).st_size
    return total


def main() -> None:
    pair = load("pair", PACKET / "source/en_first_medium_trained_pair_launch.py")
    tests = load("candidate_tests", PACKET / "tests/test_pair_launcher.py")
    results = {}

    with tempfile.TemporaryDirectory() as raw:
        monkeypatch = pytest.MonkeyPatch()
        contract, kwargs = tests.outer_fixture(Path(raw), monkeypatch)
        result = pair.run_single_attempt(**kwargs)
        actual = independent_sum(contract)
        assert actual == result["aggregate_persistent_output_bytes"]
        assert actual < contract["resources"]["persistent_output_bytes_max"]
        assert (Path(contract["planned_paths"]["root"]) / "COMPLETE.json").is_file()
        results["below"] = {"actual": actual, "cap": contract["resources"]["persistent_output_bytes_max"]}
        monkeypatch.undo()

    with tempfile.TemporaryDirectory() as raw:
        monkeypatch = pytest.MonkeyPatch()
        contract, kwargs = tests.outer_fixture(
            Path(raw), monkeypatch, pre=tests.exact_cap_padding, fixture_live_at_total=True
        )
        result = pair.run_single_attempt(**kwargs)
        actual = independent_sum(contract)
        assert actual == result["aggregate_persistent_output_bytes"] == contract["resources"]["persistent_output_bytes_max"]
        results["equal"] = {"actual": actual, "cap": contract["resources"]["persistent_output_bytes_max"]}
        monkeypatch.undo()

    with tempfile.TemporaryDirectory() as raw:
        monkeypatch = pytest.MonkeyPatch()

        def late(contract):
            path = Path(contract["planned_paths"]["logs"]) / "late-independent.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

        contract, kwargs = tests.outer_fixture(
            Path(raw), monkeypatch, pre=tests.exact_cap_padding,
            late=late, fixture_live_at_total=True,
        )
        try:
            pair.run_single_attempt(**kwargs)
        except pair.OutputCapExceeded:
            pass
        else:
            raise AssertionError("above-cap outer path returned false success")
        root = Path(contract["planned_paths"]["root"])
        attempt = Path(contract["planned_paths"]["attempt_evidence"])
        actual = independent_sum(contract)
        assert actual > contract["resources"]["persistent_output_bytes_max"]
        assert not (root / "COMPLETE.json").exists()
        assert (root / "COMPLETE.pending.json").is_file()
        assert json.loads((attempt / "FAILURE.json").read_text())["completion_written"] is False
        results["above"] = {"actual": actual, "cap": contract["resources"]["persistent_output_bytes_max"], "false_complete": False}
        monkeypatch.undo()

    print(json.dumps({"status": "PASS", "results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
