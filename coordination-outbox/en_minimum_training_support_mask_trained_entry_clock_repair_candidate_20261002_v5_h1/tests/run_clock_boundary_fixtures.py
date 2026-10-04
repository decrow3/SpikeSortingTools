#!/usr/bin/env python3
"""Saved-array-only known answers for the trained terminal clock validator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from npx_preprocessing.motion import kilosort_support_mask_trained as trained
from testing.en_minimum_training_support_mask_trained_fixture import build_fixture, invoke


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = {
        "schema": "trained-terminal-clock-boundary-fixtures-v1",
        "saved_arrays_only": True,
        "voltage_sort_training_detection_accessed": False,
    }
    with tempfile.TemporaryDirectory(prefix="trained-clock-boundary-") as temporary:
        root = Path(temporary) / "fixture"
        path, digest = build_fixture(root, "run", crop_origin=0)
        completed = invoke(path, digest, root, "success")
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        results = root / "run/native_results"
        snapshot = root / "run/pre_extraction_snapshot"
        zero_config = json.loads(path.read_text())
        receipt["zero_origin"] = (
            trained.validate_saved_artifacts(results, snapshot, zero_config)["crop_origin_samples"] == 0
        )
        origin = 208_498_882
        config = json.loads(path.read_text())
        config["crop"] = {"imin": origin, "imax": origin + 80}
        config["output_clock"] = {
            "spike_times_frame": "global_acquisition_samples",
            "full_st_frame": "crop_local_samples",
            "crop_origin_samples": origin,
            "crop_local_bounds_half_open": [0, 80],
            "global_bounds_half_open": [origin, origin + 80],
            "exact_relation": "spike_times.npy == full_st.npy[kept_spikes,0] + crop.imin",
        }
        local = np.array([1, 5, 9], dtype=np.int64)
        np.save(results / "spike_times.npy", local + origin, allow_pickle=False)
        receipt["correct_nonzero_origin"] = (
            trained.validate_saved_artifacts(results, snapshot, config)["crop_origin_samples"] == origin
        )

        def rejected(values, message):
            np.save(results / "spike_times.npy", values, allow_pickle=False)
            try:
                trained.validate_saved_artifacts(results, snapshot, config)
            except ValueError as error:
                return message in str(error)
            return False

        receipt["wrong_offset_rejected"] = rejected(local, "outside the half-open crop")
        receipt["off_by_one_rejected"] = rejected(local + origin + 1, "global ancestry differs")
        np.save(results / "spike_times.npy", local + origin, allow_pickle=False)
        full_st = np.load(results / "full_st.npy", allow_pickle=False)
        full_st[3, 0] = 80
        np.save(results / "full_st.npy", full_st, allow_pickle=False)
        try:
            trained.validate_saved_artifacts(results, snapshot, config)
        except ValueError as error:
            receipt["half_open_bound_rejected"] = "crop-local time is outside" in str(error)
        else:
            receipt["half_open_bound_rejected"] = False
        overflow = json.loads(path.read_text())
        overflow["crop"]["imax"] = int(np.iinfo(np.int64).max) + 1
        try:
            trained.prevalidate_trained_structure(overflow)
        except ValueError as error:
            receipt["int64_overflow_contract_rejected"] = "invalid for signed int64" in str(error)
        else:
            receipt["int64_overflow_contract_rejected"] = False
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    required = [value for key, value in receipt.items() if key not in {
        "schema", "saved_arrays_only", "voltage_sort_training_detection_accessed"
    }]
    return 0 if all(required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
