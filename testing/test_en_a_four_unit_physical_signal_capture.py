from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from testing.en_a_four_unit_physical_signal_capture import (
    expected_arithmetic,
    load_selections,
    window_bounds,
)


RECEIPT = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_a_four_unit_approval_metadata_20261001_v3_h5/SELECTION_RECEIPT.csv")


class FourUnitCaptureWindowTests(unittest.TestCase):
    def test_all_eight_actual_pair_lags_define_both_bounds(self) -> None:
        rows = load_selections(RECEIPT)
        pairs = [row for row in rows if row["role"] == "short_pair"]
        self.assertEqual([int(row["lag_samples"]) for row in pairs], [29, 25, 23, 9, 7, 12, 11, 11])
        for row in pairs:
            lag = int(row["lag_samples"])
            bounds = window_bounds(row)
            self.assertEqual(bounds["input_start"], int(row["frame_1"]) - 512)
            self.assertEqual(bounds["input_stop"], int(row["frame_2"]) + 513)
            self.assertEqual(bounds["input_stop"] - bounds["input_start"], 1025 + lag)
            self.assertEqual(bounds["retained_start"], int(row["frame_1"]) - 60)
            self.assertEqual(bounds["retained_stop"], int(row["frame_2"]) + 61)
            self.assertEqual(bounds["retained_stop"] - bounds["retained_start"], 121 + lag)

    def test_exact_totals_are_below_unchanged_caps(self) -> None:
        arithmetic = expected_arithmetic(load_selections(RECEIPT))
        self.assertEqual(arithmetic["planned_base_input_bytes"], 31_683_072)
        self.assertEqual(arithmetic["planned_with_duplicate_input_bytes"], 32_492_544)
        self.assertEqual(arithmetic["planned_retained_int16_bytes"], 3_912_192)
        self.assertEqual(arithmetic["cap_with_one_max_pair_duplicate_bytes"], 32_653_824)
        self.assertEqual(arithmetic["raw_read_calls_max"], 41)

    def test_inconsistent_frame_2_or_lag_fails_closed(self) -> None:
        with RECEIPT.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
            fields = list(rows[0])
        pair = next(row for row in rows if row["role"] == "short_pair" and int(row["lag_samples"]) != 29)
        pair["frame_2"] = str(int(pair["frame_1"]) + 29)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "receipt.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(RuntimeError, "endpoints/lag inconsistent"):
                load_selections(path)
        direct = dict(pair)
        direct["frame_2"] = str(int(direct["frame_1"]) + int(direct["lag_samples"]) + 1)
        with self.assertRaisesRegex(RuntimeError, "endpoints/lag inconsistent"):
            window_bounds(direct)


if __name__ == "__main__":
    unittest.main()
