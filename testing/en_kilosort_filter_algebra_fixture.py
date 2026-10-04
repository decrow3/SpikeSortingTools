#!/usr/bin/env python3
"""Prospective CPU known-answer checks for installed KS4 filter algebra."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

import kilosort
from kilosort.io import BinaryFiltered


EXPECTED_VERSION = "4.0.27"
EXPECTED_IO_SHA256 = "767b76a07647122b45048ba12cdaf49185f87f3fa33334081d4af099825414bd"
ATOL = 1e-6
OUT = (
    Path(__file__).resolve().parent
    / "outputs/en_executed_provenance_mechanism_audit_20260930_v1/"
    "filter_algebra_fixture.json"
)


def independent_expected(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=1, keepdims=True)
    # torch.median on an even channel count selects the lower of the two middle
    # order statistics. np.partition with kth=3 reproduces that for 8 rows.
    lower_median = np.partition(centered, kth=(centered.shape[0] - 1) // 2, axis=0)[
        (centered.shape[0] - 1) // 2
    ]
    return w @ (centered - lower_median[None, :])


def installed_filter(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    f = object.__new__(BinaryFiltered)
    f.chan_map = torch.arange(x.shape[0], dtype=torch.int64)
    f.invert_sign = False
    f.do_CAR = True
    f.hp_filter = None
    f.artifact_threshold = np.inf
    f.whiten_mat = torch.as_tensor(w, dtype=torch.float32)
    f.dshift = None
    return f.filter(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()


def summarize(name: str, x: np.ndarray, w: np.ndarray) -> dict:
    observed = installed_filter(x, w)
    expected = independent_expected(x, w)
    error = np.abs(observed - expected)
    if not np.allclose(observed, expected, rtol=0, atol=ATOL):
        raise AssertionError(f"{name}: installed output disagrees with independent answer")
    row_l2 = np.linalg.norm(observed, axis=1)
    return {
        "name": name,
        "max_abs_error": float(error.max(initial=0)),
        "max_abs_output": float(np.abs(observed).max(initial=0)),
        "nonzero_output_rows_gt_1e-7": np.flatnonzero(row_l2 > 1e-7).tolist(),
        "row_l2": row_l2.tolist(),
    }


def main() -> None:
    io_path = Path(inspect.getfile(kilosort)).parent / "io.py"
    io_hash = hashlib.sha256(io_path.read_bytes()).hexdigest()
    if kilosort.__version__ != EXPECTED_VERSION:
        raise AssertionError(f"unexpected Kilosort version {kilosort.__version__}")
    if io_hash != EXPECTED_IO_SHA256:
        raise AssertionError(f"unexpected io.py hash {io_hash}")

    common = np.asarray(
        [-1, 1, -2, 2, -3, 3, -4, 4, -4, 4, -3, 3, -2, 2, -1, 1],
        dtype=np.float32,
    )
    zero = np.zeros((8, 16), dtype=np.float32)
    full = np.tile(common, (8, 1))
    mask_01 = full.copy()
    mask_01[[0, 1]] = 0
    mask_45 = full.copy()
    mask_45[[4, 5]] = 0

    identity = np.eye(8, dtype=np.float32)
    wmix = identity.copy()
    for i in range(8):
        wmix[i, (i - 1) % 8] = 0.25

    cases = []
    for matrix_name, w in (("identity", identity), ("wmix", wmix)):
        cases.extend(
            [
                summarize(f"zero__{matrix_name}", zero, w),
                summarize(f"common_full__{matrix_name}", full, w),
                summarize(f"mask_01__{matrix_name}", mask_01, w),
                summarize(f"mask_45__{matrix_name}", mask_45, w),
            ]
        )

    by_name = {row["name"]: row for row in cases}
    for matrix_name in ("identity", "wmix"):
        if by_name[f"zero__{matrix_name}"]["max_abs_output"] > ATOL:
            raise AssertionError("zero control produced nonzero output")
        if by_name[f"common_full__{matrix_name}"]["max_abs_output"] > ATOL:
            raise AssertionError("full common-mode control was not removed")
    if by_name["mask_01__identity"]["nonzero_output_rows_gt_1e-7"] != [0, 1]:
        raise AssertionError("identity residual did not follow mask [0,1]")
    if by_name["mask_45__identity"]["nonzero_output_rows_gt_1e-7"] != [4, 5]:
        raise AssertionError("identity residual did not follow mask [4,5]")

    result = {
        "schema": "en-kilosort-filter-algebra-known-answer-v1",
        "scope": (
            "prospective algebra only; high-pass disabled; no voltage, waveform, "
            "noise model, learned template, detection threshold, or event claim"
        ),
        "kilosort_version": kilosort.__version__,
        "io_path": str(io_path),
        "io_sha256": io_hash,
        "device": "cpu",
        "dtype": "float32",
        "shape_channels_samples": [8, 16],
        "atol": ATOL,
        "rtol": 0,
        "cases": cases,
        "verdict": "all frozen installed-vs-independent algebra controls passed",
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
