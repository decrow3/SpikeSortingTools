#!/usr/bin/env python
"""Build AE's canonical Luke0804 imec1 censor mask.

This is intentionally a small, dependency-light program so the identical
inputs produce a byte-identical CSV on huklaban1 and huklaban5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_FIELD_SHA256 = "dc6c2a80a8a0a4ac6709720173993e084765e0b3a0cc23906ad955bc6de102e7"
EXPECTED_CATALOGUE_SHA256 = "46463cda1348c0e0b0c39ad22ae9c71bc46f7c7347797c1c10588944cfc52df1"
DT_S = 0.25
MEDIAN_SAMPLES = 241
DILATION_BINS = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dilate(mask: np.ndarray, bins: int = DILATION_BINS) -> np.ndarray:
    result = np.asarray(mask, bool).copy()
    source = result.copy()
    for offset in range(1, bins + 1):
        result[offset:] |= source[:-offset]
        result[:-offset] |= source[offset:]
    return result


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    stops = np.flatnonzero(mask & np.r_[~mask[1:], True]) + 1
    return list(zip(starts.tolist(), stops.tolist()))


def build(field_path: Path, catalogue_path: Path):
    field_hash = sha256(field_path)
    catalogue_hash = sha256(catalogue_path)
    if field_hash != EXPECTED_FIELD_SHA256:
        raise RuntimeError(f"Unexpected field SHA-256: {field_hash}")
    if catalogue_hash != EXPECTED_CATALOGUE_SHA256:
        raise RuntimeError(f"Unexpected catalogue SHA-256: {catalogue_hash}")

    with np.load(field_path, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], float)
        displacement = np.asarray(data["displacement_um"], float)
    if displacement.shape != (len(time_s), 1):
        raise RuntimeError(f"Expected displacement [T,1], got {displacement.shape}")
    if not np.allclose(np.diff(time_s), DT_S, rtol=0, atol=1e-12):
        raise RuntimeError("Field is not on AE's 0.25 s grid")
    d = displacement[:, 0]
    # pandas implements the stipulated centred window with truncated edges.
    base = pd.Series(d).rolling(MEDIAN_SAMPLES, center=True, min_periods=1).median().to_numpy()
    centred = d - base

    source_raw = {name: np.zeros(len(time_s), bool) for name in ("A", "U", "E")}
    catalogue = pd.read_csv(catalogue_path)
    for row in catalogue.itertuples(index=False):
        inside = (time_s >= float(row.start_s)) & (time_s < float(row.end_s))
        if row.status == "accepted":
            source_raw["A"] |= inside
        elif inside.any() and np.nanmax(np.abs(centred[inside])) > 60.0:
            source_raw["U"] |= inside
    source_raw["E"] = np.abs(centred) > 50.0

    # Dilating each source separately is equivalent to dilating their union,
    # while retaining unambiguous provenance for merged output runs.
    source_dilated = {name: dilate(mask) for name, mask in source_raw.items()}
    mask = np.logical_or.reduce(list(source_dilated.values()))
    rows = []
    for start, stop in runs(mask):
        contributors = [name for name in ("A", "U", "E") if source_dilated[name][start:stop].any()]
        rows.append((time_s[start], time_s[stop - 1] + DT_S, "+".join(contributors)))
    return time_s, centred, source_raw, source_dilated, mask, rows, field_hash, catalogue_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    time_s, centred, raw, expanded, mask, rows, field_hash, catalogue_hash = build(
        args.field, args.catalogue
    )
    csv_path = args.output / "censor_mask_v1.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("start_s,end_s,source\n")
        for start, stop, source in rows:
            stream.write(f"{start:.3f},{stop:.3f},{source}\n")

    receipt = {
        "schema": "luke0804-imec1-canonical-censor-mask-ae-v1",
        "input_field": str(args.field),
        "input_field_sha256": field_hash,
        "input_catalogue": str(args.catalogue),
        "input_catalogue_sha256": catalogue_hash,
        "csv": str(csv_path),
        "csv_sha256": sha256(csv_path),
        "time_grid_s": DT_S,
        "running_median_samples": MEDIAN_SAMPLES,
        "dilation_bins_each_side": DILATION_BINS,
        "n_intervals": len(rows),
        "censored_bins": int(mask.sum()),
        "censored_seconds": float(mask.sum() * DT_S),
        "grid_seconds": float(len(mask) * DT_S),
        "censored_fraction": float(mask.mean()),
        "per_source_raw_seconds": {name: float(value.sum() * DT_S) for name, value in raw.items()},
        "per_source_dilated_seconds": {name: float(value.sum() * DT_S) for name, value in expanded.items()},
        "source_seconds_overlap_note": "Per-source totals overlap and therefore need not sum to union seconds.",
        "centred_field_max_abs_um": float(np.nanmax(np.abs(centred))),
    }
    receipt_path = args.output / "censor_mask_v1_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output / "censor_mask_v1_grid.npz", time_s=time_s, mask=mask,
                        centred_displacement_um=centred,
                        source_A=raw["A"], source_U=raw["U"], source_E=raw["E"])
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
