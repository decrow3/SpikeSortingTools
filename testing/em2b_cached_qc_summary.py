#!/usr/bin/env python3
"""Summarize compact sorting-only QC tables from completed EM.2b arms."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


REQUIRED = {
    "unit_id",
    "spike_count",
    "presence_ratio",
    "raw_adjacent_isi_lt_refractory_fraction",
    "exact_duplicate_sample_count",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_arm(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("arm must be NAME=QC_CSV") from exc
    if not name or not path:
        raise argparse.ArgumentTypeError("arm must be NAME=QC_CSV")
    return name, Path(path)


def summarize(table: pd.DataFrame) -> dict[str, int | float]:
    missing = REQUIRED - set(table)
    if missing:
        raise RuntimeError(f"missing QC columns: {sorted(missing)}")
    if table.unit_id.duplicated().any():
        raise RuntimeError("unit_id is not unique")
    if (table.spike_count < 0).any() or not table.presence_ratio.between(0, 1).all():
        raise RuntimeError("invalid QC counts or presence ratios")
    isi = table.raw_adjacent_isi_lt_refractory_fraction
    if not isi.between(0, 1).all():
        raise RuntimeError("invalid raw refractory fractions")
    return {
        "units": int(len(table)),
        "accepted_events": int(table.spike_count.sum()),
        "median_events_per_unit": float(table.spike_count.median()),
        "median_presence_ratio": float(table.presence_ratio.median()),
        "full_presence_units": int(table.presence_ratio.eq(1).sum()),
        "presence_ge_five_sixths_units": int(table.presence_ratio.ge(5 / 6).sum()),
        "median_raw_refractory_fraction": float(isi.median()),
        "p90_raw_refractory_fraction": float(isi.quantile(0.9)),
        "units_raw_refractory_fraction_gt_0p01": int(isi.gt(0.01).sum()),
        "exact_duplicate_sample_units": int(table.exact_duplicate_sample_count.gt(0).sum()),
        "exact_duplicate_sample_count": int(table.exact_duplicate_sample_count.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True, type=parse_arm)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    rows, inputs = [], {}
    for name, path in args.arm:
        path = path.resolve()
        if name in inputs:
            raise RuntimeError(f"duplicate arm: {name}")
        table = pd.read_csv(path)
        row = {"arm": name, **summarize(table)}
        rows.append(row)
        inputs[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    pd.DataFrame(rows).to_csv(args.output / "SUMMARY.csv", index=False)
    result = {
        "schema": "em2b-cached-sorting-qc-summary-v1",
        "status": "complete",
        "inputs": inputs,
        "arms": rows,
        "interpretation": (
            "Arm-local descriptive sorting QC only; no unit matching, biological identity, "
            "purity, or promotion claim."
        ),
    }
    (args.output / "RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    products = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(args.output.iterdir())
    ]
    (args.output / "MANIFEST.json").write_text(
        json.dumps({"products": products}, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "COMPLETE.json").write_text(
        json.dumps(
            {"status": "complete", "manifest_sha256": sha256(args.output / "MANIFEST.json")},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
