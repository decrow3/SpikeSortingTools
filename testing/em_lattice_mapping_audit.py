#!/usr/bin/env python3
"""Write EM's geometry-only exact-versus-nearest mapping audit."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from npx_preprocessing.motion import write_mapping_audit


def shifts_from_csv(path: Path, column: str) -> np.ndarray:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"{path} does not contain column {column!r}")
        return np.asarray([float(row[column]) for row in reader], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-npy", type=Path, required=True)
    parser.add_argument("--target-geometry-npy", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--shifts-npy", type=Path)
    source.add_argument("--knots-csv", type=Path)
    parser.add_argument("--shift-column", default="q_um")
    parser.add_argument("--direction", choices=("x", "y"), default="y")
    parser.add_argument("--atol-um", type=float, default=1e-6)
    parser.add_argument("--include-spikeinterface", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    geometry = np.load(args.geometry_npy, allow_pickle=False)
    target_geometry = (
        None
        if args.target_geometry_npy is None
        else np.load(args.target_geometry_npy, allow_pickle=False)
    )
    if args.shifts_npy is not None:
        shifts = np.load(args.shifts_npy, allow_pickle=False)
    else:
        shifts = shifts_from_csv(args.knots_csv, args.shift_column)
    summary = write_mapping_audit(
        args.output_dir,
        geometry,
        shifts,
        target_locations=target_geometry,
        direction=args.direction,
        atol_um=args.atol_um,
        include_spikeinterface=args.include_spikeinterface,
    )
    print(args.output_dir / "SUMMARY.json")
    print(
        "unsupported targets nearest would substitute:",
        summary["unsupported_targets_nearest_would_substitute"],
    )


if __name__ == "__main__":
    main()
