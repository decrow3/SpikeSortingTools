#!/usr/bin/env python3
"""Compare two saved DARTsort pipeline state dictionaries exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left = torch.load(args.left, map_location="cpu", weights_only=False)
    right = torch.load(args.right, map_location="cpu", weights_only=False)
    rows = []
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key), right.get(key)
        if torch.is_tensor(a) and torch.is_tensor(b):
            same_shape = a.shape == b.shape
            exact = same_shape and torch.equal(a, b)
            rows.append({
                "key": key,
                "kind": "tensor",
                "shape_left": list(a.shape),
                "shape_right": list(b.shape),
                "exact": bool(exact),
                "max_abs_difference": float((a - b).abs().max()) if same_shape and a.numel() else 0.0,
            })
        else:
            rows.append({
                "key": key,
                "kind": "metadata",
                "exact": repr(a) == repr(b),
            })
    result = {
        "left": {"path": str(args.left), "sha256": sha256(args.left)},
        "right": {"path": str(args.right), "sha256": sha256(args.right)},
        "all_entries_exact": all(row["exact"] for row in rows),
        "differing_entries": [row for row in rows if not row["exact"]],
        "tpca_entries_exact": all(
            row["exact"] for row in rows if row["key"].startswith("transformers.1.")
        ),
        "interpretation": (
            "TPCA/common waveform basis can be exact even when adaptive localization "
            "weights differ; final-refinement differences are then not a pure CD-only contrast."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
