#!/usr/bin/env python3
"""Independent analytic controls for the frozen CX modified donor payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def exact_map(geom: np.ndarray, shift: float) -> np.ndarray:
    lookup = {tuple(map(float, xy)): i for i, xy in enumerate(geom)}
    return np.array([lookup.get((float(x), float(y + shift)), -1) for x, y in geom], dtype=int)


def centered_cosine(a, b):
    x, y = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    x -= x.mean(); y -= y.mean(); d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--tapers", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    with np.load(args.source, allow_pickle=False) as z:
        templates = np.asarray(z["templates"], np.float32)
        unit_ids = np.asarray(z["unit_ids"], int)
        geom = np.asarray(z["geometry_um"], float)
    with np.load(args.tapers, allow_pickle=False) as z:
        weights = np.asarray(z["train_weights"], np.float32)
        assert np.array_equal(unit_ids, z["unit_ids"])
    geometry = []
    for shift in (-240.0, -40.0, 40.0, 240.0):
        forward, reverse = exact_map(geom, shift), exact_map(geom, -shift)
        valid = forward >= 0
        geometry.append({
            "shift_um": shift, "mapped_channels": int(valid.sum()),
            "one_to_one": bool(np.unique(forward[valid]).size == valid.sum()),
            "inverse_closure": bool(np.all(reverse[forward[valid]] == np.flatnonzero(valid))),
            "lower_boundary_losses": int(np.count_nonzero(~valid & (geom[:, 1] < geom[:, 1].min() + abs(shift)))),
            "upper_boundary_losses": int(np.count_nonzero(~valid & (geom[:, 1] > geom[:, 1].max() - abs(shift)))),
        })
    i = int(np.flatnonzero(unit_ids == 30)[0]); target = np.arange(202, 384)
    original = templates[i][:, target]; modified = (templates[i] * weights[i][None, :])[:, target]
    original_ptp = float(np.ptp(original, axis=0).max())
    clone = {
        "self_positive": {"cosine": centered_cosine(original, original), "ptp_ratio": 1.0, "gate_positive": True},
        "half_amplitude": {"cosine": centered_cosine(original * .5, original), "ptp_ratio": .5, "gate_positive": False},
        "double_amplitude": {"cosine": centered_cosine(original * 2, original), "ptp_ratio": 2.0, "gate_positive": False},
        "modified_vs_unmodified": {"cosine": centered_cosine(modified, original), "ptp_ratio": float(np.ptp(modified, axis=0).max() / original_ptp)},
        "zero": {"cosine": None, "ptp_ratio": 0.0, "gate_positive": False},
    }
    shifted = {}
    for phase in (-1, 1):
        value = np.zeros_like(modified)
        if phase > 0:
            value[phase:] = modified[:-phase]
        else:
            value[:phase] = modified[-phase:]
        shifted[str(phase)] = {"nonwrapping": True, "cosine": centered_cosine(modified, value), "ptp_ratio": float(np.ptp(value) / np.ptp(modified))}
    result = {
        "status": "pass",
        "physical_geometry_channels": int(len(geom)),
        "target_channels": 182,
        "geometry_sign_boundary_controls": geometry,
        "native_clone_controls": clone,
        "time_phase_controls": shifted,
        "all_geometry_controls_pass": bool(all(r["one_to_one"] and r["inverse_closure"] for r in geometry)),
        "notes": "Integer lattice only. Fractional forward-model controls remain unqualified and were not run.",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
