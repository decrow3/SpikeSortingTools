"""Cached-only CK sample-size and training-noise calibration."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from testing.cg_anchor_helper import _cosine, half1_noise_std
from testing.cj_force_gate_w2 import write_csv


UNITS = (23, 120)
SAMPLE_SIZES = (25, 50, 100)
ESTIMATORS = ("median", "mean")
REPETITIONS = 100
BASE_SEED = 2026092701


def balanced_take(blocks, count, rng):
    blocks = np.asarray(blocks)
    if count == len(blocks):
        return np.arange(len(blocks))
    pools = {int(block): list(rng.permutation(np.flatnonzero(blocks == block)))
             for block in np.unique(blocks)}
    keys = list(rng.permutation(sorted(pools)))
    selected = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            if pools[key]:
                selected.append(pools[key].pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError("insufficient balanced pool")
    return np.asarray(selected)


def conditioned(snippets, support, noise, estimator):
    if estimator == "median":
        template = np.median(snippets, axis=0)[:, support]
    elif estimator == "mean":
        template = np.mean(snippets, axis=0)[:, support]
    else:
        raise ValueError(estimator)
    template = template - template.mean(axis=0, keepdims=True)
    return template / noise[support][None]


def edge_residual_segments(training):
    edges = np.concatenate((training[:, :20], training[:, -20:]), axis=1)
    edges = edges - np.median(edges, axis=1, keepdims=True)
    return np.concatenate((edges[:, :20, None], edges[:, 20:, None]), axis=2
                          ).transpose(0, 2, 1, 3).reshape(-1, 20, training.shape[2])


def synthetic_events(template, residual_segments, count, rng, zero_signal=False):
    result = np.empty((count, template.shape[0], template.shape[1]), dtype=float)
    pieces = int(np.ceil(template.shape[0] / residual_segments.shape[1]))
    for event in range(count):
        chosen = rng.integers(0, len(residual_segments), size=pieces)
        noise = np.concatenate(residual_segments[chosen], axis=0)[:template.shape[0]]
        result[event] = noise if zero_signal else template + noise
    return result


def summarize(values):
    values = np.asarray(values)
    return {
        "median": float(np.median(values)),
        "q025": float(np.quantile(values, 0.025)),
        "q975": float(np.quantile(values, 0.975)),
        "minimum": float(values.min()), "maximum": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ck-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = list(csv.DictReader(open(args.ck_root / "SELECTED_EVENTS.csv")))
    raw = np.load(args.ck_root / "RAW_SNIPPETS.npy", mmap_mode="r")
    output_rows = []
    for unit in UNITS:
        indices = [i for i, row in enumerate(rows) if int(row["unit"]) == unit]
        first_indices = [i for i in indices if int(rows[i]["half"]) == 1]
        second_indices = [i for i in indices if int(rows[i]["half"]) == 2]
        first, second = np.asarray(raw[first_indices]), np.asarray(raw[second_indices])
        blocks1 = np.array([int(rows[i]["block5"]) for i in first_indices])
        blocks2 = np.array([int(rows[i]["block5"]) for i in second_indices])
        support = np.array([int(value) for value in rows[indices[0]][
            "support_channels"].split(";")])
        noise = half1_noise_std(first)
        fixed_template = np.median(first, axis=0)
        residual_segments = edge_residual_segments(first)
        rng = np.random.default_rng(BASE_SEED + unit)
        stable1 = synthetic_events(fixed_template, residual_segments, 100, rng)
        stable2 = synthetic_events(fixed_template, residual_segments, 100, rng)
        zero1 = synthetic_events(fixed_template, residual_segments, 100, rng, True)
        zero2 = synthetic_events(fixed_template, residual_segments, 100, rng, True)
        equal_blocks = np.repeat(np.arange(10), 10)
        sources = (("observed", first, second, blocks1, blocks2),
                   ("stable_training_noise", stable1, stable2, equal_blocks, equal_blocks),
                   ("zero_signal", zero1, zero2, equal_blocks, equal_blocks))
        for source, left, right, left_blocks, right_blocks in sources:
            for sample_size in SAMPLE_SIZES:
                for estimator in ESTIMATORS:
                    values = []
                    for repetition in range(REPETITIONS):
                        take1 = balanced_take(left_blocks, sample_size, rng)
                        take2 = balanced_take(right_blocks, sample_size, rng)
                        values.append(_cosine(
                            conditioned(left[take1], support, noise, estimator),
                            conditioned(right[take2], support, noise, estimator)))
                    output_rows.append({
                        "unit": unit, "source": source,
                        "sample_size_per_half": sample_size,
                        "estimator": estimator, "repetitions": REPETITIONS,
                        **summarize(values),
                    })
    write_csv(args.output / "CALIBRATION.csv", output_rows)
    summary = {
        "status": "complete_cached_only",
        "units": list(UNITS), "sample_sizes": list(SAMPLE_SIZES),
        "estimators": list(ESTIMATORS), "repetitions": REPETITIONS,
        "monte_carlo_tail_resolution": 1 / REPETITIONS,
        "base_seed": BASE_SEED,
        "support": "unchanged CK physical support",
        "noise": "unchanged half-1 robust noise",
        "surrogate_template": "fixed half-1 median; finite-sample noise imprinted",
        "surrogate_noise": "training-half edge residuals; channel covariance and <=20-sample temporal covariance",
        "raw_voltage_reads": 0, "gpu_seconds": 0,
    }
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
