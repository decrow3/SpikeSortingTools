#!/usr/bin/env python3
"""Small reference fixture for CV event fractions, boundaries and null support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def event_has_partner_fraction(
    small: np.ndarray, partner: np.ndarray, lo: int, hi: int
) -> tuple[int, int, float]:
    hit = np.zeros(small.size, dtype=bool)
    for i, t in enumerate(small):
        dt = np.abs(partner - t)
        hit[i] = np.any((dt >= lo) & (dt <= hi))
    return int(hit.sum()), int(hit.size), float(hit.mean()) if hit.size else float("nan")


def common_support(segment: tuple[int, int], signed_offset: int) -> tuple[int, int] | None:
    start, end = segment
    lo = start + max(0, -signed_offset)
    hi = end - max(0, signed_offset)
    return None if hi <= lo else (lo, hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    args = ap.parse_args()

    small = np.array([100, 200, 300, 400])
    big = np.array([108, 207, 330, 700])
    other_small = np.array([420])
    near = event_has_partner_fraction(small, np.r_[big, other_small], 0, 7)
    target = event_has_partner_fraction(small, np.r_[big, other_small], 8, 29)
    exact8 = event_has_partner_fraction(small, np.r_[big, other_small], 8, 8)
    exact30 = event_has_partner_fraction(small, np.r_[big, other_small], 30, 30)
    assert near == (1, 4, 0.25)
    assert target == (2, 4, 0.5)
    assert exact8 == (1, 4, 0.25)
    assert exact30 == (1, 4, 0.25)

    big_small_pairs_8_29 = [(100, 108)]
    small_small_remainder_8_29 = [(400, 420)]
    all_pairs_8_29 = big_small_pairs_8_29 + small_small_remainder_8_29
    assert len(all_pairs_8_29) == 2
    assert len(big_small_pairs_8_29) + len(small_small_remainder_8_29) == len(all_pairs_8_29)

    segments = [(0, 500), (600, 700)]
    offsets = [-200, -20, 20, 200]
    support = []
    for offset in offsets:
        for segment_id, segment in enumerate(segments):
            bounds = common_support(segment, offset)
            if bounds is None:
                support.append(
                    {
                        "offset": offset,
                        "segment": segment_id,
                        "support_start": None,
                        "support_end": None,
                        "exposure": 0,
                        "denominator_events": 0,
                    }
                )
                continue
            lo, hi = bounds
            denominator = int(np.count_nonzero((small >= lo) & (small < hi)))
            support.append(
                {
                    "offset": offset,
                    "segment": segment_id,
                    "support_start": lo,
                    "support_end": hi,
                    "exposure": hi - lo,
                    "denominator_events": denominator,
                }
            )
    assert any(row["exposure"] == 0 for row in support)
    assert all(
        row["exposure"] == 0
        or (
            segments[row["segment"]][0]
            <= row["support_start"]
            < row["support_end"]
            <= segments[row["segment"]][1]
        )
        for row in support
    )

    result = {
        "status": "pass",
        "event_fraction_definition": "fraction of small-child events with at least one partner",
        "near_0_7": {"numerator": near[0], "denominator": near[1], "fraction": near[2]},
        "target_8_29": {"numerator": target[0], "denominator": target[1], "fraction": target[2]},
        "exact_8": {"numerator": exact8[0], "denominator": exact8[1], "fraction": exact8[2]},
        "exact_30": {"numerator": exact30[0], "denominator": exact30[1], "fraction": exact30[2]},
        "pair_contribution_closure": {
            "big_small": len(big_small_pairs_8_29),
            "small_small_remainder": len(small_small_remainder_8_29),
            "all_pairs": len(all_pairs_8_29),
            "additive": True,
        },
        "null_support": support,
        "notes": [
            "Exact 8 is inside 8-29; exact 30 is outside.",
            "Event fractions use an any-partner union and cannot be replaced by all-pair counts or rate-times-window exposure.",
            "Signed shifts are nonwrapping and observed/shifted evaluations must use the same per-offset within-segment support.",
            "Zero-support blocks remain explicit.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
