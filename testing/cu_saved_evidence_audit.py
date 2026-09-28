#!/usr/bin/env python3
"""Independent structural audit of the saved CU voltage evidence packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("packet", type=Path)
    ap.add_argument("template_data", type=Path)
    ap.add_argument("input_manifest", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()

    failures: list[str] = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            failures.append(name)

    manifest = json.loads((args.packet / "MANIFEST.json").read_text())
    manifest_bad = []
    for item in manifest["files"]:
        path = args.packet / item["path"]
        if (
            not path.exists()
            or path.stat().st_size != item["bytes"]
            or digest(path) != item["sha256"]
        ):
            manifest_bad.append(item["path"])
    check("manifest_products", not manifest_bad)
    complete = json.loads((args.packet / "COMPLETE.json").read_text())
    check("complete_manifest_hash", digest(args.packet / "MANIFEST.json") == complete["manifest_sha256"])

    targets = pd.read_csv(args.packet / "FROZEN_TARGET_PAIRS.csv")
    comparisons = pd.read_csv(args.packet / "FROZEN_COMPARISON_PAIRS.csv")
    refs = pd.read_csv(args.packet / "FROZEN_REFERENCE_EVENTS.csv")
    rows = pd.read_csv(args.packet / "PAIR_SNIPPET_ROWS.csv")
    metrics = pd.read_csv(args.packet / "PAIR_VOLTAGE_METRICS.csv")
    supports = pd.read_csv(args.packet / "TEMPLATE_PHYSICAL_SUPPORTS.csv")
    summaries = pd.read_csv(args.packet / "PAIR_VOLTAGE_SUMMARY.csv")
    receipt = json.loads((args.packet / "EXTRACTION_AND_ANALYSIS_RECEIPT.json").read_text())
    with np.load(args.template_data, allow_pickle=False) as z:
        registered_geom = z["registered_geom"]
    input_window = json.loads(args.input_manifest.read_text())["window"]

    pair_snippets = np.load(args.packet / "PAIR_SNIPPETS_FULL182.npy", mmap_mode="r")
    reference_snippets = np.load(args.packet / "REFERENCE_SNIPPETS_FULL182.npy", mmap_mode="r")
    check("pair_shape", pair_snippets.shape == (128, int(rows.valid_samples.max()), 182))
    check("reference_shape", reference_snippets.shape == (1500, 121, 182))
    check("pair_dtype", pair_snippets.dtype == np.float32)
    check("reference_dtype", reference_snippets.dtype == np.float32)
    check("pair_indices_dense", np.array_equal(np.sort(rows.snippet_index), np.arange(128)))

    requested_channels = input_window["requested_sort_channel_ids"]
    check("physical_channel_count", len(requested_channels) == 182)
    check("support_constituents", supports.constituent.nunique() == 79)
    check("support_16_each", bool((supports.groupby("constituent").size() == 16).all()))
    check("support_unique_channels", not supports.duplicated(["constituent", "physical_channel_index"]).any())
    check(
        "support_channel_ids_exact",
        all(
            requested_channels[int(rec.physical_channel_index)] == rec.channel_id
            for rec in supports.itertuples(index=False)
        ),
    )
    check(
        "support_registered_geometry_exact",
        all(
            np.array_equal(
                registered_geom[int(rec.registered_channel_index)],
                np.array([rec.x_um, rec.y_um]),
            )
            for rec in supports.itertuples(index=False)
        ),
    )

    requested = pd.concat(
        [
            targets.assign(kind="target", pair_id=targets.target_id)[
                ["kind", "pair_id", "row_a", "row_b", "row_a_local_pre_frame", "row_b_local_pre_frame"]
            ],
            comparisons.assign(kind="comparison")[
                ["kind", "comparison_id", "row_a", "row_b", "row_a_local_pre_frame", "row_b_local_pre_frame"]
            ].rename(columns={"comparison_id": "pair_id"}),
        ],
        ignore_index=True,
    )
    merged = rows.merge(requested, on=["kind", "pair_id", "row_a", "row_b"], validate="one_to_one")
    expected_start = merged[["row_a_local_pre_frame", "row_b_local_pre_frame"]].min(axis=1) - 42
    expected_end = merged[["row_a_local_pre_frame", "row_b_local_pre_frame"]].max(axis=1) + 79
    check("pair_window_start", np.array_equal(merged.start_frame, expected_start))
    check("pair_window_end", np.array_equal(merged.end_frame, expected_end))
    check("pair_valid_samples", np.array_equal(merged.valid_samples, expected_end - expected_start))
    check("marker_a", np.array_equal(merged.marker_a, merged.row_a_local_pre_frame - expected_start))
    check("marker_b", np.array_equal(merged.marker_b, merged.row_b_local_pre_frame - expected_start))
    padding_valid = True
    for rec in rows.itertuples(index=False):
        tail = pair_snippets[int(rec.snippet_index), int(rec.valid_samples) :]
        padding_valid &= not tail.size or bool(np.isnan(tail).all())
    check("pair_padding_nan", padding_valid)
    check("reference_all_finite", bool(np.isfinite(reference_snippets).all()))

    expected_metric_pairs = set(zip(rows.kind, rows.pair_id))
    check("metric_pair_rows_exact", set(zip(metrics.kind, metrics.pair_id)) == expected_metric_pairs)
    check("nnls_amplitudes_nonnegative", bool((metrics[["one_event_amplitude", "two_event_amplitude_a", "two_event_amplitude_b"]] >= 0).all().all()))
    check("two_event_nested_sse", bool((metrics.two_event_sse <= metrics.one_event_sse + 1e-9).all()))
    reduction = (metrics.one_event_sse - metrics.two_event_sse) / metrics.one_event_sse
    check("fractional_reduction_formula", bool(np.allclose(reduction, metrics.two_vs_one_fractional_sse_reduction, atol=1e-12)))
    support_sets = {
        int(k): set(map(int, g.physical_channel_index))
        for k, g in supports.groupby("constituent")
    }
    expected_support = [
        len(support_sets[int(a)] | support_sets[int(b)])
        for a, b in zip(metrics.constituent_a, metrics.constituent_b)
    ]
    check("metric_support_union", np.array_equal(metrics.support_channels, expected_support))

    recomputed = (
        metrics.groupby(["kind", "category"], sort=False)
        .agg(
            pairs=("pair_id", "size"),
            median_two_vs_one_reduction=("two_vs_one_fractional_sse_reduction", "median"),
            q25_two_vs_one_reduction=("two_vs_one_fractional_sse_reduction", lambda x: x.quantile(0.25)),
            q75_two_vs_one_reduction=("two_vs_one_fractional_sse_reduction", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )
    joined = summaries.merge(recomputed, on=["kind", "category"], suffixes=("_saved", "_recomputed"))
    check("summary_counts", bool((joined.pairs_saved == joined.pairs_recomputed).all()))
    for column in ("median_two_vs_one_reduction", "q25_two_vs_one_reduction", "q75_two_vs_one_reduction"):
        check(
            f"summary_{column}",
            bool(np.allclose(joined[f"{column}_saved"], joined[f"{column}_recomputed"])),
        )

    boundary = metrics[(metrics.kind == "target") & (metrics.row_a == 246684) & (metrics.row_b == 246685)]
    check("sole_boundary_row", len(boundary) == 1)
    b = boundary.iloc[0]
    check("sole_boundary_no_second_gain", bool(b.two_vs_one_fractional_sse_reduction == 0 and b.two_event_amplitude_b == 0))
    check("logical_voltage_cap", int(receipt["logical_total_bytes"]) <= 536_870_912)
    check("logical_pair_bytes", int(receipt["logical_pair_bytes"]) == pair_snippets.nbytes)
    check("logical_reference_bytes", int(receipt["logical_reference_bytes"]) == reference_snippets.nbytes)
    check("array_hashes", digest(args.packet / "PAIR_SNIPPETS_FULL182.npy") == receipt["pair_snippets_sha256"] and digest(args.packet / "REFERENCE_SNIPPETS_FULL182.npy") == receipt["reference_snippets_sha256"])

    result = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "manifest_files": len(manifest["files"]),
        "manifest_bad": manifest_bad,
        "logical_voltage_bytes": int(receipt["logical_total_bytes"]),
        "pair_windows": len(rows),
        "reference_windows": len(refs),
        "boundary_case": {
            "rows": [246684, 246685],
            "one_event_sse": float(b.one_event_sse),
            "two_event_sse": float(b.two_event_sse),
            "two_vs_one_fractional_sse_reduction": float(b.two_vs_one_fractional_sse_reduction),
            "two_event_amplitude_a": float(b.two_event_amplitude_a),
            "two_event_amplitude_b": float(b.two_event_amplitude_b),
            "reference_split_cosine_a": float(b.reference_split_cosine_a),
            "reference_split_cosine_b": float(b.reference_split_cosine_b),
            "event_b_own_reference_cosine": float(b.event_b_own_reference_cosine),
        },
        "qualifications": [
            "The descriptive residual operator source was not included in the packet, so output invariants are independently checked but implementation source cannot be hashed here.",
            "Fractional SSE is computed over lag-dependent pair-window lengths; the 5-20 ms comparison windows are systematically longer and are context, not a calibrated null.",
            "Event-centered windows overlap for close pairs, constituent references derive from sorting output, and lag-only evidence cannot classify physical duplicates.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
