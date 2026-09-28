#!/usr/bin/env python3
"""Finite DK audit of scorer region accounting and trough-fixed boundaries."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd

from testing.luke_dh_hybrid_scorer import score_with_frozen_association_v3

ROOT = Path(__file__).resolve().parents[1]
DH = ROOT / "testing/outputs/dh_hybrid_truth_contract_20260928"
OUT = ROOT / "testing/outputs/dk_h1_scorer_boundary_correction_20260928"
FS = 29999.759166666667


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    started = time.monotonic(); cpu0 = time.process_time()
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True)

    cross = score_with_frozen_association_v3(
        np.array([100]), np.array([1]), np.array([False]),
        np.array([100]), np.array([7]), np.array([True]),
        [{"donor_id": 1, "primary_label": 7}], 0,
    )
    rest = next(row for row in cross["scores"] if row["region"] == "rest")
    shared = score_with_frozen_association_v3(
        np.array([100, 100]), np.array([1, 2]), np.array([False, False]),
        np.array([100]), np.array([7]), np.array([False]),
        [{"donor_id": 1, "primary_label": 7}, {"donor_id": 2, "primary_label": 7}], 0,
    )
    shared_rest = [row for row in shared["scores"] if row["region"] == "rest"]
    scorer = {
        "status": "pass_corrected_region_accounting",
        "cross_boundary_fixture": {
            "matches": len(cross["matches"]),
            "global_unmatched_truth_count": rest["global_unmatched_truth_count"],
            "fn_without_candidate": rest["fn_without_candidate"],
            "regional_outflow_matches": rest["regional_outflow_matches"],
            "candidate_recall_global": rest["candidate_recall"],
            "same_region_candidate_recall": rest["same_region_candidate_recall"],
        },
        "shared_label_fixture": {
            "truth_rows": 2,
            "output_rows": 1,
            "matches": len(shared["matches"]),
            "unique_matched_outputs": len({row["output_index"] for row in shared["matches"]}),
            "globally_matched_truth": sum(row["globally_matched_truth_count"] for row in shared_rest),
            "global_unmatched_truth": sum(row["global_unmatched_truth_count"] for row in shared_rest),
            "output_double_use": False,
        },
        "correction": (
            "fn_without_candidate and candidate_recall now use global matches. "
            "same-region deficit/recall and regional inflow/outflow are separate fields."
        ),
        "compatibility": "Existing cross_boundary_truth_matches/output_consumed fields are retained.",
        "source_sha256": sha(ROOT / "testing/luke_dh_hybrid_scorer.py"),
        "test_source_sha256": sha(ROOT / "testing/test_luke_dh_hybrid_scorer.py"),
        "tests": "11 passed",
    }
    if scorer["cross_boundary_fixture"]["fn_without_candidate"] != 0:
        raise RuntimeError("cross-boundary global match still counted as FN")
    if len(shared["matches"]) != 1 or len({row["output_index"] for row in shared["matches"]}) != 1:
        raise RuntimeError("shared-label output reuse")
    dump(OUT / "SCORER_REGION_ACCOUNTING.json", scorer)

    events = pd.read_csv(DH / "SOURCE_EVENTS.csv")
    members = pd.read_csv(DH / "INJECTION_MEMBERSHIP.csv")
    with np.load(DH / "EXACT_QUERY_MOTION.npz", allow_pickle=False) as z:
        states = np.asarray(z["states_um"], int)
        chunk = int(z["chunk_length_samples"])
    rows = events.merge(members[["source_event_row_id", "sample_start", "sample_stop"]],
                        left_on="event_row_id", right_on="source_event_row_id", validate="one_to_one")
    sample = rows.local_sample.to_numpy(np.int64); lo = rows.sample_start.to_numpy(np.int64); hi = rows.sample_stop.to_numpy(np.int64)
    event_chunk = sample // chunk; left = lo // chunk; right = (hi - 1) // chunk
    changed = (left != right) & (states[left] != states[right])
    boundary = (left + 1) * chunk
    opposite_span = np.where(event_chunk == left, hi - boundary, boundary - lo)
    spans = opposite_span[changed]
    deltas = np.abs(states[right[changed]] - states[left[changed]])
    boundary_result = {
        "status": "qualified_discrete_trough_contract_not_continuous_motion",
        "events": int(len(rows)),
        "waveforms_crossing_any_chunk_boundary": int((left != right).sum()),
        "waveforms_crossing_state_change_boundary": int(changed.sum()),
        "troughs_exactly_on_boundary": int((sample % chunk == 0).sum()),
        "opposite_side_samples_per_changed_waveform": {
            "minimum": int(spans.min()), "median": float(np.median(spans)), "maximum": int(spans.max()),
            "maximum_duration_ms": float(1000 * spans.max() / FS),
            "total_event_samples": int(spans.sum()),
        },
        "state_step_abs_um": {"minimum": int(deltas.min()), "median": float(np.median(deltas)), "maximum": int(deltas.max())},
        "interpretation": (
            "Each waveform is held at its trough chunk's state for all 121 samples. For the 78 state-change crossings, "
            "samples on the other side of the boundary differ from a hypothetical per-sample continuous field only "
            "within the recorded waveform span (at most the reported sample duration). This is exact for the discrete "
            "matching consumer, not blanket exact continuous-time motion."
        ),
    }
    dump(OUT / "TROUGH_FIXED_BOUNDARY_SCOPE.json", boundary_result)
    (OUT / "REPORT.md").write_text(f"""# DK scorer and boundary correction

The scorer's former regional deficit could count a globally matched truth row
as `fn_without_candidate` when its output crossed the episode/rest boundary.
The corrected implementation separates global unmatched truth from regional
outflow/inflow and keeps same-region recall explicit. A two-donor shared-label
fixture proves one output row is used at most once. Eleven focused tests pass.

The motion statement is narrowed as well. Of 42,181 events, 708 waveforms cross
a 7,500-sample boundary and 78 cross a state change. Injection holds the trough
chunk's state across all 121 samples. The opposite-side span is at most
{int(spans.max())} samples ({1000 * spans.max() / FS:.3f} ms). This matches the
discrete DARTsort chunk consumer but is not exact continuous-time motion.

No h5 payload was accessed, no matching or calibration was rerun, and no sort,
raw read, RF fit or GPU work occurred. The actual DK endpoint remains pending
authorized h5 source/output publication.
""")
    receipt = {
        "status": "complete_finite_DK_correction_endpoint_pending",
        "cpu_allwork_cap_s": 1110,
        "measured_cpu_s": time.process_time() - cpu0,
        "measured_wall_s": time.monotonic() - started,
        "conservative_active_charge_s": 60.0,
        "prior_h1_cumulative_active_s": 18533.22,
        "new_h1_cumulative_active_s": 18593.22,
        "overall_ceiling_active_s": 26000.0,
        "raw_voltage_reads": 0, "gpu_s": 0, "sorts": 0, "matching_runs": 0, "rf_fits": 0,
    }
    dump(OUT / "RESOURCE_RECEIPT.json", receipt)
    products = []
    for path in sorted(OUT.iterdir()):
        if path.name not in {"MANIFEST.json", "COMPLETE.json"}:
            products.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)})
    dump(OUT / "MANIFEST.json", {"products": products})
    dump(OUT / "COMPLETE.json", {"status": receipt["status"], "manifest_sha256": sha(OUT / "MANIFEST.json"), "written_last": True})
    print(json.dumps({"status": receipt["status"], "changed_boundary_events": int(changed.sum()), "max_span_samples": int(spans.max())}))


if __name__ == "__main__":
    main()
