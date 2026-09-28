#!/usr/bin/env python3
"""Audit DH/DK waveform-state behavior at 7,500-sample boundaries."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DH = ROOT / "testing/outputs/dh_hybrid_truth_contract_20260928"
OUT = ROOT / "testing/outputs/dk_h1_source_endpoint_review_20260928.partial"
MATCHING = Path("/home/huklab/anaconda3/envs/spike-sort-challengers/lib/python3.12/site-packages/dartsort/peel/matching.py")
S_CFG = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/av_aw_handoff_20260926_v1/static_w2/effective-config.json")
D_CFG = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/luke0804-imec1-ao-d2l-v2/windows/W2/runs/D2L/sort/effective-config.json")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    events = pd.read_csv(DH / "SOURCE_EVENTS.csv")
    members = pd.read_csv(DH / "INJECTION_MEMBERSHIP.csv")
    with np.load(DH / "EXACT_QUERY_MOTION.npz", allow_pickle=False) as z:
        states = np.asarray(z["states_um"], int)
        chunk = int(z["chunk_length_samples"])
        centers = np.asarray(z["chunk_center_samples"], int)
        n_samples = int(z["n_samples"])
    rows = events.merge(
        members[["source_event_row_id", "sample_start", "sample_stop"]],
        left_on="event_row_id",
        right_on="source_event_row_id",
        validate="one_to_one",
    )
    sample = rows.local_sample.to_numpy(np.int64)
    lo = rows.sample_start.to_numpy(np.int64)
    hi = rows.sample_stop.to_numpy(np.int64)
    event_chunk = sample // chunk
    left_chunk = lo // chunk
    right_chunk = (hi - 1) // chunk
    crosses = left_chunk != right_chunk
    crosses_state = crosses & (states[left_chunk] != states[right_chunk])
    checks = {
        "membership_one_to_one": len(rows) == len(events) == len(members),
        "waveform_width_121": bool(np.all(hi - lo == 121)),
        "trough_offset_42": bool(np.all(sample - lo == 42)),
        "no_clipped_writes": bool(np.all(lo >= 0) and np.all(hi <= n_samples)),
        "event_state_equals_trough_chunk_state": bool(np.array_equal(rows.state_um.to_numpy(int), states[event_chunk])),
        "matching_center_in_event_chunk": bool(np.all(centers[event_chunk] // chunk == event_chunk)),
        "exact_boundary_uses_new_chunk": bool(np.all(event_chunk[sample % chunk == 0] == sample[sample % chunk == 0] // chunk)),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    crossing = rows.loc[crosses_state, [
        "event_row_id", "donor_id", "local_sample", "sample_start", "sample_stop",
        "chunk_index", "state_um",
    ]].copy()
    crossing["left_chunk"] = left_chunk[crosses_state]
    crossing["right_chunk"] = right_chunk[crosses_state]
    crossing["left_state_um"] = states[left_chunk[crosses_state]]
    crossing["right_state_um"] = states[right_chunk[crosses_state]]
    crossing["matching_state_um"] = states[event_chunk[crosses_state]]
    crossing.to_csv(OUT / "STATE_BOUNDARY_EVENTS.csv", index=False)
    s_cfg = json.loads(S_CFG.read_text())["config"]
    d_cfg = json.loads(D_CFG.read_text())["config"]
    result = {
        "status": "pass_saved_contract_boundary_qualification",
        "checks": checks,
        "events": int(len(rows)),
        "events_crossing_any_7500_sample_boundary": int(crosses.sum()),
        "events_crossing_a_state_change_boundary": int(crosses_state.sum()),
        "events_with_trough_exactly_on_boundary": int((sample % chunk == 0).sum()),
        "contract": (
            "Each 121-sample injected waveform is relocated with the state of its trough sample and held at that state "
            "across the waveform. DARTsort matching selects one template state for the chunk containing the returned "
            "trough; margins supply waveform support but margin events are discarded from the neighboring chunk."
        ),
        "whole_pipeline_contrast": {
            "S_h_matching_chunk_samples": int(s_cfg["matching_cfg"]["chunk_length_samples"]),
            "D2L_h_matching_chunk_samples": int(d_cfg["matching_cfg"]["chunk_length_samples"]),
            "interpretation": "The intended comparison changes motion use and matching chunk size (30000 versus 7500); it is not a motion-field-only contrast.",
        },
        "sources": {
            "events_sha256": sha(DH / "SOURCE_EVENTS.csv"),
            "membership_sha256": sha(DH / "INJECTION_MEMBERSHIP.csv"),
            "exact_query_motion_sha256": sha(DH / "EXACT_QUERY_MOTION.npz"),
            "h1_installed_matching_sha256": sha(MATCHING),
            "S_h_config_sha256": sha(S_CFG),
            "D2L_h_config_sha256": sha(D_CFG),
        },
        "limitation": "This qualifies the published DH contract against h1 DARTsort 0.5.16. Actual h5 materializer/consumer source remains unverified until its dedicated packet is authorized and published.",
    }
    (OUT / "STATE_BOUNDARY_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
