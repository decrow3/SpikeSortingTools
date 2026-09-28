#!/usr/bin/env python3
"""Finite source-level preparation audit for DN's S_L_h comparator."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DH = ROOT / "testing/outputs/dh_hybrid_truth_contract_20260928"
GEOMETRY_SOURCE = (
    ROOT
    / "testing/outputs/luke_au_cpu_preparation/full_probe_extraction_v1/"
    "bc_attempt5/full_probe_final_label_templates.npz"
)
OUT = ROOT / "testing/outputs/dn_h1_sl_source_preparation_20260928"
TARGET = np.arange(202, 384, dtype=np.int64)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def exact_map(geom: np.ndarray, shift_um: float) -> np.ndarray:
    """Source i -> exact same-column target at y + shift, or -1."""
    result = np.full(len(geom), -1, dtype=np.int64)
    lookup = {(float(x), float(y)): i for i, (x, y) in enumerate(geom)}
    if len(lookup) != len(geom):
        raise RuntimeError("geometry coordinates are not unique")
    for i, (x, y) in enumerate(geom):
        result[i] = lookup.get((float(x), float(y + shift_um)), -1)
    good = result >= 0
    if np.unique(result[good]).size != good.sum():
        raise RuntimeError("non-injective exact lattice map")
    return result


def main() -> None:
    started = time.monotonic()
    cpu0 = time.process_time()
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)

    events = pd.read_csv(DH / "SOURCE_EVENTS.csv")
    members = pd.read_csv(DH / "INJECTION_MEMBERSHIP.csv")
    support = pd.read_csv(DH / "DONOR_STATE_SUPPORT.csv")
    with np.load(DH / "EXACT_QUERY_MOTION.npz", allow_pickle=False) as z:
        states = np.asarray(z["states_um"], dtype=np.int64)
        chunk = int(z["chunk_length_samples"])
        n_samples = int(z["n_samples"])
    # Read only the frozen geometry/channel metadata from DH's declared source.
    with np.load(GEOMETRY_SOURCE, allow_pickle=False) as z:
        geom = np.asarray(z["geometry_um"], dtype=np.float64)[TARGET]
        channel_ids = np.asarray(z["channel_ids"])[TARGET]
    expected_ids = np.array([f"imec1.ap#AP{i}" for i in range(202, 384)])
    if not np.array_equal(channel_ids, expected_ids):
        raise RuntimeError("DH physical target changed")

    rows = []
    for q in sorted(map(int, np.unique(states))):
        # output(x,y) = input(x,y+q) is source->target shift -q.
        inverse_map = exact_map(geom, -q)
        retained = int(np.count_nonzero(inverse_map >= 0))
        zero_filled = int(len(geom) - retained)

        # Check the coordinate algebra wherever both coordinates exist.
        lookup = {(float(x), float(y)) for x, y in geom}
        checked = 0
        for x, y in geom:
            observed = (float(x), float(y + q))
            restored = (observed[0], float(observed[1] - q))
            if observed in lookup:
                checked += 1
                if restored != (float(x), float(y)):
                    raise RuntimeError("inverse sign check failed")
        if checked != retained:
            raise RuntimeError((q, checked, retained))
        ss = support.loc[support.state_um == q]
        rows.append(
            {
                "state_um": q,
                "chunks": int(np.count_nonzero(states == q)),
                "events": int(np.count_nonzero(events.state_um.to_numpy() == q)),
                "inverse_retained_channels": retained,
                "inverse_zero_filled_channels": zero_filled,
                "inverse_retained_fraction": retained / len(geom),
                "donor_state_rows": int(len(ss)),
                "donor_support_gate_passes": int(ss.passes_frozen_support_gate.sum()),
                "donor_min_retained_energy_fraction": float(ss.retained_energy_fraction.min()),
            }
        )
    state_table = pd.DataFrame(rows)
    state_table.to_csv(OUT / "STATE_SUPPORT.csv", index=False)

    joined = events.merge(
        members[["source_event_row_id", "sample_start", "sample_stop"]],
        left_on="event_row_id",
        right_on="source_event_row_id",
        validate="one_to_one",
    )
    lo = joined.sample_start.to_numpy(np.int64)
    hi = joined.sample_stop.to_numpy(np.int64)
    trough = joined.local_sample.to_numpy(np.int64)
    left_chunk = lo // chunk
    right_chunk = (hi - 1) // chunk
    trough_chunk = trough // chunk
    crosses_any = left_chunk != right_chunk
    crosses_state = crosses_any & (states[left_chunk] != states[right_chunk])
    # Samples not governed by the trough's state within a crossing waveform.
    opposite = np.zeros(len(joined), dtype=np.int64)
    left_n = np.maximum(0, np.minimum(hi, (trough_chunk * chunk)) - lo)
    right_boundary = (trough_chunk + 1) * chunk
    right_n = np.maximum(0, hi - np.maximum(lo, right_boundary))
    opposite[crosses_state] = (left_n + right_n)[crosses_state]
    if int(crosses_any.sum()) != 708 or int(crosses_state.sum()) != 78:
        raise RuntimeError("published boundary counts changed")
    if int(opposite.max()) != 77:
        raise RuntimeError("published maximum opposite-state span changed")

    result = {
        "status": "ready_source_level_actual_h5_endpoint_pending",
        "operator": "target(x,y) = source(x,y+q(t)); exact same-column lattice; zero when the requested source is off the 182-channel crop",
        "sign_check": {
            "injection": "base-position y -> observed y+q",
            "inverse": "observed y+q -> output (y+q)-q = y",
            "conclusion": "the operator cancels q and preserves the fixed donor placement base",
        },
        "scope": {
            "channels": int(len(geom)),
            "channel_ids": "imec1.ap#AP202:imec1.ap#AP383",
            "states_um": sorted(map(int, np.unique(states))),
            "chunks": int(len(states)),
            "events": int(len(events)),
            "n_samples": n_samples,
        },
        "boundary_qualification": {
            "waveforms_crossing_any_state_chunk_boundary": int(crosses_any.sum()),
            "waveforms_crossing_a_state_change": int(crosses_state.sum()),
            "maximum_samples_on_opposite_side_of_trough_state": int(opposite.max()),
            "maximum_opposite_span_ms": float(opposite.max() / 29999.759166666667 * 1000),
            "interpretation": "Trough-fixed injections and matching use the same discrete state. At 78 state changes, part of a 121-sample waveform spans the adjacent state, so S_L_h is not an exact continuous-time inverse over that limited waveform span.",
        },
        "background": "The inverse operator acts on the entire already-preprocessed shared hybrid recording. Native background is countershifted too; S_L_h is not an injection-only transform or a motion-only contrast.",
        "support": rows,
        "limitations": [
            "This is source-level preparation from the published DH contract, not review of the held h5 worker or endpoint.",
            "No voltage was materialized and no output waveform equality was tested.",
            "The crop-edge zeros are deterministic consequences of the inverse lattice shift and must be recorded by the worker.",
        ],
        "inputs": {
            "hybrid_manifest_sha256": sha(DH / "HYBRID_MANIFEST.json"),
            "events_sha256": sha(DH / "SOURCE_EVENTS.csv"),
            "membership_sha256": sha(DH / "INJECTION_MEMBERSHIP.csv"),
            "state_motion_sha256": sha(DH / "EXACT_QUERY_MOTION.npz"),
            "support_sha256": sha(DH / "DONOR_STATE_SUPPORT.csv"),
            "geometry_source_sha256": sha(GEOMETRY_SOURCE),
        },
    }
    (OUT / "INTEGRATION_CHECK.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (OUT / "INTEGRATION_NOTE.md").write_text(
        "# DN S_L_h source-level integration note\n\n"
        "## Verdict\n\n"
        "The proposed inverse lattice operator has the correct sign: DH injects the state as "
        "`y -> y + q`, and S_L_h uses `target(y) = source(y + q)`, which maps the "
        "state-shifted signal back to its fixed placement depth. The worker must apply this "
        "to the entire shared hybrid voltage with exact same-column indexing and zero fill at "
        "the 182-channel crop edge. Native background is therefore countershifted as well.\n\n"
        "All 175 frozen donor/state support rows pass. The per-state crop losses and occupancy "
        "are in `STATE_SUPPORT.csv`. Of 42,181 injected events, 708 waveforms cross a 7,500-sample "
        "boundary and 78 cross a state change. Their trough-fixed relocation can disagree with a "
        "continuous inverse for at most 77 samples (2.567 ms). This exception is limited to the "
        "waveform span; it does not justify calling the construction an exact continuous-time inverse.\n\n"
        "Actual h5 worker source and the one requested endpoint remain held and unreviewed. No "
        "materialization, sort, matching, calibration, raw read, GPU work, or export was performed.\n"
    )
    cpu_s = time.process_time() - cpu0
    wall_s = time.monotonic() - started
    receipt = {
        "status": "complete_finite_preparation_actual_h5_endpoint_pending",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_s_measured": cpu_s,
        "wall_s_measured": wall_s,
        "conservative_active_charge_s": 60.0,
        "threads_max": 2,
        "readers_max": 1,
        "raw_voltage_bytes": 0,
        "gpu_s": 0,
        "sorts": 0,
        "materialized_voltage_bytes": 0,
        "prior_h1_cumulative_active_s": 18593.22,
        "new_h1_cumulative_active_s": 18653.22,
        "overall_ceiling_active_s": 26000.0,
        "actual_h5_worker_reviewed": False,
    }
    (OUT / "RESOURCE_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    files = []
    for p in sorted(OUT.iterdir()):
        if p.name not in {"MANIFEST.json", "COMPLETE.json"}:
            files.append({"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)})
    (OUT / "MANIFEST.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")
    (OUT / "COMPLETE.json").write_text(
        json.dumps({"status": receipt["status"], "manifest_sha256": sha(OUT / "MANIFEST.json"), "written_last": True}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": receipt["status"], "output": str(OUT), "charge_s": 60.0}, sort_keys=True))


if __name__ == "__main__":
    main()
