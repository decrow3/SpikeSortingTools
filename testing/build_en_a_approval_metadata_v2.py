#!/usr/bin/env python3
"""Build the immutable metadata-only A four-unit approval-request repair."""

from __future__ import annotations

import argparse, csv, hashlib, json, os, shutil
from pathlib import Path
import numpy as np

from testing.en_state_support_diagnosis import field_cells, support_tables, assign_positions


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    root = Path("/media/huklaban5/Data/en_rounded_field_ks129_20260929_v1/arm_a_v1/full/sort/sorter_output")
    field_path = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_am3_imec0_field_request_20260929_v1/payload/luke0804_imec0_two_layer_motion.npz")
    fs = 29999.835983263598; num_samples = 314204894
    field_spec = {"field_path": str(field_path), "field_sha256": sha(field_path),
                  "time_key": "time_s", "displacement_key": "displacement_um",
                  "rigid_projection": "median across depth"}
    cells = field_cells(field_spec, num_samples / fs)
    times = np.load(root / "spike_times.npy", mmap_mode="r")
    units = np.load(root / "spike_clusters.npy", mmap_mode="r")
    templates = np.load(root / "spike_templates.npy", mmap_mode="r")
    positions = np.load(root / "spike_positions.npy", mmap_mode="r")
    geometry = np.load(root / "channel_positions.npy")
    support = support_tables(geometry, cells["states"])
    slots = [
      (278,"pair_primary","short_pair",-200),(278,"pair_secondary","short_pair",-80),(278,"ordinary_primary","ordinary_event",-200),(278,"ordinary_secondary","ordinary_event",-80),(278,"q0_control","q0_identity_control",0),
      (271,"pair_primary","short_pair",-160),(271,"pair_secondary","short_pair",-240),(271,"ordinary_primary","ordinary_event",-160),(271,"ordinary_secondary","ordinary_event",-240),(271,"q0_control","q0_identity_control",0),
      (588,"pair_primary","short_pair",-120),(588,"pair_secondary","short_pair",-160),(588,"ordinary_primary","ordinary_event",-120),(588,"ordinary_secondary","ordinary_event",-160),(588,"q0_control","q0_identity_control",0),
      (445,"pair_primary","short_pair",-200),(445,"pair_secondary","short_pair",-120),(445,"ordinary_primary","ordinary_event",-200),(445,"ordinary_secondary","ordinary_event",-120),(445,"q0_control","q0_identity_control",0),
    ]
    per_unit = {}; complete_event_parts = []
    for unit in sorted({s[0] for s in slots}):
        ix = np.flatnonzero(units == unit); tt = np.asarray(times[ix]); seconds = tt / fs
        knot = np.clip(np.searchsorted(cells["edges"], seconds, side="right") - 1,
                       0, len(cells["knot_state"]) - 1)
        states = cells["knot_state"][knot]; segments = cells["knot_segment"][knot]
        status, _, _ = assign_positions(np.asarray(positions[ix, 1], float), states, support)
        per_unit[unit] = (ix, tt, states, segments, status)
        complete_event_parts.append({
          "global_event_index":ix.astype(np.int64),"unit_id":np.full(len(ix),unit,np.int32),
          "unit_event_ordinal":np.arange(len(ix),dtype=np.int64),"frame":tt.astype(np.int64),
          "template_id":np.asarray(templates[ix],dtype=np.int32),
          "saved_position_y_um":np.asarray(positions[ix,1],dtype=np.float32),
          "state_um":np.asarray(cells["states"][states],dtype=np.float32),
          "segment_id":segments.astype(np.int32),"support_status":status.astype(np.uint8),
          "previous_same_unit_lag_samples":np.r_[0,np.diff(tt)].astype(np.int64),
          "following_same_unit_lag_samples":np.r_[np.diff(tt),0].astype(np.int64)})
    event_keys=list(complete_event_parts[0])
    np.savez_compressed(args.output/"FOUR_UNIT_EVENT_METADATA.npz",
      **{key:np.concatenate([part[key] for part in complete_event_parts]) for key in event_keys})
    rows, selected = [], set()
    for unit, suffix, role, state_um in slots:
        if role != "short_pair": continue
        ix, tt, state, segment, status = per_unit[unit]
        si = int(np.flatnonzero(cells["states"] == state_um)[0])
        good = (np.diff(tt) > 0) & (np.diff(tt) < 30) & (segment[1:] == segment[:-1]) & (state[1:] == si) & (status[:-1] == 0) & (status[1:] == 0)
        j = int(np.flatnonzero(good)[0]); a, b = int(ix[j]), int(ix[j+1]); selected.update((a,b))
        rows.append({"slot_id":f"u{unit}_{suffix}","unit_id":unit,"role":role,"state_um":state_um,
          "event_index_1":a,"event_index_2":b,"frame_1":int(tt[j]),"frame_2":int(tt[j+1]),
          "seconds_1":int(tt[j])/fs,"seconds_2":int(tt[j+1])/fs,"lag_samples":int(tt[j+1]-tt[j]),
          "previous_same_unit_lag_samples":"","following_same_unit_lag_samples":"",
          "segment_id":int(segment[j]),"support_1":"supported","support_2":"supported",
          "template_id_1":int(templates[a]),"template_id_2":int(templates[b]),"status":"resolved"})
    for unit, suffix, role, state_um in slots:
        if role == "short_pair": continue
        ix, tt, state, segment, status = per_unit[unit]
        si = int(np.flatnonzero(cells["states"] == state_um)[0])
        previous = np.r_[0, np.diff(tt)]; following = np.r_[np.diff(tt), 0]
        good = (state == si) & (previous >= 300) & (following >= 300)
        good &= np.asarray([int(i) not in selected for i in ix])
        j = int(np.flatnonzero(good)[0]); a = int(ix[j])
        rows.append({"slot_id":f"u{unit}_{suffix}","unit_id":unit,"role":role,"state_um":state_um,
          "event_index_1":a,"event_index_2":"","frame_1":int(tt[j]),"frame_2":"",
          "seconds_1":int(tt[j])/fs,"seconds_2":"","lag_samples":0,"segment_id":int(segment[j]),
          "previous_same_unit_lag_samples":int(previous[j]),"following_same_unit_lag_samples":int(following[j]),
          "support_1":"supported" if status[j] == 0 else f"status_{int(status[j])}","support_2":"",
          "template_id_1":int(templates[a]),"template_id_2":"","status":"resolved"})
    order = {f"u{u}_{s}":i for i,(u,s,_,_) in enumerate(slots)}; rows.sort(key=lambda r:order[r["slot_id"]])
    with (args.output / "SELECTION_RECEIPT.csv").open("w", newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    mapping=[]
    for unit in sorted(per_unit):
        ix=per_unit[unit][0]; ids,count=np.unique(np.asarray(templates[ix]),return_counts=True)
        mapping.append({"unit_id":unit,"event_count":len(ix),"template_distribution":
                        {str(int(k)):int(v) for k,v in zip(ids,count)}})
    write_json(args.output / "UNIT_TEMPLATE_MAPPING.json", mapping)

    exported=np.load(root / "templates.npy",mmap_mode="r"); footprint=[]
    for unit in sorted(per_unit):
        tid=int(next(iter(mapping[[m["unit_id"] for m in mapping].index(unit)]["template_distribution"])))
        wave=np.asarray(exported[tid],dtype=np.float64); p2p=np.ptp(wave,axis=0); peak=int(np.argmax(p2p))
        active=p2p >= 0.2*float(p2p[peak])
        footprint.append({"unit_id":unit,"exported_template_id":tid,"coordinate_frame":"arm_A_materialized_sorter_channel_coordinates",
          "provenance":"post-sort exported templates.npy; not the unavailable pre-extraction detection bank",
          "template_shape":list(wave.shape),"all_values_finite":bool(np.isfinite(wave).all()),
          "finite_channels":int(np.isfinite(wave).all(axis=0).sum()),"active_channels_at_20pct_peak_p2p":int(active.sum()),
          "peak_channel_index":peak,"peak_x_um":float(geometry[peak,0]),"peak_row_y_um":float(geometry[peak,1]),
          "p2p_min":float(p2p.min()),"p2p_median":float(np.median(p2p)),"p2p_max":float(p2p.max()),
          "interpretation":"model-derived scalar footprint summary; not measured identity or waveform export"})
    write_json(args.output / "MODEL_DERIVED_TEMPLATE_FOOTPRINTS.json", footprint)

    original_manifest=Path("/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0/recording/rescue_recording_manifest.json")
    corrected_manifest=Path("/media/huklaban5/Data/en_rounded_field_ks129_20260929_v1/arm_a_v1/full/recording/rescue_recording_manifest.json")
    om=json.loads(original_manifest.read_text()); cm=json.loads(corrected_manifest.read_text())
    inputs={"schema":"en-a-four-unit-approval-input-bindings-v3","binary_files_opened_or_hashed":False,
      "clock":{"global_frames":[0,num_samples],"sampling_frequency_hz":fs,"channels":384,"dtype":"int16"},
      "original":{"manifest_path":str(original_manifest),"manifest_sha256":sha(original_manifest),
        "binary_path":str(original_manifest.parent / om["recording_binary_files"][0]["name"]),
        "manifest_recorded_binary_sha256":om["recording_binary_files"][0]["sha256"],
        "recording_content_sha256":om["recording_content_sha256"]},
      "corrected_A":{"manifest_path":str(corrected_manifest),"manifest_sha256":sha(corrected_manifest),
        "binary_path":str(corrected_manifest.parent / cm["recording_binary_files"][0]["name"]),
        "manifest_recorded_binary_sha256":cm["recording_binary_files"][0]["sha256"],
        "recording_content_sha256":cm["recording_content_sha256"],"adapter":cm["adapter"],
        "field_sha256":cm["field_sha256"],"source_frames":cm["source_frames"]},
      "saved_metadata":[{"path":str(p),"sha256":sha(p)} for p in [field_path,root/"spike_times.npy",root/"spike_clusters.npy",root/"spike_templates.npy",root/"spike_positions.npy",root/"channel_positions.npy",root/"templates.npy",root/"ops.npy"]],
      "generator":{"path":str(Path(__file__).resolve()),"sha256":sha(Path(__file__).resolve())}}
    write_json(args.output / "INPUT_BINDINGS.json",inputs)
    contract={"schema":"en-a-four-unit-physical-signal-capture-contract-v3","status":"execution_disabled_pending_human_approval",
      "selection_receipt":"SELECTION_RECEIPT.csv","selection_slots":20,"pair_windows":8,"single_event_windows":12,
      "representations":2,"raw_read_calls_max":41,"base_window_reads":40,"duplicate_positive_control_reads":1,
      "logical_input_bytes_max":34603008,"calculated_base_input_bytes":31844352,
      "calculated_max_with_one_max_pair_duplicate":32653824,"retained_numeric_bytes_max":6291456,
      "calculated_retained_int16_bytes":4073472,"ram_bytes_max":2147483648,"cpu_threads_max":4,"wall_seconds_max":300,
      "buffer_policy":"Process one representation/window at a time. Centered/CAR/high-pass float buffers are transient and deleted before the next window; only declared retained int16 snippets may persist H5-locally.",
      "execution_enabled":False,"voltage_read_authorized":False,"numeric_waveform_transfer":False,
      "prohibitions":["sort","matcher replay","RF","holdout","threshold search","parameter sweep","numeric waveform transfer"]}
    write_json(args.output / "CAPTURE_CONTRACT.v3.json",contract)
    shutil.copy2(Path(__file__).resolve(), args.output/"build_en_a_approval_metadata_v3.py")
    (args.output / "README.md").write_text("""# A four-unit approval-request metadata repair v3\n\nStatus: metadata complete; execution and voltage access disabled pending the one authorized H1 re-review and explicit human approval.\n\nThis packet preserves all 20 frozen selection rules. `FOUR_UNIT_EVENT_METADATA.npz` contains complete non-waveform metadata for all 512,838 events from the four units: global indices, frames, unit/template label IDs, model-derived saved y positions, physical states, numeric contiguous-state segments, endpoint support, and neighboring same-unit lags. Together with the included generator source, field/geometry/array hashes, it reproduces earliest/original-adjacency claims rather than only selected rows. No waveform or numeric template array is transferred.\n\n`SELECTION_RECEIPT.csv` resolves all 20 slots. `MODEL_DERIVED_TEMPLATE_FOOTPRINTS.json` contains only scalar summaries from post-sort exported templates and is explicitly not the unavailable pre-extraction detection bank. Both recording identities are manifest-bound; neither binary was opened or newly hashed.\n\nThe disabled capture contract preserves 41 reads and a 32,653,824-byte computed maximum under a 33 MiB cap. Float stages are transient, one representation/window at a time.\n\nImplementation checks\n- Done: included generator source and complete four-unit event metadata sufficient to independently rerun all frozen selection predicates.\n- Done: resolved exact frame/time, numeric segment, support, template label, and neighbor-gap evidence for 20/20 slots.\n- Done: bound original/corrected manifests and manifest-recorded identities without binary access.\n- Done: retained only scalar, model-derived footprint summaries; no waveform/template values.\n- Not done: voltage, real covariance, sort/matcher, RF/holdout, H1 independent re-review, or human approval.\n- Can establish: complete reproducibility of metadata selections and bounded approval request inputs.\n- Cannot establish: physical signal behavior, identity/purity, preprocessing causality, or scientific benefit.\n""")
    products=[]
    for p in sorted(args.output.iterdir()): products.append({"path":p.name,"bytes":p.stat().st_size,"sha256":sha(p)})
    write_json(args.output / "MANIFEST.json",{"schema":"en-a-four-unit-approval-metadata-v3-manifest","products":products})
    write_json(args.output / "COMPLETE.json",{"schema":"en-a-four-unit-approval-metadata-v3-complete","status":"complete_metadata_execution_disabled","manifest_sha256":sha(args.output/"MANIFEST.json")})


if __name__ == "__main__": main()
