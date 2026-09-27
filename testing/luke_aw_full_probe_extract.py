#!/usr/bin/env python3
"""Authorized AW CPU-only full-probe final-label template extraction.

This reads W2 voltage without modifying it, caches exact ibllikecmr output in
/dev/shm, constructs full-probe templates for the approved scalar prescreen,
and measures support. It never launches a sort and never uses CUDA.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DARTSORT_ROOT = Path("/home/huklab/Documents/DARTsort")
HANDOFF = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/av_aw_handoff_20260926_v1")
SUPPLEMENT = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/av_aw_handoff_20260926_v1_qc_supplement"
)
RAW_ROOT = Path("/media/huklab/Expansion/Luke0804_V2V1_g0")
MASK = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/"
    "luke_imec1_medicine_reference_sweep_v2/stage4_ab/censor_mask_v1.csv"
)
HUB_MASK = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/incoming/censor_mask_v1/"
    "censor_mask_v1.csv"
)
OUT_BASE = ROOT / "testing/outputs/luke_au_cpu_preparation/full_probe_extraction_v1"
OUT = OUT_BASE / os.environ.get("LUKE_AW_ATTEMPT", "attempt2")
SCRATCH = Path("/dev/shm/luke_aw_full_probe_extract_v1")
START_S = 900.0
END_S = 1240.0
MAX_CANDIDATES = 90
MAX_WALL_S = 60.0 * 60.0
MAX_PERSISTENT_BYTES = 200_000_000
N_JOBS = 2
STATES_UM = np.array([0.0, -40.0, -80.0, -120.0, -160.0, -200.0, -240.0])


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def normalized_mask(path: Path) -> list[tuple[float, float, str]]:
    with path.open(newline="") as stream:
        return [
            (
                float(row["start_s"]),
                float(row["end_s"]),
                "".join(sorted(set(row["source"].replace("+", "")))),
            )
            for row in csv.DictReader(stream)
        ]


def inside_intervals(times_s: np.ndarray, rows: list[tuple[float, float, str]]) -> np.ndarray:
    result = np.zeros(times_s.size, dtype=bool)
    for start_s, end_s, _ in rows:
        result |= (times_s >= start_s) & (times_s < end_s)
    return result


def evenly_spaced(indices: np.ndarray, count: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size <= count:
        return indices.copy()
    take = np.rint(np.linspace(0, indices.size - 1, count)).astype(np.int64)
    if np.unique(take).size != count:
        raise RuntimeError("deterministic temporal subsample was not unique")
    return indices[take]


def centered_cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    av -= av.mean()
    bv -= bv.mean()
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    return float(np.dot(av, bv) / denom) if denom else float("nan")


def exact_map(geom: np.ndarray, shift_um: float) -> np.ndarray:
    geom = np.asarray(geom, dtype=np.float64)
    out = np.full(geom.shape[0], -1, dtype=np.int64)
    for source, (x_um, y_um) in enumerate(geom):
        hit = np.flatnonzero(
            np.isclose(geom[:, 0], x_um, atol=1e-6, rtol=0.0)
            & np.isclose(geom[:, 1], y_um + shift_um, atol=1e-6, rtol=0.0)
        )
        if hit.size > 1:
            raise RuntimeError(f"ambiguous exact map for source {source}")
        if hit.size == 1:
            out[source] = int(hit[0])
    return out


def remap(template: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    valid = mapping >= 0
    targets = mapping[valid]
    if np.unique(targets).size != targets.size:
        raise RuntimeError("exact map is many-to-one")
    out = np.zeros_like(template)
    out[:, targets] = template[:, valid]
    return out


def build_sorting(times: np.ndarray, channels: np.ndarray, labels: np.ndarray, fs: float):
    from dartsort.util.data_util import DARTsortSorting

    return DARTsortSorting(
        times_samples=np.asarray(times, dtype=np.int64),
        channels=np.asarray(channels, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int32),
        sampling_frequency=fs,
    )


def ibllikecmr_retain_measured_bad_channels(rec):
    """Exact ibllikecmr on good channels while retaining real bad-channel traces.

    DARTsort's standard strategy removes detected bad channels. The AZ support
    denominator is explicitly all 384 measured channels. We therefore use the
    same detected-good set for both common references and the same per-channel
    noise scaling, but retain the real trace of any detected bad channel rather
    than padding or extrapolating it. The returned good-channel traces are
    numerically checked against the ordinary removed-channel construction.
    """
    import spikeinterface.full as si

    filtered = rec.astype(np.float32)
    filtered = si.highpass_filter(filtered)
    if "inter_sample_shift" in filtered.get_property_keys():
        filtered = si.phase_shift(filtered)
    bad_ids, bad_labels = si.detect_bad_channels(filtered, seed=0)
    bad_ids = np.asarray(bad_ids)
    all_ids = np.asarray(filtered.get_channel_ids())
    good_ids = all_ids[~np.isin(all_ids, bad_ids)]
    if not good_ids.size:
        raise RuntimeError("bad-channel detector removed every channel")

    first_full = si.common_reference(filtered, ref_channel_ids=good_ids.tolist())
    noise = si.get_noise_levels(
        first_full,
        return_in_uV=False,
        random_slices_kwargs=dict(seed=0, num_chunks_per_segment=100),
    )
    retained = si.scale(first_full, gain=1.0 / noise)
    retained = si.common_reference(retained, ref_channel_ids=good_ids.tolist())
    retained = retained.astype("float32")

    # Independent operator construction for the good subset, reusing only the
    # already measured per-channel noise values. This does not trigger a second
    # noise/random-read pass.
    good_pos = np.flatnonzero(np.isin(all_ids, good_ids))
    ordinary = filtered.select_channels(good_ids.tolist())
    ordinary = si.common_reference(ordinary)
    ordinary = si.scale(ordinary, gain=1.0 / noise[good_pos])
    ordinary = si.common_reference(ordinary).astype("float32")
    return retained, ordinary, bad_ids, np.asarray(bad_labels)


def build_templates(recording, sorting, template_cfg, waveform_cfg, computation_cfg, tsvd):
    from dartsort.templates import TemplateData
    from dartsort.util.motion import MotionInfo

    motion = MotionInfo.from_motion_est(geom=recording.get_channel_locations())
    return TemplateData.from_config(
        recording=recording,
        sorting=sorting,
        template_cfg=template_cfg,
        waveform_cfg=waveform_cfg,
        motion=motion,
        tsvd=tsvd,
        computation_cfg=computation_cfg,
        save_folder=None,
        show_progress=True,
    )


def select_event_rows(
    times: np.ndarray,
    channels: np.ndarray,
    labels: np.ndarray,
    unit_to_indices: dict[int, np.ndarray],
    per_unit_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chosen = []
    for unit_id in sorted(unit_to_indices):
        chosen.append(evenly_spaced(unit_to_indices[unit_id], per_unit_count))
    ix = np.sort(np.concatenate(chosen))
    return times[ix], channels[ix], labels[ix]


def support_rows(templates: np.ndarray, geom: np.ndarray, crop_ix: np.ndarray, unit_ids: np.ndarray):
    total = np.square(templates, dtype=np.float64).sum(axis=(1, 2))
    rows = []
    mappings = {float(s): exact_map(geom, float(s)) for s in STATES_UM}
    reverse = {float(s): exact_map(geom, float(-s)) for s in STATES_UM}
    crop_keep = np.zeros(geom.shape[0], dtype=bool)
    crop_keep[crop_ix] = True
    for ui, unit_id in enumerate(unit_ids):
        template = templates[ui]
        source_ptp = float(np.ptp(template, axis=0).max())
        for state in STATES_UM:
            moved = remap(template, mappings[float(state)])
            retained = float(np.square(moved[:, crop_ix], dtype=np.float64).sum() / total[ui])
            target_only = moved.copy()
            target_only[:, ~crop_keep] = 0
            restored = remap(target_only, reverse[float(state)])
            restored_ptp = float(np.ptp(restored, axis=0).max())
            ratio = restored_ptp / source_ptp if source_ptp else float("nan")
            cosine = centered_cosine(template, restored)
            rows.append(
                {
                    "unit_id": int(unit_id),
                    "state_um": float(state),
                    "retained_energy_fraction": retained,
                    "roundtrip_centered_cosine": cosine,
                    "roundtrip_ptp_ratio": ratio,
                    "support_pass": bool(retained >= 0.99),
                    "operator_pass": bool(cosine >= 0.99 and 0.98 <= ratio <= 1.02),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run() -> None:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(
        OUT / "RUNNING.json",
        {
            "status": "running",
            "pid": os.getpid(),
            "started_unix": time.time(),
            "authorization": "user approved 2026-09-26",
        },
    )
    masks = normalized_mask(MASK)
    hub_masks = normalized_mask(HUB_MASK)
    if masks != hub_masks:
        raise RuntimeError("prescreen mask is not numerically equal to canonical AE mask")

    # Import the sealed DARTsort environment only after basic preflight.
    sys.path.insert(0, str(DARTSORT_ROOT))
    sys.path.insert(0, str(DARTSORT_ROOT / "src"))
    from experiments.dataset_pipeline.pipeline import resolved_sorter_config
    from spikeinterface.extractors import read_spikeglx
    from dartsort.templates import TemplateData

    static = HANDOFF / "static_w2"
    config = json.loads((static / "config.json").read_text())
    input_manifest = json.loads((static / "input-manifest.json").read_text())
    window = input_manifest["window"]
    fs_expected = float(window["sampling_frequency"])
    source_ids = np.asarray(window["source_channel_ids"])
    sort_ids = np.asarray(window["requested_sort_channel_ids"])
    if source_ids.size != 384 or sort_ids.size != 182:
        raise RuntimeError("unexpected 384/182 channel contract")

    with np.load(static / "dartsort_sorting.npz", allow_pickle=False) as z:
        times = np.asarray(z["times_samples"], dtype=np.int64)
        crop_channels = np.asarray(z["channels"], dtype=np.int64)
        labels = np.asarray(z["labels"], dtype=np.int32)
        fs = float(z["sampling_frequency"])
    if not np.isclose(fs, fs_expected, rtol=0, atol=1e-9):
        raise RuntimeError("sorting/input sampling-frequency mismatch")

    qc_rows = {}
    with (SUPPLEMENT / "sorting_qc.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            qc_rows[int(row["unit_id"])] = row
    session_times = START_S + times / fs
    rest = ~inside_intervals(session_times, masks)
    edge = (times >= 42) & (times + 79 <= int(window["end_frame"] - window["start_frame"]))
    final = labels >= 0
    candidates = []
    unit_to_rest = {}
    for unit_id in sorted(qc_rows):
        unit = labels == unit_id
        n = int(unit.sum())
        rest_ix = np.flatnonzero(unit & rest & edge)
        rest_fraction = float((unit & rest).sum() / n) if n else 0.0
        isi = float(qc_rows[unit_id]["raw_adjacent_isi_lt_refractory_fraction"])
        passed_first_three = n >= 800 and isi <= 0.005 and rest_fraction >= 0.80
        passed = passed_first_three and rest_ix.size >= 400
        candidates.append(
            {
                "unit_id": unit_id,
                "spike_count": n,
                "raw_adjacent_isi_lt_1ms_fraction": isi,
                "rest_spike_fraction": rest_fraction,
                "valid_rest_waveforms": int(rest_ix.size),
                "first_three_pass": passed_first_three,
                "scalar_and_waveform_support_pass": passed,
            }
        )
        if passed:
            unit_to_rest[unit_id] = rest_ix
    if len(unit_to_rest) > MAX_CANDIDATES:
        raise RuntimeError(f"candidate count {len(unit_to_rest)} exceeds approved cap")
    write_csv(OUT / "candidate_prescreen.csv", candidates)

    raw = read_spikeglx(RAW_ROOT, stream_id="imec1.ap")
    if not np.isclose(raw.get_sampling_frequency(), fs, rtol=0, atol=1e-9):
        raise RuntimeError("local raw/source clock mismatch")
    if not np.array_equal(raw.get_channel_ids(), source_ids):
        raise RuntimeError("local raw/source channel IDs mismatch")
    start_frame = int(window["start_frame"])
    end_frame = int(window["end_frame"])
    rec = raw.frame_slice(start_frame=start_frame, end_frame=end_frame)
    rec.reset_times()
    pre, ordinary_good, bad_ids, bad_labels = ibllikecmr_retain_measured_bad_channels(rec)
    if pre.get_num_channels() != 384 or not np.array_equal(pre.get_channel_ids(), source_ids):
        raise RuntimeError("retained-channel preprocessing changed the sealed 384-channel contract")
    good_ids = source_ids[~np.isin(source_ids, bad_ids)]
    check_chunks = [
        (0, min(round(fs), pre.get_num_frames())),
        (pre.get_num_frames() // 2, pre.get_num_frames() // 2 + round(fs)),
        (max(0, pre.get_num_frames() - round(fs)), pre.get_num_frames()),
    ]
    good_equivalence_max_abs = 0.0
    for a, b in check_chunks:
        retained_good = pre.get_traces(a, b, channel_ids=good_ids.tolist())
        ordinary_trace = ordinary_good.get_traces(a, b)
        good_equivalence_max_abs = max(
            good_equivalence_max_abs,
            float(np.max(np.abs(retained_good - ordinary_trace))),
        )
    if good_equivalence_max_abs > 1e-5:
        raise RuntimeError(
            f"retained-channel repair changed ordinary good traces by {good_equivalence_max_abs}"
        )
    atomic_json(
        OUT / "preprocessing_repair_receipt.json",
        {
            "status": "equivalent_good_channels_plus_measured_bad_channels",
            "detected_bad_channel_ids": [str(x) for x in bad_ids],
            "channel_labels_in_source_order": [str(x) for x in bad_labels],
            "good_channel_count": int(good_ids.size),
            "retained_channel_count": int(pre.get_num_channels()),
            "good_trace_equivalence_max_abs_three_1s_chunks": good_equivalence_max_abs,
            "bad_channel_policy": "retain actual filtered/referenced/noise-scaled trace; no padding or extrapolation",
        },
    )

    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    cache_dir = SCRATCH / "preprocessed"
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    cached = pre.save(
        folder=cache_dir,
        format="binary",
        n_jobs=N_JOBS,
        chunk_duration="1s",
        progress_bar=True,
    )
    if cached.get_num_channels() != 384:
        raise RuntimeError("cached preprocessing lost channels")

    full_index = {str(cid): i for i, cid in enumerate(cached.get_channel_ids())}
    crop_ix = np.asarray([full_index[str(cid)] for cid in sort_ids], dtype=np.int64)
    full_channels = np.asarray([crop_ix[ch] for ch in crop_channels], dtype=np.int64)
    crop = cached.select_channels(sort_ids.tolist())

    cfg = resolved_sorter_config(config, 384, "sort")
    computation_cfg = dataclasses.replace(
        cfg.computation_cfg,
        device="cpu",
        n_jobs_cpu=N_JOBS,
        n_jobs_gpu=0,
        n_jobs_small=N_JOBS,
        n_jobs_small_gpu=0,
    )
    # Full-probe residual whitening cannot be reconstructed from the compact
    # handoff. Support is therefore measured before spatial whitening, while
    # retaining the sealed temporal SVD and DARTsort peelreduce-median builder.
    template_cfg = dataclasses.replace(
        cfg.template_cfg,
        whitening=dataclasses.replace(cfg.template_cfg.whitening, strategy="none"),
        registered_templates=False,
    )
    sealed_td = TemplateData.from_npz(static / "template_data.npz")
    tsvd = sealed_td.tsvd
    if tsvd is None or tsvd.components_.shape != (5, 121):
        raise RuntimeError("sealed temporal SVD is missing or has wrong shape")

    if not unit_to_rest:
        raise RuntimeError("no candidate has 400 valid rest waveforms")
    first_unit = sorted(unit_to_rest)[0]
    preflight_ix = evenly_spaced(unit_to_rest[first_unit], 400)
    pre_full = build_sorting(times[preflight_ix], full_channels[preflight_ix], labels[preflight_ix], fs)
    pre_crop = build_sorting(times[preflight_ix], crop_channels[preflight_ix], labels[preflight_ix], fs)
    td_full = build_templates(cached, pre_full, template_cfg, cfg.waveform_cfg, computation_cfg, tsvd)
    td_crop = build_templates(crop, pre_crop, template_cfg, cfg.waveform_cfg, computation_cfg, tsvd)
    full_crop_template = td_full.templates[0][:, crop_ix]
    crop_template = td_crop.templates[0]
    preflight_cos = centered_cosine(full_crop_template, crop_template)
    preflight_ratio = float(
        np.ptp(full_crop_template, axis=0).max() / np.ptp(crop_template, axis=0).max()
    )
    preflight_max = float(np.max(np.abs(full_crop_template - crop_template)))
    preflight_pass = bool(preflight_cos >= 0.99 and 0.98 <= preflight_ratio <= 1.02)
    atomic_json(
        OUT / "crop_overlap_preflight.json",
        {
            "unit_id": first_unit,
            "waveforms": int(preflight_ix.size),
            "centered_cosine": preflight_cos,
            "ptp_ratio_full_crop_over_crop": preflight_ratio,
            "max_abs_difference": preflight_max,
            "pass": preflight_pass,
            "builder": "DARTsort peelreduce median, sealed temporal SVD, spatial whitening disabled for full-probe support measurement",
        },
    )
    if not preflight_pass:
        raise RuntimeError("crop-overlap builder preflight failed")
    if time.monotonic() - started >= MAX_WALL_S:
        raise TimeoutError("wall cap reached after preflight")

    main_times, main_channels, main_labels = select_event_rows(
        times, full_channels, labels, unit_to_rest, 500
    )
    main_sorting = build_sorting(main_times, main_channels, main_labels, fs)
    td = build_templates(cached, main_sorting, template_cfg, cfg.waveform_cfg, computation_cfg, tsvd)
    if set(map(int, td.unit_ids)) != set(unit_to_rest):
        raise RuntimeError("template builder output units differ from candidate set")

    split_source = {u: evenly_spaced(ix, 400) for u, ix in unit_to_rest.items()}
    left = {u: ix[::2] for u, ix in split_source.items()}
    right = {u: ix[1::2] for u, ix in split_source.items()}
    split_data = []
    split_templates = []
    for name, selection in (("left", left), ("right", right)):
        st, sc, sl = select_event_rows(times, full_channels, labels, selection, 200)
        split_td = build_templates(
            cached,
            build_sorting(st, sc, sl, fs),
            template_cfg,
            cfg.waveform_cfg,
            computation_cfg,
            tsvd,
        )
        split_templates.append(split_td.templates)
        if not np.array_equal(split_td.unit_ids, td.unit_ids):
            raise RuntimeError(f"{name} split output unit order mismatch")
    for i, unit_id in enumerate(td.unit_ids):
        split_data.append(
            {
                "unit_id": int(unit_id),
                "split_centered_cosine": centered_cosine(split_templates[0][i], split_templates[1][i]),
                "left_to_full_centered_cosine": centered_cosine(split_templates[0][i], td.templates[i]),
                "right_to_full_centered_cosine": centered_cosine(split_templates[1][i], td.templates[i]),
                "left_to_full_ptp_ratio": float(np.ptp(split_templates[0][i], axis=0).max() / np.ptp(td.templates[i], axis=0).max()),
                "right_to_full_ptp_ratio": float(np.ptp(split_templates[1][i], axis=0).max() / np.ptp(td.templates[i], axis=0).max()),
            }
        )
    write_csv(OUT / "split_half_diagnostics.csv", split_data)

    support = support_rows(td.templates, cached.get_channel_locations(), crop_ix, td.unit_ids)
    write_csv(OUT / "actual_state_support.csv", support)
    support_by_unit = {}
    for unit_id in td.unit_ids:
        ur = [row for row in support if row["unit_id"] == int(unit_id)]
        support_by_unit[int(unit_id)] = all(r["support_pass"] and r["operator_pass"] for r in ur)

    template_rows = []
    geom = cached.get_channel_locations()
    for i, unit_id in enumerate(td.unit_ids):
        channel_ptp = np.ptp(td.templates[i], axis=0)
        peak_channel = int(np.argmax(channel_ptp))
        scalar = next(r for r in candidates if r["unit_id"] == int(unit_id))
        template_rows.append(
            {
                **scalar,
                "peak_channel_index": peak_channel,
                "peak_channel_id": str(source_ids[peak_channel]),
                "depth_um": float(geom[peak_channel, 1]),
                "full_probe_ptp_standardized": float(channel_ptp[peak_channel]),
                "all_state_support_and_operator_pass": support_by_unit[int(unit_id)],
            }
        )
    write_csv(OUT / "full_probe_candidate_measurements.csv", template_rows)
    np.savez_compressed(
        OUT / "full_probe_final_label_templates.npz",
        templates=np.asarray(td.templates, dtype=np.float32),
        unit_ids=np.asarray(td.unit_ids, dtype=np.int64),
        spike_counts=np.asarray(td.spike_counts, dtype=np.int32),
        channel_ids=source_ids,
        geometry_um=np.asarray(geom, dtype=np.float64),
        sampling_frequency=np.asarray(fs),
        trough_offset_samples=np.asarray(td.trough_offset_samples),
        measurement_domain=np.asarray("384-channel ibllikecmr, no spatial whitening"),
    )

    elapsed = time.monotonic() - started
    if elapsed > MAX_WALL_S:
        raise TimeoutError("wall cap exceeded")
    files = [p for p in OUT.iterdir() if p.is_file() and p.name != "RUNNING.json"]
    persistent = sum(p.stat().st_size for p in files)
    if persistent > MAX_PERSISTENT_BYTES:
        raise RuntimeError(f"persistent output {persistent} exceeds approved cap")
    summary = {
        "schema": "luke-aw-full-probe-extraction-v1",
        "status": "measurement_complete_selection_pending",
        "elapsed_s": elapsed,
        "candidate_first_three_count": int(sum(r["first_three_pass"] for r in candidates)),
        "candidate_with_400_valid_rest_count": len(unit_to_rest),
        "all_state_support_and_operator_pass_count": int(sum(support_by_unit.values())),
        "selection_performed": False,
        "sorting_performed": False,
        "gpu_used": False,
        "voltage_modified": False,
        "sampling_frequency_hz": fs,
        "measurement_channels": 384,
        "deployed_sorter_channels": 182,
        "mask_sha256": digest(MASK),
        "hub_mask_sha256": digest(HUB_MASK),
        "masks_numerically_equal": True,
        "spatial_whitening": "disabled because compact handoff lacks a full-probe residual model; support measured on standardized temporally denoised templates",
        "retained_detected_bad_channel_ids": [str(x) for x in bad_ids],
        "ordinary_good_trace_equivalence_max_abs": good_equivalence_max_abs,
        "persistent_bytes": persistent,
        "scratch_removed_on_exit": True,
        "outputs": {p.name: digest(p) for p in files},
    }
    atomic_json(OUT / "extraction_summary.json", summary)
    (OUT / "RUNNING.json").unlink(missing_ok=True)
    atomic_json(OUT / "COMPLETE.json", {"status": "complete", "summary_sha256": digest(OUT / "extraction_summary.json")})


def main() -> None:
    def stop(_signum, _frame):
        raise TimeoutError("terminated by wall-time supervisor")

    signal.signal(signal.SIGTERM, stop)
    try:
        run()
    except Exception as exc:
        atomic_json(
            OUT / "FAILED.json",
            {"status": "failed", "type": type(exc).__name__, "message": str(exc), "time_unix": time.time()},
        )
        raise
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    main()
