"""Prepare the bounded Luke0804 imec0 MEDiCINe-aware merge replay.

This command never reads or rewrites AP voltage.  ``audit`` builds the geometry,
support, and per-spike-depth sidecars.  ``cluster`` reruns only Kilosort4's final
clustering on the saved 930--1230 s detections/features and stops before merging.
Both stages are atomic and provenance guarded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from pipeline.config import fingerprint
from pipeline.motion_aware_merge import (
    MERGE_REPLAY_SCHEMA,
    aligned_template_similarity,
    exact_same_column_pairs,
    medicine_basis_weights,
    squared_pc_feature_depths,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SORT = Path(
    "/mnt/NPX/Luke/20250804/"
    "rescue_pipeline_results_Luke0804_V2V1_g0_imec0/kilosort4/sorter_output"
)
DEFAULT_FIELD = (
    ROOT / "testing/outputs/luke_screened_medicine_300s_v1/"
    "fit_5sigma_relaxed_10000/field.npz"
)
DEFAULT_FIELD_INPUT = (
    ROOT / "testing/outputs/luke_screened_medicine_300s_v1/input_5sigma_relaxed"
)
DEFAULT_OUTPUT = ROOT / "testing/outputs/luke_medicine_motion_aware_merge_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def input_record(path: Path, *, content_hash: bool) -> dict:
    stat = path.stat()
    result = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if content_hash:
        result["sha256"] = sha256(path)
    return result


def begin_atomic_stage(output: Path, request: dict) -> tuple[Path, str]:
    digest = fingerprint(request)
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise RuntimeError(f"Incomplete stage requires inspection: {partial}")
    if output.exists():
        receipt = json.loads((output / "receipt.json").read_text())
        if receipt.get("request_digest") != digest or not receipt.get("complete"):
            raise RuntimeError("Existing output belongs to another or incomplete request")
        return output, digest
    partial.mkdir(parents=True)
    atomic_json(partial / "request.json", {**request, "request_digest": digest})
    return partial, digest


def finish_atomic_stage(partial: Path, output: Path, receipt: dict) -> dict:
    atomic_json(partial / "receipt.json", receipt)
    os.replace(partial, output)
    return receipt


def window_indices(st: np.ndarray, fs: float, start_s: float, stop_s: float) -> tuple[int, int]:
    if st.ndim != 2 or st.shape[1] < 2 or stop_s <= start_s:
        raise ValueError("invalid st or window")
    times = st[:, 0]
    if np.any(np.diff(times[: min(len(times), 1_000_000)]) < 0):
        raise ValueError("full_st must be time sorted")
    lo = int(np.searchsorted(times, round(start_s * fs), side="left"))
    hi = int(np.searchsorted(times, round(stop_s * fs), side="left"))
    if hi <= lo:
        raise ValueError("selected window contains no spikes")
    return lo, hi


def geometry_audit(positions: np.ndarray, shanks: np.ndarray) -> dict:
    tested = {}
    for shift in (0.0, 20.0, 40.0, 80.0, -40.0, -80.0):
        target, source = exact_same_column_pairs(
            positions, shift, channel_shanks=shanks
        )
        tested[str(int(shift))] = int(len(source))
        if len(source):
            if not np.allclose(positions[target, 0], positions[source, 0]):
                raise RuntimeError("x identity failed")
            if not np.allclose(positions[target, 1], positions[source, 1] + shift):
                raise RuntimeError("y translation failed")
    if tested["20"] != 0 or tested["40"] == 0:
        raise RuntimeError("imec0 staggered-geometry premise failed")
    return {
        "channel_count": int(len(positions)),
        "depth_row_spacing_um": float(np.min(np.diff(np.unique(positions[:, 1])))),
        "smallest_exact_same_column_shift_um": 40.0,
        "coordinate_matched_channel_counts": tested,
    }


def synthetic_translation_audit(positions: np.ndarray, shanks: np.ndarray) -> dict:
    rng = np.random.default_rng(990804)
    base = rng.normal(size=(len(positions), 6))
    base[np.linalg.norm(base, axis=1) < 1.0] = 0
    results = {}
    for shift in (40.0, 80.0):
        target, source = exact_same_column_pairs(positions, shift, channel_shanks=shanks)
        shifted = np.zeros_like(base)
        shifted[target] = base[source]
        correct = aligned_template_similarity(
            shifted, base, positions, shift, channel_shanks=shanks,
            temporal_lags=(0,), min_channels=8,
        )
        wrong = aligned_template_similarity(
            shifted, base, positions, 0, channel_shanks=shanks,
            temporal_lags=(0,), min_channels=8,
        )
        if correct["score"] < 1 - 1e-12:
            raise RuntimeError("known translation was not recovered")
        results[str(int(shift))] = {"correct": correct, "zero_offset": wrong}
    return results


def run_audit(args) -> dict:
    output = args.output / "audit"
    required = [
        args.sort / name for name in
        ("full_st.npy", "tF.npy", "ops.npy", "channel_positions.npy", "channel_shanks.npy")
    ] + [args.field, args.field_input / "peaks.npy", args.field_input / "locations.npy", args.field_input / "input.json"]
    request = {
        "schema": MERGE_REPLAY_SCHEMA,
        "stage": "audit",
        "window_s": [args.start_s, args.stop_s],
        "inputs": [input_record(p, content_hash=args.hash_large or p.stat().st_size < 2_000_000_000) for p in required],
        "observed_depth": "pinned_ks4_squared_pc_feature_energy_centroid",
        "field_support": "medicine_triangular_basis_kish_effective_sample_size",
        "voltage_modified": False,
    }
    stage, request_digest = begin_atomic_stage(output, request)
    if stage == output:
        return json.loads((output / "receipt.json").read_text())
    try:
        positions = np.load(args.sort / "channel_positions.npy")
        shanks = np.load(args.sort / "channel_shanks.npy").reshape(-1)
        geometry = geometry_audit(positions, shanks)
        synthetic = synthetic_translation_audit(positions, shanks)

        raw_ops = np.load(args.sort / "ops.npy", allow_pickle=True).item()
        fs = float(raw_ops["fs"])
        st = np.load(args.sort / "full_st.npy", mmap_mode="r")
        t_f = np.load(args.sort / "tF.npy", mmap_mode="r")
        if len(st) != len(t_f):
            raise RuntimeError("full_st/tF length mismatch")
        lo, hi = window_indices(st, fs, args.start_s, args.stop_s)
        detection_templates = np.asarray(st[lo:hi, 1], dtype=np.int64)
        depths, depth_eligible = squared_pc_feature_depths(
            t_f[lo:hi], detection_templates, raw_ops["iCC"], raw_ops["iU"],
            raw_ops["yc"], chunk_spikes=args.chunk_spikes,
        )
        np.save(stage / "source_spike_indices.npy", np.arange(lo, hi, dtype=np.int64))
        np.save(stage / "observed_depth_um.npy", depths)
        np.save(stage / "observed_depth_eligible.npy", depth_eligible)
        np.save(stage / "detection_templates.npy", detection_templates)

        field = np.load(args.field)
        if set(field.files) != {"time_s", "depth_um", "displacement_um"}:
            raise RuntimeError(f"unexpected saved field members: {field.files}")
        input_cfg = json.loads((args.field_input / "input.json").read_text())
        peaks = np.load(args.field_input / "peaks.npy", mmap_mode="r")
        locations = np.load(args.field_input / "locations.npy", mmap_mode="r")
        peak_times = peaks["sample_index"] / float(input_cfg["sampling_frequency_hz"]) + float(input_cfg["start_s"])
        raw, total, effective = medicine_basis_weights(
            peak_times, locations["y"], field["time_s"], field["depth_um"],
            time_kernel_width_s=1.0,
        )
        np.savez_compressed(
            stage / "field_support_audit.npz", raw_peak_count=raw,
            total_basis_weight=total, kish_effective_sample_size=effective,
            time_s=field["time_s"], depth_um=field["depth_um"],
        )
        receipt = {
            **request,
            "request_digest": request_digest,
            "complete": True,
            "window_source_indices": [lo, hi],
            "window_spikes": hi - lo,
            "sampling_frequency_hz": fs,
            "geometry": geometry,
            "synthetic_translation": synthetic,
            "observed_depth_eligible_fraction": float(np.mean(depth_eligible)),
            "support_summary": {
                "raw_count_min": int(raw.min()),
                "raw_count_median": float(np.median(raw)),
                "effective_min": float(effective.min()),
                "effective_median": float(np.median(effective)),
            },
            "qualification_status": "support_audited_split_half_not_yet_run",
        }
        return finish_atomic_stage(stage, output, receipt)
    except Exception:
        # Preserve the partial directory as required by the experiment contract.
        raise


def run_cluster(args) -> dict:
    audit = args.output / "audit"
    if not (audit / "receipt.json").exists():
        raise RuntimeError("Run and accept the audit stage first")
    audit_receipt = json.loads((audit / "receipt.json").read_text())
    if not audit_receipt.get("complete"):
        raise RuntimeError("Audit is incomplete")
    output = args.output / "bounded_premerge"
    request = {
        "schema": MERGE_REPLAY_SCHEMA,
        "stage": "bounded_final_clustering_before_merge",
        "audit_request_digest": audit_receipt["request_digest"],
        "window_s": [args.start_s, args.stop_s],
        "source_ops": input_record(args.sort / "ops.npy", content_hash=True),
        "source_st": input_record(args.sort / "full_st.npy", content_hash=args.hash_large),
        "source_tF": input_record(args.sort / "tF.npy", content_hash=args.hash_large),
        "kilosort_operation": "clustering_qr.run(mode=template); merging_function_not_called",
        "voltage_read": False,
        "voltage_modified": False,
    }
    stage, request_digest = begin_atomic_stage(output, request)
    if stage == output:
        return json.loads((output / "receipt.json").read_text())
    try:
        import torch
        from kilosort import clustering_qr
        from kilosort.io import load_ops

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ops = load_ops(args.sort / "ops.npy", device=torch.device("cpu"))
        fs = float(ops["fs"])
        full_st = np.load(args.sort / "full_st.npy", mmap_mode="r")
        full_t_f = np.load(args.sort / "tF.npy", mmap_mode="r")
        lo, hi = window_indices(full_st, fs, args.start_s, args.stop_s)
        st = np.asarray(full_st[lo:hi]).copy()
        t_f = torch.from_numpy(np.asarray(full_t_f[lo:hi]).copy())
        clu, wall = clustering_qr.run(ops, st, t_f, mode="template", device=device)
        np.save(stage / "st.npy", st)
        np.save(stage / "clu.npy", clu)
        np.save(stage / "Wall.npy", wall.detach().cpu().numpy())
        np.save(stage / "source_spike_indices.npy", np.arange(lo, hi, dtype=np.int64))
        receipt = {
            **request,
            "request_digest": request_digest,
            "complete": True,
            "device": str(device),
            "spikes": int(len(st)),
            "premerge_clusters": int(np.unique(clu).size),
            "wall_shape": list(wall.shape),
            "merging_function_called": False,
        }
        return finish_atomic_stage(stage, output, receipt)
    except Exception:
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "cluster"))
    parser.add_argument("--sort", type=Path, default=DEFAULT_SORT)
    parser.add_argument("--field", type=Path, default=DEFAULT_FIELD)
    parser.add_argument("--field-input", type=Path, default=DEFAULT_FIELD_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-s", type=float, default=930.0)
    parser.add_argument("--stop-s", type=float, default=1230.0)
    parser.add_argument("--chunk-spikes", type=int, default=100_000)
    parser.add_argument("--hash-large", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = run_audit(args) if args.stage == "audit" else run_cluster(args)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
