"""Read-only validation of the AV -> AW compact handoff.

This never reads voltage, launches a sort, or uses CUDA.  It seals hashes,
records the actual S/D2L matching operators, audits the exact W2 trajectory,
and reports what the shallow saved bank can and cannot establish.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from testing.luke_au_cpu_preparation import (
        centered_cosine,
        exact_remap,
        exact_same_column_map,
        sample_and_quantize_trajectory,
    )
except ModuleNotFoundError:  # direct ``python testing/...py`` execution
    from luke_au_cpu_preparation import (
        centered_cosine,
        exact_remap,
        exact_same_column_map,
        sample_and_quantize_trajectory,
    )


DEFAULT_HANDOFF = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/av_aw_handoff_20260926_v1"
)
DEFAULT_LOCAL_MOTION = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "luke0804-imec1-ao-d2l-v2/windows/W2/runs/D2L/motion"
)
DEFAULT_MASK = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/"
    "luke_imec1_medicine_reference_sweep_v2/stage4_ab/censor_mask_v1.csv"
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def q(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size),
        "median": float(np.median(values)) if values.size else None,
        "p95": float(np.percentile(values, 95)) if values.size else None,
        "max": float(values.max()) if values.size else None,
    }


def load_mask(path: Path) -> list[tuple[float, float]]:
    with path.open(newline="") as stream:
        return [
            (float(row["start_s"]), float(row["end_s"]))
            for row in csv.DictReader(stream)
        ]


def mask_at(times_s: np.ndarray, intervals: list[tuple[float, float]]) -> np.ndarray:
    result = np.zeros(times_s.size, dtype=bool)
    for start_s, end_s in intervals:
        result |= (times_s >= start_s) & (times_s < end_s)
    return result


def matching_summary(path: Path) -> dict:
    payload = json.loads(path.read_text())
    cfg = payload["config"]
    matching = cfg["matching_cfg"]
    return {
        "effective_config_sha256": digest(path),
        "chunk_length_samples": int(matching["chunk_length_samples"]),
        "template_type": matching["template_type"],
        "channel_selection": matching["channel_selection"],
        "threshold": float(matching["threshold"]),
        "up_factor": int(matching["up_factor"]),
        "drift_interp_params": matching["drift_interp_params"],
        "internal_motion_estimation": bool(
            cfg["motion_estimation_cfg"]["do_motion_estimation"]
        ),
        "matching_iterations": int(cfg["matching_iterations"]),
    }


def crop_qualification(
    templates: np.ndarray,
    geom: np.ndarray,
    unit_ids: np.ndarray,
    occupied_states_um: np.ndarray,
    *,
    interpolation_radius_um: float,
) -> dict:
    energy_by_channel = np.sum(np.square(templates, dtype=np.float64), axis=1)
    total_energy = energy_by_channel.sum(axis=1)
    margin_um = float(np.max(np.abs(occupied_states_um)) + interpolation_radius_um)
    y = geom[:, 1]
    core = (y >= y.min() + margin_um) & (y <= y.max() - margin_um)
    interior_fraction = np.divide(
        energy_by_channel[:, core].sum(axis=1),
        total_energy,
        out=np.full(templates.shape[0], np.nan),
        where=total_energy > 0,
    )

    all_states_pass = np.ones(templates.shape[0], dtype=bool)
    state_counts = {}
    for state_um in occupied_states_um:
        forward = exact_same_column_map(geom, float(state_um))
        reverse = exact_same_column_map(geom, -float(state_um))
        passed = np.zeros(templates.shape[0], dtype=bool)
        for i, template in enumerate(templates):
            moved = exact_remap(template, forward)
            restored = exact_remap(moved, reverse)
            retained = float(np.square(moved, dtype=np.float64).sum() / total_energy[i])
            source_ptp = float(np.ptp(template, axis=0).max())
            restored_ptp = float(np.ptp(restored, axis=0).max())
            ptp_ratio = restored_ptp / source_ptp if source_ptp else np.nan
            cosine = centered_cosine(template, restored)
            passed[i] = bool(
                retained > 0.99
                and abs(ptp_ratio - 1.0) <= 0.02
                and cosine >= 0.99
            )
        all_states_pass &= passed
        state_counts[f"{float(state_um):g}"] = int(passed.sum())

    interior_pass = interior_fraction > 0.99
    both = interior_pass & all_states_pass
    return {
        "scope": "observed 182-channel crop only",
        "full_probe_energy_known": False,
        "interpolation_radius_um": float(interpolation_radius_um),
        "max_abs_occupied_state_um": float(np.max(np.abs(occupied_states_um))),
        "margin_um_each_crop_edge": margin_um,
        "core_channel_count": int(core.sum()),
        "templates_total": int(templates.shape[0]),
        "observed_interior_energy_gt_0p99": int(interior_pass.sum()),
        "all_occupied_exact_states_pass": int(all_states_pass.sum()),
        "both_crop_screens_pass": int(both.sum()),
        "both_crop_screens_pass_unit_ids": [int(x) for x in unit_ids[both]],
        "frozen_30_donor_target_feasible_from_crop": bool(both.sum() >= 30),
        "per_state_exact_pass_counts": state_counts,
        "observed_interior_energy_fraction": {
            "min": float(np.nanmin(interior_fraction)),
            "median": float(np.nanmedian(interior_fraction)),
            "p95": float(np.nanpercentile(interior_fraction, 95)),
            "max": float(np.nanmax(interior_fraction)),
        },
    }


def run(handoff: Path, local_motion: Path, mask_path: Path) -> dict:
    manifest_path = handoff / "HANDOFF_MANIFEST.json"
    complete_path = handoff / "COMPLETE.json"
    manifest = json.loads(manifest_path.read_text())
    file_rows = []
    for item in manifest["files"]:
        path = handoff / item["relative_path"]
        actual_hash = digest(path)
        actual_bytes = path.stat().st_size
        file_rows.append(
            {
                "relative_path": item["relative_path"],
                "expected_sha256": item["sha256"],
                "actual_sha256": actual_hash,
                "expected_bytes": int(item["bytes"]),
                "actual_bytes": int(actual_bytes),
                "pass": actual_hash == item["sha256"] and actual_bytes == item["bytes"],
            }
        )

    field_path = handoff / "d2l_field/luke0804_imec1_two_layer_motion.npz"
    local_field_path = local_motion / "fields.npz"
    interface_path = local_motion / "interface-audit.json"
    interface = json.loads(interface_path.read_text())
    with np.load(field_path, allow_pickle=False) as z:
        global_t = np.asarray(z["time_s"], dtype=np.float64)
        global_d = np.asarray(z["displacement_um"][:, 0], dtype=np.float64)
        sign = str(np.asarray(z["sign_convention"]).item())
    with np.load(local_field_path, allow_pickle=False) as z:
        local_t = np.asarray(z["time_bin_centers_s"], dtype=np.float64)
        local_d = np.asarray(z["displacement"][0], dtype=np.float64)
        source_t = np.asarray(z["source_time_s"], dtype=np.float64)

    static_npz = handoff / "static_w2/template_data.npz"
    sorting_npz = handoff / "static_w2/dartsort_sorting.npz"
    with np.load(static_npz, allow_pickle=False) as z:
        templates = np.asarray(z["templates"], dtype=np.float32)
        geom = np.asarray(z["registered_geom"], dtype=np.float64)
        unit_ids = np.asarray(z["unit_ids"], dtype=np.int64)
        fs_hz = float(z["sampling_frequency"])
        template_keys = sorted(z.files)
    with np.load(sorting_npz, allow_pickle=False) as z:
        sorting_keys = sorted(z.files)

    cfg = json.loads((handoff / "d2l_acceptance/config.json").read_text())
    start_s, end_s = map(float, cfg["window_seconds"])
    start_frame = round(start_s * fs_hz)
    end_frame = round(end_s * fs_hz)
    starts = np.arange(
        0, end_frame - start_frame, int(cfg["matching_chunk_length_samples"]), dtype=np.int64
    )
    trajectory = sample_and_quantize_trajectory(
        local_t,
        local_d,
        starts,
        int(cfg["matching_chunk_length_samples"]),
        fs_hz=fs_hz,
    )
    absolute_centres_s = start_frame / fs_hz + trajectory["chunk_center_s"]
    episode = mask_at(absolute_centres_s, load_mask(mask_path))
    occupied_states = np.unique(trajectory["quantized_displacement_um"])
    copied_expected = np.interp(source_t, global_t, global_d)
    full_expected = np.interp(absolute_centres_s, global_t, global_d)
    adapter_sampled = trajectory["sampled_displacement_um"]

    quality_columns = {
        "unit_id": "direct: template_data.unit_ids",
        "depth_um": "derivable from registered_geom and template support; no sealed column",
        "ptp_uv": "template PTP derivable, but physical-uV identity is not sealed as a column",
        "refractory_violation_fraction": "derivable only after freezing a metric definition; no sealed column",
        "rest_spike_fraction": "derivable from labels/times plus the verified mask; no sealed column",
        "isolation_score": "absent and not uniquely derivable from bundled arrays",
        "quality": "absent; qc-phy outputs are referenced by receipt but not bundled",
    }
    directly_available = ["unit_id"]
    missing_hard = ["quality", "isolation_score"]

    return {
        "schema": "luke-aw-handoff-validation-v1",
        "status": "handoff_valid_full_probe_worker_blocked",
        "handoff": {
            "root": str(handoff),
            "manifest_sha256": digest(manifest_path),
            "complete_sha256": digest(complete_path),
            "files_expected": len(file_rows),
            "files_hash_and_size_valid": int(sum(row["pass"] for row in file_rows)),
            "all_files_pass": bool(all(row["pass"] for row in file_rows)),
            "file_validation": file_rows,
        },
        "field_authority": {
            "source_sha256": digest(field_path),
            "authoritative_sha256": manifest["accepted_d2l_v1_field"]["source_sha256"],
            "matches": digest(field_path)
            == manifest["accepted_d2l_v1_field"]["source_sha256"],
            "time_origin": manifest["accepted_d2l_v1_field"]["time_origin"],
            "sign_convention": sign,
            "local_adapter_field_sha256": digest(local_field_path),
            "local_interface_audit_sha256": digest(interface_path),
            "filters_applied": interface["filters_applied"],
            "resampling_applied": interface["resampling_applied"],
            "max_abs_copied_sample_difference_um": float(
                np.max(np.abs(local_d - copied_expected))
            ),
        },
        "matching_operator": {
            "dartsort_commit": "edcfe1b51d672b4136eb13cc78c0875da804b851",
            "static_s": matching_summary(handoff / "static_w2/effective-config.json"),
            "d2l": matching_summary(handoff / "d2l_acceptance/effective-config.json"),
            "interpretation": (
                "Both are drifty matchers. D2L evaluates the unrounded rigid field at "
                "0.25 s chunk centres and continuously spatially kernel-interpolates the "
                "registered template basis; integer pitch support is a separate path."
            ),
        },
        "trajectory": {
            "window_seconds": [start_s, end_s],
            "sampling_frequency_hz": fs_hz,
            "window_samples": int(end_frame - start_frame),
            "chunk_count": int(starts.size),
            "chunk_length_samples": int(cfg["matching_chunk_length_samples"]),
            "occupied_quantized_states_um": {
                f"{float(state):g}": int(count)
                for state, count in zip(
                    *np.unique(trajectory["quantized_displacement_um"], return_counts=True)
                )
            },
            "quantization_abs_error_um": q(np.abs(trajectory["quantization_error_um"])),
            "episode_quantization_abs_error_um": q(
                np.abs(trajectory["quantization_error_um"])[episode]
            ),
            "rest_quantization_abs_error_um": q(
                np.abs(trajectory["quantization_error_um"])[~episode]
            ),
            "max_abs_local_vs_authoritative_chunk_sample_um": float(
                np.max(np.abs(adapter_sampled - full_expected))
            ),
            "canonical_mask_sha256": digest(mask_path),
        },
        "saved_bank": {
            "templates_shape": list(templates.shape),
            "unit_id_count": int(unit_ids.size),
            "unique_unit_id_count": int(np.unique(unit_ids).size),
            "geometry_shape": list(geom.shape),
            "geometry_depth_span_um": [float(geom[:, 1].min()), float(geom[:, 1].max())],
            "template_npz_keys": template_keys,
            "sorting_npz_keys": sorting_keys,
            "required_quality_columns": quality_columns,
            "directly_available_required_columns": directly_available,
            "missing_hard_columns": missing_hard,
            "crop_qualification": crop_qualification(
                templates,
                geom,
                unit_ids,
                occupied_states,
                interpolation_radius_um=200.0,
            ),
        },
        "worker_ready": False,
        "remaining_blockers": [
            "The only saved donor bank omits AP0-AP201, so full-probe energy is not identifiable.",
            "The frozen donor quality label and isolation score are not present in the bundle.",
        ],
        "sealed_population_constraint": (
            "Production uses seeded independent per-unit trains with >=3 ms "
            "refractory; the regular 5 Hz train is fixture-only."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--local-motion", type=Path, default=DEFAULT_LOCAL_MOTION)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.handoff, args.local_motion, args.mask)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "worker_ready", "remaining_blockers")}, indent=2))


if __name__ == "__main__":
    main()
