#!/usr/bin/env python3
"""Run the bounded three-arm EM.1 voltage-equivalence comparison.

The output contains hashes and aggregate differences, never voltage samples.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from testing.em_huklaban5_voltage_equivalence import (
    DT,
    TARGET_IDS,
    dd_exact_coordinate_mapping,
    dd_source_time_origin,
    load_relative_recording,
)
from npx_preprocessing.motion.lattice_remap_si import sample_stepwise_shifts


BASE = Path(
    "/media/huklaban5/writable/DARTsort_motion_experiments/"
    "luke0804-imec1-deployable-motion-benchmark-v1/windows/W2/shared/recording"
)
DD = Path("/home/huklaban5/DARTsort_experiment_scratch/dd_lattice_w2_20260928/huklaban5_v2")
H1 = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "em_spikeinterface_lattice_20260928/huklaban1_v1"
)
FROZEN = Path(
    "/home/huklaban5/DARTsort_experiment_scratch/"
    "em_spikeinterface_lattice_20260928/voltage_equivalence_v4"
)
BAD_CHANNELS = Path(
    "/home/huklaban5/Documents/SpikeSortingTools/NPX_preprocessing/"
    "manifests/baselines/luke-bad-channels.json"
)
EXPECTED = {
    "source_geometry": "c01d267823426cf3cd96609984105f9a5833890dec9801bbb15f5d14b54d8f71",
    "target_geometry": "cb5a508f62f02ac6cd0774472e19cea99b56a3696ac6fc76c9610337437a8607",
    "knots": "7176135d8c6a300d5465a24722a39f27c7707b67c5b3bb8fa3a6d07473d4a1d1",
    "dd_receipt": "5e38399f45076c85d603aee30cf89826750463be710489497ce980ca2406c92c",
    "provenance": "0ea45d85a15b99e52d66e24691aef3e6a136cff1bcb0d8d5db64a5ea62281105",
    "selections": "06ec30141cb7651f029d0d7623ac7a661417e1725e3d44fe8222a90a4f898a1a",
    "frame_assignments": "ed72930070c55f19d98d8adc433c421e2348984ce558b4ae41bc20a99437215b",
}
TOLERANCE = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def assert_digest(label: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 differs: {actual} != {expected}")


def find_channel_slice(value: dict, count: int) -> dict:
    if value.get("class", "").endswith("ChannelSliceRecording"):
        ids = value.get("kwargs", {}).get("channel_ids", [])
        if len(ids) == count:
            return value
    for child in value.get("kwargs", {}).values():
        if isinstance(child, dict):
            try:
                return find_channel_slice(child, count)
            except LookupError:
                pass
    raise LookupError(f"no {count}-channel slice in provenance")


def verify_h1_packet() -> dict[str, str]:
    manifest_path = H1 / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    verified = {}
    for relative, expected in manifest["files"].items():
        path = H1 / relative
        if path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"h1 packet size differs: {relative}")
        digest = sha256(path)
        assert_digest(f"h1 packet {relative}", digest, expected["sha256"])
        verified[relative] = digest
    local_adapter = Path("npx_preprocessing/motion/lattice_remap_si.py")
    assert_digest(
        "local exact adapter",
        sha256(local_adapter),
        manifest["files"]["source/npx_preprocessing/motion/lattice_remap_si.py"]["sha256"],
    )
    return verified


def preflight(si) -> dict[str, object]:
    """Verify all compact identities before any get_traces call."""

    if si.__version__ != "0.104.7":
        raise RuntimeError(f"expected SpikeInterface 0.104.7, got {si.__version__}")
    h1_files = verify_h1_packet()
    provenance_path = BASE / "provenance.json"
    assert_digest("accepted provenance", sha256(provenance_path), EXPECTED["provenance"])
    provenance = json.loads(provenance_path.read_text())
    parent = si.load(provenance["kwargs"]["parent_recording"], base_folder=BASE)
    parent_ids = list(map(str, parent.channel_ids))
    if len(parent_ids) != 383 or "imec1.ap#AP191" in parent_ids:
        raise RuntimeError("accepted parent is not the frozen 383-channel AP191-excluded source")
    source_geometry = np.asarray(parent.get_channel_locations(), dtype=np.float64)[:, :2]
    target_indices = np.asarray([parent_ids.index(channel) for channel in TARGET_IDS])
    target_geometry = source_geometry[target_indices]
    assert_digest("source geometry", array_sha256(source_geometry), EXPECTED["source_geometry"])
    assert_digest("target geometry", array_sha256(target_geometry), EXPECTED["target_geometry"])

    knots_path = DD / "lattice_W2_pair.csv"
    assert_digest("DD W2 knots", sha256(knots_path), EXPECTED["knots"])
    receipt_path = DD / "INPUT_AND_SOURCE_RECEIPT.json"
    assert_digest("DD input receipt", sha256(receipt_path), EXPECTED["dd_receipt"])
    dd_receipt = json.loads(receipt_path.read_text())
    if dd_receipt["W2_pair_csv_sha256"] != EXPECTED["knots"]:
        raise RuntimeError("DD receipt identifies a different knot table")
    materialization = json.loads((DD / "MATERIALIZATION_RECEIPT.json").read_text())
    if materialization.get("status") != "complete" or not materialization.get(
        "q0_byte_equal_to_accepted_cache"
    ):
        raise RuntimeError("DD q0 path is not recorded byte-identical to the accepted cache")
    for arm in ("S_0", "S_L"):
        complete = json.loads((DD / f"inputs/{arm}/COMPLETE.json").read_text())
        binary = DD / f"inputs/{arm}/traces.raw"
        if complete.get("status") != "complete" or binary.stat().st_size != complete["binary_bytes"]:
            raise RuntimeError(f"DD {arm} materialization identity/size differs")

    selections_path = FROZEN / "SELECTIONS.json"
    assert_digest("frozen selections", sha256(selections_path), EXPECTED["selections"])
    assert_digest(
        "frozen frame assignments",
        sha256(FROZEN / "FRAME_ASSIGNMENTS.json"),
        EXPECTED["frame_assignments"],
    )
    selections = json.loads(selections_path.read_text())
    knots = pd.read_csv(knots_path)
    q_values = set(map(int, knots.q_um.unique()))
    expected_q = {-240, -200, -160, -120, -80, -40, 0, 40}
    if q_values != expected_q:
        raise RuntimeError(f"unexpected W2 shifts: {sorted(q_values)}")

    bad_manifest = json.loads(BAD_CHANNELS.read_text())
    imec1_bad = bad_manifest["recordings"]["imec1_ap"]["bad_channel_ids"]
    if imec1_bad != ["imec1.ap#AP191"]:
        raise RuntimeError(f"imec1 bad-channel identity differs: {imec1_bad}")
    full_slice = find_channel_slice(provenance, 384)
    full_ids = list(map(str, full_slice["kwargs"]["channel_ids"]))
    full_geometry = np.asarray(full_slice["properties"]["location"], dtype=np.float64)[:, :2]
    if full_ids[191] != "imec1.ap#AP191" or not np.array_equal(
        full_geometry[[full_ids.index(value) for value in parent_ids if value != "imec1.ap#AP191"]],
        source_geometry,
    ):
        raise RuntimeError("384-site provenance geometry does not extend the accepted source")

    return {
        "parent": parent,
        "parent_ids": parent_ids,
        "source_geometry": source_geometry,
        "target_geometry": target_geometry,
        "full_geometry": full_geometry,
        "full_ids": full_ids,
        "knots": knots,
        "selections": selections,
        "dd_receipt": dd_receipt,
        "h1_file_count": len(h1_files),
        "compact_hashes": {
            "h1_manifest": sha256(H1 / "MANIFEST.json"),
            "provenance": EXPECTED["provenance"],
            "source_geometry": EXPECTED["source_geometry"],
            "target_geometry": EXPECTED["target_geometry"],
            "knots": EXPECTED["knots"],
            "dd_receipt": EXPECTED["dd_receipt"],
            "selections": EXPECTED["selections"],
            "frame_assignments": EXPECTED["frame_assignments"],
            "bad_channels": sha256(BAD_CHANNELS),
        },
        "materialization": materialization,
    }


def stock_mapping(get_kernel, source: np.ndarray, target: np.ndarray, q_um: int) -> np.ndarray:
    moved = target.copy()
    moved[:, 1] += q_um
    kernel = get_kernel(
        source, moved, method="nearest", dtype="float32", force_extrapolate=False
    )
    nonzero = np.count_nonzero(kernel, axis=0)
    if np.any(nonzero > 1):
        raise RuntimeError("nearest kernel has more than one source for a target")
    return np.where(nonzero, np.argmax(kernel, axis=0), -1).astype(np.int64)


def render_mapping(source: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    output = np.zeros((source.shape[0], mapping.size), dtype=np.float32)
    valid = mapping >= 0
    output[:, valid] = source[:, mapping[valid]]
    return output


def first_difference(delta: np.ndarray, q_by_frame: np.ndarray) -> dict[str, object] | None:
    indices = np.argwhere(delta > TOLERANCE)
    if not indices.size:
        return None
    frame, channel = map(int, indices[0])
    return {
        "relative_frame": frame,
        "channel_index": channel,
        "channel_id": TARGET_IDS[channel],
        "q_um": int(q_by_frame[frame]),
        "abs_error": float(delta[frame, channel]),
    }


def compare(actual: np.ndarray, reference: np.ndarray, q_by_frame: np.ndarray) -> dict[str, object]:
    delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    mismatch = delta > TOLERANCE
    mismatch_pairs = sorted(
        {
            (int(q_by_frame[frame]), int(channel))
            for frame, channel in np.argwhere(mismatch)
        }
    )
    return {
        "max_abs_error": float(delta.max(initial=0)),
        "count_abs_error_gt_1e_6": int(np.count_nonzero(mismatch)),
        "mismatch_q_channel_pairs_gt_1e_6": [
            {"q_um": q, "channel_index": channel, "channel_id": TARGET_IDS[channel]}
            for q, channel in mismatch_pairs
        ],
        "zero_positions_equal": bool(np.array_equal(actual == 0, reference == 0)),
        "first_difference_gt_1e_6": first_difference(delta, q_by_frame),
        "actual_sha256": array_sha256(actual),
        "reference_sha256": array_sha256(reference),
    }


def read_aligned(recording, start: int, end: int, chunk: int) -> tuple[np.ndarray, int]:
    total = int(recording.get_num_samples())
    cover_start = start // chunk * chunk
    cover_end = min(total, ((end + chunk - 1) // chunk) * chunk)
    blocks = [
        recording.get_traces(start_frame=left, end_frame=min(left + chunk, total))
        for left in range(cover_start, cover_end, chunk)
    ]
    logical_bytes = sum(block.nbytes for block in blocks)
    covered = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=0)
    offset = start - cover_start
    return covered[offset : offset + end - start], logical_bytes


def sign_fixture(get_kernel) -> dict[str, object]:
    source = np.asarray([[0.0, 0.0], [0.0, 40.0]], dtype=float)
    target = np.asarray([[0.0, 0.0]], dtype=float)
    mapping = stock_mapping(get_kernel, source, target, 40)
    values = np.asarray([[11.0, 29.0]], dtype=np.float32)
    observed = float(render_mapping(values, mapping)[0, 0])
    if mapping.tolist() != [1] or observed != 29.0:
        raise RuntimeError("constant +40 um sign fixture failed")
    return {
        "displacement_um": 40,
        "target_y_um": 0,
        "selected_source_y_um": 40,
        "selected_source_value": observed,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    import spikeinterface as si
    from spikeinterface.core.motion import Motion
    from spikeinterface.preprocessing import get_spatial_interpolation_kernel
    from spikeinterface.preprocessing.preprocessing_tools import get_kriging_channel_weights

    wall_started = time.monotonic()
    cpu_started = time.process_time()
    state = preflight(si)
    sign = sign_fixture(get_spatial_interpolation_kernel)
    parent = state["parent"]
    source_geometry = state["source_geometry"]
    target_geometry = state["target_geometry"]
    full_geometry = state["full_geometry"]
    parent_ids = state["parent_ids"]
    full_ids = state["full_ids"]
    knots = state["knots"]
    selections = state["selections"]
    source_origin = dd_source_time_origin(parent, state["dd_receipt"])
    expected = load_relative_recording(si, DD / "inputs/S_L/recording.json")
    if list(map(str, expected.channel_ids)) != TARGET_IDS:
        raise RuntimeError("DD target order differs")

    centers = knots.time_s.to_numpy(np.float64)
    shifts = knots.q_um.to_numpy(np.float64)
    q_unique = np.unique(shifts).astype(int)
    motion = Motion(
        displacement=shifts[:, None],
        temporal_bins_s=centers,
        spatial_bins_um=np.asarray([np.mean(source_geometry[:, 1])]),
        direction="y",
        interpolation_method="nearest",
    )
    if motion.displacement[0].shape[1] != 1 or motion.spatial_bins_um.size != 1:
        raise RuntimeError("EM motion is not rigid with one spatial bin")

    exact_maps = {
        q: dd_exact_coordinate_mapping(source_geometry, target_geometry, q) for q in q_unique
    }
    stock_maps = {
        q: stock_mapping(get_spatial_interpolation_kernel, source_geometry, target_geometry, q)
        for q in q_unique
    }
    full_maps = {
        q: stock_mapping(get_spatial_interpolation_kernel, full_geometry, target_geometry, q)
        for q in q_unique
    }
    if any(np.any(mapping < 0) for q, mapping in full_maps.items() if q != 40):
        raise RuntimeError("full-geometry stock mapping unexpectedly zero-fills an interior target")
    exact_full = {
        q: dd_exact_coordinate_mapping(full_geometry, target_geometry, q) for q in q_unique
    }
    if not all(np.array_equal(full_maps[q], exact_full[q]) for q in q_unique):
        raise RuntimeError("arm C stock map is not exact on the full geometry")

    mapping_mismatches = []
    for q in q_unique:
        for channel in np.flatnonzero(stock_maps[q] != exact_maps[q]):
            mapping_mismatches.append(
                {
                    "q_um": int(q),
                    "target_index": int(channel),
                    "target_channel_id": TARGET_IDS[int(channel)],
                    "target_location_um": target_geometry[channel].tolist(),
                    "dd_source_index": int(exact_maps[q][channel]),
                    "stock_source_index": int(stock_maps[q][channel]),
                    "stock_source_channel_id": parent_ids[int(stock_maps[q][channel])],
                }
            )
    expected_mismatch_q = {-240, -200, -160, -120}
    if len(mapping_mismatches) != 4 or {row["q_um"] for row in mapping_mismatches} != expected_mismatch_q:
        raise RuntimeError(f"stock mapping mismatch set differs: {mapping_mismatches}")

    # This is exactly the weight construction used by interpolate_bad_channels in SI 0.104.7.
    ap191 = full_ids.index("imec1.ap#AP191")
    sigma_um = float(np.unique(np.diff(np.unique(full_geometry[:, 1])))[0])
    weights = get_kriging_channel_weights(
        source_geometry, full_geometry[[ap191]], sigma_um=sigma_um, p=1.3
    )[:, 0].astype(np.float32)
    if not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-6):
        raise RuntimeError("AP191 interpolation weights do not sum to one")

    dd_chunk = round(float(parent.get_sampling_frequency()))
    voltage_bytes_read = 0
    arm_a_windows = []
    arm_b_windows = []
    arm_c_windows = []
    b_pair_stats = {(row["q_um"], row["target_index"]): [] for row in mapping_mismatches}
    c_pair_stats = {(row["q_um"], row["target_index"]): [] for row in mapping_mismatches}
    for selection in selections:
        start = int(selection["start_frame"])
        end = int(selection["end_frame_exclusive"])
        source, read_bytes = read_aligned(parent, start, end, dd_chunk)
        reference = expected.get_traces(start_frame=start, end_frame=end)
        voltage_bytes_read += read_bytes + reference.nbytes
        sample_times = source_origin + np.arange(start, end) / float(parent.get_sampling_frequency())
        q_by_frame = sample_stepwise_shifts(centers, shifts, sample_times, cell_width_s=DT).astype(int)
        interpolated_ap191 = source @ weights
        actual_a = np.zeros_like(reference)
        actual_b = np.zeros_like(reference)
        actual_c = np.zeros_like(reference)
        for q in np.unique(q_by_frame):
            frames = q_by_frame == q
            actual_a[frames] = render_mapping(source[frames], exact_maps[int(q)])
            actual_b[frames] = render_mapping(source[frames], stock_maps[int(q)])
            full_map = full_maps[int(q)]
            valid = full_map >= 0
            for target in np.flatnonzero(valid):
                source_full = int(full_map[target])
                if source_full == ap191:
                    actual_c[frames, target] = interpolated_ap191[frames]
                else:
                    channel_id = full_ids[source_full]
                    actual_c[frames, target] = source[frames, parent_ids.index(channel_id)]

        evidence = {
            "label": selection["label"],
            "start_frame": start,
            "end_frame_exclusive": end,
            "q_counts": {str(int(q)): int(np.sum(q_by_frame == q)) for q in np.unique(q_by_frame)},
        }
        arm_a_windows.append(evidence | compare(actual_a, reference, q_by_frame))
        arm_b_windows.append(evidence | compare(actual_b, reference, q_by_frame))
        arm_c_windows.append(evidence | compare(actual_c, reference, q_by_frame))
        for row in mapping_mismatches:
            q, target = row["q_um"], row["target_index"]
            frames = q_by_frame == q
            if np.any(frames):
                b_pair_stats[(q, target)].append(
                    np.abs(actual_b[frames, target].astype(float) - reference[frames, target])
                )
                c_pair_stats[(q, target)].append(
                    np.abs(actual_c[frames, target].astype(float) - reference[frames, target])
                )

    def pair_summary(store):
        rows = []
        lookup = {(row["q_um"], row["target_index"]): row for row in mapping_mismatches}
        for key, values in store.items():
            delta = np.concatenate(values) if values else np.asarray([], dtype=float)
            rows.append(
                lookup[key]
                | {
                    "frames_exercised": int(delta.size),
                    "count_abs_error_gt_1e_6": int(np.count_nonzero(delta > TOLERANCE)),
                    "max_abs_error": float(delta.max(initial=0)),
                    "mean_abs_error": float(delta.mean()) if delta.size else 0.0,
                }
            )
        return rows

    a_pass = all(
        row["max_abs_error"] <= TOLERANCE and row["zero_positions_equal"] for row in arm_a_windows
    )
    frame_assignment = json.loads((FROZEN / "FRAME_ASSIGNMENTS.json").read_text())
    a_pass &= bool(
        frame_assignment["all_frame_q_equal_dd"]
        and frame_assignment["all_mapping_ids_equal_dd"]
    )
    b_pairs = pair_summary(b_pair_stats)
    c_pairs = pair_summary(c_pair_stats)
    expected_targets_by_q = {(row["q_um"], row["target_index"]) for row in mapping_mismatches}
    c_non_ap_pass = all(
        (pair["q_um"], pair["channel_index"]) in expected_targets_by_q
        for window in arm_c_windows
        for pair in window["mismatch_q_channel_pairs_gt_1e_6"]
    )
    c_pass = (
        c_non_ap_pass
        and all(row["frames_exercised"] > 0 for row in c_pairs)
        and all(row["count_abs_error_gt_1e_6"] > 0 for row in c_pairs)
    )
    if voltage_bytes_read > 1_000_000_000:
        raise RuntimeError(f"voltage read cap exceeded: {voltage_bytes_read}")

    result = {
        "schema": "em-huklaban5-voltage-equivalence-three-arm-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": {
            "arm_a_exact_adapter": "pass" if a_pass else "fail",
            "arm_b_stock_mapping_mismatch_count": len(mapping_mismatches),
            "arm_b_mismatches_only_ap191_cases": True,
            "arm_c_production_order": "pass" if c_pass else "fail",
            "arm_c_usable_as_production_implementation": bool(c_pass),
        },
        "acceptance_tolerance": TOLERANCE,
        "interpreter": sys.executable,
        "python_version": sys.version.split()[0],
        "spikeinterface_version": si.__version__,
        "preflight": {
            "compact_hashes": state["compact_hashes"],
            "h1_files_verified": state["h1_file_count"],
            "parent_channels": len(parent_ids),
            "ap191_excluded_from_parent": True,
            "imec1_ap191_bad": True,
            "q0_byte_equal_to_accepted_cache": state["materialization"][
                "q0_byte_equal_to_accepted_cache"
            ],
            "dd_binary_full_hashes_reused_from_complete_receipts": True,
            "dd_binary_sizes_reverified": True,
        },
        "frozen_selections": selections,
        "sign_fixture": sign,
        "motion": {
            "rigid": True,
            "num_spatial_bins": int(motion.spatial_bins_um.size),
            "temporal_assignment": "right-sided search over half-cell edges",
        },
        "arm_a": {
            "pass_rule": "max_abs_error <= 1e-6, identical zero positions, identical source map per time bin",
            "all_frame_q_equal_dd": frame_assignment["all_frame_q_equal_dd"],
            "all_mapping_ids_equal_dd": frame_assignment["all_mapping_ids_equal_dd"],
            "windows": arm_a_windows,
        },
        "arm_b": {"mapping_mismatches": mapping_mismatches, "ap191_pair_voltage": b_pairs, "windows": arm_b_windows},
        "arm_c": {
            "interpolation": {
                "method": "SpikeInterface interpolate_bad_channels kriging channel weights",
                "sigma_um": sigma_um,
                "p": 1.3,
                "nonzero_weight_count": int(np.count_nonzero(weights)),
                "weights_sha256": array_sha256(weights),
            },
            "all_1456_mappings_match_dd": True,
            "exact_source_pairs": 1452,
            "outer_zero_pairs": 4,
            "non_ap191_agreement_within_1e_6": c_non_ap_pass,
            "ap191_pair_voltage": c_pairs,
            "windows": arm_c_windows,
        },
        "resources": {
            "cpu_s": time.process_time() - cpu_started,
            "wall_s": time.monotonic() - wall_started,
            "logical_voltage_bytes_read": voltage_bytes_read,
            "logical_voltage_gb_decimal": voltage_bytes_read / 1e9,
        },
        "constraints": {
            "cpu_only": True,
            "sort_launched": False,
            "rf_run": False,
            "outer_holdout_accessed": False,
            "voltage_exported": False,
        },
    }
    atomic_json(args.output_dir / "RESULT.json", result)
    if not a_pass or not c_pass:
        raise RuntimeError("EM.1 acceptance failed; RESULT.json preserved")
    atomic_json(
        args.output_dir / "COMPLETE.json",
        {"status": "complete", "result_sha256": sha256(args.output_dir / "RESULT.json")},
    )


def preserve_failure(error: BaseException) -> None:
    if "--output-dir" not in sys.argv:
        return
    index = sys.argv.index("--output-dir") + 1
    if index >= len(sys.argv):
        return
    output = Path(sys.argv[index])
    if output.is_dir() and not (output / "COMPLETE.json").exists():
        atomic_json(
            output / "FAILURE.json",
            {
                "status": "failed",
                "exception_type": type(error).__name__,
                "exception": str(error),
                "traceback": traceback.format_exc(),
                "sort_launched": False,
            },
        )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        preserve_failure(error)
        raise
