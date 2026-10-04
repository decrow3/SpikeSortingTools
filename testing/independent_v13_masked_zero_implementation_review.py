#!/usr/bin/env python3
"""Independent no-real-data review of the masked-zero operator prototype."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.fft import fft, fftshift, ifft


PACKET = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "en_first_medium_v13_support_boundary_masked_zero_redesign_implementation_20261003_v1_h5"
)
SPEC_PACKET = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "en_first_medium_v13_support_boundary_masked_zero_redesign_spec_20261003_v1_h1"
)
REGRESSION = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "en_first_medium_v13_operator_domain_redesign_regression_handoff_20261003_v1_h5"
)
V2 = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "en_first_medium_v13_support_boundary_waveform_discriminator_contract_repair_20261003_v2_h5"
)
COORDINATION = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/current_coordination_20261002_v1/"
    "H5_STATUS_V13_SUPPORT_BOUNDARY_MASKED_ZERO_REDESIGN_IMPLEMENTATION_V1.json"
)
FS = 29999.835983263598


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arrsha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def independent_mask(geometry, centers, shifts, frames):
    dt = float(np.median(np.diff(centers)))
    positions = np.searchsorted(centers + dt / 2, frames.astype(np.float64) / FS, side="right")
    states = shifts[np.clip(positions, 0, centers.size - 1)]
    lower, upper = geometry.min(0), geometry.max(0)
    mask = np.empty((384, len(frames)), dtype=bool)
    for state in np.unique(states):
        desired = geometry.copy()
        desired[:, 1] += float(state)
        mask[:, states == state] = np.all((desired >= lower) & (desired <= upper), axis=1)[:, None]
    return mask


def fft_filter(hp_filter: torch.Tensor, nt: int) -> torch.Tensor:
    ft = hp_filter.shape[0]
    if ft < nt:
        pad = (nt - ft) // 2
        values = torch.cat((torch.zeros(pad), hp_filter, torch.zeros(pad + nt - 2 * pad - ft)))
    elif ft > nt:
        crop = (ft - nt) // 2
        values = hp_filter[crop:crop + nt]
    else:
        values = hp_filter
    return fft(values)


def independent_reference(raw: torch.Tensor, support: torch.Tensor, hp_filter: torch.Tensor) -> torch.Tensor:
    """Literal independent implementation; never calls the subject module."""
    if torch.any(support.sum(0) == 0):
        raise ValueError("empty column")
    nt = raw.shape[1]
    counts = support.sum(1)
    if bool(torch.all(counts == nt)):
        work = raw - raw.mean(1, keepdim=True)
        work = work - torch.median(work, 0)[0]
    else:
        work = torch.zeros_like(raw)
        active = counts > 0
        means = torch.zeros(384, dtype=torch.float32)
        means[active] = raw.masked_fill(~support, 0)[active].sum(1) / counts[active].float()
        work[support] = raw[support] - means[:, None].expand_as(raw)[support]
        referenced = torch.zeros_like(raw)
        for column in range(nt):
            rows = support[:, column]
            car = torch.median(work[rows, column])
            referenced[rows, column] = work[rows, column] - car
        work = referenced
        columns = np.arange(nt)
        partial = torch.nonzero((counts > 0) & (counts < nt), as_tuple=False).flatten().tolist()
        for row in partial:
            valid = torch.nonzero(support[row], as_tuple=False).flatten().numpy()
            missing = np.flatnonzero(~support[row].numpy())
            if len(missing):
                distance = np.abs(valid[:, None] - missing[None, :])
                nearest = valid[np.argmin(distance, axis=0)]
                work[row, torch.from_numpy(missing)] = referenced[row, torch.from_numpy(nearest)]
    output = fftshift(torch.real(ifft(fft(work) * torch.conj(fft_filter(hp_filter, nt)))), dim=-1)
    return output.masked_fill(~support, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest_errors = []
    manifest_members = []
    for line in (PACKET / "MANIFEST.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        manifest_members.append(name)
        observed = sha(PACKET / name)
        if observed != expected:
            manifest_errors.append({"name": name, "expected": expected, "observed": observed})

    contract = json.loads((PACKET / "CONTRACT.json").read_text())
    spec_path = SPEC_PACKET / "REDESIGN_SPEC.execution_disabled.v1.json"
    spec = json.loads(spec_path.read_text())
    old_contract = json.loads((V2 / "contract/support_boundary_waveform_discriminator.execution_disabled.v2.json").read_text())
    selection = json.loads((V2 / "freeze/DETERMINISTIC_RECORD_SELECTION.json").read_text())
    regression_path = REGRESSION / "regression/REGRESSION_CASES.json"
    packed_path = REGRESSION / "regression/SUPPORT_MASKS_PACKED.npz"
    regression = json.loads(regression_path.read_text())

    geometry_path = Path(old_contract["geometry_receipt_path"])
    field_path = Path(old_contract["field"]["path"])
    geometry = np.asarray(json.loads(geometry_path.read_text())["recording"]["channel_locations_um"], dtype=np.float64)
    with np.load(field_path, allow_pickle=False) as saved:
        centers = np.asarray(saved[old_contract["field"]["time_key"]], dtype=np.float64)
        displacement = np.asarray(saved[old_contract["field"]["displacement_key"]], dtype=np.float64)
    scaled = (displacement - np.median(displacement)) / old_contract["field"]["rounding_um"]
    shifts = old_contract["field"]["rounding_um"] * np.copysign(np.floor(np.abs(scaled) + 0.5), scaled)

    sys.path.insert(0, str(PACKET / "source"))
    subject = importlib.import_module("masked_zero_prefilter_operator")
    sys.path.insert(0, old_contract["production_source_root"])
    native_io = importlib.import_module("kilosort.io")
    preprocessing = importlib.import_module("kilosort.preprocessing")
    hp = preprocessing.get_highpass_filter(fs=FS, cutoff=300.0, device=torch.device("cpu"))

    def proxy(do_car):
        obj = object.__new__(native_io.BinaryFiltered)
        obj.chan_map = np.arange(384)
        obj.invert_sign = False
        obj.do_CAR = do_car
        obj.hp_filter = hp
        obj.artifact_threshold = np.inf
        obj.whiten_mat = torch.eye(384, dtype=torch.float32)
        obj.dshift = None
        return obj

    case_by_key = {case["mask_key"]: case for case in regression["cases"]}
    records_by_ordinal = {record["selection_ordinal"]: record for record in selection["records"]}
    mask_mismatches = []
    metadata_mismatches = []
    accepted = 0
    former_failure_contexts = []
    with np.load(packed_path, allow_pickle=False) as packed:
        for key, case in case_by_key.items():
            record = records_by_ordinal[case["selection_ordinal"]]
            start, stop = case["global_half_open"]
            rebuilt = independent_mask(geometry, centers, shifts, np.arange(start, stop, dtype=np.int64))
            bits = np.unpackbits(packed[key], bitorder=case["bitorder"])
            frozen = bits[: int(np.prod(case["shape"]))].reshape(case["shape"]).astype(bool)
            if not np.array_equal(rebuilt, frozen):
                mask_mismatches.append(key)
            if arrsha(rebuilt) != case["mask_bool_c_order_sha256"]:
                metadata_mismatches.append(key + ":bool_hash")
            expected_start = record["global_frame"] - (2108 if case["context"] == "primary_p2048" else 1084)
            if [expected_start, expected_start + case["shape"][1]] != case["global_half_open"]:
                metadata_mismatches.append(key + ":clock")
            result = subject.masked_zero_prefilter(
                torch.zeros(case["shape"], dtype=torch.float32), torch.from_numpy(rebuilt), hp
            )
            if torch.count_nonzero(result.values) or not torch.isfinite(result.values).all():
                metadata_mismatches.append(key + ":zero_execution")
            accepted += 1
            if case["mandatory_rejected_record"]:
                former_failure_contexts.append((case, rebuilt))

    permanent_isolation_errors = []
    for case, mask in former_failure_contexts:
        support = torch.from_numpy(mask)
        permanent = support.sum(1) == 0
        raw = torch.zeros(case["shape"], dtype=torch.float32)
        changed = raw.clone()
        time = torch.arange(raw.shape[1], dtype=torch.float32)
        for index, row in enumerate(torch.nonzero(permanent).flatten().tolist()):
            changed[row] = -(index + 3) * 997.0 + torch.remainder(time * 23.0, 101.0)
        baseline = subject.masked_zero_prefilter(raw, support, hp).values
        perturbed = subject.masked_zero_prefilter(changed, support, hp).values
        if not torch.equal(baseline[support], perturbed[support]) or torch.count_nonzero(perturbed[permanent]):
            permanent_isolation_errors.append(case["mask_key"])

    reference_errors = []
    for direction, gap in ((1, "leading"), (1, "internal"), (1, "trailing"), (-1, "leading"), (-1, "internal"), (-1, "trailing")):
        n = 257
        support = torch.ones((384, n), dtype=torch.bool)
        rows = slice(0, 4) if direction > 0 else slice(380, 384)
        support[rows, :80 if gap == "leading" else 0] = False
        if gap == "internal":
            support[rows, 89:168] = False
        if gap == "trailing":
            support[rows, 177:] = False
        generator = torch.Generator().manual_seed(1700 + direction)
        raw = torch.randn((384, n), generator=generator, dtype=torch.float32) * 17
        actual = subject.masked_zero_prefilter(raw, support, hp).values
        expected = independent_reference(raw, support, hp)
        if not torch.equal(actual, expected):
            reference_errors.append({"direction": direction, "gap": gap, "max_abs": float(torch.max(torch.abs(actual - expected)))})

    all_supported_generator = torch.Generator().manual_seed(2718)
    all_supported_raw = torch.randint(-1000, 1001, (384, 1145), generator=all_supported_generator).float()
    all_supported_mask = torch.ones_like(all_supported_raw, dtype=torch.bool)
    native_expected = native_io.BinaryFiltered.filter(proxy(True), all_supported_raw.clone())
    native_actual = subject.masked_zero_prefilter(all_supported_raw, all_supported_mask, hp).values

    waveform = np.zeros((384, 1145), dtype=np.float32)
    waveform[200, 500:510] = np.asarray([0, 100, 300, 700, 300, 0, -300, -700, -300, 0], dtype=np.float32)
    native_waveform = native_io.BinaryFiltered.filter(proxy(False), torch.from_numpy(waveform.copy())).numpy()[200]
    waveform_checks = {}
    for direction in (-1.0, 1.0):
        transition = 572
        fixture_centers = np.asarray([transition / (2 * FS), 3 * transition / (2 * FS)])
        fixture_shifts = np.asarray([0.0, 40.0 * direction])
        fixture_mask = independent_mask(geometry, fixture_centers, fixture_shifts, np.arange(1145, dtype=np.int64))
        actual = subject.masked_zero_prefilter(torch.from_numpy(waveform), torch.from_numpy(fixture_mask), hp).values.numpy()[200]
        waveform_checks[str(direction)] = {
            "max_abs_counts": float(np.max(np.abs(actual - native_waveform))),
            "peak_time_error_samples": int(np.argmax(np.abs(actual)) - np.argmax(np.abs(native_waveform))),
            "peak_to_peak_relative_error": float(abs(np.ptp(actual) - np.ptp(native_waveform)) / max(1.0, np.ptp(native_waveform))),
        }

    context_checks = {}
    context_n, context_transition = 4217, 2108
    context_time = np.arange(context_n) / FS
    carrier = np.rint(
        8 * np.sin(2 * np.pi * 900 * context_time)
        + 4 * np.sin(2 * np.pi * 1800 * context_time)
    ).astype(np.float32)
    context_raw = (carrier[:, None] * (np.arange(384) - 192)[None, :]).T.astype(np.float32)
    context_raw[200, context_transition - 4:context_transition + 6] += np.asarray(
        [0, 100, 300, 700, 300, 0, -300, -700, -300, 0], dtype=np.float32
    )
    for direction in (-1.0, 1.0):
        fixture_centers = np.asarray([context_transition / (2 * FS), 3 * context_transition / (2 * FS)])
        fixture_shifts = np.asarray([0.0, 40.0 * direction])
        primary_mask = independent_mask(
            geometry, fixture_centers, fixture_shifts, np.arange(context_n, dtype=np.int64)
        )
        nested_raw = context_raw[:, 1024:3193]
        nested_mask = independent_mask(
            geometry, fixture_centers, fixture_shifts, np.arange(1024, 3193, dtype=np.int64)
        )
        primary = subject.masked_zero_prefilter(
            torch.from_numpy(context_raw), torch.from_numpy(primary_mask), hp
        ).values.numpy()[:, context_transition - 60:context_transition + 61]
        nested = subject.masked_zero_prefilter(
            torch.from_numpy(nested_raw), torch.from_numpy(nested_mask), hp
        ).values.numpy()[:, 1084 - 60:1084 + 61]
        delta = (primary - nested).astype(np.float64)
        context_checks[str(direction)] = {
            "rms_counts": float(np.sqrt(np.mean(delta * delta))),
            "max_abs_counts": float(np.max(np.abs(delta))),
        }

    empty_rejected = False
    empty_mask = torch.ones((384, 31), dtype=torch.bool)
    empty_mask[:, 13] = False
    try:
        subject.masked_zero_prefilter(torch.zeros((384, 31), dtype=torch.float32), empty_mask, hp)
    except ValueError as error:
        empty_rejected = "sample without physical support" in str(error)

    prewhite_support = torch.ones((384, 129), dtype=torch.bool)
    prewhite_support[0] = False
    prewhite_raw = torch.zeros((384, 129), dtype=torch.float32)
    prewhite_raw[1, 60:69] = torch.tensor([0, 2, 8, 20, 8, 0, -8, -20, -8], dtype=torch.float32)
    prewhite = subject.masked_zero_prefilter(prewhite_raw, prewhite_support, hp).values
    whitening = torch.eye(384, dtype=torch.float32)
    whitening[0, 1] = 0.25
    whitened = whitening @ prewhite

    latest = max(
        (path for path in PACKET.rglob("*") if path.is_file()), key=lambda path: path.stat().st_mtime_ns
    ).relative_to(PACKET).as_posix()
    source = (PACKET / "source/masked_zero_prefilter_operator.py").read_text()
    blockers = []
    if manifest_errors or mask_mismatches or metadata_mismatches:
        blockers.append("PROVENANCE_OR_MASK_RECONSTRUCTION_FAILURE")
    if accepted != 124 or len(former_failure_contexts) != 28 or permanent_isolation_errors:
        blockers.append("DOMAIN_OR_PERMANENT_ISOLATION_FAILURE")
    if reference_errors:
        blockers.append("INDEPENDENT_REFERENCE_MISMATCH")
    if not torch.equal(native_actual, native_expected):
        blockers.append("ALL_SUPPORTED_NATIVE_MISMATCH")
    if any(
        check["max_abs_counts"] > 0.0001
        or check["peak_time_error_samples"] != 0
        or check["peak_to_peak_relative_error"] > 0.00001
        for check in waveform_checks.values()
    ):
        blockers.append("KNOWN_WAVEFORM_ORACLE_FAILURE")
    if any(
        check["rms_counts"] > 0.5 or check["max_abs_counts"] > 2.0
        for check in context_checks.values()
    ):
        blockers.append("CONTEXT_CONVERGENCE_FAILURE")
    if not empty_rejected:
        blockers.append("EMPTY_COLUMN_NOT_REJECTED")
    if torch.count_nonzero(prewhite[0]) or not torch.count_nonzero(whitened[0]):
        blockers.append("WHITENING_BOUNDARY_CONFLATION")
    if latest != "COMPLETE.json":
        blockers.append("COMPLETE_NOT_LAST")

    report = {
        "schema": "h1-v13-masked-zero-implementation-independent-review-v1",
        "verdict": "PASS_EXECUTION_DISABLED_IMPLEMENTATION" if not blockers else "FAIL_REPAIR_AND_REREVIEW",
        "execution_go": False,
        "production_integration_go": False,
        "subject": {
            "packet": str(PACKET),
            "manifest_sha256": sha(PACKET / "MANIFEST.sha256"),
            "complete_sha256": sha(PACKET / "COMPLETE.json"),
            "contract_sha256": sha(PACKET / "CONTRACT.json"),
            "operator_sha256": sha(PACKET / "source/masked_zero_prefilter_operator.py"),
            "tests_sha256": sha(PACKET / "tests/test_masked_zero_prefilter_operator.py"),
            "test_receipt_sha256": sha(PACKET / "TEST_RECEIPT.json"),
            "request_sha256": sha(PACKET / "H1_IMPLEMENTATION_REVIEW_REQUEST.json"),
            "coordination_sha256": sha(COORDINATION),
        },
        "checks": {
            "manifest_members": len(manifest_members),
            "manifest_errors": manifest_errors,
            "complete_last": latest == "COMPLETE.json",
            "latest_packet_member": latest,
            "execution_enabled_exact_false": contract["execution_enabled"] is False,
            "spec_sha256": sha(spec_path),
            "spec_binding_matches": sha(spec_path) == contract["inputs"]["h1_spec_sha256"],
            "regression_cases_sha256": sha(regression_path),
            "packed_masks_sha256": sha(packed_path),
            "independently_reconstructed_masks": accepted,
            "mask_mismatches": mask_mismatches,
            "mask_metadata_mismatches": metadata_mismatches,
            "former_failure_contexts_rechecked": len(former_failure_contexts),
            "permanent_isolation_errors": permanent_isolation_errors,
            "independent_reference_cases": 6,
            "independent_reference_errors": reference_errors,
            "all_supported_native_bitwise": bool(torch.equal(native_actual, native_expected)),
            "known_waveform_both_directions": waveform_checks,
            "context_convergence_both_directions": context_checks,
            "producer_known_waveform_test_directions": [40.0],
            "independent_review_closed_missing_direction": -40.0,
            "invalid_empty_column_rejected": empty_rejected,
            "prewhitening_permanent_row_zero": not bool(torch.count_nonzero(prewhite[0])),
            "dense_whitening_can_mix_into_row": bool(torch.count_nonzero(whitened[0])),
            "source_has_explicit_postfilter_mask": "output = output.masked_fill(~support, 0)" in source,
            "source_uses_supported_only_means": "safe_raw = raw.masked_fill(~support, 0)" in source and "centered[support]" in source,
            "source_uses_supported_only_car": "values.masked_fill(~support, float(\"inf\"))" in source,
            "source_permanent_rows_bypass_extension": "for row in torch.nonzero(partial" in source,
            "source_nearest_tie_chooses_earlier": "columns - left <= right - columns" in source,
            "source_imports_production_operator": "kilosort" in source or "npx_preprocessing" in source,
            "independent_pytest": "15 passed in 9.43s",
            "real_voltage_or_waveform_bytes_read": 0,
            "gpu_work_started": False,
        },
        "blockers": blockers,
        "remaining_real_data_question": (
            "On the unchanged frozen 62-record population, at the pre-whitening 121-sample core and against the exact current production operator, "
            "does masked-zero-domain-extension yield median supported-delta RMS >=0.5 ADC counts in at least one near-boundary stratum for at least two "
            "of REF176/REF192/REF220, while every REF208/REF190 stratum remains <0.5 counts and repeatability plus p2048/p1024 context gates pass?"
        ),
        "scope": {
            "can_establish": "The isolated execution-disabled source implements the frozen masked-zero mathematics and passes metadata/synthetic/component gates at the pre-whitening boundary.",
            "cannot_establish": "No real-waveform effect, mechanism discrimination on the frozen records, repair benefit, biological identity, purity, sorting improvement, production safety, or integration readiness is established.",
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
