#!/usr/bin/env python3
"""Qualify AZ donors from saved full-probe templates without reading voltage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from testing.luke_au_cpu_preparation import seeded_independent_population
except ModuleNotFoundError:
    from luke_au_cpu_preparation import seeded_independent_population


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION = (
    ROOT
    / "testing/outputs/luke_au_cpu_preparation/full_probe_extraction_v1/ba_attempt4"
)
DEFAULT_PREREG = ROOT / "testing/inputs/luke_az_relocation_preregistration_v1.json"
DEFAULT_OUTPUT = ROOT / "testing/outputs/luke_au_cpu_preparation/donor_qualification_v1"
STATES_UM = np.asarray([0.0, -40.0, -80.0, -120.0, -160.0, -200.0, -240.0])
TARGET_FIRST_AP = 202
MIN_ENERGY = 0.99
MIN_COSINE = 0.99
PTP_RANGE = (0.98, 1.02)
CLONE_COSINE = 0.99
FS_EXPECTED = 29999.759166666667


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing empty table: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def centered_cosines(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    q = np.asarray(query, dtype=np.float64).reshape(-1)
    b = np.asarray(bank, dtype=np.float64).reshape(bank.shape[0], -1)
    q = q - q.mean()
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(b, axis=1) * np.linalg.norm(q)
    return np.divide(b @ q, denom, out=np.full(b.shape[0], np.nan), where=denom > 0)


def exact_map(geom: np.ndarray, shift_um: float) -> np.ndarray:
    lookup = {(round(float(x), 6), round(float(y), 6)): i for i, (x, y) in enumerate(geom)}
    if len(lookup) != geom.shape[0]:
        raise ValueError("geometry coordinates are not unique")
    result = np.full(geom.shape[0], -1, dtype=np.int64)
    for source, (x, y) in enumerate(geom):
        result[source] = lookup.get(
            (round(float(x), 6), round(float(y + shift_um), 6)), -1
        )
    valid = result >= 0
    if np.unique(result[valid]).size != valid.sum():
        raise ValueError("exact map is not one-to-one")
    return result


def remap_to_target(template: np.ndarray, mapping: np.ndarray, target_ix: np.ndarray) -> np.ndarray:
    target_lookup = {int(channel): i for i, channel in enumerate(target_ix)}
    out = np.zeros((template.shape[0], target_ix.size), dtype=template.dtype)
    for source, destination in enumerate(mapping):
        local = target_lookup.get(int(destination))
        if local is not None:
            out[:, local] = template[:, source]
    return out


def state_metrics(
    templates: np.ndarray,
    mapping: np.ndarray,
    target_mask: np.ndarray,
    source_energy: np.ndarray,
    source_sum: np.ndarray,
    source_centered_norm2: np.ndarray,
    source_ptp: np.ndarray,
) -> dict[str, np.ndarray]:
    keep = (mapping >= 0) & target_mask[np.maximum(mapping, 0)]
    kept = templates[:, :, keep].astype(np.float64, copy=False)
    kept_energy = np.square(kept).sum(axis=(1, 2))
    kept_sum = kept.sum(axis=(1, 2))
    n = templates.shape[1] * templates.shape[2]
    restored_norm2 = kept_energy - np.square(kept_sum) / n
    dot = kept_energy - source_sum * kept_sum / n
    cosine = np.divide(
        dot,
        np.sqrt(source_centered_norm2 * restored_norm2),
        out=np.full(templates.shape[0], np.nan),
        where=(source_centered_norm2 > 0) & (restored_norm2 > 0),
    )
    if keep.any():
        restored_ptp = np.ptp(templates[:, :, keep], axis=1).max(axis=1)
    else:
        restored_ptp = np.zeros(templates.shape[0])
    return {
        "retained_energy_fraction": kept_energy / source_energy,
        "roundtrip_centered_cosine": cosine,
        "roundtrip_ptp_ratio": restored_ptp / source_ptp,
        "retained_source_channel_count": np.full(templates.shape[0], int(keep.sum())),
    }


def placement_bases(geom: np.ndarray, target_mask: np.ndarray) -> list[float]:
    span = float(geom[:, 1].max() - geom[:, 1].min())
    bases = np.arange(-span, span + 20.0, 40.0)
    accepted = []
    for base in bases:
        totals = base + STATES_UM
        if np.any(np.isclose(totals, 0.0, atol=1e-9, rtol=0.0)):
            continue
        if all(
            np.any((mapping := exact_map(geom, float(total))) >= 0)
            and np.any(target_mask[mapping[mapping >= 0]])
            for total in totals
        ):
            accepted.append(float(base))
    return sorted(
        accepted,
        key=lambda base: (
            float(np.max(np.abs(base + STATES_UM))),
            abs(base),
            base,
        ),
    )


def rank_key(row: dict) -> tuple[float, float, float, int]:
    return (
        -float(row["full_probe_ptp_standardized"]),
        float(row["raw_adjacent_isi_lt_1ms_fraction"]),
        -float(row["valid_rest_waveforms"]),
        int(row["unit_id"]),
    )


def select_cohort(qualified: list[dict], target_depths: np.ndarray) -> list[dict]:
    if len(qualified) <= 30:
        return sorted(qualified, key=lambda row: int(row["unit_id"]))
    edges = np.linspace(float(target_depths.min()), float(target_depths.max()), 7)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    selected = []
    selected_ids = set()
    for stratum in range(6):
        rows = [
            row
            for row in qualified
            if edges[stratum] <= float(row["placed_peak_depth_um"]) < edges[stratum + 1]
        ]
        for row in sorted(rows, key=rank_key)[:5]:
            selected.append(dict(row, depth_stratum=stratum))
            selected_ids.add(int(row["unit_id"]))
    if len(selected) < 30:
        remainder = [row for row in qualified if int(row["unit_id"]) not in selected_ids]
        for row in sorted(remainder, key=rank_key)[: 30 - len(selected)]:
            stratum = int(
                np.clip(
                    np.searchsorted(edges, float(row["placed_peak_depth_um"]), side="right") - 1,
                    0,
                    5,
                )
            )
            selected.append(dict(row, depth_stratum=stratum))
    return sorted(selected, key=lambda row: (int(row["depth_stratum"]), *rank_key(row)))


def run(extraction: Path, prereg: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    prereg_payload = json.loads(prereg.read_text())
    if prereg_payload.get("status") != "frozen_before_template_outcomes":
        raise RuntimeError("placement preregistration is not frozen")
    template_path = extraction / "full_probe_final_label_templates.npz"
    with np.load(template_path, allow_pickle=False) as z:
        templates = np.asarray(z["templates"], dtype=np.float32)
        unit_ids = np.asarray(z["unit_ids"], dtype=np.int64)
        channel_ids = np.asarray(z["channel_ids"])
        geom = np.asarray(z["geometry_um"], dtype=np.float64)
        fs = float(z["sampling_frequency"])
        trough_offset = int(z["trough_offset_samples"])
    if templates.shape != (90, 121, 384) or unit_ids.size != 90:
        raise RuntimeError(f"unexpected measured bank shape: {templates.shape}")
    if not np.isclose(fs, FS_EXPECTED, atol=1e-9, rtol=0):
        raise RuntimeError(f"unexpected sampling frequency: {fs}")
    expected_ids = np.asarray([f"imec1.ap#AP{i}" for i in range(384)])
    if not np.array_equal(channel_ids, expected_ids):
        raise RuntimeError("source channel IDs are not the sealed AP0-AP383 order")
    target_ix = np.arange(TARGET_FIRST_AP, 384, dtype=np.int64)
    target_mask = np.zeros(384, dtype=bool)
    target_mask[target_ix] = True
    measurements = {int(row["unit_id"]): row for row in read_csv(extraction / "full_probe_candidate_measurements.csv")}
    if set(measurements) != set(map(int, unit_ids)):
        raise RuntimeError("measurement table and template unit IDs differ")

    source64 = templates.astype(np.float64)
    source_energy = np.square(source64).sum(axis=(1, 2))
    source_sum = source64.sum(axis=(1, 2))
    n_values = templates.shape[1] * templates.shape[2]
    source_centered_norm2 = source_energy - np.square(source_sum) / n_values
    source_ptp = np.ptp(templates, axis=1).max(axis=1).astype(np.float64)
    native_crop = templates[:, :, target_ix]
    native_ptp = np.ptp(native_crop, axis=1).max(axis=1).astype(np.float64)

    bases = placement_bases(geom, target_mask)
    if not bases:
        raise RuntimeError("geometric placement enumeration is empty")
    unique_shifts = sorted({float(base + state) for base in bases for state in STATES_UM} | set(map(float, STATES_UM)))
    mappings = {shift: exact_map(geom, shift) for shift in unique_shifts}
    metrics = {
        shift: state_metrics(
            templates,
            mappings[shift],
            target_mask,
            source_energy,
            source_sum,
            source_centered_norm2,
            source_ptp,
        )
        for shift in unique_shifts
    }

    native_rows = []
    for i, unit_id in enumerate(unit_ids):
        state_passes = []
        for state in STATES_UM:
            sm = metrics[float(state)]
            passed = bool(
                sm["retained_energy_fraction"][i] >= MIN_ENERGY
                and sm["roundtrip_centered_cosine"][i] >= MIN_COSINE
                and PTP_RANGE[0] <= sm["roundtrip_ptp_ratio"][i] <= PTP_RANGE[1]
            )
            state_passes.append(passed)
        native_rows.append(
            {
                "unit_id": int(unit_id),
                "all_states_support_operator_pass": all(state_passes),
                "minimum_retained_energy_fraction": min(
                    float(metrics[float(state)]["retained_energy_fraction"][i]) for state in STATES_UM
                ),
                "minimum_roundtrip_centered_cosine": min(
                    float(metrics[float(state)]["roundtrip_centered_cosine"][i]) for state in STATES_UM
                ),
                "minimum_roundtrip_ptp_ratio": min(
                    float(metrics[float(state)]["roundtrip_ptp_ratio"][i]) for state in STATES_UM
                ),
                "maximum_roundtrip_ptp_ratio": max(
                    float(metrics[float(state)]["roundtrip_ptp_ratio"][i]) for state in STATES_UM
                ),
            }
        )

    attempt_rows = []
    chosen = []
    chosen_state_rows = []
    failure_summary = []
    for i, unit_id in enumerate(unit_ids):
        candidate_failures = Counter()
        chosen_row = None
        for rank, base in enumerate(bases, start=1):
            support_ok = True
            minimum_energy = np.inf
            minimum_cosine = np.inf
            minimum_ptp = np.inf
            maximum_ptp = -np.inf
            for state in STATES_UM:
                total = float(base + state)
                sm = metrics[total]
                energy = float(sm["retained_energy_fraction"][i])
                cosine = float(sm["roundtrip_centered_cosine"][i])
                ratio = float(sm["roundtrip_ptp_ratio"][i])
                minimum_energy = min(minimum_energy, energy)
                minimum_cosine = min(minimum_cosine, cosine)
                minimum_ptp = min(minimum_ptp, ratio)
                maximum_ptp = max(maximum_ptp, ratio)
                support_ok &= bool(
                    energy >= MIN_ENERGY
                    and cosine >= MIN_COSINE
                    and PTP_RANGE[0] <= ratio <= PTP_RANGE[1]
                )
            clone_ok = True
            worst_clone_cosine = -np.inf
            worst_clone_unit = None
            worst_clone_state = None
            if support_ok:
                for state in STATES_UM:
                    total = float(base + state)
                    moved_crop = remap_to_target(templates[i], mappings[total], target_ix)
                    moved_ptp = float(np.ptp(moved_crop, axis=0).max())
                    cosines = centered_cosines(moved_crop, native_crop)
                    ratios = np.divide(
                        moved_ptp,
                        native_ptp,
                        out=np.full(native_ptp.shape, np.nan),
                        where=native_ptp > 0,
                    )
                    clone = (cosines >= CLONE_COSINE) & (ratios >= PTP_RANGE[0]) & (ratios <= PTP_RANGE[1])
                    if np.any(clone):
                        clone_ok = False
                        clone_i = int(np.flatnonzero(clone)[np.argmax(cosines[clone])])
                        if float(cosines[clone_i]) > worst_clone_cosine:
                            worst_clone_cosine = float(cosines[clone_i])
                            worst_clone_unit = int(unit_ids[clone_i])
                            worst_clone_state = float(state)
            passed = bool(support_ok and clone_ok)
            reasons = []
            if not support_ok:
                reasons.append("support_or_inverse_operator")
            if support_ok and not clone_ok:
                reasons.append("native_clone")
            if passed:
                reasons.append("pass")
            for reason in reasons:
                candidate_failures[reason] += 1
            attempt = {
                "unit_id": int(unit_id),
                "base_rank": rank,
                "base_shift_um": base,
                "total_shift_min_um": float(np.min(base + STATES_UM)),
                "total_shift_max_um": float(np.max(base + STATES_UM)),
                "minimum_retained_energy_fraction": minimum_energy,
                "minimum_roundtrip_centered_cosine": minimum_cosine,
                "minimum_roundtrip_ptp_ratio": minimum_ptp,
                "maximum_roundtrip_ptp_ratio": maximum_ptp,
                "support_operator_pass": support_ok,
                "native_clone_pass": clone_ok if support_ok else "not_evaluated",
                "worst_clone_centered_cosine": worst_clone_cosine if np.isfinite(worst_clone_cosine) else "",
                "worst_clone_unit_id": worst_clone_unit if worst_clone_unit is not None else "",
                "worst_clone_state_um": worst_clone_state if worst_clone_state is not None else "",
                "placement_pass": passed,
                "failure_reason": "+".join(reason for reason in reasons if reason != "pass"),
            }
            attempt_rows.append(attempt)
            if passed:
                scalar = measurements[int(unit_id)]
                chosen_row = {
                    **scalar,
                    "base_rank": rank,
                    "base_shift_um": base,
                    "placed_peak_depth_um": float(scalar["depth_um"]) + base,
                    "minimum_retained_energy_fraction": minimum_energy,
                    "minimum_roundtrip_centered_cosine": minimum_cosine,
                    "minimum_roundtrip_ptp_ratio": minimum_ptp,
                    "maximum_roundtrip_ptp_ratio": maximum_ptp,
                    "native_clone_avoidance_pass": True,
                }
                chosen.append(chosen_row)
                for state in STATES_UM:
                    total = float(base + state)
                    sm = metrics[total]
                    moved_crop = remap_to_target(templates[i], mappings[total], target_ix)
                    moved_ptp = float(np.ptp(moved_crop, axis=0).max())
                    cosines = centered_cosines(moved_crop, native_crop)
                    ratios = np.divide(moved_ptp, native_ptp, out=np.full(native_ptp.shape, np.nan), where=native_ptp > 0)
                    clone = (cosines >= CLONE_COSINE) & (ratios >= PTP_RANGE[0]) & (ratios <= PTP_RANGE[1])
                    chosen_state_rows.append(
                        {
                            "unit_id": int(unit_id),
                            "base_shift_um": base,
                            "motion_state_um": float(state),
                            "total_shift_um": total,
                            "retained_energy_fraction": float(sm["retained_energy_fraction"][i]),
                            "roundtrip_centered_cosine": float(sm["roundtrip_centered_cosine"][i]),
                            "roundtrip_ptp_ratio": float(sm["roundtrip_ptp_ratio"][i]),
                            "native_clone_count": int(clone.sum()),
                            "maximum_native_centered_cosine": float(np.nanmax(cosines)),
                        }
                    )
                break
        if chosen_row is None:
            failure_summary.append(
                {
                    "unit_id": int(unit_id),
                    "placement_bases_evaluated": len(bases),
                    "support_or_inverse_operator_failed_bases": candidate_failures["support_or_inverse_operator"],
                    "native_clone_failed_bases": candidate_failures["native_clone"],
                    "reason": "no_frozen_geometric_placement_passed",
                }
            )

    cohort = select_cohort(chosen, geom[target_ix, 1]) if len(chosen) >= 20 else []
    write_csv(output / "native_position_screen.csv", native_rows)
    write_csv(output / "placement_attempts.csv", attempt_rows)
    write_csv(output / "qualified_relocated_donors.csv", chosen)
    write_csv(output / "qualified_relocated_state_audit.csv", chosen_state_rows)
    if failure_summary:
        write_csv(output / "unqualified_candidate_reasons.csv", failure_summary)
    if cohort:
        write_csv(output / "selected_donor_cohort.csv", cohort)
        cohort_ids = np.asarray([int(row["unit_id"]) for row in cohort], dtype=np.int64)
        source_pos = {int(unit_id): i for i, unit_id in enumerate(unit_ids)}
        cohort_templates = np.asarray([templates[source_pos[int(unit_id)]] for unit_id in cohort_ids])
        bases_by_unit = np.asarray([float(row["base_shift_um"]) for row in cohort], dtype=np.float64)
        state_bank = np.empty((cohort_ids.size, STATES_UM.size, 121, target_ix.size), dtype=np.float32)
        for di, unit_id in enumerate(cohort_ids):
            original = templates[source_pos[int(unit_id)]]
            for si, state in enumerate(STATES_UM):
                state_bank[di, si] = remap_to_target(
                    original, mappings[float(bases_by_unit[di] + state)], target_ix
                )
        np.savez_compressed(
            output / "qualified_donor_templates.npz",
            unit_ids=cohort_ids,
            original_templates_384=cohort_templates,
            target_state_templates=state_bank,
            target_channel_ids=channel_ids[target_ix],
            target_geometry_um=geom[target_ix],
            base_shift_um=bases_by_unit,
            occupied_motion_states_um=STATES_UM,
            sampling_frequency_hz=np.asarray(fs),
            trough_offset_samples=np.asarray(trough_offset),
            measurement_domain=np.asarray("post-ibllikecmr standardized float32 before spatial whitening"),
        )
        population = seeded_independent_population(
            cohort_ids, fs_hz=fs, start_s=900.0, end_s=1240.0
        )
        all_samples = np.concatenate([population[int(unit_id)] for unit_id in cohort_ids])
        all_units = np.concatenate(
            [np.full(population[int(unit_id)].size, int(unit_id), dtype=np.int64) for unit_id in cohort_ids]
        )
        order = np.lexsort((all_units, all_samples))
        np.savez_compressed(
            output / "immutable_ground_truth.npz",
            unit_ids=cohort_ids,
            spike_samples_w2_local=all_samples[order],
            spike_unit_ids=all_units[order],
            sampling_frequency_hz=np.asarray(fs),
            window_start_s=np.asarray(900.0),
            window_end_s=np.asarray(1240.0),
            rate_hz=np.asarray(5.0),
            refractory_ms=np.asarray(3.0),
            base_seed=np.asarray(20260926),
        )

    native_saved = read_csv(extraction / "actual_state_support.csv")
    saved_by_unit = Counter()
    for row in native_saved:
        if row["support_pass"] == "True" and row["operator_pass"] == "True":
            saved_by_unit[int(row["unit_id"])] += 1
    saved_native_all = sorted(unit for unit, count in saved_by_unit.items() if count == STATES_UM.size)
    recomputed_native_all = sorted(int(row["unit_id"]) for row in native_rows if row["all_states_support_operator_pass"])
    if saved_native_all != recomputed_native_all:
        raise RuntimeError("native-position screen does not reproduce extraction receipt")

    status = "QUALIFIED" if 20 <= len(cohort) <= 30 else "SCIENTIFIC_FEASIBILITY_BLOCKED"
    summary = {
        "schema": "luke-az-relocated-donor-qualification-v1",
        "status": status,
        "hybrid_launch_permitted": False,
        "reason_hybrid_not_yet_permitted": (
            "query-time injection and matching transition operator audit remains on the GPU worker"
            if status == "QUALIFIED"
            else "fewer than 20 donors passed the frozen relocation and native-clone gates"
        ),
        "source_template_sha256": sha256(template_path),
        "source_summary_sha256": sha256(extraction / "extraction_summary.json"),
        "preregistration_sha256": sha256(prereg),
        "measured_candidate_count": int(unit_ids.size),
        "native_position_all_state_pass_count": len(recomputed_native_all),
        "native_position_all_state_pass_unit_ids": recomputed_native_all,
        "geometric_base_count": len(bases),
        "first_ten_geometric_bases_um": bases[:10],
        "qualified_relocated_count": len(chosen),
        "unqualified_count": len(failure_summary),
        "selected_cohort_count": len(cohort),
        "selected_unit_ids": [int(row["unit_id"]) for row in cohort],
        "occupied_states_um": STATES_UM.tolist(),
        "target_channel_count": int(target_ix.size),
        "target_channel_span": [str(channel_ids[target_ix[0]]), str(channel_ids[target_ix[-1]])],
        "sampling_frequency_hz": fs,
        "no_voltage_read": True,
        "sorting_performed": False,
        "gpu_used": False,
    }
    files = [path for path in output.iterdir() if path.is_file()]
    summary["artifacts"] = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(files)
    }
    write_json(output / "qualification_summary.json", summary)
    write_json(
        output / "COMPLETE.json",
        {
            "status": status,
            "qualification_summary_sha256": sha256(output / "qualification_summary.json"),
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction", type=Path, default=DEFAULT_EXTRACTION)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.extraction, args.preregistration, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "QUALIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
