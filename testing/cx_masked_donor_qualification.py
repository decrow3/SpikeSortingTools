#!/usr/bin/env python3
"""One frozen saved-array qualification of tapered Luke imec1 donor templates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x, y = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    x -= x.mean(); y -= y.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d else np.nan


def load_az(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("cx_az_source", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def connected_peak_component(active: np.ndarray, geom: np.ndarray, peak: int) -> np.ndarray:
    keep = np.zeros(len(active), dtype=bool)
    if not active[peak]:
        active = active.copy(); active[peak] = True
    frontier = [peak]; keep[peak] = True
    while frontier:
        i = frontier.pop()
        near = np.flatnonzero(active & ~keep & (np.linalg.norm(geom - geom[i], axis=1) <= 45.0))
        keep[near] = True; frontier.extend(map(int, near))
    return keep


def support_weights(template: np.ndarray, geom: np.ndarray) -> tuple[np.ndarray, dict]:
    centered = template.astype(np.float64) - template.mean(axis=0, keepdims=True)
    energy = np.square(centered).sum(axis=0)
    dist = np.linalg.norm(geom[:, None, :] - geom[None, :, :], axis=2)
    kernel = np.exp(-0.5 * np.square(dist / 40.0))
    smooth = kernel @ energy / kernel.sum(axis=1)
    peak = int(np.argmax(smooth))
    amplitude_ratio = np.sqrt(np.divide(smooth, smooth[peak], out=np.zeros_like(smooth), where=smooth[peak] > 0))
    phase = np.clip((amplitude_ratio - 0.05) / 0.05, 0.0, 1.0)
    weights = 0.5 - 0.5 * np.cos(np.pi * phase)
    component = connected_peak_component(weights > 0, geom, peak)
    weights[~component] = 0.0
    active = weights > 0
    return weights.astype(np.float32), {
        "peak_channel": peak,
        "nonzero_channels": int(active.sum()),
        "fully_weighted_channels": int((weights >= 1 - 1e-7).sum()),
        "effective_channels": float(weights.sum()),
        "depth_span_um": float(np.ptp(geom[active, 1])) if active.any() else 0.0,
    }


def bank_metrics(az, templates, mappings, target_mask):
    source64 = templates.astype(np.float64)
    energy = np.square(source64).sum(axis=(1, 2))
    sums = source64.sum(axis=(1, 2))
    n = templates.shape[1] * templates.shape[2]
    norms = energy - np.square(sums) / n
    ptp = np.ptp(templates, axis=1).max(axis=1).astype(np.float64)
    return {
        shift: az.state_metrics(templates, mapping, target_mask, energy, sums, norms, ptp)
        for shift, mapping in mappings.items()
    }


def qualify_bank(az, templates, original_native_crop, geom, target_ix, bases, states):
    target_mask = np.zeros(len(geom), dtype=bool); target_mask[target_ix] = True
    shifts = sorted({float(base + state) for base in bases for state in states})
    mappings = {shift: az.exact_map(geom, shift) for shift in shifts}
    metrics = bank_metrics(az, templates, mappings, target_mask)
    native_ptp = np.ptp(original_native_crop, axis=1).max(axis=1).astype(float)
    rows = []
    for i in range(len(templates)):
        chosen = None
        best_min_energy = -np.inf
        best_base = None
        for rank, base in enumerate(bases, 1):
            values = [metrics[float(base + state)] for state in states]
            min_energy = min(float(v["retained_energy_fraction"][i]) for v in values)
            min_cos = min(float(v["roundtrip_centered_cosine"][i]) for v in values)
            min_ptp = min(float(v["roundtrip_ptp_ratio"][i]) for v in values)
            max_ptp = max(float(v["roundtrip_ptp_ratio"][i]) for v in values)
            if min_energy > best_min_energy:
                best_min_energy, best_base = min_energy, float(base)
            support_pass = min_energy >= .99 and min_cos >= .99 and min_ptp >= .98 and max_ptp <= 1.02
            clone_pass = True
            max_clone_cos = -np.inf
            if support_pass:
                for state in states:
                    shift = float(base + state)
                    moved = az.remap_to_target(templates[i], mappings[shift], target_ix)
                    moved_ptp = float(np.ptp(moved, axis=0).max())
                    cos = az.centered_cosines(moved, original_native_crop)
                    ratios = np.divide(moved_ptp, native_ptp, out=np.full(len(native_ptp), np.nan), where=native_ptp > 0)
                    clone = (cos >= .99) & (ratios >= .98) & (ratios <= 1.02)
                    max_clone_cos = max(max_clone_cos, float(np.nanmax(cos)))
                    if clone.any():
                        clone_pass = False
            if support_pass and clone_pass:
                chosen = (rank, float(base), min_energy, min_cos, min_ptp, max_ptp, max_clone_cos)
                break
        rows.append({
            "source_index": i, "qualified": chosen is not None,
            "chosen_base_rank": None if chosen is None else chosen[0],
            "chosen_base_shift_um": None if chosen is None else chosen[1],
            "minimum_retained_energy_fraction": None if chosen is None else chosen[2],
            "minimum_roundtrip_centered_cosine": None if chosen is None else chosen[3],
            "minimum_roundtrip_ptp_ratio": None if chosen is None else chosen[4],
            "maximum_roundtrip_ptp_ratio": None if chosen is None else chosen[5],
            "maximum_native_background_cosine": None if chosen is None else chosen[6],
            "best_minimum_energy_any_base": best_min_energy,
            "best_energy_base_um": best_base,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor-root", type=Path, required=True)
    ap.add_argument("--bc-root", type=Path, required=True)
    ap.add_argument("--scorer", type=Path, required=True)
    ap.add_argument("--preregistration", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    if os.statvfs(args.output).f_bavail * os.statvfs(args.output).f_frsize < 30_000_000_000:
        raise RuntimeError("30 GB free-space gate")
    prereg = json.loads(args.preregistration.read_text())
    if prereg["status"] != "frozen_before_modified_donor_outcomes":
        raise RuntimeError("CX preregistration is not frozen")
    source_path = args.donor_root / "extraction/full_probe_final_label_templates.npz"
    split_path = args.bc_root / "extraction/split_half_spatial_templates.npz"
    old_code = args.donor_root / "code/luke_az_qualify_donors.py"
    az = load_az(old_code)
    with np.load(source_path, allow_pickle=False) as z:
        full = np.asarray(z["templates"], np.float32); unit_ids = np.asarray(z["unit_ids"], int)
        geom = np.asarray(z["geometry_um"], float); channel_ids = np.asarray(z["channel_ids"])
    with np.load(split_path, allow_pickle=False) as z:
        left = np.asarray(z["left_templates"], np.float32); right = np.asarray(z["right_templates"], np.float32)
        if not np.array_equal(unit_ids, z["unit_ids"]) or not np.array_equal(geom, z["geometry_um"]):
            raise RuntimeError("split/full donor lineage mismatch")
        shared_basis = bool(not z["statistical_independence"])
    weights_left, weights_right, support_rows = [], [], []
    for i, unit in enumerate(unit_ids):
        wl, lm = support_weights(left[i], geom); wr, rm = support_weights(right[i], geom)
        weights_left.append(wl); weights_right.append(wr)
        support_rows.append({"unit_id": int(unit), **{f"train_{k}": v for k, v in lm.items()}, **{f"swap_{k}": v for k, v in rm.items()}, "weight_cosine_train_swap": cosine(wl, wr)})
    weights_left = np.asarray(weights_left); weights_right = np.asarray(weights_right)
    modified = full * weights_left[:, None, :]
    modified_swap = full * weights_right[:, None, :]
    target_ix = np.arange(202, 384, dtype=int)
    target_mask = np.zeros(384, bool); target_mask[target_ix] = True
    bases = az.placement_bases(geom, target_mask)
    states = np.asarray(prereg["qualification"]["occupied_states_um"], float)
    original_native_crop = full[:, :, target_ix]
    q_train = qualify_bank(az, modified, original_native_crop, geom, target_ix, bases, states)
    q_swap = qualify_bank(az, modified_swap, original_native_crop, geom, target_ix, bases, states)
    old_ids = set(prereg["old_result_preserved"]["unit_ids"])
    rows = []
    for i, unit in enumerate(unit_ids):
        fc = full[i].astype(float); mc = modified[i].astype(float)
        rc = right[i].astype(float); lc = left[i].astype(float); diff = lc - rc
        def energy(x): return float(np.square(x - x.mean(axis=0, keepdims=True)).sum())
        right_before, right_after = energy(rc), energy(rc * weights_left[i][None, :])
        diff_before, diff_after = energy(diff), energy(diff * weights_left[i][None, :])
        rows.append({
            "unit_id": int(unit), "old_source_qualified": int(unit) in old_ids,
            "modified_source_qualified": bool(q_train.iloc[i].qualified),
            "swapped_half_qualified": bool(q_swap.iloc[i].qualified),
            "qualification_stable_across_swap": bool(q_train.iloc[i].qualified == q_swap.iloc[i].qualified),
            "heldout_stable_energy_removed_fraction": 1 - right_after / right_before if right_before else np.nan,
            "train_heldout_difference_energy_removed_fraction": 1 - diff_after / diff_before if diff_before else np.nan,
            "full_source_energy_removed_fraction": 1 - np.square(mc).sum() / np.square(fc).sum(),
            "modified_over_original_ptp": float(np.ptp(mc, axis=0).max() / np.ptp(fc, axis=0).max()),
            "modified_original_centered_cosine": cosine(mc, fc),
            "heldout_split_cosine_before": cosine(lc, rc),
            "heldout_split_cosine_after_train_mask": cosine(lc * weights_left[i][None,:], rc * weights_left[i][None,:]),
            **support_rows[i],
            **{f"qualification_{k}": v for k, v in q_train.iloc[i].to_dict().items() if k != "source_index"},
            **{f"swap_qualification_{k}": v for k, v in q_swap.iloc[i].to_dict().items() if k not in ("source_index", "qualified")},
        })
    per = pd.DataFrame(rows)
    per.to_csv(args.output / "PER_DONOR_MODIFIED_QUALIFICATION.csv", index=False)
    np.savez_compressed(args.output / "DONOR_COORDINATE_TAPERS.npz", unit_ids=unit_ids, train_weights=weights_left, swapped_weights=weights_right, channel_ids=channel_ids, geometry_um=geom)
    qualified = per[per.modified_source_qualified].copy()
    qualified.to_csv(args.output / "QUALIFIED_MODIFIED_DONORS.csv", index=False)

    # Frozen stress cases: descriptive waveform checks only, no sorter outcome.
    id_to_i = {int(u): i for i, u in enumerate(unit_ids)}
    stress = []
    for a, b in [(484, 487)]:
        ia, ib = id_to_i[a], id_to_i[b]
        stress.append({"case": "similar_distinct_real_overlap", "unit_a": a, "unit_b": b, "variant": "unmodified", "cosine": cosine(full[ia], full[ib]), "ptp_ratio": float(np.ptp(full[ia])/np.ptp(full[ib]))})
        stress.append({"case": "similar_distinct_real_overlap", "unit_a": a, "unit_b": b, "variant": "modified", "cosine": cosine(modified[ia], modified[ib]), "ptp_ratio": float(np.ptp(modified[ia])/np.ptp(modified[ib]))})
    i = id_to_i[30]; base = modified[i]
    for scale in (.5, 1., 2.):
        stress.append({"case":"isolated_strong_amplitude", "unit_a":30, "unit_b":None, "variant":f"scale_{scale:g}", "cosine":cosine(base,base*scale), "ptp_ratio":float(np.ptp(base*scale)/np.ptp(base))})
    for shift in (-1,1):
        shifted=np.zeros_like(base)
        if shift>0: shifted[shift:]=base[:-shift]
        else: shifted[:shift]=base[-shift:]
        stress.append({"case":"isolated_strong_morphology", "unit_a":30, "unit_b":None, "variant":f"nonwrapping_time_shift_{shift}", "cosine":cosine(base,shifted), "ptp_ratio":float(np.ptp(shifted)/np.ptp(base))})
    stress.append({"case":"zero_control", "unit_a":None, "unit_b":None, "variant":"zero", "cosine":None, "ptp_ratio":0.0})
    pd.DataFrame(stress).to_csv(args.output / "FROZEN_STRESS_CASES.csv", index=False)

    scorer_text = args.scorer.read_text()
    truth_audit = {
        "source": str(args.scorer), "sha256": sha256(args.scorer),
        "finding": "score_with_frozen_association defines fp as every primary-label output in a region minus injected-truth matches; it has no background-event provenance input.",
        "pooled_background_stealing_risk": "A label with pre-existing background spikes can win or lose association through precision ranking, and all unmatched background events inside the chosen label are charged as injection false positives.",
        "unmatched_background_not_all_fp": "Events in unassigned labels are absent from donor FP counts; therefore pooled background is neither uniformly nor causally assigned.",
        "required_benchmark_fix": "Carry immutable background row identity through injection/sort, report injected TP/FN separately from background preservation/stealing/novel events, and keep unmatched background as unassigned rather than automatic FP.",
        "source_tokens_present": all(token in scorer_text for token in ["output_count", '"fp"', "associate_full_train"]),
    }
    (args.output / "TRUTH_ASSIGNMENT_AUDIT.json").write_text(json.dumps(truth_audit, indent=2, sort_keys=True) + "\n")
    summary = {
        "status": "complete", "interpretation": "exploratory modified-donor qualification; not pristine validation",
        "old_source_qualified": int(per.old_source_qualified.sum()), "old_source_denominator": len(per),
        "modified_source_qualified": int(per.modified_source_qualified.sum()),
        "swapped_half_qualified": int(per.swapped_half_qualified.sum()),
        "qualification_agreement_count": int(per.qualification_stable_across_swap.sum()),
        "modified_and_swap_qualified": int((per.modified_source_qualified & per.swapped_half_qualified).sum()),
        "median_stable_energy_removed_fraction": float(per.heldout_stable_energy_removed_fraction.median()),
        "median_unstable_difference_energy_removed_fraction": float(per.train_heldout_difference_energy_removed_fraction.median()),
        "median_modified_original_cosine": float(per.modified_original_centered_cosine.median()),
        "median_modified_over_original_ptp": float(per.modified_over_original_ptp.median()),
        "shared_basis_dependency": shared_basis, "sorter_launched": False, "raw_voltage_bytes": 0, "gpu_s": 0,
        "source_bytes_read": int(source_path.stat().st_size + split_path.stat().st_size),
        "wall_s": time.monotonic() - started,
        "inputs": {"full_templates_sha256": sha256(source_path), "split_halves_sha256": sha256(split_path), "old_qualification_code_sha256": sha256(old_code), "preregistration_sha256": sha256(args.preregistration)},
    }
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.output / "README.md").write_text(
        "# CX masked donor qualification\n\n"
        f"Preserved baseline: **5/90**. The frozen modified-source rule qualifies **{summary['modified_source_qualified']}/90**; "
        f"the swapped-half rule qualifies **{summary['swapped_half_qualified']}/90**, with "
        f"**{summary['modified_and_swap_qualified']}** passing both. This is exploratory because the disjoint spike halves share a temporal basis and preprocessing.\n\n"
        "No sorter, raw voltage, fractional shift, or outcome-tuned mask was used. The support/taper is learned in donor coordinates and travels with the waveform. `TRUTH_ASSIGNMENT_AUDIT.json` records why the existing pooled-background FP definition must be separated before a benchmark launch.\n"
    )
    files=[]
    for path in sorted(args.output.iterdir()):
        if path.name not in ("MANIFEST.json","COMPLETE.json"):
            files.append({"path":path.name,"bytes":path.stat().st_size,"sha256":sha256(path)})
    (args.output / "MANIFEST.json").write_text(json.dumps({"files":files,"total_bytes":sum(x["bytes"] for x in files)},indent=2,sort_keys=True)+"\n")
    (args.output / "COMPLETE.json").write_text(json.dumps({"status":"complete","manifest_sha256":sha256(args.output/"MANIFEST.json")},indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
