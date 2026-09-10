"""Qualify the five-minute MEDiCINe field with deterministic split-half refits."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from pipeline.config import fingerprint
from pipeline.motion_coordinates import MOTION_FIELD_SCHEMA
from testing.luke_medicine_motion_aware_merge import DEFAULT_FIELD, DEFAULT_FIELD_INPUT, DEFAULT_OUTPUT


OUT = DEFAULT_OUTPUT / "field_qualification"
MIN_EFFECTIVE_SUPPORT = 20.0
MAX_SPLIT_HALF_DISCREPANCY_UM = 20.0


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def deterministic_split(peaks, locations, fs: float) -> np.ndarray:
    times = peaks["sample_index"] / fs
    amplitudes = np.abs(peaks["amplitude"].astype(float))
    aq = np.quantile(amplitudes, [0.25, 0.5, 0.75])
    zq = np.quantile(locations["y"], [0.25, 0.5, 0.75])
    strata = (
        np.floor(times / 5.0).astype(np.int64) * 16
        + np.searchsorted(zq, locations["y"]) * 4
        + np.searchsorted(aq, amplitudes)
    )
    order = np.lexsort((peaks["channel_index"], peaks["sample_index"], strata))
    half = np.zeros(len(peaks), dtype=bool)
    previous, parity = None, 0
    for index in order:
        key = int(strata[index])
        if key != previous:
            previous, parity = key, 0
        half[index] = bool(parity)
        parity ^= 1
    return half


def interp_field(field, times, depths):
    out = np.full((len(times), len(depths)), np.nan)
    for j, depth in enumerate(depths):
        if depth < field["depth_um"][0] or depth > field["depth_um"][-1]:
            continue
        curve = np.array([np.interp(depth, field["depth_um"], row) for row in field["displacement_um"]])
        inside = (times >= field["time_s"][0]) & (times <= field["time_s"][-1])
        out[inside, j] = np.interp(times[inside], field["time_s"], curve)
    return out


def main() -> None:
    partial = OUT.with_name(OUT.name + ".partial")
    if partial.exists():
        raise RuntimeError(f"Partial qualification requires inspection: {partial}")
    if OUT.exists():
        receipt = json.loads((OUT / "receipt.json").read_text())
        if not receipt.get("complete"):
            raise RuntimeError("Existing qualification is incomplete")
        print(json.dumps(receipt, indent=2))
        return
    partial.mkdir(parents=True)
    cfg = json.loads((DEFAULT_FIELD_INPUT / "input.json").read_text())
    peaks = np.load(DEFAULT_FIELD_INPUT / "peaks.npy")
    locations = np.load(DEFAULT_FIELD_INPUT / "locations.npy")
    if len(peaks) != len(locations):
        raise RuntimeError("peak/location length mismatch")
    split_b = deterministic_split(peaks, locations, float(cfg["sampling_frequency_hz"]))
    if abs(np.mean(split_b) - 0.5) > 0.001:
        raise RuntimeError("split halves are unexpectedly imbalanced")
    if np.any(split_b & ~split_b):
        raise AssertionError("unreachable split overlap")
    for label, keep in (("a", ~split_b), ("b", split_b)):
        input_dir = partial / f"input_{label}"
        fit_dir = partial / f"fit_{label}"
        input_dir.mkdir()
        np.save(input_dir / "peaks.npy", peaks[keep])
        np.save(input_dir / "locations.npy", locations[keep])
        save_json(input_dir / "input.json", {**cfg, "split": label, "peaks": int(np.sum(keep))})
        command = [
            sys.executable, "-u", "-m", "testing.luke_screened_medicine_fit",
            "--input", str(input_dir), "--output", str(fit_dir), "--steps", "10000",
        ]
        save_json(partial / f"command_{label}.json", {"command": command})
        with (partial / f"fit_{label}.stdout.log").open("w") as stdout, (partial / f"fit_{label}.stderr.log").open("w") as stderr:
            subprocess.run(command, cwd=Path(__file__).resolve().parents[1], stdout=stdout, stderr=stderr, check=True)

    full = np.load(DEFAULT_FIELD)
    half_a = np.load(partial / "fit_a/field.npz")
    half_b = np.load(partial / "fit_b/field.npz")
    times, depths = full["time_s"], full["depth_um"]
    a = interp_field(half_a, times, depths)
    b = interp_field(half_b, times, depths)
    displacement = full["displacement_um"]
    discrepancy = np.maximum.reduce((np.abs(displacement - a), np.abs(displacement - b), 0.5 * np.abs(a - b)))
    support = np.load(DEFAULT_OUTPUT / "audit/field_support_audit.npz")["kish_effective_sample_size"]
    if support.shape != displacement.shape:
        raise RuntimeError("support/field shape mismatch")
    eligible = (
        np.isfinite(discrepancy)
        & (support >= MIN_EFFECTIVE_SUPPORT)
        & (discrepancy <= MAX_SPLIT_HALF_DISCREPANCY_UM)
    )
    contract = {
        "schema": "medicine-field-split-half-qualification-v1",
        "source_field_sha256": hashlib.sha256(DEFAULT_FIELD.read_bytes()).hexdigest(),
        "split": "deterministic alternating within 5s x depth-quartile x amplitude-quartile strata",
        "half_counts": [int(np.sum(~split_b)), int(np.sum(split_b))],
        "minimum_kish_effective_support": MIN_EFFECTIVE_SUPPORT,
        "maximum_split_half_discrepancy_um": MAX_SPLIT_HALF_DISCREPANCY_UM,
        "confidence_semantics": "binary eligibility, not a calibrated probability",
        "time_reference_transform": f"saved absolute selected-recording seconds minus {cfg['start_s']}",
    }
    qualification_digest = fingerprint(contract)
    np.savez_compressed(
        partial / "qualified_field.npz",
        schema_version=MOTION_FIELD_SCHEMA,
        qualification_passed=np.array(True),
        qualification_digest=np.array(qualification_digest),
        time_reference=np.array("selected_recording_start"),
        depth_reference=np.array("probe_y_um"),
        displacement_convention=np.array("observed_depth_offset_um"),
        estimator=np.array("MEDiCINe-5sigma-relaxed-10k-split-half-qualified"),
        polarity=np.array("negative"),
        displacement_um=displacement,
        time_s=times - float(cfg["start_s"]),
        depth_um=depths,
        support=support,
        confidence=eligible.astype(float),
    )
    np.savez_compressed(
        partial / "qualification_evidence.npz", full=displacement, half_a=a,
        half_b=b, discrepancy_um=discrepancy, support=support, eligible=eligible,
        time_s=times, depth_um=depths,
    )
    receipt = {
        **contract,
        "qualification_digest": qualification_digest,
        "complete": True,
        "field_knots": int(eligible.size),
        "eligible_knots": int(np.sum(eligible)),
        "eligible_fraction": float(np.mean(eligible)),
        "discrepancy_median_um": float(np.nanmedian(discrepancy)),
        "discrepancy_p95_um": float(np.nanquantile(discrepancy, 0.95)),
        "discrepancy_max_um": float(np.nanmax(discrepancy)),
        "scientific_status": "qualified_for_bounded_merge_replay_not_motion_ground_truth",
    }
    save_json(partial / "receipt.json", receipt)
    os.replace(partial, OUT)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
