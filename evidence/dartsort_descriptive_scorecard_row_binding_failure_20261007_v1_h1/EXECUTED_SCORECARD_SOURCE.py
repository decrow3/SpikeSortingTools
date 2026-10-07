#!/usr/bin/env python3
"""Frozen saved-output DARTsort descriptive scorecard (no voltage access)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import shutil
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
from spikeinterface.qualitymetrics.misc_metrics import amplitude_cutoff, slidingRP_violations


FS = 29999.759166666667
N_SAMPLES = 314204094
DURATION_S = 10473.553879363088
N_EVENTS = 66286937
N_ACCEPTED = 65287555
N_NEGATIVE = 999382
N_UNITS = 1872
Q_STATES = np.array([-280, -240, -200, -160, -120, -80, -40, 0, 40, 80], dtype=np.int64)
Q_SHA256 = "92d7a28ccc808f76924a6ba392eb71c1a9b0b867796615a15307d1c59bc7df4f"
H5_SHA256 = "cc70606126e94367bdab76fc46afee08164838de55f1801d367a83a660969731"
NPZ_SHA256 = "772acc6a810bd65ea3dd8b99f197c4b8d3fcfb9248091027877d26fca50e7aa7"
CONTRACT_SHA256 = "21f21e26404172e7a369d806d30ab2966421a3b473305253aea50141b854e992"


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def quantiles(values: np.ndarray) -> dict[str, float | None]:
    if not len(values):
        return {key: None for key in ("q10", "q25", "median", "q75", "q90")}
    q = np.quantile(values.astype(np.float64, copy=False), [0.1, 0.25, 0.5, 0.75, 0.9])
    return {key: float(value) for key, value in zip(("q10", "q25", "median", "q75", "q90"), q)}


def round_half_away(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def load_and_validate_lattice(path: Path) -> dict[str, np.ndarray]:
    if sha256(path) != Q_SHA256:
        raise ValueError("q lattice SHA-256 mismatch")
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    expected = ("knot_index", "time_s", "field_um", "reference_median_um", "inside_canonical_mask", "q_um")
    if raw.dtype.names != expected or len(raw) != 41895:
        raise ValueError(f"q lattice schema mismatch: {raw.dtype.names}, rows={len(raw)}")
    knot = raw["knot_index"].astype(np.int64)
    times = raw["time_s"].astype(np.float64)
    field = raw["field_um"].astype(np.float64)
    reference = raw["reference_median_um"].astype(np.float64)
    inside = raw["inside_canonical_mask"].astype(bool)
    q = raw["q_um"].astype(np.int64)
    if not np.array_equal(knot, np.arange(41895)) or not np.allclose(times, knot * 0.25, atol=1e-12, rtol=0):
        raise ValueError("q lattice knot/time grid mismatch")
    if not np.all(np.isfinite(field)) or not np.allclose(reference, -4.695588242, atol=5e-10, rtol=0):
        raise ValueError("q lattice field/reference mismatch")
    recomputed = np.zeros(len(q), dtype=np.int64)
    recomputed[inside] = (40 * round_half_away((field[inside] - reference[inside]) / 40)).astype(np.int64)
    if not np.array_equal(q, recomputed) or not np.array_equal(np.unique(q), Q_STATES):
        raise ValueError("q lattice quantization/states mismatch")
    segment = np.cumsum(np.r_[True, q[1:] != q[:-1]]).astype(np.int64) - 1
    ends = np.minimum(times + 0.25, DURATION_S)
    durations = np.maximum(0.0, ends - times)
    exposures = {int(state): float(durations[q == state].sum()) for state in Q_STATES}
    if not math.isclose(sum(exposures.values()), DURATION_S, abs_tol=1e-8):
        raise ValueError("q lattice exposure does not cover recording duration")
    return {"q": q, "segment": segment, "exposure": exposures}


def validate_h5(path: Path) -> None:
    if path.stat().st_size != 43817258584:
        raise ValueError("matching1.h5 size mismatch")
    with h5py.File(path, "r") as h5:
        specs = {
            "times_samples": ((N_EVENTS,), np.dtype("int64")),
            "labels": ((N_EVENTS,), np.dtype("int32")),
            "denoised_ptp_amplitudes": ((N_EVENTS,), np.dtype("float32")),
            "point_source_localizations": ((N_EVENTS, 3), np.dtype("float32")),
            "geom": ((383, 2), np.dtype("float64")),
            "channel_index": ((383, 18), np.dtype("int64")),
        }
        for key, (shape, dtype) in specs.items():
            if h5[key].shape != shape or h5[key].dtype != dtype:
                raise ValueError(f"HDF5 schema mismatch for {key}: {h5[key].shape}, {h5[key].dtype}")
        if float(h5["sampling_frequency"][()]) != FS:
            raise ValueError("HDF5 sampling frequency mismatch")


def validate_npz_row_binding(npz_path: Path, h5_path: Path) -> None:
    if npz_path.stat().st_size != 10937366891:
        raise ValueError("sorting NPZ size mismatch")
    with np.load(npz_path) as sorting, h5py.File(h5_path, "r") as h5:
        if sorting["times_samples"].shape != (N_EVENTS,) or sorting["labels"].shape != (N_EVENTS,):
            raise ValueError("sorting NPZ event-array shape mismatch")
        if sorting["times_samples"].dtype != np.dtype("int64") or sorting["labels"].dtype != np.dtype("int32"):
            raise ValueError("sorting NPZ event-array dtype mismatch")
        if not np.array_equal(sorting["times_samples"], h5["times_samples"][:]):
            raise ValueError("sorting NPZ/HDF5 times_samples row binding mismatch")
        if not np.array_equal(sorting["labels"], h5["labels"][:]):
            raise ValueError("sorting NPZ/HDF5 labels row binding mismatch")
        if float(sorting["sampling_frequency"]) != FS:
            raise ValueError("sorting NPZ sampling frequency mismatch")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def event_bin(times: np.ndarray) -> np.ndarray:
    return np.clip(np.floor((times.astype(np.float64) / FS) / 0.25).astype(np.int64), 0, 41894)


def analyze(args: argparse.Namespace) -> None:
    started = time.monotonic()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"fresh output namespace required: {output}")
    output.mkdir(parents=True)
    try:
        lattice = load_and_validate_lattice(args.lattice)
        validate_h5(args.h5)
        validate_npz_row_binding(args.sorting_npz, args.h5)
        input_identity = json.loads(args.identity.read_text())
        if input_identity["matching1_h5"]["sha256"] != H5_SHA256:
            raise ValueError("identity receipt HDF5 hash mismatch")
        if input_identity["q_lattice"]["sha256"] != Q_SHA256:
            raise ValueError("identity receipt q hash mismatch")
        shutil.copy2(args.identity, output / "INPUT_IDENTITY.json")
        shutil.copy2(args.contract, output / "SCORECARD_CONTRACT.json")
        shutil.copy2(Path(__file__), output / "EXECUTED_SOURCE.py")
        effective = {
            "schema": "dartsort-descriptive-scorecard-effective-config-v1",
            "h5": str(args.h5), "sorting_npz": str(args.sorting_npz),
            "lattice": str(args.lattice), "fs": FS,
            "num_samples": N_SAMPLES, "duration_s": DURATION_S,
            "states_um": Q_STATES.tolist(), "raw_voltage_bytes": 0, "gpu": False,
            "threads": int(args.threads), "max_ram_gib": 32, "max_wall_hours": 2,
            "sliding_rp": {"bin_size_ms": 0.25, "window_size_s": 1.0,
                           "exclude_ref_period_below_ms": 0.5, "max_ref_period_ms": 10.0,
                           "pass_contamination": 0.10, "confidence": 0.90},
            "noise_cutoff": {"implementation": "SpikeInterface amplitude_cutoff",
                             "num_histogram_bins": 500, "histogram_smoothing_value": 3,
                             "amplitudes_bins_min_ratio": 5},
        }
        atomic_json(output / "EFFECTIVE_CONFIG.json", effective)

        with h5py.File(args.h5, "r") as h5:
            labels_all = h5["labels"][:]
            accepted = labels_all >= 0
            if int(accepted.sum()) != N_ACCEPTED or int((~accepted).sum()) != N_NEGATIVE:
                raise ValueError("accepted/negative event counts mismatch")
            labels = labels_all[accepted].astype(np.int32, copy=False)
            del labels_all
            times = h5["times_samples"][:][accepted].astype(np.int64, copy=False)
            amps = h5["denoised_ptp_amplitudes"][:][accepted].astype(np.float32, copy=False)
            z_abs = h5["point_source_localizations"][:, 2][accepted].astype(np.float32, copy=False)
        if labels.min() != 0 or labels.max() != N_UNITS - 1 or len(np.unique(labels)) != N_UNITS:
            raise ValueError("unit cardinality/labels mismatch")
        if times.min() < 0 or times.max() >= N_SAMPLES:
            raise ValueError("event sample bounds mismatch")
        if not np.all(np.isfinite(amps)) or not np.all(np.isfinite(z_abs)):
            raise ValueError("non-finite saved features")

        bins = event_bin(times)
        q = lattice["q"][bins]
        segments = lattice["segment"][bins]
        state_index = np.searchsorted(Q_STATES, q).astype(np.int8)
        if not np.array_equal(Q_STATES[state_index], q):
            raise ValueError("event q mapping mismatch")

        # Stable unit/time ordering supports exact adjacent-ISI and presence calculations.
        order = np.lexsort((times, labels))
        labels = labels[order]; times = times[order]; amps = amps[order]; z_abs = z_abs[order]
        q = q[order]; state_index = state_index[order]; segments = segments[order]
        counts = np.bincount(labels, minlength=N_UNITS)
        offsets = np.r_[0, np.cumsum(counts)]
        exposure = lattice["exposure"]
        displaced_exposure = sum(exposure[int(s)] for s in Q_STATES if s != 0)
        n60 = int(math.ceil(DURATION_S / 60.0))
        per_state_rows: list[dict] = []
        per_unit_rows: list[dict] = []

        for unit in range(N_UNITS):
            if time.monotonic() - started > 7100:
                raise TimeoutError("frozen two-hour stop condition approached")
            lo, hi = int(offsets[unit]), int(offsets[unit + 1])
            ut = times[lo:hi]; ua = amps[lo:hi]; uz = z_abs[lo:hi]
            uq = q[lo:hi]; usi = state_index[lo:hi]; useg = segments[lo:hi]
            dt = np.diff(ut)
            same_segment = (np.diff(useg) == 0) & (np.diff(uq) == 0)
            safe_dt = dt[same_segment]
            positive_safe = int(np.count_nonzero(safe_dt > 0))
            lag_9_29 = int(np.count_nonzero((safe_dt >= 9) & (safe_dt <= 29)))
            raw_lt_1ms = float(np.mean(dt < FS / 1000.0)) if len(dt) else math.nan
            sliding = slidingRP_violations([ut], FS, DURATION_S, 0.25, 1.0, 0.5, 10.0, None)
            q0_count = int(np.count_nonzero(uq == 0))
            displaced_count = int(len(uq) - q0_count)
            q0_rate = q0_count / exposure[0]
            displaced_rate = displaced_count / displaced_exposure
            q0_amp = ua[uq == 0]; displaced_amp = ua[uq != 0]
            q0_z = uz[uq == 0]; displaced_z = uz[uq != 0]
            unit_z_median = float(np.median(uz))
            bins60 = np.unique(np.minimum((ut.astype(np.float64) / FS / 60.0).astype(np.int64), n60 - 1)).size
            bins100 = np.unique(np.minimum((ut.astype(np.float64) / FS / (DURATION_S / 100)).astype(np.int64), 99)).size
            per_unit_rows.append({
                "unit_id": unit, "event_count": len(ut),
                "presence_ratio_60s_full_session": bins60 / n60,
                "presence_ratio_100_equal_bins": bins100 / 100.0,
                "exact_duplicate_count": int(np.count_nonzero(safe_dt == 0)),
                "lag_1_8_count": int(np.count_nonzero((safe_dt >= 1) & (safe_dt <= 8))),
                "lag_9_29_count": lag_9_29,
                "positive_segment_safe_interval_count": positive_safe,
                "lag_9_29_fraction": lag_9_29 / positive_safe if positive_safe else None,
                "raw_adjacent_isi_lt_1ms_fraction": finite_or_none(raw_lt_1ms),
                "sliding_rp_min_contamination": finite_or_none(sliding),
                "sliding_rp_pass_10pct_90conf": bool(np.isfinite(sliding) and sliding <= 0.10),
                "q0_event_count": q0_count, "displaced_event_count": displaced_count,
                "q0_to_displaced_rate_ratio": q0_rate / displaced_rate if displaced_rate else None,
                "q0_to_displaced_median_ptp_ratio": float(np.median(q0_amp) / np.median(displaced_amp)) if len(q0_amp) and len(displaced_amp) and np.median(displaced_amp) else None,
                "q0_to_displaced_median_z_shift_um": float(np.median(displaced_z) - np.median(q0_z)) if len(q0_z) and len(displaced_z) else None,
                "unit_median_observed_z_abs_um": unit_z_median,
                "interior_support": bool(300 <= unit_z_median <= 3540),
                "state_supported": bool(q0_count >= 100 and displaced_count >= 100),
            })
            for si, state in enumerate(Q_STATES):
                keep = usi == si
                sa = ua[keep]; sz = uz[keep]; count = int(keep.sum())
                aq = quantiles(sa); zq = quantiles(sz)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cutoff = amplitude_cutoff(sa) if count else math.nan
                per_state_rows.append({
                    "unit_id": unit, "state_um": int(state), "event_count": count,
                    "exposure_s": exposure[int(state)], "rate_hz": count / exposure[int(state)],
                    "ptp_q10": aq["q10"], "ptp_q25": aq["q25"], "ptp_median": aq["median"],
                    "ptp_q75": aq["q75"], "ptp_q90": aq["q90"],
                    "dartsort_denoised_ptp_noise_cutoff": finite_or_none(cutoff),
                    "observed_z_abs_q25_um": zq["q25"], "observed_z_abs_median_um": zq["median"],
                    "observed_z_abs_q75_um": zq["q75"],
                })

        state_summary = []
        for si, state in enumerate(Q_STATES):
            keep = state_index == si
            aq = quantiles(amps[keep]); zq = quantiles(z_abs[keep])
            state_summary.append({
                "state_um": int(state), "state_exposure_s": exposure[int(state)],
                "events": int(keep.sum()), "units_with_events": int(np.unique(labels[keep]).size),
                "events_per_exposure_hz": int(keep.sum()) / exposure[int(state)],
                "ptp_q10": aq["q10"], "ptp_q25": aq["q25"], "ptp_median": aq["median"],
                "ptp_q75": aq["q75"], "ptp_q90": aq["q90"],
                "observed_z_abs_q25_um": zq["q25"], "observed_z_abs_median_um": zq["median"],
                "observed_z_abs_q75_um": zq["q75"],
            })

        write_csv(output / "PER_UNIT_STATE.csv", per_state_rows, list(per_state_rows[0]))
        write_csv(output / "PER_UNIT_SUMMARY.csv", per_unit_rows, list(per_unit_rows[0]))
        write_csv(output / "STATE_SUMMARY.csv", state_summary, list(state_summary[0]))
        runtime = time.monotonic() - started
        summary = {
            "schema": "dartsort-descriptive-scorecard-summary-v1",
            "status": "DESCRIPTIVE_ONLY_NO_PRODUCTION_VERDICT",
            "population": {"accepted_events": N_ACCEPTED, "units": N_UNITS, "curation": "uncurated"},
            "state_supported_units": int(sum(row["state_supported"] for row in per_unit_rows)),
            "interior_support_units": int(sum(row["interior_support"] for row in per_unit_rows)),
            "states": state_summary,
            "can_establish": "state-conditioned descriptive changes in saved DARTsort events and model-derived features",
            "cannot_establish": ["R1 or WIU", "production benefit or harm", "physical amplitude completeness", "curated yield", "biological identity or purity", "sorter-only causality"],
        }
        atomic_json(output / "SCORECARD_SUMMARY.json", summary)
        atomic_json(output / "COST_RECEIPT.json", {
            "runtime_s": runtime, "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            "raw_voltage_bytes": 0, "gpu": False, "threads": int(args.threads),
            "h5_feature_rows_read": N_EVENTS, "outcome_rows": N_ACCEPTED,
        })
        checks = """Implementation checks
- Done: exact q lattice hash/schema/grid/quantization validated before event reads.
- Done: full matching1.h5 SHA-256 was independently computed before launch; size, datasets, dtypes, shapes, clock, event bounds, label population, and finite feature values validated in-run.
- Done: event states use the frozen AP-frame-zero 0.25 s half-open assignment; adjacent short intervals require the same exact q and contiguous q segment.
- Done: outputs remain descriptive for all uncurated nonnegative DARTsort labels; no external sorter attribution is made.
- Not done: raw-waveform or physical-amplitude validation (forbidden by this saved-output contract).
- Not done: biological identity/purity, curated yield, sorter-only causality, or inferential uncertainty.
- Can establish: exact saved-output counts and state-conditioned descriptive distributions for this one DARTsort imec1 run.
- Cannot establish: R1/WIU, production benefit/harm, physical 50 uV completeness, or generalization beyond this run.
"""
        (output / "IMPLEMENTATION_CHECKS.md").write_text(checks)
        payload_names = ["INPUT_IDENTITY.json", "SCORECARD_CONTRACT.json", "EXECUTED_SOURCE.py", "EFFECTIVE_CONFIG.json", "PER_UNIT_STATE.csv", "PER_UNIT_SUMMARY.csv", "STATE_SUMMARY.csv", "SCORECARD_SUMMARY.json", "COST_RECEIPT.json", "IMPLEMENTATION_CHECKS.md"]
        manifest = {"schema": "immutable-evidence-manifest-v1", "files": []}
        for name in payload_names:
            path = output / name
            manifest["files"].append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
        atomic_json(output / "MANIFEST.json", manifest)
        atomic_json(output / "COMPLETE.json", {"status": "complete", "manifest_sha256": sha256(output / "MANIFEST.json"), "complete": True})
    except Exception as exc:
        atomic_json(output / "FAILED.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "elapsed_s": time.monotonic() - started})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--sorting-npz", type=Path, required=True)
    parser.add_argument("--lattice", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    analyze(args)


if __name__ == "__main__":
    main()
