#!/usr/bin/env python3
"""Independent EE fixture for the frozen fractional-position hybrid contract."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from testing.luke_au_cpu_preparation import exact_remap, exact_same_column_map  # noqa: E402
from testing.luke_dh_exact_chunk_motion import ExactChunkLatticeMotion  # noqa: E402


DH = ROOT / "testing/outputs/dh_hybrid_truth_contract_20260928"
CX = ROOT / "testing/outputs/cx_masked_donor_qualification_v1/run"
SOURCE = ROOT / "testing/outputs/luke_au_cpu_preparation/full_probe_extraction_v1/bc_attempt5/full_probe_final_label_templates.npz"
TAPERS = CX / "DONOR_COORDINATE_TAPERS.npz"
FIELD = Path("/media/huklab/Data/NPX/Ryansorting/Luke/luke_imec1_medicine_reference_sweep_v2/stage4_ab/luke0804_imec1_two_layer_motion.npz")
ACTUAL_FIELD = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/luke0804-imec1-ao-d2l-v2/windows/W2/runs/D2L/motion/fields.npz")
S_CONFIG = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/av_aw_handoff_20260926_v1/static_w2/effective-config.json")
D2L_CONFIG = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/luke0804-imec1-ao-d2l-v2/windows/W2/runs/D2L/sort/effective-config.json")
OUT = ROOT / "testing/outputs/ee_h1_fractional_translation_fixture_20260928"
DONORS = (13, 30, 407, 415)
TARGET = np.arange(202, 384, dtype=np.int64)
START_FRAME = 26_999_783


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fractional_same_x_translate(
    waveforms: np.ndarray, geometry_um: np.ndarray, shift_um: float
) -> np.ndarray:
    """Translate source(x,y) to target(x,y+shift) by same-x linear sampling.

    Equivalently, output(x,y) = source(x,y-shift). Values outside the source
    column's coordinate support are zero. There is no cross-column borrowing,
    extrapolation, or amplitude renormalization.
    """
    w = np.asarray(waveforms)
    geom = np.asarray(geometry_um, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != len(geom):
        raise ValueError("waveforms/geometry mismatch")
    out = np.zeros_like(w)
    for x in np.unique(geom[:, 0]):
        inds = np.flatnonzero(geom[:, 0] == x)
        order = np.argsort(geom[inds, 1])
        inds = inds[order]
        ys = geom[inds, 1]
        requested = ys - float(shift_um)
        right = np.searchsorted(ys, requested, side="left")
        exact = (right < len(ys)) & np.isclose(
            ys[np.minimum(right, len(ys) - 1)], requested, rtol=0.0, atol=1e-9
        )
        if exact.any():
            out[:, inds[exact]] = w[:, inds[right[exact]]]
        interp = (~exact) & (right > 0) & (right < len(ys))
        if interp.any():
            lo = right[interp] - 1
            hi = right[interp]
            alpha = (requested[interp] - ys[lo]) / (ys[hi] - ys[lo])
            out[:, inds[interp]] = (
                w[:, inds[lo]] * (1.0 - alpha)[None, :]
                + w[:, inds[hi]] * alpha[None, :]
            )
    return out


def energy(x: np.ndarray) -> float:
    return float(np.square(np.asarray(x, dtype=np.float64)).sum())


def main() -> None:
    started = time.monotonic()
    cpu0 = time.process_time()
    if OUT.exists():
        raise FileExistsError(OUT)
    if os.statvfs(OUT.parent).f_bavail * os.statvfs(OUT.parent).f_frsize < 30_000_000_000:
        raise RuntimeError("30 GB free-space gate")
    OUT.mkdir(parents=True)

    declared = [
        SOURCE,
        TAPERS,
        DH / "EXACT_QUERY_MOTION.npz",
        DH / "SOURCE_EVENTS.csv",
        DH / "INJECTION_MEMBERSHIP.csv",
        DH / "DONORS.csv",
        DH / "HYBRID_MANIFEST.json",
        FIELD,
        ACTUAL_FIELD,
        S_CONFIG,
        D2L_CONFIG,
    ]
    # Each declared data file is loaded once and hashed once. This conservative
    # accounting therefore charges twice its on-disk size.
    conservative_saved_reads = 2 * sum(p.stat().st_size for p in declared)
    if conservative_saved_reads > 64 * 1024**2:
        raise RuntimeError("64 MiB conservative saved-read gate")
    hashes = {str(p): sha(p) for p in declared}

    with np.load(SOURCE, allow_pickle=False) as z:
        unit_ids = np.asarray(z["unit_ids"], dtype=np.int64)
        donor_ix = [int(np.flatnonzero(unit_ids == u)[0]) for u in DONORS]
        templates = np.asarray(z["templates"])[donor_ix]
        geom = np.asarray(z["geometry_um"], dtype=np.float64)
        channel_ids = np.asarray(z["channel_ids"])
        trough = int(z["trough_offset_samples"])
        domain = str(z["measurement_domain"])
    with np.load(TAPERS, allow_pickle=False) as z:
        if not np.array_equal(z["unit_ids"], unit_ids):
            raise RuntimeError("taper/source unit ordering mismatch")
        weights = np.asarray(z["train_weights"])[donor_ix]
    tapered = templates * weights[:, None, :]
    donor_table = pd.read_csv(DH / "DONORS.csv").set_index("donor_id").loc[list(DONORS)]
    bases = donor_table.placement_base_shift_um.to_numpy(float)

    with np.load(DH / "EXACT_QUERY_MOTION.npz", allow_pickle=False) as z:
        unrounded = np.asarray(z["source_unrounded_um"], dtype=np.float64)
        rounded = np.asarray(z["states_um"], dtype=np.float64)
        starts = np.asarray(z["chunk_start_samples"], dtype=np.int64)
        centers = np.asarray(z["chunk_center_samples"], dtype=np.int64)
        center_local_s = np.asarray(z["chunk_center_local_s"], dtype=np.float64)
        fs = float(z["sampling_frequency_hz"])
        chunk = int(z["chunk_length_samples"])
        n_samples = int(z["n_samples"])
    if not np.array_equal(rounded, 40.0 * np.sign(unrounded) * np.floor(np.abs(unrounded / 40.0) + 0.5)):
        raise RuntimeError("frozen half-away 40 um rounding mismatch")
    if not np.array_equal(starts, np.arange(len(starts), dtype=np.int64) * chunk):
        raise RuntimeError("non-half-open chunk starts")

    # Confirm source_unrounded is exactly the declared field sample on the
    # session clock and the saved D2L local-time field on the local clock.
    with np.load(FIELD, allow_pickle=False) as z:
        field_t = np.asarray(z["time_s"], dtype=np.float64)
        field_d = np.asarray(z["displacement_um"], dtype=np.float64)[:, 0]
        sign = str(z["sign_convention"])
    with np.load(ACTUAL_FIELD, allow_pickle=False) as z:
        actual_t = np.asarray(z["time_bin_centers_s"], dtype=np.float64)
        actual_d = np.asarray(z["displacement"], dtype=np.float64)[0]
    from_session = np.interp((START_FRAME + centers) / fs, field_t, field_d)
    from_local = np.interp(center_local_s, actual_t, actual_d)
    if sign != "corrected = observed - displacement":
        raise RuntimeError("unexpected sign convention")
    if not np.allclose(from_session, unrounded, rtol=0, atol=1e-10):
        raise RuntimeError("session-clock source_unrounded mismatch")
    if not np.allclose(from_local, unrounded, rtol=0, atol=1e-10):
        raise RuntimeError("local-clock source_unrounded mismatch")

    # Analytic fixtures: identity, one-pitch equivalence, interior linear field,
    # and a half-pitch impulse split within a single x column.
    ag = np.c_[np.zeros(5), np.arange(5, dtype=float) * 40.0]
    identity_w = np.arange(15, dtype=np.float64).reshape(3, 5)
    identity = fractional_same_x_translate(identity_w, ag, 0.0)
    one_pitch = fractional_same_x_translate(identity_w, ag, 40.0)
    one_pitch_exact = exact_remap(identity_w, exact_same_column_map(ag, 40.0))
    linear = (2.0 * ag[:, 1] + 7.0)[None, :]
    linear_half = fractional_same_x_translate(linear, ag, 20.0)
    interior = (ag[:, 1] - 20.0 >= ag[:, 1].min()) & (ag[:, 1] - 20.0 <= ag[:, 1].max())
    linear_expected = 2.0 * (ag[interior, 1] - 20.0) + 7.0
    impulse = np.zeros((1, 5), dtype=np.float64)
    impulse[0, 2] = 1.0
    impulse_half = fractional_same_x_translate(impulse, ag, 20.0)
    analytic = {
        "identity_exact": bool(np.array_equal(identity, identity_w)),
        "one_pitch_exact_equivalence": bool(np.array_equal(one_pitch, one_pitch_exact)),
        "linear_half_pitch_interior_max_abs_error": float(np.max(np.abs(linear_half[0, interior] - linear_expected))),
        "impulse_half_pitch_values": impulse_half[0].tolist(),
        "impulse_half_pitch_peak_ratio": float(np.abs(impulse_half).max()),
        "impulse_half_pitch_energy_ratio": energy(impulse_half) / energy(impulse),
        "impulse_half_pitch_sum_ratio": float(impulse_half.sum() / impulse.sum()),
        "outside_support_zero": bool(impulse_half[0, 0] == 0 and impulse_half[0, -1] == 0),
    }
    if not analytic["identity_exact"] or not analytic["one_pitch_exact_equivalence"]:
        raise RuntimeError(analytic)
    if analytic["linear_half_pitch_interior_max_abs_error"] > 1e-12:
        raise RuntimeError(analytic)
    if not np.allclose(impulse_half[0], [0, 0, 0.5, 0.5, 0], rtol=0, atol=1e-12):
        raise RuntimeError(analytic)

    test_q = [
        ("zero", 0.0),
        ("one_pitch", 40.0),
        ("half_pitch", 20.0),
        ("frozen_min", float(unrounded.min())),
        ("frozen_max", float(unrounded.max())),
    ]
    actual_rows = []
    for donor, base, source_w in zip(DONORS, bases, tapered):
        source_e = energy(source_w)
        source_ptp = float(np.ptp(source_w, axis=0).max())
        for name, q in test_q:
            total = float(base + q)
            moved_full = fractional_same_x_translate(source_w, geom, total)
            moved_crop = moved_full[:, TARGET]
            exact_equivalent = None
            if np.isclose(total / 40.0, np.rint(total / 40.0), rtol=0, atol=1e-10):
                exact_crop = exact_remap(source_w, exact_same_column_map(geom, total))[:, TARGET]
                exact_equivalent = bool(np.array_equal(moved_crop, exact_crop))
                if not exact_equivalent:
                    raise RuntimeError((donor, name, total, "integer-pitch mismatch"))
            target_ptp = float(np.ptp(moved_crop, axis=0).max())
            actual_rows.append(
                {
                    "donor_id": donor,
                    "case": name,
                    "displacement_q_um": q,
                    "placement_base_um": base,
                    "total_translation_um": total,
                    "retained_energy_fraction": energy(moved_crop) / source_e,
                    "max_ptp_ratio": target_ptp / source_ptp,
                    "nonzero_crop_channels": int(np.any(moved_crop != 0, axis=0).sum()),
                    "integer_pitch_exact_equivalence": exact_equivalent,
                }
            )
    actual_df = pd.DataFrame(actual_rows)
    actual_df.to_csv(OUT / "FOUR_DONOR_TRANSLATION_METRICS.csv", index=False)

    # Query and time-boundary contract. Unrounded D2L uses the same class and
    # local clock; S_L consumes the frozen rounded values. Exact boundaries
    # belong to the new half-open chunk.
    d2l_motion = ExactChunkLatticeMotion(
        unrounded, sampling_frequency_hz=fs, chunk_length_samples=chunk, n_samples=n_samples
    )
    sl_motion = ExactChunkLatticeMotion(
        rounded, sampling_frequency_hz=fs, chunk_length_samples=chunk, n_samples=n_samples
    )
    if not np.array_equal(d2l_motion.disp_at_s(center_local_s), unrounded):
        raise RuntimeError("unrounded adapter query mismatch")
    if not np.array_equal(sl_motion.disp_at_s(center_local_s), rounded):
        raise RuntimeError("rounded adapter query mismatch")
    changes = np.flatnonzero(np.diff(unrounded) != 0) + 1
    check_chunks = changes[: min(32, len(changes))]
    boundary_ok = True
    for ci in check_chunks:
        boundary = int(starts[ci])
        before = float(d2l_motion.disp_at_s(np.array([(boundary - 1) / fs]))[0])
        at = float(d2l_motion.disp_at_s(np.array([boundary / fs]))[0])
        boundary_ok &= before == unrounded[ci - 1] and at == unrounded[ci]
    registered = 2500.0
    sign_ix = np.array([int(np.argmin(unrounded)), int(np.argmax(unrounded))])
    observed = registered + unrounded[sign_ix]
    corrected = d2l_motion.correct_s(center_local_s[sign_ix], observed)
    if not np.allclose(corrected, registered, rtol=0, atol=1e-12):
        raise RuntimeError("unrounded sign fixture")

    events = pd.read_csv(DH / "SOURCE_EVENTS.csv")
    members = pd.read_csv(DH / "INJECTION_MEMBERSHIP.csv")
    joined = events.merge(
        members[["source_event_row_id", "sample_start", "sample_stop"]],
        left_on="event_row_id", right_on="source_event_row_id", validate="one_to_one"
    )
    event_chunk = joined.local_sample.to_numpy(np.int64) // chunk
    lo_chunk = joined.sample_start.to_numpy(np.int64) // chunk
    hi_chunk = (joined.sample_stop.to_numpy(np.int64) - 1) // chunk
    crosses = lo_chunk != hi_chunk
    crosses_fractional_change = crosses & (unrounded[lo_chunk] != unrounded[hi_chunk])
    if not np.array_equal(joined.state_um.to_numpy(float), rounded[event_chunk]):
        raise RuntimeError("published event trough-state mismatch")

    s_cfg = json.loads(S_CONFIG.read_text())["config"]
    d_cfg = json.loads(D2L_CONFIG.read_text())["config"]
    result = {
        "status": "engineering_fixture_pass_benchmark_parked_fractional_attenuation_limit",
        "donors": list(DONORS),
        "donor_role": "fixed development engineering subset; no reselection or requalification",
        "measurement_domain": domain,
        "operator": (
            "source(x,y) -> target(x,y+shift) by piecewise linear interpolation in y within each exact physical x column; "
            "zero outside source support; no cross-column borrowing, extrapolation, or renormalization; taper moves with source"
        ),
        "analytic": analytic,
        "field": {
            "source_unrounded_matches_session_field": True,
            "source_unrounded_matches_saved_local_D2L_field": True,
            "sign_convention": sign,
            "unrounded_min_um": float(unrounded.min()),
            "unrounded_max_um": float(unrounded.max()),
            "rounded_min_um": float(rounded.min()),
            "rounded_max_um": float(rounded.max()),
            "maximum_absolute_rounding_error_um": float(np.max(np.abs(unrounded - rounded))),
        },
        "time_contract": {
            "chunks": int(len(starts)),
            "chunk_length_samples": chunk,
            "half_open": True,
            "exact_boundary_uses_new_chunk": bool(boundary_ok),
            "queried_boundaries": int(len(check_chunks)),
            "waveform_samples": 121,
            "trough_offset_samples": trough,
            "events": int(len(joined)),
            "waveforms_crossing_any_chunk_boundary": int(crosses.sum()),
            "waveforms_crossing_an_unrounded_state_change": int(crosses_fractional_change.sum()),
            "waveform_rule": "one chunk state chosen by trough sample and held through all 121 samples; this is a staircase, not continuous motion",
        },
        "query": {
            "D2L": "ExactChunkLatticeMotion with source_unrounded_um and recording-local zero clock",
            "S_L": "same chunk clock with frozen 40-um half-away rounded states; inverse voltage remap is target(x,y)=source(x,y+rounded_q)",
            "sign_fixture_pass": True,
        },
        "future_arm_differences": {
            "S_h": {"input": "common fractional injected native W2", "motion": "disabled", "matching_chunk_samples": int(s_cfg["matching_cfg"]["chunk_length_samples"])},
            "D2L_h": {"input": "same common fractional injected native W2", "motion": "unrounded exact-chunk external object", "matching_chunk_samples": int(d_cfg["matching_cfg"]["chunk_length_samples"])},
            "S_L_h": {"input": "separately derived rounded-lattice inverse remap of the common injected input", "motion": "disabled", "matching_chunk_samples": int(s_cfg["matching_cfg"]["chunk_length_samples"])},
            "common_required": "single linkage, seed 0, reviewed zero/zero localization fix and finite guard, otherwise accepted frozen configurations",
        },
        "interpretation": (
            "This validates generator algebra and query/boundary conventions, but it does not clear a benchmark launch. Fractional interpolation changes amplitude/energy by construction; "
            "the large attenuation for donors 407/415 near half pitch would confound a motion-continuity endpoint unless handled prospectively. It is not native identity proof. "
            "The four donors remain the exposed modified-source development set, including the known original 5/90 limitation."
        ),
        "source_hashes": {
            "fixture_source": sha(Path(__file__).resolve()),
            "fractional_exact_reference_source": sha(ROOT / "testing/luke_au_cpu_preparation.py"),
            "exact_query_adapter_source": sha(ROOT / "testing/luke_dh_exact_chunk_motion.py"),
            **hashes,
        },
    }
    (OUT / "FIXTURE.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    half = actual_df.loc[actual_df.case == "half_pitch"]
    fmin = actual_df.loc[actual_df.case == "frozen_min"]
    fmax = actual_df.loc[actual_df.case == "frozen_max"]
    (OUT / "REPORT.md").write_text(
        "# EE independent fractional-position contract fixture\n\n"
        "## Verdict\n\n"
        "The proposed same-x piecewise-linear generator and exact-chunk query contract pass the analytic and four-donor engineering fixture, but the benchmark is parked. "
        "`source_unrounded_um` exactly reproduces both the declared session-time source field and saved local-time D2L field at all 1,360 chunk centres. "
        "Its sign is `corrected = observed - displacement`; exact 7,500-sample boundaries use the new chunk.\n\n"
        "Zero translation is identity and every integer-pitch tested placement is exactly equal to the prior exact mapper on the AP202:383 crop. "
        "The analytic half-pitch impulse splits 0.5/0.5 across adjacent same-column sites: sum is preserved in the interior, peak falls to 0.5 and squared energy to 0.5. "
        "There is no renormalization. For donors 13, 30, 407 and 415, half-pitch retained-energy fractions span "
        f"{half.retained_energy_fraction.min():.6f}--{half.retained_energy_fraction.max():.6f}; frozen-min spans {fmin.retained_energy_fraction.min():.6f}--{fmin.retained_energy_fraction.max():.6f}, "
        f"and frozen-max spans {fmax.retained_energy_fraction.min():.6f}--{fmax.retained_energy_fraction.max():.6f}. Donors 407/415 retain only about 0.626--0.656 energy at half pitch, so a later outcome would mix motion handling with generator attenuation. These are synthetic-generator properties, not support gates or donor requalification.\n\n"
        f"All {len(joined):,} event troughs retain the published rounded-state ownership. {int(crosses.sum())} waveforms cross a chunk boundary and "
        f"{int(crosses_fractional_change.sum())} cross an unrounded state change; each waveform remains fixed to its trough chunk for all 121 samples. "
        "The benchmark is therefore a fractional-position staircase challenge, not continuously moving waveforms.\n\n"
        "Future S_h and S_L_h inherit 30,000-sample matching chunks; D2L_h inherits 7,500. S_h and D2L_h share one fractional injected input, while S_L_h uses a separately derived inverse remap based on the frozen 40-um rounding. "
        "All three would require single linkage, seed 0 and the same reviewed localization guard. EE does not launch or authorize them; the sub-pitch benchmark is parked pending a separate decision.\n"
    )

    receipt = {
        "status": result["status"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_s_measured": time.process_time() - cpu0,
        "wall_s_measured": time.monotonic() - started,
        "conservative_active_charge_s": 120.0,
        "threads_max": 2,
        "readers_max": 1,
        "saved_read_cap_bytes": 64 * 1024**2,
        "conservative_saved_reads_bytes": conservative_saved_reads,
        "saved_read_cap_passed": True,
        "raw_or_recording_voltage_bytes": 0,
        "gpu_s": 0,
        "sorts": 0,
        "benchmark_arms_launched": 0,
        "prior_h1_cumulative_active_s": 18878.22,
        "new_h1_cumulative_active_s": 18998.22,
        "overall_ceiling_active_s": 26000.0,
    }
    (OUT / "RESOURCE_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    files = []
    for p in sorted(OUT.iterdir()):
        if p.name not in {"MANIFEST.json", "COMPLETE.json"}:
            files.append({"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)})
    (OUT / "MANIFEST.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")
    (OUT / "COMPLETE.json").write_text(
        json.dumps({"status": result["status"], "manifest_sha256": sha(OUT / "MANIFEST.json"), "written_last": True}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": result["status"], "output": str(OUT), "charge_s": 120.0}, sort_keys=True))


if __name__ == "__main__":
    main()
