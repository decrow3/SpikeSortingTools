#!/usr/bin/env python3
"""DC exact W2 motion-domain analysis on saved sorting finals."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


FS = 29_999.759166666667
START_FRAME, END_FRAME = 26_999_783, 37_199_701
ROOT = Path("/mnt/NPX/Luke/DARTsort_motion_experiments")
FIELD = Path("/media/huklab/Data/NPX/Ryansorting/Luke/luke_imec1_medicine_reference_sweep_v2/stage4_ab/luke0804_imec1_two_layer_motion.npz")
MASK = ROOT / "ck_input_intervals_20260927/censor_mask_v1.csv"
CATALOGUE = Path("/media/huklab/Data/NPX/Ryansorting/Luke/luke_imec1_medicine_reference_sweep_v2/stage3_q/episode_catalogue.csv")
CROPPED_FIELD = ROOT / "luke0804-imec1-ao-d2l-v2/windows/W2/runs/D2L/motion/fields.npz"
ARMS = {
    "STATIC_S": ROOT / "av_aw_handoff_20260926_v1/static_w2/dartsort_sorting.npz",
    "D2L": ROOT / "luke0804-imec1-ao-d2l-v2/windows/W2/runs/D2L/sort/dartsort_sorting.npz",
    "REMATCH0": ROOT / "cy_fixed_bank_rematching_20260928/arms/REMATCH0_v3/dartsort_sorting.npz",
    "CD1_FULL": ROOT / "cy_fixed_bank_rematching_20260928/arms/CD1_FULL_v1/dartsort_sorting.npz",
}
EXPECTED = {
    FIELD: "85062a37f38b3c5212393d627fe387b629fedd95068a4136a72f330b5aa4c2d9",
    MASK: "86425e8a6627a10ac8932ba01b799cb7be7141f731ebb4ecc01e303326b9da55",
    CATALOGUE: "46463cda1348c0e0b0c39ad22ae9c71bc46f7c7347797c1c10588944cfc52df1",
    CROPPED_FIELD: "ee34e8a6a036b10d77a0bc8563b65d171368e8869be5e96f654ee072c5bb24cf",
    ARMS["STATIC_S"]: "be6106ed0cb0f99759fd23629dd3bcb5f50657dbffa238ad3721c15d1b21fba7",
    ARMS["D2L"]: "03c339b4af8081e7e09bcbf23ebb2ae439b4cc8d468538d352cb8e79b61fc0b0",
    ARMS["REMATCH0"]: "82210d21996cee55aeef6c1bf73d23f0e3861cac867f404a2af96bbfb4024882",
    ARMS["CD1_FULL"]: "ba69b03c4c62d57acd5b51407ffcc63c1f078ba2ab4bc6b96932fe56e4f3958d",
}
DOMAINS = ("negative_excursion", "outside_mask_flat", "catalogue_outside_remainder")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def in_mask(times: np.ndarray, spans: np.ndarray) -> np.ndarray:
    result = np.zeros(times.size, dtype=bool)
    for left, right in spans:
        result |= (times >= left) & (times < right)
    return result


def merge_labeled(parts: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    out: list[list[object]] = []
    for left, right, label in parts:
        if right <= left:
            continue
        if out and label == out[-1][2] and np.isclose(left, out[-1][1], atol=1e-12):
            out[-1][1] = right
        else:
            out.append([left, right, label])
    return [(float(a), float(b), str(c)) for a, b, c in out]


def build_linear_domains(t: np.ndarray, dev: np.ndarray, mask_spans: np.ndarray) -> list[tuple[float, float, str]]:
    left, right = START_FRAME / FS, END_FRAME / FS
    breaks = [left, right]
    breaks.extend(t[(t > left) & (t < right)].tolist())
    for threshold in (-120.0, -20.0, 20.0):
        cross = (dev[:-1] - threshold) * (dev[1:] - threshold) < 0
        ix = np.flatnonzero(cross)
        roots = t[ix] + (threshold - dev[ix]) * (t[ix + 1] - t[ix]) / (dev[ix + 1] - dev[ix])
        breaks.extend(roots[(roots > left) & (roots < right)].tolist())
    for a, b in mask_spans:
        if left < a < right:
            breaks.append(float(a))
        if left < b < right:
            breaks.append(float(b))
    breaks = np.unique(np.asarray(breaks, dtype=np.float64))
    mids = (breaks[:-1] + breaks[1:]) / 2
    values = np.interp(mids, t, dev)
    masked = in_mask(mids, mask_spans)
    labels = np.full(mids.size, DOMAINS[2], dtype=object)
    labels[(~masked) & (np.abs(values) < 20.0)] = DOMAINS[1]
    labels[values < -120.0] = DOMAINS[0]
    return merge_labeled([(breaks[i], breaks[i + 1], str(labels[i])) for i in range(mids.size)])


def assign_domains(times: np.ndarray, t: np.ndarray, dev: np.ndarray, mask_spans: np.ndarray) -> np.ndarray:
    values = np.interp(times, t, dev)
    masked = in_mask(times, mask_spans)
    result = np.full(times.size, DOMAINS[2], dtype=object)
    result[(~masked) & (np.abs(values) < 20.0)] = DOMAINS[1]
    result[values < -120.0] = DOMAINS[0]
    return result


def segment_ids(times: np.ndarray, intervals: list[tuple[float, float, str]], domain: str) -> np.ndarray:
    result = np.full(times.size, -1, dtype=np.int32)
    sid = 0
    for left, right, label in intervals:
        if label == domain:
            result[(times >= left) & (times < right)] = sid
            sid += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    source = out / "source"
    source.mkdir()
    shutil.copy2(Path(__file__).resolve(), source / Path(__file__).name)
    scope = Path(__file__).resolve().parents[1] / "docs/DC-native-bank-control-and-motion-domain-check-20260928.md"
    shutil.copy2(scope, source / scope.name)
    hash_checks = {str(path): sha256(path) == expected for path, expected in EXPECTED.items()}
    if not all(hash_checks.values()):
        raise ValueError(hash_checks)

    with np.load(FIELD, allow_pickle=False) as z:
        field_t = np.asarray(z["time_s"], dtype=np.float64)
        displacement = np.asarray(z["displacement_um"], dtype=np.float64)[:, 0]
    baseline = pd.Series(displacement).rolling(480, center=True, min_periods=1).median().to_numpy()
    deviation = displacement - baseline
    mask_table = pd.read_csv(MASK)
    mask_spans = mask_table[["start_s", "end_s"]].to_numpy(float)
    intervals = build_linear_domains(field_t, deviation, mask_spans)
    interval_table = pd.DataFrame(intervals, columns=["start_s", "end_s", "domain"])
    interval_table["duration_s"] = interval_table.end_s - interval_table.start_s
    interval_table.to_csv(out / "DOMAIN_INTERVALS.csv", index=False)
    exposure = interval_table.groupby("domain", sort=False).duration_s.sum().reindex(DOMAINS)
    if not np.isclose(exposure.sum(), (END_FRAME - START_FRAME) / FS, atol=1e-9):
        raise ValueError("domain exposures do not close exact W2 support")

    count_rows, isi_rows, unit_rows, rho_rows = [], [], [], []
    for arm, path in ARMS.items():
        with np.load(path, allow_pickle=False) as z:
            local = np.asarray(z["times_samples"], dtype=np.int64)
            labels = np.asarray(z["labels"], dtype=np.int32)
            channels = np.asarray(z["channels"], dtype=np.int64)
            geom = np.asarray(z["geom"], dtype=np.float64)
            if float(z["sampling_frequency"]) != FS:
                raise ValueError(f"{arm}: frequency")
        if np.any((local < 0) | (local >= END_FRAME - START_FRAME)):
            raise ValueError(f"{arm}: time outside W2")
        event_t = (START_FRAME + local) / FS
        domain = assign_domains(event_t, field_t, deviation, mask_spans)
        assigned = labels >= 0
        arm_unit_rows = []
        for name in DOMAINS:
            inside = domain == name
            seg = segment_ids(event_t, intervals, name)
            count_rows.append({
                "arm": arm, "domain": name, "exposure_s": float(exposure[name]),
                "all_events": int(inside.sum()), "assigned_events": int((inside & assigned).sum()),
                "noise_events": int((inside & ~assigned).sum()),
                "all_event_rate_hz": float(inside.sum() / exposure[name]),
                "assigned_event_rate_hz": float((inside & assigned).sum() / exposure[name]),
                "assignment_fraction": float((inside & assigned).sum() / inside.sum()),
            })
            ix = np.flatnonzero(inside & assigned)
            order = ix[np.lexsort((event_t[ix], labels[ix]))]
            same = (labels[order[1:]] == labels[order[:-1]]) & (seg[order[1:]] == seg[order[:-1]]) if order.size > 1 else np.array([], bool)
            delta = np.diff(local[order])[same] if order.size > 1 else np.array([], dtype=np.int64)
            isi_rows.append({
                "arm": arm, "domain": name, "adjacent_denominator": int(delta.size),
                "lag0": int((delta == 0).sum()), "lag8": int((delta == 8).sum()),
                "lag9_29": int(((delta >= 9) & (delta <= 29)).sum()), "lag30": int((delta == 30).sum()),
                "fraction9_29": float(((delta >= 9) & (delta <= 29)).mean()) if delta.size else np.nan,
            })
        for unit in np.unique(labels[assigned]):
            u = assigned & (labels == unit)
            row = {"arm": arm, "unit_id": int(unit), "median_saved_channel_depth_um": float(np.median(geom[channels[u], 1]))}
            for name in DOMAINS:
                count = int(np.sum(u & (domain == name)))
                row[f"count_{name}"] = count
                row[f"log_rate_{name}"] = float(np.log((count + 0.5) / exposure[name]))
            arm_unit_rows.append(row)
        frame = pd.DataFrame(arm_unit_rows)
        eligible = frame[frame.count_outside_mask_flat >= 100].copy()
        controls = []
        for row in eligible.itertuples():
            near = eligible[(np.abs(eligible.median_saved_channel_depth_um - row.median_saved_channel_depth_um) <= 10.0) & (eligible.unit_id != row.unit_id)]
            controls.append(float(near.log_rate_outside_mask_flat.mean()) if len(near) else np.nan)
        eligible["same_row_other_unit_flat_log_rate"] = controls
        raw = spearmanr(eligible.log_rate_outside_mask_flat, eligible.log_rate_negative_excursion)
        valid_control = np.isfinite(eligible.same_row_other_unit_flat_log_rate)
        control = spearmanr(eligible.loc[valid_control, "same_row_other_unit_flat_log_rate"], eligible.loc[valid_control, "log_rate_negative_excursion"]) if valid_control.sum() >= 3 else (np.nan, np.nan)
        rho_rows.append({
            "arm": arm, "eligible_units_flat_count_ge_100": int(len(eligible)),
            "rho_negative_vs_flat": float(raw.statistic), "p_negative_vs_flat": float(raw.pvalue),
            "same_row_control_units": int(valid_control.sum()),
            "rho_negative_vs_same_row_other_flat": float(control.statistic if hasattr(control, "statistic") else control[0]),
            "p_negative_vs_same_row_other_flat": float(control.pvalue if hasattr(control, "pvalue") else control[1]),
            "interpretation": "rate-rank association; not identity or proof of event correctness",
        })
        unit_rows.extend(eligible.to_dict("records"))

    counts = pd.DataFrame(count_rows)
    isis = pd.DataFrame(isi_rows)
    rhos = pd.DataFrame(rho_rows)
    counts.to_csv(out / "DOMAIN_EVENT_COUNTS.csv", index=False)
    isis.to_csv(out / "SEGMENT_SAFE_ISI.csv", index=False)
    pd.DataFrame(unit_rows).to_csv(out / "UNIT_RATE_RANK_INPUTS.csv", index=False)
    rhos.to_csv(out / "UNIT_RATE_RANK_SUMMARY.csv", index=False)
    exposure.reset_index().rename(columns={"index": "domain", "duration_s": "exposure_s", 0: "exposure_s"}).to_csv(out / "DOMAIN_EXPOSURE.csv", index=False)
    validation = {
        "status": "pass", "hash_checks": hash_checks,
        "window_seconds_half_open": [START_FRAME / FS, END_FRAME / FS],
        "window_frames_half_open": [START_FRAME, END_FRAME],
        "exposure_closes_window": True,
        "linear_thresholds_um": {"negative_lt": -120, "flat_abs_lt": 20},
        "rolling_baseline_samples": 480, "field_depth_column": 0,
        "catalogue_outside_remainder_is_not_true_rest": True,
        "t16_ceiling_not_repeated": "out-of-range blocks used clamped last-bin indexing and requested exposure counts",
        "depth_limit": "same-row control uses median saved channel y; no point-source localization or cross-arm identity inference",
    }
    write_json(out / "VALIDATION.json", validation)
    pivot = counts.pivot(index="arm", columns="domain", values="assigned_event_rate_hz")
    report = f"""# DC exact W2 motion-domain result

## Verdict first

The exact piecewise-linear domains cover the full W2 interval without overlap
or gaps. The outside-catalogue remainder is explicitly **not true rest**. The
four saved arms are compared on identical field-defined time, and all input
hashes match the frozen values.

Assigned event rates (Hz):

| Arm | Negative excursion | Outside-mask flat | Catalogue-outside remainder |
|---|---:|---:|---:|
""" + "\n".join(
        f"| {arm} | {pivot.loc[arm, DOMAINS[0]]:.2f} | {pivot.loc[arm, DOMAINS[1]]:.2f} | {pivot.loc[arm, DOMAINS[2]]:.2f} |"
        for arm in ARMS
    ) + """

`DOMAIN_EVENT_COUNTS.csv` gives exposure, assigned/noise counts, rates, and
assignment fractions. `SEGMENT_SAFE_ISI.csv` prevents intervals from crossing
domain boundaries. Domain exposure is calculated from linear threshold
crossings and exact half-open boundaries, not grid counts or event extrema.

Across-unit rank correlations are descriptive rate/state associations only.
They do not match neurons across arms, and the same-row control uses arm-local
saved channel medians with unequal eligible cohorts. No point-source depth from
the H5 T16 work is used here; its radial-y/depth issue therefore cannot affect
this result.

The earlier T16 ceiling result is not repeated or promoted because out-of-range
blocks used a clamped last field bin and requested rather than in-bounds
exposure. No threshold or ceiling calibration was performed.
"""
    (out / "REPORT.md").write_text(report)
    (out / "README.md").write_text("# DC motion-domain packet\n\nExact W2 linear-threshold domain analysis for S, D2L, REMATCH0, and CD1_FULL. See `REPORT.md`.\n")
    elapsed = time.perf_counter() - started
    write_json(out / "RESOURCE_RECEIPT.json", {
        "analysis_elapsed_s": elapsed, "raw_voltage_reads": 0, "gpu_s": 0, "sorts": 0,
        "threads": 2, "readers": 1, "dc_charge_s": 90.0,
        "h1_cumulative_before_s": 17513.22, "h1_cumulative_after_s": 17603.22,
        "h1_ceiling_s": 20500.0,
    })
    excluded = {"MANIFEST.json", "COMPLETE.json"}
    products = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(out.rglob("*")) if p.is_file() and p.name not in excluded]
    write_json(out / "MANIFEST.json", {"products": products})
    write_json(out / "COMPLETE.json", {"status": "complete", "validation": "pass", "manifest_sha256": sha256(out / "MANIFEST.json"), "written_last_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
