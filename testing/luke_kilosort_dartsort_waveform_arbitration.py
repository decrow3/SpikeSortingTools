"""Bounded raw-waveform arbitration for selected KS/DARTsort families."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from testing.luke_kilosort_dartsort_detailed_qc import (
    ARMS, BASE_DEPTH_UM, BASE_TOLERANCE_MS, FS, atomic_json,
    load_dartsort_detailed, load_ks_detailed, subset,
)
from testing.luke_kilosort_dartsort_unit_comparison import exclusive_pairs


SCHEMA = "luke-kilosort-dartsort-waveform-arbitration-v1"
N_FAMILIES_PER_ARM = 5
MAX_EVENTS_PER_PARTITION = 256
PRE_SAMPLES = 30
POST_SAMPLES = 31
RADIUS_UM = 100.0
GAIN_UV_PER_BIT = 2.34375


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ids(value: str) -> list[int]:
    return [int(item) for item in str(value).split(";") if item]


def choose_candidates(families: pd.DataFrame) -> pd.DataFrame:
    eligible = families.loc[(families.ks_good_unit_count > 0) & (families.matched_events >= 50)].copy()
    eligible["arbitration_priority"] = (
        (1.0 - eligible.jaccard) * np.log1p(eligible.ks_spikes + eligible.dartsort_spikes)
    )
    return (
        eligible.sort_values(["arm", "arbitration_priority"], ascending=[True, False])
        .groupby("arm", sort=False).head(N_FAMILIES_PER_ARM).reset_index(drop=True)
    )


def waveform_stack(raw: np.memmap, times: np.ndarray, channels: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    times = np.asarray(times, dtype=np.int64)
    times = times[(times >= PRE_SAMPLES) & (times < raw.shape[0] - POST_SAMPLES)]
    if len(times) > MAX_EVENTS_PER_PARTITION:
        times = np.sort(rng.choice(times, MAX_EVENTS_PER_PARTITION, replace=False))
    waves = np.empty((len(times), PRE_SAMPLES + POST_SAMPLES, len(channels)), dtype=np.float32)
    for index, time in enumerate(times):
        wave = np.asarray(raw[time - PRE_SAMPLES:time + POST_SAMPLES, channels], dtype=np.float32)
        wave *= GAIN_UV_PER_BIT
        wave -= np.median(wave[:15], axis=0, keepdims=True)
        waves[index] = wave
    return waves, len(times)


def cosine_at_best_lag(reference: np.ndarray, target: np.ndarray, max_lag: int = 2) -> tuple[float, int]:
    best = (-np.inf, 0)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = reference[-lag:], target[:lag]
        elif lag > 0:
            a, b = reference[:-lag], target[lag:]
        else:
            a, b = reference, target
        av, bv = a.reshape(-1).astype(float), b.reshape(-1).astype(float)
        denom = np.linalg.norm(av) * np.linalg.norm(bv)
        value = float(np.dot(av, bv) / denom) if denom else np.nan
        if np.isfinite(value) and value > best[0]:
            best = (value, lag)
    return best


def waveform_metrics(waves: np.ndarray, reference: np.ndarray | None) -> tuple[dict, np.ndarray | None]:
    if len(waves) == 0:
        return {"sampled_events": 0, "median_waveform_ptp_uv": np.nan,
                "median_waveform_snr": np.nan, "reference_cosine": np.nan,
                "reference_best_lag_samples": np.nan}, None
    median = np.median(waves.astype(np.float64), axis=0)
    ptp = float(np.max(np.ptp(median, axis=0)))
    noise = float(np.median(np.abs(waves[:, :15])) / 0.6744897501960817)
    cosine, lag = cosine_at_best_lag(reference, median) if reference is not None else (1.0, 0)
    return {"sampled_events": len(waves), "median_waveform_ptp_uv": ptp,
            "median_waveform_snr": ptp / noise if noise else np.nan,
            "reference_cosine": cosine, "reference_best_lag_samples": lag}, median


def run(detailed_dir: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError("output exists; preserve prior evidence")
    output.mkdir(parents=True)
    matrix = Path("/media/huklab/Data/luke_motion_348ch_matrix_v1")
    lfp = Path("/media/huklab/Data/luke_lfp_native_sg_348ch_v1")
    dart_root = Path("/media/huklab/Data/NPX/Ryansorting/Luke/DARTsort_motion_experiments/luke0804-imec0-930-1230-v1/native_motion")
    raw_path = matrix / "arms/unwarped/recording/traces_cached_seg0.raw"
    binary_json = matrix / "arms/unwarped/recording/binary.json"
    binary = json.loads(binary_json.read_text())
    n_channels = int(binary["kwargs"]["num_channels"])
    dtype = np.dtype(binary["kwargs"]["dtype"])
    n_frames = raw_path.stat().st_size // (dtype.itemsize * n_channels)
    raw = np.memmap(raw_path, mode="r", dtype=dtype, shape=(n_frames, n_channels))
    geom = np.load(matrix / "arms/unwarped/kilosort4/sorter_output/ops.npy", allow_pickle=True).item()
    geom = np.c_[np.asarray(geom["xc"], float), np.asarray(geom["yc"], float)]
    fields = np.load(matrix / "resolved_candidate_fields.npz", allow_pickle=False)
    roots = {"unwarped": matrix / "arms/unwarped", "ap_rigid": matrix / "arms/ap_rigid",
             "dartsort_native": matrix / "arms/dartsort_native", "lfp_native": lfp / "arms/lfp_native",
             "lfp_savgol": lfp / "arms/lfp_savgol"}
    dart = load_dartsort_detailed(dart_root / "dartsort_sorting.npz", dart_root / "matching1.h5", dart_root / "receipt.json")
    candidates = choose_candidates(pd.read_csv(detailed_dir / "correspondence_family_qc.csv"))
    candidates.to_csv(output / "waveform_arbitration_candidates.csv", index=False)
    rng = np.random.default_rng(20260910)
    rows, medians = [], {}
    tolerance = int(round(BASE_TOLERANCE_MS * 1e-3 * FS))
    for candidate in candidates.itertuples():
        ks = load_ks_detailed(candidate.arm, roots[candidate.arm], fields)
        kmask = np.isin(ks["labels"], parse_ids(candidate.ks_units))
        dmask = np.isin(dart["labels"], parse_ids(candidate.dartsort_units))
        ksub, dsub = subset(ks, kmask), subset(dart, dmask)
        khit, dhit = exclusive_pairs(ksub, dsub, tolerance, BASE_DEPTH_UM)
        kshared = np.zeros(len(ksub["times"]), bool); kshared[khit] = True
        dshared = np.zeros(len(dsub["times"]), bool); dshared[dhit] = True
        center_depth = float(np.median(np.r_[ksub["depth"][kshared], dsub["depth"][dshared]]))
        channels = np.flatnonzero(np.abs(geom[:, 1] - center_depth) <= RADIUS_UM)
        if not len(channels):
            channels = np.array([int(np.argmin(np.abs(geom[:, 1] - center_depth)))])
        partitions = {
            "ks_shared": ksub["times"][kshared], "ks_only": ksub["times"][~kshared],
            "dartsort_shared": dsub["times"][dshared], "dartsort_only": dsub["times"][~dshared],
        }
        wave_cache = {}
        for name, times in partitions.items():
            wave_cache[name] = waveform_stack(raw, times, channels, rng)
        _, reference = waveform_metrics(wave_cache["ks_shared"][0], None)
        for name, (waves, _) in wave_cache.items():
            metrics, median = waveform_metrics(waves, reference)
            rows.append({"arm": candidate.arm, "family_id": candidate.family_id,
                         "family_shape": candidate.family_shape, "ks_units": candidate.ks_units,
                         "dartsort_units": candidate.dartsort_units, "partition": name,
                         "center_depth_um": center_depth, "channel_count": len(channels), **metrics})
            medians[(candidate.arm, int(candidate.family_id), name)] = median
    table = pd.DataFrame(rows)
    table.to_csv(output / "waveform_partition_qc.csv", index=False)
    pivot = table.pivot(index=["arm", "family_id"], columns="partition", values="reference_cosine").reset_index()
    pivot.to_csv(output / "waveform_family_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    order = ARMS
    data = [pivot.loc[pivot.arm == arm, "ks_only"].dropna() for arm in order]
    axes[0].boxplot(data, labels=[a.replace("_", "\n") for a in order], showfliers=False)
    axes[0].set_title("KS-only median waveform vs shared reference"); axes[0].set_ylabel("Best-lag cosine")
    data = [pivot.loc[pivot.arm == arm, "dartsort_only"].dropna() for arm in order]
    axes[1].boxplot(data, labels=[a.replace("_", "\n") for a in order], showfliers=False)
    axes[1].set_title("DARTsort-only median waveform vs shared reference"); axes[1].set_ylabel("Best-lag cosine")
    for axis in axes:
        axis.set_ylim(-1, 1.02); axis.axhline(0.9, color="#666666", lw=1, ls="--"); axis.grid(axis="y", alpha=.2)
    fig.suptitle("Bounded waveform arbitration on the common unwarped recording")
    fig.savefig(output / "01_waveform_arbitration.png", dpi=180)
    fig.savefig(output / "01_waveform_arbitration.pdf")
    plt.close(fig)
    result = {"schema": SCHEMA, "status": "complete", "detailed_source": str(detailed_dir.resolve()),
              "families_per_arm": N_FAMILIES_PER_ARM, "max_events_per_partition": MAX_EVENTS_PER_PARTITION,
              "waveform_samples": [PRE_SAMPLES, POST_SAMPLES], "radius_um": RADIUS_UM,
              "gain_uv_per_bit": GAIN_UV_PER_BIT, "raw_recording": str(raw_path),
              "source_hashes": {"binary_json": sha(binary_json), "detailed_summary": sha(detailed_dir / "summary.json")},
              "row_count": len(table), "candidate_count": len(candidates),
              "limitations": ["Families were selected for high disagreement and KS-good membership, not as a population-random sample.",
                              "Median-waveform similarity can reject gross waveform mismatch but does not estimate NN isolation or missed spikes."]}
    atomic_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detailed-dir", type=Path, default=Path("testing/outputs/luke_kilosort_dartsort_detailed_qc_v5"))
    parser.add_argument("--output", type=Path, default=Path("testing/outputs/luke_kilosort_dartsort_waveform_arbitration_v1"))
    args = parser.parse_args()
    print(json.dumps(run(args.detailed_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
