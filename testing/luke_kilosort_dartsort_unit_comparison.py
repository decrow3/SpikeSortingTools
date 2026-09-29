"""Match native DARTsort units to the five 348-channel KS4 correction arms."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


SCHEMA = "luke-kilosort-dartsort-unit-comparison-v1"
FS = 29999.835983263598
DURATION_S = 100.0
MIN_UNIT_SPIKES = 20
MIN_PAIR_EVENTS = 20
BASE_TOLERANCE_MS = 0.5
BASE_DEPTH_UM = 100.0
BASE_MATCH_FRACTION = 0.5
ARMS = ("unwarped", "ap_rigid", "dartsort_native", "lfp_native", "lfp_savgol")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corrected_to_observed_depth(
    arm: str,
    times: np.ndarray,
    corrected_depth: np.ndarray,
    fields: np.lib.npyio.NpzFile,
) -> np.ndarray:
    """Invert z_corrected = z_observed - displacement(t, z_observed)."""
    absolute_s = 930.0 + times / FS
    if arm == "unwarped":
        return corrected_depth.copy()
    if arm == "ap_rigid":
        disp = np.interp(absolute_s, fields["ap_time_s"], fields["ap_rigid_displacement_um"])
        return corrected_depth + disp
    if arm == "lfp_native":
        disp = np.interp(absolute_s, fields["lfp_native_time_s"], fields["lfp_native_displacement_um"])
        return corrected_depth + disp
    if arm == "lfp_savgol":
        disp = np.interp(absolute_s, fields["lfp_savgol_time_s"], fields["lfp_savgol_displacement_um"])
        return corrected_depth + disp
    if arm == "dartsort_native":
        sampler = RegularGridInterpolator(
            (fields["dartsort_time_s"], fields["dartsort_depth_um"]),
            fields["dartsort_displacement_um"],
            bounds_error=False,
            fill_value=None,
        )
        observed = corrected_depth.copy()
        for _ in range(8):
            updated = corrected_depth + sampler(np.column_stack([absolute_s, observed]))
            if np.max(np.abs(updated - observed)) < 1e-5:
                observed = updated
                break
            observed = updated
        return observed
    raise ValueError(arm)


def load_ks(arm: str, root: Path, fields: np.lib.npyio.NpzFile) -> dict:
    path = root / "kilosort4/sorter_output"
    times = np.load(path / "spike_times.npy").reshape(-1).astype(np.int64)
    labels = np.load(path / "spike_clusters.npy").reshape(-1).astype(np.int64)
    corrected_depth = np.load(path / "spike_positions.npy")[:, 1].astype(float)
    if not (len(times) == len(labels) == len(corrected_depth)):
        raise RuntimeError(f"KS arrays differ in length: {arm}")
    order = np.argsort(times, kind="stable")
    times, labels, corrected_depth = times[order], labels[order], corrected_depth[order]
    observed_depth = corrected_to_observed_depth(arm, times, corrected_depth, fields)
    label_table = pd.read_csv(path / "cluster_KSLabel.tsv", sep="\t")
    label_column = next(column for column in label_table if column != "cluster_id")
    good = set(label_table.loc[label_table[label_column].astype(str).str.lower().eq("good"), "cluster_id"].astype(int))
    ops = np.load(path / "ops.npy", allow_pickle=True).item()
    if int(ops["nblocks"]) != 0 or ops["dshift"] is not None or not bool(ops["do_CAR"]):
        raise RuntimeError(f"unexpected saved KS settings: {arm}")
    return {"name": arm, "times": times, "labels": labels, "depth": observed_depth, "good": good}


def load_dartsort(npz_path: Path, h5_path: Path, receipt_path: Path) -> dict:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "load_dartsort requires the optional h5py dependency to read "
            f"{h5_path}"
        ) from exc
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "complete" or receipt.get("returncode") != 0:
        raise RuntimeError("DARTsort source receipt is not complete")
    z = np.load(npz_path, allow_pickle=False)
    with h5py.File(h5_path, "r") as handle:
        h5_times = np.asarray(handle["times_samples"], dtype=np.int64)
        h5_channels = np.asarray(handle["channels"], dtype=np.int64)
        raw_depth = np.asarray(handle["point_source_localizations"][:, 2], dtype=float)
    times = np.asarray(z["times_samples"], dtype=np.int64)
    labels = np.asarray(z["labels"], dtype=np.int64)
    if not np.array_equal(np.asarray(z["channels"], dtype=np.int64), h5_channels):
        raise RuntimeError("DARTsort NPZ and localization H5 row channels differ")
    time_delta = times - h5_times
    if np.max(np.abs(time_delta)) > 1:
        raise RuntimeError("DARTsort final events do not align to localization rows")
    keep = (labels >= 0) & (times >= 0) & (times < int(round(DURATION_S * FS))) & np.isfinite(raw_depth)
    times, labels, raw_depth = times[keep], labels[keep], raw_depth[keep]
    order = np.argsort(times, kind="stable")
    times, labels, raw_depth = times[order], labels[order], raw_depth[order]
    counts = Counter(map(int, labels))
    median_depth = {int(unit): float(np.median(raw_depth[labels == unit])) for unit in counts}
    eligible_depth = {unit for unit, depth in median_depth.items() if 260.0 <= depth <= 3720.0}
    keep = np.isin(labels, list(eligible_depth))
    return {
        "name": "dartsort",
        "times": times[keep],
        "labels": labels[keep],
        "depth": raw_depth[keep],
        "good": set(),
        "median_depth": median_depth,
    }


def exclusive_pairs(first: dict, second: dict, tolerance: int, depth_um: float) -> tuple[np.ndarray, np.ndarray]:
    """Greedy nearest time/depth event pairing with no event reuse."""
    a_t, b_t = first["times"], second["times"]
    a_z, b_z = first["depth"], second["depth"]
    candidates: list[tuple[int, float, int, int]] = []
    left = 0
    for ai, time in enumerate(a_t):
        while left < len(b_t) and b_t[left] < time - tolerance:
            left += 1
        right = left
        while right < len(b_t) and b_t[right] <= time + tolerance:
            dz = abs(float(a_z[ai]) - float(b_z[right]))
            if dz <= depth_um:
                candidates.append((abs(int(time) - int(b_t[right])), dz, ai, right))
            right += 1
    candidates.sort()
    used_a = np.zeros(len(a_t), dtype=bool)
    used_b = np.zeros(len(b_t), dtype=bool)
    a_hit, b_hit = [], []
    for _, _, ai, bi in candidates:
        if not used_a[ai] and not used_b[bi]:
            used_a[ai] = True
            used_b[bi] = True
            a_hit.append(ai)
            b_hit.append(bi)
    return np.asarray(a_hit, dtype=np.int64), np.asarray(b_hit, dtype=np.int64)


def pair_table(ks: dict, dart: dict, a_hit: np.ndarray, b_hit: np.ndarray) -> pd.DataFrame:
    ks_ids, ks_count = np.unique(ks["labels"], return_counts=True)
    dart_ids, dart_count = np.unique(dart["labels"], return_counts=True)
    ki = {int(unit): index for index, unit in enumerate(ks_ids)}
    di = {int(unit): index for index, unit in enumerate(dart_ids)}
    counts = np.zeros((len(ks_ids), len(dart_ids)), dtype=np.int64)
    if len(a_hit):
        np.add.at(counts, ([ki[int(x)] for x in ks["labels"][a_hit]], [di[int(x)] for x in dart["labels"][b_hit]]), 1)
    rows = []
    nz = np.argwhere(counts > 0)
    for i, j in nz:
        count = int(counts[i, j])
        ka, db = int(ks_count[i]), int(dart_count[j])
        rows.append({
            "arm": ks["name"],
            "ks_unit": int(ks_ids[i]),
            "dartsort_unit": int(dart_ids[j]),
            "coincident_events": count,
            "ks_spikes": ka,
            "dartsort_spikes": db,
            "ks_coverage": count / ka,
            "dartsort_coverage": count / db,
            "smaller_unit_fraction": count / min(ka, db),
            "jaccard": count / (ka + db - count),
            "ks_good": int(ks_ids[i]) in ks["good"],
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["mutual_best"] = False
    best_dart = table.sort_values(["smaller_unit_fraction", "coincident_events"], ascending=False).drop_duplicates("ks_unit")
    best_ks = table.sort_values(["smaller_unit_fraction", "coincident_events"], ascending=False).drop_duplicates("dartsort_unit")
    mutual = set(zip(best_dart.ks_unit, best_dart.dartsort_unit)) & set(zip(best_ks.ks_unit, best_ks.dartsort_unit))
    table["mutual_best"] = [(int(a), int(b)) in mutual for a, b in zip(table.ks_unit, table.dartsort_unit)]
    return table.sort_values(["smaller_unit_fraction", "coincident_events"], ascending=False)


def unit_metrics(spikes: dict) -> pd.DataFrame:
    refractory = int(round(1.5e-3 * FS))
    bin_size = int(round(10.0 * FS))
    rows = []
    for unit in np.unique(spikes["labels"]):
        times = spikes["times"][spikes["labels"] == unit]
        rows.append({
            "unit": int(unit),
            "spikes": int(len(times)),
            "rate_hz": len(times) / DURATION_S,
            "refractory_fraction": float(np.mean(np.diff(times) < refractory)) if len(times) > 1 else np.nan,
            "presence_fraction": np.unique(times // bin_size).size / 10.0,
        })
    return pd.DataFrame(rows)


def mapping_shapes(table: pd.DataFrame, ks_units: set[int], dart_units: set[int]) -> dict:
    edges = table.loc[
        (table.coincident_events >= MIN_PAIR_EVENTS)
        & ((table[["ks_coverage", "dartsort_coverage"]].max(axis=1) >= 0.5) | (table[["ks_coverage", "dartsort_coverage"]].min(axis=1) >= 0.2))
    ]
    parent = {("ks", unit): ("ks", unit) for unit in ks_units} | {("dart", unit): ("dart", unit) for unit in dart_units}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for row in edges.itertuples():
        a, b = find(("ks", int(row.ks_unit))), find(("dart", int(row.dartsort_unit)))
        if a != b:
            parent[b] = a
    groups: defaultdict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    for node in parent:
        groups[find(node)].append(node)
    shapes = Counter(f"{sum(x[0]=='ks' for x in members)}x{sum(x[0]=='dart' for x in members)}" for members in groups.values())
    return {"qualified_edges": int(len(edges)), "mapping_shapes": dict(sorted(shapes.items()))}


def summarize(ks: dict, dart: dict, table: pd.DataFrame, null_counts: list[int]) -> tuple[dict, pd.DataFrame]:
    ks_metrics = unit_metrics(ks).set_index("unit")
    dart_metrics = unit_metrics(dart).set_index("unit")
    eligible_ks = set(map(int, ks_metrics.loc[ks_metrics.spikes >= MIN_UNIT_SPIKES].index))
    eligible_dart = set(map(int, dart_metrics.loc[dart_metrics.spikes >= MIN_UNIT_SPIKES].index))
    baseline = table.loc[
        table.mutual_best
        & table.coincident_events.ge(MIN_PAIR_EVENTS)
        & table.smaller_unit_fraction.ge(BASE_MATCH_FRACTION)
        & table.ks_unit.isin(eligible_ks)
        & table.dartsort_unit.isin(eligible_dart)
    ].copy()
    for prefix, metrics, key in [("ks", ks_metrics, "ks_unit"), ("dartsort", dart_metrics, "dartsort_unit")]:
        for column in ["spikes", "rate_hz", "refractory_fraction", "presence_fraction"]:
            baseline[f"{prefix}_{column}"] = baseline[key].map(metrics[column])
    shapes = mapping_shapes(table, eligible_ks, eligible_dart)
    summary = {
        "arm": ks["name"],
        "ks_spikes": int(len(ks["times"])),
        "dartsort_spikes": int(len(dart["times"])),
        "ks_eligible_units": len(eligible_ks),
        "ks_good_units": len(ks["good"]),
        "dartsort_eligible_units": len(eligible_dart),
        "matched_unit_pairs": int(len(baseline)),
        "matched_ks_good_units": int(baseline.ks_good.sum()),
        "ks_eligible_unit_match_fraction": len(baseline) / max(len(eligible_ks), 1),
        "ks_good_unit_match_fraction": baseline.ks_good.sum() / max(len(ks["good"]), 1),
        "dartsort_eligible_unit_match_fraction": len(baseline) / max(len(eligible_dart), 1),
        "median_smaller_unit_fraction": float(baseline.smaller_unit_fraction.median()) if len(baseline) else np.nan,
        "median_ks_coverage": float(baseline.ks_coverage.median()) if len(baseline) else np.nan,
        "median_dartsort_coverage": float(baseline.dartsort_coverage.median()) if len(baseline) else np.nan,
        "median_ks_refractory": float(baseline.ks_refractory_fraction.median()) if len(baseline) else np.nan,
        "median_dartsort_refractory": float(baseline.dartsort_refractory_fraction.median()) if len(baseline) else np.nan,
        "median_ks_presence": float(baseline.ks_presence_fraction.median()) if len(baseline) else np.nan,
        "median_dartsort_presence": float(baseline.dartsort_presence_fraction.median()) if len(baseline) else np.nan,
        "null_matched_pairs_mean": float(np.mean(null_counts)),
        "null_matched_pairs_max": int(max(null_counts)),
        **shapes,
    }
    return summary, baseline


def plot(summary: pd.DataFrame, output: Path) -> None:
    labels = [x.replace("_", "\n") for x in summary.arm]
    colors = ["#6B7280", "#D97706", "#2563EB", "#7C3AED", "#0F9D8A"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].bar(labels, summary.matched_unit_pairs, color=colors, edgecolor="#374151")
    axes[0, 0].plot(labels, summary.null_matched_pairs_mean, "o", color="#111827", label="shifted null mean")
    axes[0, 0].set_title("Mutual-best matched unit pairs")
    axes[0, 0].legend()
    axes[0, 1].bar(np.arange(len(summary)) - 0.18, 100 * summary.ks_eligible_unit_match_fraction, 0.36, label="all eligible KS", color="#0F9D8A")
    axes[0, 1].bar(np.arange(len(summary)) + 0.18, 100 * summary.ks_good_unit_match_fraction, 0.36, label="KS-good", color="#D97706")
    axes[0, 1].set_xticks(range(len(summary)), labels)
    axes[0, 1].set_ylabel("Percent matched")
    axes[0, 1].set_title("Kilosort units with a DARTsort counterpart")
    axes[0, 1].legend()
    axes[1, 0].bar(np.arange(len(summary)) - 0.18, 100 * summary.median_ks_coverage, 0.36, label="KS event coverage", color="#7C3AED")
    axes[1, 0].bar(np.arange(len(summary)) + 0.18, 100 * summary.median_dartsort_coverage, 0.36, label="DARTsort event coverage", color="#2563EB")
    axes[1, 0].set_xticks(range(len(summary)), labels)
    axes[1, 0].set_ylabel("Median percent")
    axes[1, 0].set_title("Within matched unit pairs")
    axes[1, 0].legend()
    axes[1, 1].bar(np.arange(len(summary)) - 0.18, 100 * summary.median_ks_refractory, 0.36, label="KS", color="#7C3AED")
    axes[1, 1].bar(np.arange(len(summary)) + 0.18, 100 * summary.median_dartsort_refractory, 0.36, label="DARTsort", color="#2563EB")
    axes[1, 1].set_xticks(range(len(summary)), labels)
    axes[1, 1].set_ylabel("Median ISI <1.5 ms, percent")
    axes[1, 1].set_title("Paired refractory burden")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Kilosort–DARTsort unit reconciliation\nExclusive 0.5 ms / 100 µm event pairs, 930–1030 s")
    fig.savefig(output / "01_kilosort_dartsort_unit_comparison.png", dpi=180)
    fig.savefig(output / "01_kilosort_dartsort_unit_comparison.pdf")
    plt.close(fig)


def run(output: Path) -> dict:
    if output.exists():
        raise RuntimeError("output exists; preserve prior evidence")
    output.mkdir(parents=True)
    matrix = Path("/media/huklab/Data/luke_motion_348ch_matrix_v1")
    lfp = Path("/media/huklab/Data/luke_lfp_native_sg_348ch_v1")
    dart_root = Path("/media/huklab/Data/NPX/Ryansorting/Luke/DARTsort_motion_experiments/luke0804-imec0-930-1230-v1/native_motion")
    fields = np.load(matrix / "resolved_candidate_fields.npz", allow_pickle=False)
    roots = {
        "unwarped": matrix / "arms/unwarped",
        "ap_rigid": matrix / "arms/ap_rigid",
        "dartsort_native": matrix / "arms/dartsort_native",
        "lfp_native": lfp / "arms/lfp_native",
        "lfp_savgol": lfp / "arms/lfp_savgol",
    }
    dart = load_dartsort(dart_root / "dartsort_sorting.npz", dart_root / "matching1.h5", dart_root / "receipt.json")
    dart_metrics = unit_metrics(dart)
    summaries, baselines, all_pairs, sensitivity, depth_sensitivity = [], [], [], [], []
    null_shifts_s = (17.0, 31.0, 47.0)
    for arm in ARMS:
        ks = load_ks(arm, roots[arm], fields)
        a_hit, b_hit = exclusive_pairs(ks, dart, int(round(BASE_TOLERANCE_MS * 1e-3 * FS)), BASE_DEPTH_UM)
        pairs = pair_table(ks, dart, a_hit, b_hit)
        null_counts = []
        for shift_s in null_shifts_s:
            shifted = dict(dart)
            shifted["times"] = (dart["times"] + int(round(shift_s * FS))) % int(round(DURATION_S * FS))
            order = np.argsort(shifted["times"], kind="stable")
            for key in ["times", "labels", "depth"]:
                shifted[key] = shifted[key][order]
            nh_a, nh_b = exclusive_pairs(ks, shifted, int(round(BASE_TOLERANCE_MS * 1e-3 * FS)), BASE_DEPTH_UM)
            null_table = pair_table(ks, shifted, nh_a, nh_b)
            eligible_ks = set(unit_metrics(ks).query("spikes >= @MIN_UNIT_SPIKES").unit)
            eligible_dart = set(dart_metrics.query("spikes >= @MIN_UNIT_SPIKES").unit)
            qualified = null_table.loc[
                null_table.mutual_best & null_table.coincident_events.ge(MIN_PAIR_EVENTS)
                & null_table.smaller_unit_fraction.ge(BASE_MATCH_FRACTION)
                & null_table.ks_unit.isin(eligible_ks) & null_table.dartsort_unit.isin(eligible_dart)
            ]
            null_counts.append(len(qualified))
        summary, baseline = summarize(ks, dart, pairs, null_counts)
        summaries.append(summary)
        baseline["arm"] = arm
        baselines.append(baseline)
        all_pairs.append(pairs)
        eligible_ks = set(unit_metrics(ks).query("spikes >= @MIN_UNIT_SPIKES").unit)
        eligible_dart = set(dart_metrics.query("spikes >= @MIN_UNIT_SPIKES").unit)
        for depth_tolerance_um in (60.0, 100.0, 200.0, 400.0, 10000.0):
            if depth_tolerance_um == BASE_DEPTH_UM:
                depth_table = pairs
            else:
                dh_a, dh_b = exclusive_pairs(
                    ks, dart, int(round(BASE_TOLERANCE_MS * 1e-3 * FS)), depth_tolerance_um
                )
                depth_table = pair_table(ks, dart, dh_a, dh_b)
            depth_matched = depth_table.loc[
                depth_table.mutual_best
                & depth_table.coincident_events.ge(MIN_PAIR_EVENTS)
                & depth_table.smaller_unit_fraction.ge(BASE_MATCH_FRACTION)
                & depth_table.ks_unit.isin(eligible_ks)
                & depth_table.dartsort_unit.isin(eligible_dart)
            ]
            depth_sensitivity.append({
                "arm": arm,
                "depth_tolerance_um": depth_tolerance_um,
                "matched_pairs": int(len(depth_matched)),
            })
        for tolerance_ms in (0.25, 0.5, 1.0):
            if tolerance_ms == BASE_TOLERANCE_MS:
                table = pairs
            else:
                sh_a, sh_b = exclusive_pairs(ks, dart, int(round(tolerance_ms * 1e-3 * FS)), BASE_DEPTH_UM)
                table = pair_table(ks, dart, sh_a, sh_b)
            for threshold in (0.3, 0.5, 0.7):
                matched = table.loc[table.mutual_best & table.coincident_events.ge(MIN_PAIR_EVENTS) & table.smaller_unit_fraction.ge(threshold)]
                sensitivity.append({"arm": arm, "tolerance_ms": tolerance_ms, "smaller_unit_fraction_threshold": threshold, "matched_pairs": int(len(matched))})
    summary_table = pd.DataFrame(summaries)
    baseline_table = pd.concat(baselines, ignore_index=True)
    pair_edges = pd.concat(all_pairs, ignore_index=True)
    sensitivity_table = pd.DataFrame(sensitivity)
    depth_sensitivity_table = pd.DataFrame(depth_sensitivity)
    summary_table.to_csv(output / "sorter_pair_summary.csv", index=False)
    baseline_table.to_csv(output / "matched_unit_pairs.csv", index=False)
    pair_edges.to_csv(output / "unit_pair_edges.csv", index=False)
    sensitivity_table.to_csv(output / "threshold_sensitivity.csv", index=False)
    depth_sensitivity_table.to_csv(output / "depth_sensitivity.csv", index=False)
    dart_metrics.to_csv(output / "dartsort_unit_metrics.csv", index=False)
    plot(summary_table, output)
    provenance = {
        "schema": SCHEMA,
        "status": "complete",
        "interval_s": [930.0, 1030.0],
        "channel_contract": "AP26-AP373; DARTsort units retained by median raw localization depth 260-3720 um",
        "event_pairing": "exclusive greedy assignment ordered by absolute time difference then depth difference",
        "baseline": {"tolerance_ms": BASE_TOLERANCE_MS, "depth_tolerance_um": BASE_DEPTH_UM, "minimum_pair_events": MIN_PAIR_EVENTS, "minimum_unit_spikes": MIN_UNIT_SPIKES, "smaller_unit_fraction": BASE_MATCH_FRACTION, "unit_assignment": "mutual best"},
        "null_circular_shifts_s": list(null_shifts_s),
        "coordinate_policy": "invert each KS voltage warp so all event depths are compared in observed source coordinates",
        "source_provenance": {
            "matrix_contract_sha256": sha(matrix / "contract.json"),
            "lfp_contract_sha256": sha(lfp / "contract.json"),
            "dartsort_npz_sha256": sha(dart_root / "dartsort_sorting.npz"),
            "dartsort_receipt_sha256": sha(dart_root / "receipt.json"),
            "dartsort_h5": {"path": str(dart_root / "matching1.h5"), "size_bytes": (dart_root / "matching1.h5").stat().st_size, "mtime_ns": (dart_root / "matching1.h5").stat().st_mtime_ns},
            "dartsort_localization_row_alignment": "final NPZ channels equal matching1 H5 channels rowwise; final time differs by at most one sample after template realignment; final labels intentionally supersede matching-stage labels",
        },
        "limitations": [
            "Spike-time/depth agreement is strong identity evidence but not waveform confirmation.",
            "DARTsort has no KS-good label directly comparable to Kilosort's KSLabel.",
            "The DARTsort run spans 300 s but this comparison uses only its first matched 100 s.",
            "DARTsort was sorted on 384 channels; units are restricted post hoc to the 348-channel depth range.",
        ],
        "summaries": json.loads(summary_table.to_json(orient="records")),
    }
    (output / "summary.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("testing/outputs/luke_kilosort_dartsort_unit_comparison_v1"))
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result["summaries"], indent=2))


if __name__ == "__main__":
    main()
