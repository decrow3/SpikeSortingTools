"""Draw the sealed Luke v2 holdout events from raw AP voltage.

The v2 window manifest left "matched local reference" underspecified.  This
script first writes an immutable method addendum fixing the operation to the
same 300--6000 Hz filter and 100-um local median reference used in the matched
Luke--Yates raw audit.  Event drawing is a separate explicit mode and never
opens sorter output.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_yates_raw_voltage_audit import (
    extrema_candidates,
    load_specs,
    local_median_reference,
    spatial_neighbors,
)


OUTPUT = Path("testing/outputs/luke_prospective_holdout")
MANIFEST = OUTPUT / "holdout_manifest_v2.json"
PARENT_SEAL = OUTPUT / "seal_v2.json"
SEALED_WINDOWS = OUTPUT / "sealed_windows_v2.csv"
METHOD = OUTPUT / "event_sampling_method_addendum_v2.json"
SEED = "luke-20250804-prospective-holdout-v2"
PROBE_SAMPLE_RATES_HZ = {
    "imec0": 29999.835983263598,
    "imec1": 29999.759166666667,
}
DEPTH_THIRDS = (1, 2, 3)
POLARITIES = ("negative", "positive")
AMPLITUDE_STRATA = ("50_to_75", "75_to_100", "at_least_100")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_parent_seal(
    manifest_path: Path, seal_path: Path, windows_path: Path
) -> dict[str, str]:
    """Fail closed unless the sealed v2 inputs are intact and mutually consistent."""
    seal = json.loads(seal_path.read_text())
    observed = sha256_file(manifest_path)
    if seal.get("sealed") is not True or seal.get("event_indices_drawn") is not False:
        raise RuntimeError("Parent v2 seal is not an undrawn immutable seal")
    if seal.get("sha256") != observed:
        raise RuntimeError(
            f"Parent manifest SHA mismatch: seal={seal.get('sha256')} observed={observed}"
        )

    manifest = json.loads(manifest_path.read_text())
    table = pd.read_csv(windows_path)
    manifest_windows = {
        (row["window_id"], float(row["start_s"]), float(row["stop_s"]), row["motion_stratum"])
        for row in manifest["windows"]
    }
    table_windows = {
        (row.window_id, float(row.start_s), float(row.stop_s), row.motion_stratum)
        for row in table.itertuples(index=False)
    }
    if manifest_windows != table_windows:
        raise RuntimeError("sealed_windows_v2.csv does not reproduce the manifest windows")
    probes = {row["probe"] for row in manifest["windows"]}
    if probes != set(PROBE_SAMPLE_RATES_HZ) or len(manifest["windows"]) != 12:
        raise RuntimeError("Expected both probes across the six sealed windows")
    return {
        "manifest_sha256": observed,
        "seal_sha256": sha256_file(seal_path),
        "sealed_windows_sha256": sha256_file(windows_path),
    }


def method_payload(manifest: Path) -> dict:
    return {
        "version": 2,
        "parent_manifest": str(manifest),
        "parent_manifest_sha256": sha256_file(manifest),
        "sealed_before_raw_voltage_access": True,
        "filter": {
            "kind": "scipy.signal.sosfiltfilt",
            "design": "third-order Butterworth bandpass",
            "low_hz": 300.0,
            "high_hz": 6000.0,
            "sampling_rate_design_hz": 30000.0,
            "chunk_core_s": 5.0,
            "chunk_padding_s": 0.1,
        },
        "reference": "For every physical channel, subtract at each sample the median of all same-shank contacts within 100 um, including the target contact.",
        "candidate": "Positive or negative temporal local extremum with absolute referenced amplitude at least 50 uV.",
        "deduplication": "Greedy descending-amplitude suppression within +/-0.5 ms and <=100 um; ties break by sample then physical channel.",
        "depth_strata": "Equal thirds of the probe's physical y-coordinate span; boundary values enter the higher third except the maximum.",
        "amplitude_strata_uv": {"50_to_75": [50.0, 75.0], "75_to_100": [75.0, 100.0], "at_least_100": [100.0, None]},
        "selection": "Within every probe/window/depth/polarity/amplitude cell, rank SHA256(seed|probe|window_id|sample_index|physical_channel) ascending and take four; no replacement across sparse cells.",
        "forbidden_inputs": "All sorter outputs, reviewed-event labels, recovery metrics and candidate motion-branch results.",
    }


def seal_method(manifest: Path, method: Path) -> dict:
    payload = method_payload(manifest)
    encoded = (json.dumps(payload, indent=2) + "\n").encode()
    method.parent.mkdir(parents=True, exist_ok=True)
    method.write_bytes(encoded)
    seal = {
        "method": str(method),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "sealed": True,
        "event_indices_drawn": False,
    }
    (method.parent / "event_sampling_method_seal_v2.json").write_text(
        json.dumps(seal, indent=2) + "\n"
    )
    return seal


def greedy_deduplicate(
    samples: np.ndarray,
    channels: np.ndarray,
    amplitudes: np.ndarray,
    neighbors: list[np.ndarray],
    temporal_radius: int,
) -> np.ndarray:
    """Return indices kept by deterministic descending-amplitude suppression."""
    order = np.lexsort((channels, samples, -amplitudes))
    accepted: list[list[int]] = [[] for _ in neighbors]
    kept: list[int] = []
    for index in order:
        sample, channel = int(samples[index]), int(channels[index])
        blocked = False
        for neighbor in neighbors[channel]:
            values = accepted[int(neighbor)]
            position = bisect.bisect_left(values, sample - temporal_radius)
            if position < len(values) and values[position] <= sample + temporal_radius:
                blocked = True
                break
        if blocked:
            continue
        bisect.insort(accepted[channel], sample)
        kept.append(int(index))
    return np.asarray(kept, dtype=np.int64)


def amplitude_stratum(amplitude: float) -> str:
    if amplitude < 75:
        return "50_to_75"
    if amplitude < 100:
        return "75_to_100"
    return "at_least_100"


def hash_rank(probe: str, window_id: str, sample: int, channel: int) -> str:
    value = f"{SEED}|{probe}|{window_id}|{sample}|{channel}"
    return hashlib.sha256(value.encode()).hexdigest()


def torch_local_median_reference(
    values: np.ndarray, neighbors: list[np.ndarray], device: str
) -> np.ndarray:
    """Match NumPy's odd/even median definition using Torch."""
    import torch

    source = torch.as_tensor(values, device=device)
    result = torch.empty_like(source)
    for channel, neighborhood in enumerate(neighbors):
        local = source[:, torch.as_tensor(neighborhood, device=device)]
        count = local.shape[1]
        upper = torch.kthvalue(local, count // 2 + 1, dim=1).values
        if count % 2:
            median = upper
        else:
            lower = torch.kthvalue(local, count // 2, dim=1).values
            median = 0.5 * (lower + upper)
        result[:, channel] = source[:, channel] - median
    return result.cpu().numpy()


def apply_local_reference(
    values: np.ndarray, neighbors: list[np.ndarray], backend: str
) -> np.ndarray:
    if backend == "numpy":
        return local_median_reference(values, neighbors)
    if backend == "torch_cuda":
        return torch_local_median_reference(values, neighbors, "cuda")
    raise ValueError(f"Unknown reference backend: {backend}")


def scan_window(
    spec,
    window: dict,
    core_s: float = 5.0,
    padding_s: float = 0.1,
    reference_backend: str = "numpy",
) -> pd.DataFrame:
    probe = str(window["probe"])
    fs = PROBE_SAMPLE_RATES_HZ[probe]
    raw = np.memmap(spec.binary, dtype="int16", mode="r", shape=(spec.n_frames, spec.n_channels_file))
    locations = np.asarray(spec.locations_um)
    reference_neighbors = spatial_neighbors(locations, spec.shanks, 100.0)
    dedup_neighbors = spatial_neighbors(locations, spec.shanks, 100.0)
    sos = butter(3, (300.0, 6000.0), btype="bandpass", fs=30000.0, output="sos")
    pad = int(round(padding_s * fs))
    rows = []
    start_s, stop_s = float(window["start_s"]), float(window["stop_s"])
    for core_start_s in np.arange(start_s, stop_s, core_s):
        core_stop_s = min(stop_s, core_start_s + core_s)
        core_start = int(round(core_start_s * fs))
        core_stop = int(round(core_stop_s * fs))
        read_start, read_stop = max(0, core_start - pad), min(spec.n_frames, core_stop + pad)
        values = np.asarray(raw[read_start:read_stop, : spec.neural_channels], dtype=np.float32)
        values *= float(spec.gain_uv_per_count)
        filtered = sosfiltfilt(sos, values, axis=0).astype(np.float32)
        referenced = apply_local_reference(
            filtered, reference_neighbors, reference_backend
        )
        lo, hi = core_start - read_start, core_stop - read_start
        for polarity, negative in (("negative", True), ("positive", False)):
            # Find extrema on the padded trace, then retain only the core. This
            # preserves candidates at core boundaries while the later global
            # suppression handles physical duplicates across chunks.
            times, channels, amplitudes = extrema_candidates(referenced, negative)
            keep = (times >= lo) & (times < hi) & (amplitudes >= 50.0)
            for local_time, channel, amplitude in zip(times[keep], channels[keep], amplitudes[keep]):
                rows.append((read_start + int(local_time), int(channel), float(amplitude), polarity))
    candidates = pd.DataFrame(rows, columns=["sample_index", "physical_channel", "amplitude_uv", "polarity"])
    if candidates.empty:
        return candidates
    kept = greedy_deduplicate(
        candidates.sample_index.to_numpy(), candidates.physical_channel.to_numpy(),
        candidates.amplitude_uv.to_numpy(), dedup_neighbors, int(round(0.5e-3 * fs)),
    )
    result = candidates.iloc[kept].copy()
    y = locations[result.physical_channel.to_numpy(), 1]
    edges = np.linspace(locations[:, 1].min(), locations[:, 1].max(), 4)
    result["depth_third"] = np.clip(np.digitize(y, edges[1:-1]), 0, 2) + 1
    result["depth_um"] = y
    result["amplitude_stratum"] = [amplitude_stratum(value) for value in result.amplitude_uv]
    result["probe"] = probe
    result["window_id"] = window["window_id"]
    result["window_start_s"] = start_s
    result["motion_stratum"] = window["motion_stratum"]
    result["time_s"] = result.sample_index / fs
    result["rank_sha256"] = [hash_rank(result.probe.iloc[0], window["window_id"], int(sample), int(channel)) for sample, channel in zip(result.sample_index, result.physical_channel)]
    return result


def expected_cells(windows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "probe": window["probe"],
                "window_id": window["window_id"],
                "depth_third": depth,
                "polarity": polarity,
                "amplitude_stratum": amplitude,
            }
            for window in windows
            for depth in DEPTH_THIRDS
            for polarity in POLARITIES
            for amplitude in AMPLITUDE_STRATA
        ]
    )


def select_events(
    candidates: pd.DataFrame, windows: list[dict], count: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["probe", "window_id", "depth_third", "polarity", "amplitude_stratum"]
    observed = candidates.groupby(keys, observed=True).size().rename("candidate_count").reset_index()
    chosen = candidates.sort_values("rank_sha256").groupby(keys, observed=True, as_index=False).head(count)
    selected_counts = chosen.groupby(keys, observed=True).size().rename("selected_count").reset_index()
    counts = expected_cells(windows).merge(observed, on=keys, how="left")
    counts = counts.merge(selected_counts, on=keys, how="left").fillna(
        {"candidate_count": 0, "selected_count": 0}
    )
    counts["candidate_count"] = counts.candidate_count.astype(int)
    counts["selected_count"] = counts.selected_count.astype(int)
    counts["quota"] = count
    counts["deficit"] = count - counts.selected_count
    counts["quota_met"] = counts.selected_count == count
    return chosen.sort_values(keys + ["rank_sha256"]), counts


def make_blinded_outputs(chosen: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create an opaque reviewer list and a separate sealed lookup key."""
    chosen = chosen.copy()
    chosen["candidate_id"] = [
        "LH2-" + hashlib.sha256(f"candidate-id|{value}".encode()).hexdigest()[:16]
        for value in chosen.rank_sha256
    ]
    if chosen.candidate_id.duplicated().any():
        raise RuntimeError("Candidate ID collision")
    blinded = chosen[["candidate_id"]].sort_values("candidate_id")
    key = chosen[
        [
            "candidate_id",
            "probe",
            "window_id",
            "motion_stratum",
            "depth_third",
            "polarity",
            "amplitude_stratum",
            "sample_index",
            "time_s",
            "physical_channel",
            "depth_um",
            "amplitude_uv",
            "rank_sha256",
        ]
    ].sort_values("candidate_id")
    return blinded, key


def draw(
    manifest_path: Path,
    method_path: Path,
    output: Path,
    limit_windows: int | None = None,
    seal_path: Path = PARENT_SEAL,
    windows_path: Path = SEALED_WINDOWS,
    reference_backend: str = "numpy",
) -> dict:
    immutable_before = validate_parent_seal(manifest_path, seal_path, windows_path)
    manifest = json.loads(manifest_path.read_text())
    expected = method_payload(manifest_path)
    if json.loads(method_path.read_text()) != expected:
        raise RuntimeError("Method addendum differs from the current sealed method")
    method_seal_path = method_path.parent / "event_sampling_method_seal_v2.json"
    method_seal = json.loads(method_seal_path.read_text())
    if method_seal.get("sealed") is not True or method_seal.get("sha256") != sha256_file(method_path):
        raise RuntimeError("Event sampling method addendum does not match its seal")
    specs = {spec.name.split()[1]: spec for spec in load_specs() if spec.name.endswith("session") and spec.name.startswith("Luke")}
    windows = manifest["windows"]
    if limit_windows is not None:
        windows = windows[:limit_windows]
    started = time.perf_counter()
    frames = []
    for index, window in enumerate(windows, 1):
        print(f"Scanning {index}/{len(windows)} {window['probe']} {window['window_id']}", flush=True)
        frames.append(
            scan_window(
                specs[window["probe"]],
                window,
                reference_backend=reference_backend,
            )
        )
    candidates = pd.concat(frames, ignore_index=True)
    chosen, counts = select_events(candidates, windows)
    blinded, key = make_blinded_outputs(chosen)
    suffix = "_pilot" if limit_windows is not None else "_v2"
    chosen_path = output / f"holdout_reviewer_candidates{suffix}.csv"
    key_path = output / f"holdout_candidate_key{suffix}.csv"
    counts_path = output / f"holdout_cell_deficits{suffix}.csv"
    blinded.to_csv(chosen_path, index=False)
    key.to_csv(key_path, index=False)
    counts.to_csv(counts_path, index=False)
    immutable_after = validate_parent_seal(manifest_path, seal_path, windows_path)
    if immutable_after != immutable_before:
        raise RuntimeError("A sealed v2 input changed during event drawing")
    summary = {
        "complete_draw": limit_windows is None,
        "windows_scanned": len(windows),
        "deduplicated_candidates": int(len(candidates)),
        "events_selected": int(len(chosen)),
        "cells_present": int(len(counts)),
        "cells_meeting_quota": int(counts.quota_met.sum()),
        "cells_below_quota": int((~counts.quota_met).sum()),
        "total_quota_deficit": int(counts.deficit.sum()),
        "elapsed_s": time.perf_counter() - started,
        **immutable_after,
        "method_sha256": sha256_file(method_path),
        "blinded_candidates_sha256": sha256_file(chosen_path),
        "candidate_key_sha256": sha256_file(key_path),
        "cell_deficits_sha256": sha256_file(counts_path),
        "sorter_outputs_accessed": False,
        "labels_accessed": False,
        "reference_backend": reference_backend,
    }
    (output / f"event_draw_summary{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("seal-method", "draw"), required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--parent-seal", type=Path, default=PARENT_SEAL)
    parser.add_argument("--sealed-windows", type=Path, default=SEALED_WINDOWS)
    parser.add_argument("--method", type=Path, default=METHOD)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--limit-windows", type=int)
    parser.add_argument(
        "--reference-backend", choices=("numpy", "torch_cuda"), default="numpy"
    )
    args = parser.parse_args()
    result = (
        seal_method(args.manifest, args.method)
        if args.mode == "seal-method"
        else draw(
            args.manifest,
            args.method,
            args.output_dir,
            args.limit_windows,
            args.parent_seal,
            args.sealed_windows,
            args.reference_backend,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
