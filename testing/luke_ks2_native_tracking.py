"""Guarded runner and batch-boundary audit for pinned native KS2 v2.0.2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-ks2-numba-cache")


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_RESCUE_ROOT = Path(
    "/mnt/NPX/Luke/20250804/"
    "rescue_pipeline_results_Luke0804_V2V1_g0_imec1"
)
DEFAULT_KS2 = Path("/home/huklab/Documents/Kilosort2-v2.0.2")
DEFAULT_FIXTURE = REPO_ROOT / "testing/outputs/luke_ks2_installation_fixture"
EXPECTED_COMMIT = "0ce102799e69b97e3364ae47b403a809712d7e15"
NT = 64 * 1024 - 64
NTBUFF = 64
STRIDE = NT - NTBUFF
ALIGNED_NT = 64 * 1024 + 64
FS = 30_000.0
HISTORICAL_HALF_WIDTH_SAMPLES = int(round(3.5e-3 * FS))
PIN_PATH = REPO_ROOT / "testing/luke_ks2_v2.0.2_pin.json"
MATLAB_RUNNER = REPO_ROOT / "testing/matlab/run_luke_ks2_native.m"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def verify_installation(kilosort_dir: Path) -> dict[str, Any]:
    commit = _git_output(kilosort_dir, "rev-parse", "HEAD")
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"KS2 commit {commit} is not pinned {EXPECTED_COMMIT}")
    tracked_diff = _git_output(kilosort_dir, "status", "--porcelain", "--untracked-files=no")
    if tracked_diff:
        raise RuntimeError("Pinned KS2 has modified tracked files")
    modules = sorted((kilosort_dir / "CUDA").glob("*.mexa64"))
    if len(modules) != 8:
        raise RuntimeError(f"Expected 8 KS2 MEX modules, found {len(modules)}")
    pin = json.loads(PIN_PATH.read_text())
    actual = {path.name: sha256_file(path) for path in modules}
    if actual != pin["mex_sha256"]:
        raise RuntimeError("Compiled KS2 MEX hashes differ from the frozen pin")
    return {
        "root": str(kilosort_dir.resolve()),
        "tag": _git_output(kilosort_dir, "describe", "--tags", "--exact-match"),
        "commit": commit,
        "tracked_source_clean": True,
        "mex_sha256": actual,
    }


def _si_components():
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    import spikeinterface as si
    from spikeinterface.sorters.external.kilosort2 import Kilosort2Sorter
    import spikeinterface.sorters as sorters_pkg

    utils_dir = Path(sorters_pkg.__file__).resolve().parent / "utils"
    return si, Kilosort2Sorter, utils_dir


def make_fixture_recording(
    seed: int = 2102, duration_s: float = 60.0, stride: int = STRIDE
):
    si, _, _ = _si_components()
    rng = np.random.default_rng(seed)
    n_channels = 32
    n_samples = int(round(duration_s * FS))
    traces = rng.normal(0, 18, size=(n_samples, n_channels)).astype(np.int16)
    wave_t = np.arange(61) - 20
    temporal = (
        -520 * np.exp(-0.5 * (wave_t / 2.2) ** 2)
        + 145 * np.exp(-0.5 * ((wave_t - 8) / 4.5) ** 2)
    )
    centers = np.arange(2, n_channels - 2, 3)
    events: list[tuple[int, int]] = []
    for batch in range(1, int(np.floor(n_samples / stride))):
        boundary = batch * stride
        for offset in (-90, -30, 30, 90, stride // 2 - 90, stride // 2, stride // 2 + 90):
            sample = boundary + offset
            if 30 <= sample < n_samples - 40:
                events.append((sample, int(centers[(batch + offset) % len(centers)])))
        # Dense deterministic support makes every batch clusterable.
        for phase in np.linspace(600, stride - 600, 42, dtype=int):
            sample = (batch - 1) * stride + int(phase)
            if 30 <= sample < n_samples - 40:
                events.append((sample, int(centers[(batch + phase) % len(centers)])))
    for sample, center in events:
        spatial = np.exp(-0.5 * ((np.arange(n_channels) - center) / 1.15) ** 2)
        patch = np.rint(temporal[:, None] * spatial[None, :]).astype(np.int16)
        traces[sample - 20 : sample + 41] += patch
    recording = si.NumpyRecording(traces, sampling_frequency=FS)
    locations = np.column_stack(
        ((np.arange(n_channels) % 2) * 32.0, (np.arange(n_channels) // 2) * 20.0)
    )
    recording.set_channel_locations(locations)
    truth = np.asarray([sample for sample, _ in events], dtype=np.int64)
    return recording, truth


def load_luke_window(rescue_root: Path, start_s: float, duration_s: float):
    from pipeline.bakeoff import _load_si_extractor

    recording = _load_si_extractor(rescue_root / "recording")
    start = int(round(start_s * recording.get_sampling_frequency()))
    end = start + int(round(duration_s * recording.get_sampling_frequency()))
    if start < 0 or end > recording.get_num_samples():
        raise ValueError("Requested Luke window is outside the accepted recording")
    return recording.frame_slice(start_frame=start, end_frame=end)


def prepare_native_input(
    recording, output_dir: Path, fixture: bool, nt: int = NT
) -> dict[str, Any]:
    _, sorter, utils_dir = _si_components()
    output_dir.mkdir(parents=True, exist_ok=False)
    params = sorter.default_params()
    params.update(
        {
            "NT": nt,
            "ntbuff": NTBUFF,
            "save_rez_to_mat": False,
            "delete_tmp_files": False,
            "delete_recording_dat": False,
            "n_jobs": 1,
            "chunk_duration": "1s",
            "progress_bar": True,
        }
    )
    if fixture:
        params["minfr_goodchannels"] = 0.0
        params["minFR"] = 0.0
    params = sorter._check_params(recording, output_dir, params)
    if params["NT"] != nt:
        raise RuntimeError("SpikeInterface changed the explicitly requested NT")
    sorter._setup_recording(recording, output_dir, params, True)
    for name in ("writeNPY.m", "constructNPYheader.m"):
        shutil.copy2(utils_dir / name, output_dir / name)
    binary_path = output_dir / "recording.dat"
    if not binary_path.exists():
        # A compatible source binary may be referenced directly in ops.mat.
        from scipy.io import loadmat

        binary_path = Path(str(loadmat(output_dir / "ops.mat", simplify_cells=True)["ops"]["fbinary"]))
    return {
        "params": params,
        "binary_path": str(binary_path.resolve()),
        "binary_bytes": binary_path.stat().st_size,
        "binary_sha256": sha256_file(binary_path),
        "num_channels": int(recording.get_num_channels()),
        "num_samples": int(recording.get_num_samples()),
        "sampling_frequency_hz": float(recording.get_sampling_frequency()),
    }


def run_matlab(output_dir: Path, kilosort_dir: Path) -> dict[str, Any]:
    command = [
        "matlab",
        "-batch",
        (
            f"addpath('{MATLAB_RUNNER.parent}'); "
            f"run_luke_ks2_native('{output_dir.resolve()}','{kilosort_dir.resolve()}')"
        ),
    ]
    started = time.time()
    completed = subprocess.run(command, text=True, capture_output=True)
    (output_dir / "matlab_stdout.log").write_text(completed.stdout)
    (output_dir / "matlab_stderr.log").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(
            f"MATLAB KS2 failed with {completed.returncode}; inspect {output_dir}"
        )
    return {"command": command, "elapsed_s": time.time() - started}


def _circular_distance(phases: np.ndarray, center: int, period: int) -> np.ndarray:
    raw = np.abs(phases - center)
    return np.minimum(raw, period - raw)


def boundary_ratio(
    times: np.ndarray, stride: int = STRIDE, sampling_frequency: float = FS
) -> dict[str, Any]:
    phases = np.asarray(times, dtype=np.int64) % stride
    width = int(round(3.5e-3 * sampling_frequency))
    boundary = _circular_distance(phases, 0, stride) <= width
    # Same total phase width at the maximally separated interior location.
    interior = _circular_distance(phases, stride // 2, stride) <= width
    boundary_rate = float(boundary.mean())
    interior_rate = float(interior.mean())
    ratio = boundary_rate / interior_rate if interior_rate else float("nan")
    shifts = np.linspace(0, stride - 1, 1001, dtype=np.int64)[1:]
    null = np.empty(shifts.size, dtype=float)
    for index, shift in enumerate(shifts):
        a = (_circular_distance(phases, int(shift), stride) <= width).mean()
        b = (_circular_distance(phases, int((shift + stride // 2) % stride), stride) <= width).mean()
        null[index] = a / b if b else np.nan
    finite = null[np.isfinite(null)]
    percentile = float((finite <= ratio).mean()) if finite.size and np.isfinite(ratio) else float("nan")
    return {
        "spike_count": int(times.size),
        "stride_samples": stride,
        "historical_boundary_total_width_samples": 2 * width + 1,
        "historical_boundary_total_width_ms": (2 * width + 1) / sampling_frequency * 1000,
        "boundary_count": int(boundary.sum()),
        "matched_interior_count": int(interior.sum()),
        "boundary_ratio": ratio,
        "pseudo_boundary_percentile": percentile,
        "pooled_gate_pass": bool(not (ratio < 0.98 and percentile < 0.01)),
    }


def audit_output(
    output_dir: Path,
    sampling_frequency: float = FS,
    stride: int = STRIDE,
) -> dict[str, Any]:
    import csv

    times = np.load(output_dir / "spike_times.npy").reshape(-1).astype(np.int64)
    labels = np.load(output_dir / "spike_clusters.npy").reshape(-1).astype(np.int64)
    if times.size != labels.size or np.any(np.diff(times) < 0):
        raise RuntimeError("KS2 normalized spike arrays are inconsistent")
    pooled = boundary_ratio(
        times, stride=stride, sampling_frequency=sampling_frequency
    )
    groups: dict[str, np.ndarray] = {"all": np.ones(times.size, dtype=bool)}

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    if unique_labels.size:
        median_count = float(np.median(label_counts))
        high_units = unique_labels[label_counts >= median_count]
        groups["unit_rate_high"] = np.isin(labels, high_units)
        groups["unit_rate_low"] = ~groups["unit_rate_high"]

    ks_labels: dict[int, str] = {}
    label_path = output_dir / "cluster_KSLabel.tsv"
    if label_path.exists():
        with label_path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                ks_labels[int(row["cluster_id"])] = row["KSLabel"]
        for name in ("good", "mua"):
            groups[f"ks_label_{name}"] = np.asarray(
                [ks_labels.get(int(value)) == name for value in labels], dtype=bool
            )

    templates = np.load(output_dir / "templates.npy", mmap_mode="r")
    positions = np.load(output_dir / "channel_positions.npy")
    spike_templates = np.load(output_dir / "spike_templates.npy").reshape(-1).astype(int)
    template_ptp = np.ptp(np.asarray(templates), axis=1)
    template_depth = positions[np.argmax(template_ptp, axis=1), 1]
    cluster_depth = {
        int(unit): float(np.median(template_depth[spike_templates[labels == unit]]))
        for unit in unique_labels
    }
    spike_depth = np.asarray([cluster_depth[int(value)] for value in labels])
    if np.unique(spike_depth).size >= 4:
        edges = np.quantile(spike_depth, [0.25, 0.5, 0.75])
        depth_group = np.digitize(spike_depth, edges, right=True)
        for quartile in range(4):
            groups[f"depth_quartile_{quartile + 1}"] = depth_group == quartile

    subgroup_rows = []
    for name, mask in groups.items():
        if mask.sum() == 0:
            continue
        result = boundary_ratio(
            times[mask], stride=stride, sampling_frequency=sampling_frequency
        )
        result["group"] = name
        result["major_subgroup"] = bool(mask.sum() >= max(100, 0.05 * times.size))
        result["major_gate_pass"] = bool(
            not (
                result["major_subgroup"]
                and result["boundary_ratio"] < 0.95
                and result["pseudo_boundary_percentile"] < 0.01
            )
        )
        subgroup_rows.append(result)

    bin_samples = int(round(0.5e-3 * sampling_frequency))
    edges = np.arange(0, stride + bin_samples, bin_samples, dtype=np.int64)
    edges[-1] = stride
    histogram_rows = []
    for name, mask in groups.items():
        if mask.sum() == 0:
            continue
        counts, _ = np.histogram(times[mask] % stride, bins=edges)
        for index, count in enumerate(counts):
            histogram_rows.append(
                {
                    "group": name,
                    "phase_start_sample": int(edges[index]),
                    "phase_end_sample": int(edges[index + 1]),
                    "phase_center_ms": float(
                        (edges[index] + edges[index + 1])
                        / 2
                        / sampling_frequency
                        * 1000
                    ),
                    "count": int(count),
                }
            )
    with (output_dir / "batch_phase_histogram_0p5ms.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(histogram_rows[0]))
        writer.writeheader()
        writer.writerows(histogram_rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        all_rows = [row for row in histogram_rows if row["group"] == "all"]
        figure, axis = plt.subplots(figsize=(9, 3.5))
        axis.plot(
            [row["phase_center_ms"] for row in all_rows],
            [row["count"] for row in all_rows],
            linewidth=1,
        )
        half_width = int(round(3.5e-3 * sampling_frequency))
        axis.axvspan(0, half_width / sampling_frequency * 1000, alpha=0.2, color="tab:red")
        axis.axvspan(
            (stride - half_width) / sampling_frequency * 1000,
            stride / sampling_frequency * 1000,
            alpha=0.2,
            color="tab:red",
        )
        axis.set(xlabel="KS2 batch phase (ms)", ylabel="spikes / 0.5 ms bin")
        figure.tight_layout()
        figure.savefig(output_dir / "batch_phase_histogram_0p5ms.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass

    unit_rows = []
    for unit in np.unique(labels):
        selected = times[labels == unit]
        if selected.size >= 20:
            unit_rows.append(
                {
                    "unit_id": int(unit),
                    **boundary_ratio(
                        selected,
                        stride=stride,
                        sampling_frequency=sampling_frequency,
                    ),
                }
            )
    audit = {
        "pooled": pooled,
        "subgroups": subgroup_rows,
        "units_ge_20_spikes": unit_rows,
        "overall_gate_pass": bool(
            pooled["pooled_gate_pass"]
            and all(row["major_gate_pass"] for row in subgroup_rows)
        ),
        "phase_bin_width_samples": bin_samples,
        "phase_bin_width_ms": bin_samples / sampling_frequency * 1000,
        "sampling_frequency_hz": sampling_frequency,
    }
    (output_dir / "batch_phase_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def output_hashes(output_dir: Path) -> dict[str, str]:
    names = (
        "spike_times.npy",
        "spike_clusters.npy",
        "spike_templates.npy",
        "templates.npy",
        "channel_map.npy",
        "whitening_mat.npy",
    )
    return {name: sha256_file(output_dir / name) for name in names}


def execute(
    recording,
    output_dir: Path,
    kilosort_dir: Path,
    *,
    fixture: bool,
    context: dict[str, Any],
    nt: int = NT,
):
    installation = verify_installation(kilosort_dir)
    partial = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial.exists():
        raise RuntimeError(f"Refusing to overwrite existing output: {output_dir} or {partial}")
    prepared = prepare_native_input(recording, partial, fixture, nt=nt)
    prepared_binary = Path(prepared["binary_path"])
    try:
        relative_binary = prepared_binary.relative_to(partial.resolve())
    except ValueError:
        pass
    else:
        prepared["binary_path"] = str((output_dir.resolve() / relative_binary))
    request = {
        "schema_version": "luke-ks2-native-tracking-v1",
        "status": "installation_fixture" if fixture else "bounded_smoke",
        "raw_voltage_warp": False,
        "ks2_installation": installation,
        "executed_batching": {
            "NT": nt,
            "ntbuff": NTBUFF,
            "stride": nt - NTBUFF,
        },
        "input": prepared,
        "context": context,
    }
    (partial / "run_request.json").write_text(json.dumps(request, indent=2, default=str) + "\n")
    runtime = run_matlab(partial, kilosort_dir)
    audit = audit_output(
        partial,
        prepared["sampling_frequency_hz"],
        stride=nt - NTBUFF,
    )
    receipt = {
        **request,
        "runtime": runtime,
        "batch_phase_audit": audit,
        "output_sha256": output_hashes(partial),
        "complete": True,
    }
    (partial / "run_manifest.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    os.replace(partial, output_dir)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kilosort-dir", type=Path, default=DEFAULT_KS2)
    parser.add_argument("--rescue-root", type=Path, default=DEFAULT_RESCUE_ROOT)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start-s", type=float, default=5910.0)
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument(
        "--nt",
        type=int,
        choices=(NT, ALIGNED_NT),
        default=NT,
        help="Installation audit value; scientific runs require a separately accepted gate.",
    )
    args = parser.parse_args()
    if args.fixture == args.smoke:
        raise SystemExit("Choose exactly one of --fixture or --smoke")
    if args.fixture:
        output = args.output_dir or DEFAULT_FIXTURE
        recording, truth = make_fixture_recording(stride=args.nt - NTBUFF)
        receipt = execute(
            recording,
            output,
            args.kilosort_dir,
            fixture=True,
            context={"seed": 2102, "synthetic_truth_spike_count": int(truth.size)},
            nt=args.nt,
        )
    else:
        output = args.output_dir or (
            args.rescue_root
            / "sorter_bakeoff/windows/rapid_motion-8b4978262d/ks2_native"
        )
        recording = load_luke_window(args.rescue_root, args.start_s, args.duration_s)
        receipt = execute(
            recording,
            output,
            args.kilosort_dir,
            fixture=False,
            context={"start_s": args.start_s, "duration_s": args.duration_s},
            nt=args.nt,
        )
    print(json.dumps(receipt, indent=2, default=str))


if __name__ == "__main__":
    main()
