#!/usr/bin/env python3
"""Finite DG source comparison and Kilosort staged-adapter fixture.

This does not read voltage, detect peaks, fit a model, or launch a sort.  It
checks the published source and demonstrates the required native operation
ordering on a deterministic synthetic array.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch

from kilosort.io import BinaryFiltered
from kilosort.preprocessing import fft_highpass, get_highpass_filter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "testing/outputs/dd_dg_source_and_ks_adapter_review_20260928"
T8 = Path("/media/huklab/Data/NPX/Ryansorting/Luke/incoming/hub_analysis_v1/t8_episode_shift.py")
T8B = Path("/media/huklab/Data/NPX/Ryansorting/Luke/incoming/hub_analysis_v1/t8b_null.py")
H5 = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/dd_lattice_w2_20260928/huklaban5_v2")
H5_GATE = H5 / "source/dd_voltage_gate.py"
H5_INPUTS = H5 / "source/dd_lattice_inputs.py"
H5_WORKER = H5 / "source/dd_sort_worker.py"
KS_IO = Path(inspect.getsourcefile(BinaryFiltered))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()


def highpass_native(x: torch.Tensor, hp: torch.Tensor) -> torch.Tensor:
    """The temporal part of BinaryFiltered.filter, before CAR/whitening."""
    centered = x - x.mean(1, keepdim=True)
    fwav = fft_highpass(hp, NT=centered.shape[1])
    return torch.fft.fftshift(
        torch.real(torch.fft.ifft(torch.fft.fft(centered) * torch.conj(fwav))), dim=-1
    )


def remap_timewise(x: torch.Tensor, maps: dict[int, np.ndarray], q: np.ndarray) -> torch.Tensor:
    out = torch.zeros((len(next(iter(maps.values()))), x.shape[1]), dtype=x.dtype)
    for value, mapping in maps.items():
        columns = np.flatnonzero(q == value)
        valid = np.flatnonzero(mapping >= 0)
        if columns.size and valid.size:
            out[np.ix_(valid, columns)] = x[np.ix_(mapping[valid], columns)]
    return out


def downstream_native_policy(x: torch.Tensor, whiten: torch.Tensor) -> torch.Tensor:
    """Stock post-load mean/CAR/whitening policy, with no second high-pass."""
    x = x - x.mean(1, keepdim=True)
    x = x - torch.median(x, dim=0).values
    return whiten @ x


def main() -> None:
    started = time.process_time()
    OUT.mkdir(parents=True, exist_ok=False)

    # Deterministic full-parent synthetic batch and a changing lattice map.
    generator = torch.Generator(device="cpu").manual_seed(20260928)
    samples = 30122
    parent = torch.randn((8, samples), generator=generator, dtype=torch.float32)
    parent += torch.linspace(-3, 3, samples)[None, :] * torch.linspace(0.2, 1.0, 8)[:, None]
    hp = get_highpass_filter(fs=30_000, cutoff=300, device=torch.device("cpu"))
    maps = {
        0: np.array([1, 2, 3, 4, 5, 6], dtype=np.int64),
        20: np.array([2, 3, 4, 5, 6, 7], dtype=np.int64),
        -20: np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
    }
    q = np.zeros(samples, dtype=np.int16)
    q[samples // 3 : 2 * samples // 3] = 20
    q[2 * samples // 3 :] = -20
    whiten = torch.eye(6, dtype=torch.float32)
    whiten += 0.01 * torch.tril(torch.ones((6, 6), dtype=torch.float32), diagonal=-1)

    # Explicit desired order and an independently spelled equivalent path.
    temporal = highpass_native(parent, hp)
    remapped = remap_timewise(temporal, maps, q)
    desired = downstream_native_policy(remapped, whiten)

    centered = parent - parent.mean(1, keepdim=True)
    fwav = fft_highpass(hp, NT=samples)
    independently_temporal = torch.fft.fftshift(
        torch.fft.ifft(torch.fft.fft(centered) * torch.conj(fwav)).real, dim=-1
    )
    independently_remapped = remap_timewise(independently_temporal, maps, q)
    independent = whiten @ (
        independently_remapped
        - independently_remapped.mean(1, keepdim=True)
        - torch.median(
            independently_remapped - independently_remapped.mean(1, keepdim=True), dim=0
        ).values
    )

    # A hand-written route which omits stock BinaryFiltered's downstream
    # per-channel mean is not identical.
    no_second_mean = whiten @ (remapped - torch.median(remapped, dim=0).values)
    fixture = {
        "status": "pass" if torch.equal(desired, independent) else "fail",
        "shape_channels_by_samples": list(desired.shape),
        "desired_sha256": tensor_sha(desired),
        "independent_sha256": tensor_sha(independent),
        "bitwise_equal": bool(torch.equal(desired, independent)),
        "max_abs_error": float(torch.max(torch.abs(desired - independent))),
        "handwritten_path_max_abs_error_when_downstream_mean_is_omitted": float(
            torch.max(torch.abs(desired - no_second_mean))
        ),
        "operation_order": [
            "per-channel temporal mean subtraction on full parent",
            "native Kilosort FFT high-pass",
            "time-varying exact-coordinate remap and crop",
            "per-channel temporal mean subtraction on remapped crop",
            "median CAR on remapped crop",
            "whitening matrix multiplication",
        ],
        "conclusion": (
            "The array transformation is expressible with installed native functions, but the stock "
            "SpikeInterface wrapper cannot inject the remap between high-pass and CAR/whitening. "
            "A frozen custom BinaryFiltered/staged runner is required. skip_kilosort_preprocessing "
            "alone is not equivalent because it supplies neither high-pass nor adaptive whitening; "
            "the desired downstream mean/CAR must remain active after remapping."
        ),
        "prototype_resource_measurement": {
            "cpu_s_including_source_review": time.process_time() - started,
            "synthetic_parent_bytes": int(parent.nelement() * parent.element_size()),
            "synthetic_output_bytes": int(desired.nelement() * desired.element_size()),
            "projected_parent_batch_bytes_383x60122_float32": 383 * 60122 * 4,
            "projected_crop_batch_bytes_182x60122_float32": 182 * 60122 * 4,
            "extra_full_window_materialization_required": False,
        },
    }
    (OUT / "KS_NATIVE_STAGE_FIXTURE.json").write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")

    gate = json.loads((H5 / "gate_voltage_v1/GATE.json").read_text())
    comparison = {
        "status": "complete",
        "original_dd_gate_status_preserved": gate["status"],
        "no_new_detection_null_or_voltage_measurement": True,
        "sources": {
            "original_t8": {"path": str(T8), "sha256": sha256(T8)},
            "original_t8b": {"path": str(T8B), "sha256": sha256(T8B)},
            "h5_actual_gate": {"path": str(H5_GATE), "sha256": sha256(H5_GATE)},
            "h5_mapping": {"path": str(H5_INPUTS), "sha256": sha256(H5_INPUTS)},
            "h5_worker": {"path": str(H5_WORKER), "sha256": sha256(H5_WORKER)},
            "kilosort_binary_filtered": {"path": str(KS_IO), "sha256": sha256(KS_IO)},
        },
        "finite_source_differences": [
            {
                "topic": "map support",
                "original": "caller zlim; x edges -8..64 inclusive (T8 lines 42-44)",
                "h5": "fixed depth edges 0..3800 and x edges 8..48 (dd_voltage_gate lines 145-151)",
                "impact": "H5 omits physical x=0 and y=3820 sites for the 2020..3820 crop while adding empty depth support below it",
            },
            {
                "topic": "baseline exclusion",
                "original": "other field-candidate core bins excluded from 4..1 s baseline (T8 lines 65-70)",
                "h5": "no other-candidate exclusion in measure (dd_voltage_gate lines 174-183)",
                "impact": "catalogue agreement is not an identical T8 reproduction",
            },
            {
                "topic": "peak floor",
                "original": "150 peaks in both maps (T8 lines 60,71)",
                "h5": "100 peaks in both maps (dd_voltage_gate lines 174-183)",
                "impact": "resolved denominators differ",
            },
            {
                "topic": "quiet null",
                "original": "random field-quiet placements with episode-count thinning (T8b lines 16-43)",
                "h5": "deterministic rate-nearest candidates outside accepted episodes, without matched thinning (dd_voltage_gate lines 190-209)",
                "impact": "the 198/200 H5 null is informative for that diagnostic, not the original T8b null",
            },
            {
                "topic": "detector/crop",
                "original": "pilot raw NPX peaks selected by caller depth range",
                "h5": "new frozen detector on each 182-site materialized voltage",
                "impact": "population and support differ by design",
            },
        ],
        "h5_input_integrity": {
            "q0_sha256": gate["source_byte_assertions"]["S_0"]["binary_sha256"],
            "remap_sha256": gate["source_byte_assertions"]["S_L"]["binary_sha256"],
            "frames_each": gate["source_byte_assertions"]["S_0"]["frames"],
            "channels_each": gate["source_byte_assertions"]["S_0"]["channels"],
            "interpretation": "valid paired materialization; failed catalogue gate is preserved and not relabelled",
        },
        "cpu_s": time.process_time() - started,
    }
    (OUT / "SOURCE_COMPARISON.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    (OUT / "README.md").write_text(
        "# DG H1 finite review\n\n"
        "The original DD actual-voltage gate remains **failed**. Its 6/10 q0 and 9/10 remap "
        "catalogue results are not an identical implementation of original T8 because map support, "
        "baseline exclusion, peak floor, null construction, detector, and crop differ. This limits "
        "interpretation but does not show corrupt paired inputs.\n\n"
        "The native Kilosort fixture proves the required numerical order can be expressed from installed "
        "functions. The stock SpikeInterface wrapper cannot insert a time-varying remap between high-pass "
        "and CAR/whitening; a separately frozen custom `BinaryFiltered`/staged runner is required before a "
        "full Kilosort pair is scientifically interpretable. No sort is authorized or launched here.\n"
    )
    if fixture["status"] != "pass":
        raise RuntimeError("native-stage fixture failed")


if __name__ == "__main__":
    main()
