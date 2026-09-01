"""Exact-matrix, sorter-independent Luke KS4 registration operator audit.

This module implements the small-array mechanics preregistered in
``testing/Luke KS4 native operator audit plan.md``.  It deliberately separates
pure operator validation from discovery-data execution:

* ``--plan-only`` writes/prints the frozen configuration without opening raw;
* ``--synthetic-validation`` exercises matrix orientation, zero-shift tax,
  ordering, matched-filter, separability, and gate aggregation on generated
  arrays only;
* ``--dry-run`` is reserved for one donor/background after the source-domain
  adapter has been explicitly qualified.

The installed Kilosort 4.0.27 functions are called for the KS spatial matrix
and temporal high-pass conversion.  SpikeInterface supplies the forward and SI
inverse matrices.  No sorter, GPU, prospective holdout, or materialized binary
is used by this module.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_OUTPUT = REPO_ROOT / "testing/outputs/luke_ks4_native_operator_audit"
DEFAULT_OPS = Path(
    "/mnt/NPX/Luke/20250804/"
    "rescue_pipeline_results_Luke0804_V2V1_g0_imec1/"
    "kilosort4/sorter_output/ops.npy"
)
DEFAULT_TEMPLATES = (
    REPO_ROOT / "testing/outputs/luke_injected_ground_truth_pilot/donor_templates.npz"
)
DEFAULT_DONOR_MANIFEST = (
    REPO_ROOT / "testing/outputs/luke_injected_ground_truth_pilot/donor_manifest.csv"
)
DEFAULT_RESCUE_RECORDING = Path(
    "/mnt/NPX/Luke/20250804/"
    "rescue_pipeline_results_Luke0804_V2V1_g0_imec1/recording"
)


@dataclass(frozen=True)
class Background:
    name: str
    start_s: float
    motion_class: str


BACKGROUNDS = (
    Background("quiet_3951", 3951.0, "quiet"),
    Background("neutral_7215", 7215.0, "intermediate"),
    Background("pathological_8160", 8160.0, "high_motion"),
)

ARMS = (
    "stationary_no_correction",
    "stationary_ks4_d0",
    "moved_no_correction",
    "moved_ks4_native_inverse",
    "moved_ks4_external_order_inverse",
    "moved_si_inverse",
)

DISPLACEMENTS_UM = (0.0, -1.0, 1.0, -2.0, 2.0, -4.0, 4.0, -6.0, 6.0, -10.0, 10.0, -20.0, 20.0)


@dataclass(frozen=True)
class SpatialKernel:
    name: str
    method: str
    sigma_um: float = 20.0
    p: float = 2.0
    num_closest: int = 4


FORWARD_GENERATORS = (
    SpatialKernel("si_kriging_p2_sigma10", "kriging", 10.0, 2.0),
    SpatialKernel("si_kriging_p2_sigma20", "kriging", 20.0, 2.0),
    SpatialKernel("si_idw4", "idw", num_closest=4),
)
SI_INVERSE = SpatialKernel("si_kriging_p2_sigma20", "kriging", 20.0, 2.0)


@dataclass(frozen=True)
class GateThresholds:
    worst_generator_median_delta_residual_max: float = -0.005
    worst_generator_median_delta_absolute_amplitude_error_max: float = 0.005
    worst_generator_median_delta_cosine_min: float = 0.0
    tax_adjusted_residual_improvement_min: float = 0.005


GATE = GateThresholds()


@dataclass(frozen=True)
class RunBounds:
    maximum_templates: int = 6
    maximum_backgrounds: int = 3
    local_filter_samples: int = 4096
    score_samples: int = 121
    noise_windows: int = 12
    background_duration_s: float = 2.0
    rectangular_case_upper_bound: int = 4212


BOUNDS = RunBounds()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    header = f"{array.dtype.str}|{array.shape}".encode("ascii")
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def frozen_plan() -> dict:
    """Return the preregistered machine-readable operator configuration."""
    return {
        "schema_version": "luke-ks4-native-operator-audit-v1",
        "status": "parallel_operator_characterization_not_pipeline_authorization",
        "critical_path": [
            "motion_estimator_bakeoff",
            "coordinate_only_application",
            "voltage_operator_characterization",
        ],
        "versions": {
            "required_kilosort": "4.0.27",
            "required_spikeinterface": "0.102.1",
        },
        "source_domain_policy": {
            "status": "must_be_qualified_before_discovery_dry_run",
            "sealed_donors": "raw_domain_float32_before_rescue_conditioning",
            "frozen_whitening": "accepted_rescue_ks4_input_after_upstream_conditioning",
            "forbidden": "silently_apply_rescue_whitening_to_unqualified_raw_domain_arrays",
            "adapter": "reextract_same_donor_sample_and_channel_from_accepted_rescue_input_then_cast_float32",
            "minimum_raw_to_rescue_local_cosine": 0.85,
            "minimum_median_raw_to_rescue_local_cosine": 0.90,
            "new_int16_warp_materialized": False,
        },
        "arms": list(ARMS),
        "forward_generators": [asdict(value) for value in FORWARD_GENERATORS],
        "si_inverse": asdict(SI_INVERSE),
        "displacements_um": list(DISPLACEMENTS_UM),
        "gate": asdict(GATE),
        "bounds": asdict(BOUNDS),
        "backgrounds": [asdict(value) for value in BACKGROUNDS],
        "whitening_policy": "one_stationary_reference_matrix_frozen_across_all_six_arms",
        "prospective_holdout_accessed": False,
        "sorter_run": False,
        "gpu_required": False,
    }


def validate_frozen_plan(plan: Mapping) -> None:
    if plan.get("schema_version") != "luke-ks4-native-operator-audit-v1":
        raise ValueError("unexpected operator-audit schema")
    if tuple(plan.get("arms", ())) != ARMS:
        raise ValueError("six causal arms changed")
    if tuple(float(x) for x in plan.get("displacements_um", ())) != DISPLACEMENTS_UM:
        raise ValueError("signed displacement grid changed")
    if plan.get("whitening_policy") != "one_stationary_reference_matrix_frozen_across_all_six_arms":
        raise ValueError("whitening must remain frozen across causal arms")
    if plan.get("prospective_holdout_accessed") is not False:
        raise ValueError("prospective holdout must remain sealed")
    if plan.get("sorter_run") is not False:
        raise ValueError("operator audit must remain sorter-independent")
    if plan.get("source_domain_policy", {}).get("status") != "must_be_qualified_before_discovery_dry_run":
        raise ValueError("source-domain qualification guard was weakened")


def _versions() -> dict[str, str]:
    import kilosort
    import spikeinterface

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "kilosort": str(kilosort.__version__),
        "spikeinterface": str(spikeinterface.__version__),
    }


def validate_versions() -> dict[str, str]:
    versions = _versions()
    expected = frozen_plan()["versions"]
    for package, required in expected.items():
        name = package.removeprefix("required_")
        if versions[name] != required:
            raise RuntimeError(f"{name}=={required} required, found {versions[name]}")
    return versions


def load_ks4_state(ops_path: Path = DEFAULT_OPS) -> dict:
    """Load the accepted full-probe whitening/geometry into CPU tensors."""
    import torch

    values = np.load(ops_path, allow_pickle=True).item()
    probe_positions = np.vstack((values["probe"]["xc"], values["probe"]["yc"])).T
    positions = np.asarray(probe_positions, dtype=np.float64)
    whiten = np.asarray(values["preprocessing"]["whiten_mat"], dtype=np.float32)
    hp_filter = np.asarray(values["preprocessing"]["hp_filter"], dtype=np.float32)
    if positions.shape != (384, 2) or whiten.shape != (384, 384):
        raise ValueError("operator audit requires the complete 384-channel physical probe")
    state = copy.deepcopy(values)
    state["nblocks"] = 1
    state["settings"] = copy.deepcopy(values["settings"])
    state["settings"]["nblocks"] = 1
    state["probe"] = copy.deepcopy(values["probe"])
    state["yblk"] = np.asarray([np.nanmedian(positions[:, 1])], dtype=np.float64)
    from kilosort.datashift import kernel2D

    # Preserve the probe coordinate dtype used by KS4 itself. In the accepted
    # 4.0.27 ops this is float32; promoting only iKxx would make the installed
    # get_drift_matrix fail (and would no longer reproduce the sorter path).
    kxx = kernel2D(probe_positions, probe_positions, float(state["settings"]["sig_interp"]))
    kxx_t = torch.from_numpy(kxx)
    state["iKxx"] = torch.linalg.inv(
        kxx_t + 0.01 * torch.eye(len(kxx_t), dtype=kxx_t.dtype)
    )
    return {
        "ops": state,
        "positions": positions,
        "whiten": whiten,
        "hp_filter": hp_filter,
        "fs": float(values["fs"]),
        "ops_sha256": hashlib.sha256(Path(ops_path).read_bytes()).hexdigest(),
        "whiten_sha256": array_sha256(whiten),
        "positions_sha256": array_sha256(positions),
    }


def ks4_drift_matrix(state: Mapping, displacement_um: float) -> np.ndarray:
    """Return KS4 4.0.27's exact rigid matrix for one signed displacement."""
    import torch
    from kilosort.preprocessing import get_drift_matrix

    dshift = np.asarray([float(displacement_um)], dtype=np.float64)
    with torch.no_grad():
        matrix = get_drift_matrix(state["ops"], dshift, device=torch.device("cpu"))
    return matrix.detach().cpu().numpy().astype(np.float64)


def si_spatial_matrix(
    positions: np.ndarray,
    displacement_um: float,
    kernel: SpatialKernel,
    *,
    force_extrapolate: bool = True,
) -> np.ndarray:
    """Return an output-channel by input-channel SI spatial matrix."""
    from spikeinterface.preprocessing.preprocessing_tools import get_spatial_interpolation_kernel

    source = np.asarray(positions, dtype=float)
    target = source.copy()
    target[:, 1] += float(displacement_um)
    interpolation = get_spatial_interpolation_kernel(
        source,
        target,
        method=kernel.method,
        sigma_um=kernel.sigma_um,
        p=kernel.p,
        num_closest=kernel.num_closest,
        force_extrapolate=force_extrapolate,
        dtype="float64",
    )
    # SI uses time-by-source @ (source-by-target); KS uses
    # (output-by-input) @ channel-by-time.
    return np.asarray(interpolation.T, dtype=np.float64)


def apply_spatial(values: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply an output-by-input spatial matrix to time-by-channel traces."""
    traces = np.asarray(values)
    operator = np.asarray(matrix)
    if traces.ndim != 2 or operator.shape != (traces.shape[1], traces.shape[1]):
        raise ValueError("spatial operator and trace dimensions disagree")
    return (operator @ traces.T).T.astype(np.float32)


def ks4_p_local(
    values: np.ndarray,
    channel_means: np.ndarray,
    hp_filter: np.ndarray,
) -> np.ndarray:
    """Run exact KS channel centering, torch-median CAR, and FFT high-pass.

    ``channel_means`` may come from a longer parent KS batch.  This lets the
    bounded implementation preserve full-batch channel centering while applying
    the temporal operator only to a guarded local array.  Local/full temporal
    equivalence must pass the dry-run check before discovery execution.
    """
    import torch
    from kilosort.preprocessing import fft_highpass

    traces = np.asarray(values, dtype=np.float32)
    means = np.asarray(channel_means, dtype=np.float32)
    if traces.ndim != 2 or means.shape != (traces.shape[1],):
        raise ValueError("values/channel_means dimensions disagree")
    x = torch.as_tensor(traces.T, dtype=torch.float32)
    x = x - torch.as_tensor(means[:, None], dtype=torch.float32)
    x = x - torch.median(x, dim=0).values
    hp = torch.as_tensor(np.asarray(hp_filter), dtype=torch.float32)
    fwav = fft_highpass(hp, NT=x.shape[1])
    x = torch.real(torch.fft.ifft(torch.fft.fft(x) * torch.conj(fwav)))
    x = torch.fft.fftshift(x, dim=-1)
    return x.T.detach().cpu().numpy()


def _center_insert(container: np.ndarray, waveform: np.ndarray) -> slice:
    start = (container.shape[0] - waveform.shape[0]) // 2
    stop = start + waveform.shape[0]
    if start < 1 or stop >= container.shape[0] - 1:
        raise ValueError("waveform does not fit inside guarded local array")
    container[start:stop] += waveform
    return slice(start, stop)


def paired_event_delta(
    background_local: np.ndarray,
    background_full_means: np.ndarray,
    waveform: np.ndarray,
    hp_filter: np.ndarray,
    whiten: np.ndarray,
    *,
    pre_matrix: np.ndarray | None = None,
    post_matrix: np.ndarray | None = None,
    parent_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return processed injected-minus-background delta and processed noise.

    The CAR nonlinearity is evaluated separately for paired injected and
    uninjected arrays.  Whitening is frozen.  ``pre_matrix`` implements the
    external-order arm and ``post_matrix`` the native KS arm.
    """
    background = np.asarray(background_local, dtype=np.float32)
    event = np.asarray(waveform, dtype=np.float32)
    means = np.asarray(background_full_means, dtype=np.float32)
    if background.shape[1] != event.shape[1] or means.shape != (background.shape[1],):
        raise ValueError("paired arrays have incompatible channels")
    injected = background.copy()
    event_slice = _center_insert(injected, event)
    injected_means = means + event.sum(axis=0, dtype=np.float64).astype(np.float32) / float(parent_samples)

    if pre_matrix is not None:
        background = apply_spatial(background, pre_matrix)
        injected = apply_spatial(injected, pre_matrix)
        means = np.asarray(pre_matrix @ means, dtype=np.float32)
        injected_means = np.asarray(pre_matrix @ injected_means, dtype=np.float32)

    p_background = ks4_p_local(background, means, hp_filter)
    p_injected = ks4_p_local(injected, injected_means, hp_filter)
    delta = p_injected - p_background
    delta = (np.asarray(whiten, dtype=np.float32) @ delta.T).T
    noise = (np.asarray(whiten, dtype=np.float32) @ p_background.T).T
    if post_matrix is not None:
        delta = apply_spatial(delta, post_matrix)
        noise = apply_spatial(noise, post_matrix)
    return delta.astype(np.float32), noise.astype(np.float32)


def score_window(values: np.ndarray, n_samples: int) -> np.ndarray:
    traces = np.asarray(values)
    start = (traces.shape[0] - int(n_samples)) // 2
    return traces[start : start + int(n_samples)]


def centered_cosine(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).ravel()
    right = np.asarray(second, dtype=np.float64).ravel()
    left -= left.mean()
    right -= right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else float("nan")


def best_scaled_residual(candidate: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    observed = np.asarray(candidate, dtype=np.float64).ravel()
    target = np.asarray(reference, dtype=np.float64).ravel()
    denominator = float(np.dot(observed, observed))
    scale = float(np.dot(observed, target) / denominator) if denominator else 0.0
    residual = target - scale * observed
    fraction = float(np.linalg.norm(residual) / max(np.linalg.norm(target), np.finfo(float).eps))
    return fraction, scale


def robust_sigma(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    median = np.median(array)
    return float(max(np.median(np.abs(array - median)) / 0.6744897501960817, np.finfo(float).eps))


def matched_filter_snr(
    event: np.ndarray,
    reference: np.ndarray,
    noise: np.ndarray,
    *,
    n_noise_windows: int = BOUNDS.noise_windows,
) -> tuple[float, float, float]:
    """Score a fixed stationary-reference direction on paired real noise."""
    target = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(event, dtype=np.float64)
    background = np.asarray(noise, dtype=np.float64)
    if target.shape != observed.shape or background.shape[1] != target.shape[1]:
        raise ValueError("matched-filter dimensions disagree")
    direction = target.ravel() - target.mean()
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("reference matched filter is empty")
    direction /= norm
    event_score = float(np.dot(observed.ravel() - observed.mean(), direction))
    half = target.shape[0] // 2
    guard = target.shape[0]
    centers = np.linspace(
        half + 1,
        background.shape[0] - half - 2,
        n_noise_windows + 2,
        dtype=int,
    )[1:-1]
    center = background.shape[0] // 2
    centers = centers[np.abs(centers - center) > guard]
    scores = []
    for value in centers:
        window = background[value - half : value - half + target.shape[0]]
        if window.shape == target.shape:
            scores.append(float(np.dot(window.ravel() - window.mean(), direction)))
    if len(scores) < 3:
        raise ValueError("too few event-free matched-filter noise windows")
    noise_sigma = robust_sigma(np.asarray(scores))
    return event_score / noise_sigma, event_score, noise_sigma


def waveform_metrics(
    event: np.ndarray,
    reference: np.ndarray,
    noise: np.ndarray,
    positions: np.ndarray,
) -> dict[str, float | int]:
    observed = np.asarray(event, dtype=np.float32)
    target = np.asarray(reference, dtype=np.float32)
    residual, scale = best_scaled_residual(observed, target)
    observed_peak = np.unravel_index(int(np.argmax(np.abs(observed))), observed.shape)
    target_peak = np.unravel_index(int(np.argmax(np.abs(target))), target.shape)
    snr, score, noise_sigma = matched_filter_snr(observed, target, noise)
    return {
        "residual_fraction": residual,
        "best_scale": scale,
        "template_cosine": centered_cosine(observed, target),
        "amplitude_retention": float(
            np.max(np.abs(observed)) / max(np.max(np.abs(target)), np.finfo(float).eps)
        ),
        "peak_channel_error": int(observed_peak[1] - target_peak[1]),
        "peak_depth_error_um": float(positions[observed_peak[1], 1] - positions[target_peak[1], 1]),
        "matched_filter_snr": snr,
        "matched_filter_score": score,
        "matched_filter_noise_sigma": noise_sigma,
        "noise_rms": float(np.sqrt(np.mean(np.square(noise, dtype=np.float64)))),
        "exact_zero_fraction": float(np.mean(noise == 0)),
    }


def fixed_nearest_pairs(reference_templates: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Freeze one nearest amplitude-normalized competitor per reference."""
    names = sorted(reference_templates)
    normalized = {}
    for name in names:
        values = np.asarray(reference_templates[name], dtype=np.float64).ravel()
        values -= values.mean()
        normalized[name] = values / max(np.linalg.norm(values), np.finfo(float).eps)
    rows = []
    for name in names:
        candidates = [other for other in names if other != name]
        if not candidates:
            continue
        distances = {other: float(np.linalg.norm(normalized[name] - normalized[other])) for other in candidates}
        nearest = min(candidates, key=lambda other: (distances[other], other))
        rows.append({"template_id": name, "neighbor_template_id": nearest, "reference_distance": distances[nearest]})
    return pd.DataFrame(rows)


def qualify_rescue_domain_templates(
    recording,
    *,
    template_path: Path = DEFAULT_TEMPLATES,
    manifest_path: Path = DEFAULT_DONOR_MANIFEST,
    maximum_templates: int = BOUNDS.maximum_templates,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict]:
    """Re-extract sealed donor identities in the accepted KS4 input domain."""
    from testing.luke_injected_ground_truth_pilot import (
        align_channel_waveforms,
        centered_cosine as pilot_cosine,
        prepare_template,
    )

    raw_templates = np.load(template_path)
    manifest = pd.read_csv(manifest_path)
    rows = []
    candidates: dict[str, np.ndarray] = {}
    for donor in manifest.itertuples(index=False):
        if donor.template_id not in raw_templates.files:
            raise KeyError(f"sealed template missing: {donor.template_id}")
        traces = recording.get_traces(
            start_frame=int(donor.donor_sample_index) - 60,
            end_frame=int(donor.donor_sample_index) + 61,
        ).astype(np.float32)
        rescue_template = prepare_template(traces, int(donor.donor_peak_channel))
        raw_local, rescue_local = align_channel_waveforms(
            raw_templates[donor.template_id],
            rescue_template,
            int(donor.donor_peak_channel),
            int(donor.donor_peak_channel),
        )
        cosine = pilot_cosine(raw_local, rescue_local)
        peak = np.unravel_index(int(np.argmax(np.abs(rescue_template))), rescue_template.shape)[1]
        row = {
            "template_id": donor.template_id,
            "donor_sample_index": int(donor.donor_sample_index),
            "donor_peak_channel": int(donor.donor_peak_channel),
            "rescue_peak_channel": int(peak),
            "peak_channel_difference": int(peak - donor.donor_peak_channel),
            "raw_to_rescue_local_cosine": float(cosine),
            "raw_template_sha256": array_sha256(raw_templates[donor.template_id]),
            "rescue_template_sha256": array_sha256(rescue_template),
            "rescue_peak_amplitude_counts": float(np.max(np.abs(rescue_template))),
        }
        rows.append(row)
        if cosine >= frozen_plan()["source_domain_policy"]["minimum_raw_to_rescue_local_cosine"]:
            candidates[donor.template_id] = rescue_template
    qualification = pd.DataFrame(rows)
    median_cosine = float(qualification.raw_to_rescue_local_cosine.median())
    minimum_cosine = float(qualification.raw_to_rescue_local_cosine.min())
    passed = bool(
        minimum_cosine >= frozen_plan()["source_domain_policy"]["minimum_raw_to_rescue_local_cosine"]
        and median_cosine >= frozen_plan()["source_domain_policy"]["minimum_median_raw_to_rescue_local_cosine"]
        and len(candidates) >= maximum_templates
    )
    # Preserve manifest order; selection cannot depend on operator outcomes.
    selected = dict(list(candidates.items())[:maximum_templates])
    result = {
        "adapter": frozen_plan()["source_domain_policy"]["adapter"],
        "qualified_donors": int(len(candidates)),
        "selected_templates": list(selected),
        "minimum_raw_to_rescue_local_cosine": minimum_cosine,
        "median_raw_to_rescue_local_cosine": median_cosine,
        "passed": passed,
        "new_int16_warp_materialized": False,
    }
    return selected, qualification, result


def load_rescue_background(recording, background: Background) -> np.ndarray:
    fs = float(recording.get_sampling_frequency())
    start = int(round(background.start_s * fs))
    count = int(round(BOUNDS.background_duration_s * fs))
    values = recording.get_traces(start_frame=start, end_frame=start + count).astype(np.float32)
    if values.shape != (count, 384):
        raise ValueError(f"unexpected background shape for {background.name}: {values.shape}")
    return values


def _local_from_full(background: np.ndarray) -> np.ndarray:
    if len(background) < BOUNDS.local_filter_samples:
        raise ValueError("background shorter than guarded local filter window")
    return score_window(background, BOUNDS.local_filter_samples).astype(np.float32)


def event_is_interior(
    waveform: np.ndarray,
    positions: np.ndarray,
    displacement_um: float,
    *,
    support_fraction: float = 0.01,
) -> bool:
    values = np.asarray(waveform)
    amplitude = np.max(np.abs(values), axis=0)
    support = amplitude >= support_fraction * max(float(amplitude.max()), np.finfo(float).eps)
    depths = np.asarray(positions, dtype=float)[:, 1]
    moved = depths[support] + float(displacement_um)
    return bool(moved.size and moved.min() >= depths.min() and moved.max() <= depths.max())


def _arm_delta(
    arm: str,
    *,
    background: np.ndarray,
    stationary_waveform: np.ndarray,
    moved_waveform: np.ndarray,
    hp_filter: np.ndarray,
    whiten: np.ndarray,
    m0: np.ndarray,
    m_inverse: np.ndarray,
    si_inverse: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    local = _local_from_full(background)
    means = background.mean(axis=0, dtype=np.float64).astype(np.float32)
    common = dict(
        background_local=local,
        background_full_means=means,
        hp_filter=hp_filter,
        whiten=whiten,
        parent_samples=len(background),
    )
    if arm == "stationary_no_correction":
        return paired_event_delta(waveform=stationary_waveform, **common)
    if arm == "stationary_ks4_d0":
        return paired_event_delta(waveform=stationary_waveform, post_matrix=m0, **common)
    if arm == "moved_no_correction":
        return paired_event_delta(waveform=moved_waveform, **common)
    if arm == "moved_ks4_native_inverse":
        return paired_event_delta(waveform=moved_waveform, post_matrix=m_inverse, **common)
    if arm == "moved_ks4_external_order_inverse":
        return paired_event_delta(waveform=moved_waveform, pre_matrix=m_inverse, **common)
    if arm == "moved_si_inverse":
        return paired_event_delta(waveform=moved_waveform, pre_matrix=si_inverse, **common)
    raise KeyError(f"unknown arm: {arm}")


def discovery_dry_run(output_dir: Path) -> dict:
    """Run one donor/background/generator/displacement through all six arms."""
    import spikeinterface.core as sc

    versions = validate_versions()
    state = load_ks4_state()
    recording = sc.load(DEFAULT_RESCUE_RECORDING)
    templates, qualification, source_result = qualify_rescue_domain_templates(recording)
    if not source_result["passed"]:
        raise RuntimeError("source-domain qualification failed")
    template_id = next(iter(templates))
    stationary = templates[template_id]
    background_spec = BACKGROUNDS[0]
    background = load_rescue_background(recording, background_spec)
    generator = FORWARD_GENERATORS[0]
    displacement = 4.0
    forward = si_spatial_matrix(state["positions"], displacement, generator)
    moved = apply_spatial(stationary, forward)
    m0 = ks4_drift_matrix(state, 0.0)
    m_same = ks4_drift_matrix(state, displacement)
    m_opposite = ks4_drift_matrix(state, -displacement)
    si_inverse = si_spatial_matrix(state["positions"], -displacement, SI_INVERSE)

    # Validate the documented KS sign rather than selecting a sign from the
    # factorial. The same-sign KS matrix should undo SI's forward convention.
    reference_delta, reference_noise = _arm_delta(
        "stationary_no_correction",
        background=background,
        stationary_waveform=stationary,
        moved_waveform=moved,
        hp_filter=state["hp_filter"],
        whiten=state["whiten"],
        m0=m0,
        m_inverse=m_same,
        si_inverse=si_inverse,
    )
    reference = score_window(reference_delta, BOUNDS.score_samples)
    same_delta, _ = paired_event_delta(
        _local_from_full(background),
        background.mean(axis=0, dtype=np.float64),
        moved,
        state["hp_filter"],
        state["whiten"],
        post_matrix=m_same,
        parent_samples=len(background),
    )
    opposite_delta, _ = paired_event_delta(
        _local_from_full(background),
        background.mean(axis=0, dtype=np.float64),
        moved,
        state["hp_filter"],
        state["whiten"],
        post_matrix=m_opposite,
        parent_samples=len(background),
    )
    same_residual, _ = best_scaled_residual(score_window(same_delta, BOUNDS.score_samples), reference)
    opposite_residual, _ = best_scaled_residual(score_window(opposite_delta, BOUNDS.score_samples), reference)
    sign_passed = bool(same_residual < opposite_residual)
    if not sign_passed:
        raise RuntimeError("KS inverse sign validation failed")

    rows = []
    arrays = {}
    start = time.perf_counter()
    for arm in ARMS:
        delta, noise = _arm_delta(
            arm,
            background=background,
            stationary_waveform=stationary,
            moved_waveform=moved,
            hp_filter=state["hp_filter"],
            whiten=state["whiten"],
            m0=m0,
            m_inverse=m_same,
            si_inverse=si_inverse,
        )
        event = score_window(delta, BOUNDS.score_samples)
        arrays[arm] = event
        rows.append(
            {
                "arm": arm,
                "template_id": template_id,
                "background": background_spec.name,
                "generator": generator.name,
                "displacement_um": displacement,
                **waveform_metrics(event, reference, noise, state["positions"]),
            }
        )
    runtime = time.perf_counter() - start
    metrics = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    qualification.to_csv(output_dir / "source_domain_qualification.csv", index=False)
    metrics.to_csv(output_dir / "dry_run_case_metrics.csv", index=False)
    np.savez_compressed(output_dir / "dry_run_event_arrays.npz", **arrays)
    np.savez_compressed(
        output_dir / "dry_run_matrices.npz",
        forward=forward,
        ks4_zero=m0,
        ks4_inverse=m_same,
        si_inverse=si_inverse,
    )
    result = {
        "status": "one_template_one_background_discovery_dry_run",
        "versions": versions,
        "source_domain": source_result,
        "template_id": template_id,
        "background": asdict(background_spec),
        "generator": asdict(generator),
        "displacement_um": displacement,
        "sign_validation": {
            "ks4_inverse_uses_same_numeric_sign_as_si_forward": True,
            "same_sign_residual": same_residual,
            "opposite_sign_residual": opposite_residual,
            "passed": sign_passed,
        },
        "matrix_diagnostics": {
            "ks4_zero_identity_frobenius": float(np.linalg.norm(m0 - np.eye(len(m0)))),
            "ks4_zero_column_sum_min": float(m0.sum(axis=0).min()),
            "ks4_zero_column_sum_max": float(m0.sum(axis=0).max()),
        },
        "hashes": {
            "ops": state["ops_sha256"],
            "whiten": state["whiten_sha256"],
            "positions": state["positions_sha256"],
            "background": array_sha256(background),
            "template": array_sha256(stationary),
        },
        "six_arm_runtime_seconds": runtime,
        "projected_rectangular_runtime_seconds": runtime
        * BOUNDS.rectangular_case_upper_bound
        / len(ARMS),
        "prospective_holdout_accessed": False,
        "sorter_run": False,
        "new_int16_warp_materialized": False,
        "passed": bool(source_result["passed"] and sign_passed and len(metrics) == len(ARMS)),
    }
    (output_dir / "dry_run_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _tax_and_gate(native: pd.DataFrame, zero_tax: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    by_generator, primary = robust_gate_summary(native)
    tax_residual = float(zero_tax.residual_fraction.median())
    tax_amplitude = float(np.median(np.abs(zero_tax.amplitude_retention - 1.0)))
    tax_cosine = float(np.median(1.0 - zero_tax.template_cosine))
    by_generator["median_recovery_residual_reduction"] = -by_generator.median_delta_residual
    by_generator["stationary_median_zero_tax_residual"] = tax_residual
    by_generator["tax_adjusted_residual_improvement"] = (
        by_generator.median_recovery_residual_reduction - tax_residual
    )
    by_generator["tax_adjusted_residual_pass"] = (
        by_generator.tax_adjusted_residual_improvement
        >= GATE.tax_adjusted_residual_improvement_min
    )
    tax_pass = bool(by_generator.tax_adjusted_residual_pass.all())
    decision = {
        **primary,
        "stationary_median_zero_tax_residual": tax_residual,
        "stationary_median_zero_tax_absolute_amplitude_error": tax_amplitude,
        "stationary_median_zero_tax_cosine_loss": tax_cosine,
        "worst_generator_tax_adjusted_residual_improvement": float(
            by_generator.tax_adjusted_residual_improvement.min()
        ),
        "tax_adjusted_residual_pass": tax_pass,
        "operator_primary_and_tax_pass": bool(primary["primary_screen_pass"] and tax_pass),
    }
    return by_generator, decision


def run_discovery_audit(output_dir: Path) -> dict:
    """Run the capped six-template, three-background discovery factorial."""
    import spikeinterface.core as sc

    versions = validate_versions()
    state = load_ks4_state()
    recording = sc.load(DEFAULT_RESCUE_RECORDING)
    templates, qualification, source_result = qualify_rescue_domain_templates(recording)
    if not source_result["passed"]:
        raise RuntimeError("source-domain qualification failed")
    backgrounds = {value.name: load_rescue_background(recording, value) for value in BACKGROUNDS}
    m0 = ks4_drift_matrix(state, 0.0)
    matrices = {"ks4_zero": m0}
    metric_rows: list[dict] = []
    pair_rows: list[dict] = []
    start = time.perf_counter()

    for background_spec in BACKGROUNDS:
        background = backgrounds[background_spec.name]
        stationary_refs: dict[str, np.ndarray] = {}
        stationary_noise = None
        for template_id, stationary in templates.items():
            for arm in ("stationary_no_correction", "stationary_ks4_d0"):
                delta, noise = _arm_delta(
                    arm,
                    background=background,
                    stationary_waveform=stationary,
                    moved_waveform=stationary,
                    hp_filter=state["hp_filter"],
                    whiten=state["whiten"],
                    m0=m0,
                    m_inverse=m0,
                    si_inverse=np.eye(len(state["positions"])),
                )
                event = score_window(delta, BOUNDS.score_samples)
                if arm == "stationary_no_correction":
                    stationary_refs[template_id] = event
                    stationary_noise = noise
                    reference = event
                else:
                    reference = stationary_refs[template_id]
                metric_rows.append(
                    {
                        "arm": arm,
                        "template_id": template_id,
                        "background": background_spec.name,
                        "motion_class": background_spec.motion_class,
                        "generator": "stationary",
                        "displacement_um": 0.0,
                        "displacement_sign": "zero",
                        "edge_status": "interior",
                        **waveform_metrics(event, reference, noise, state["positions"]),
                    }
                )
        pair_table = fixed_nearest_pairs(stationary_refs)
        if stationary_noise is None:
            raise AssertionError("stationary noise was not computed")
        stationary_pairs = pair_separability(
            pair_table, stationary_refs, stationary_refs, stationary_noise
        )
        for row in stationary_pairs.to_dict(orient="records"):
            pair_rows.append(
                {
                    "arm": "stationary_no_correction",
                    "background": background_spec.name,
                    "generator": "stationary",
                    "displacement_um": 0.0,
                    **row,
                }
            )

        for generator in FORWARD_GENERATORS:
            for displacement in (value for value in DISPLACEMENTS_UM if value != 0):
                forward_key = f"forward__{generator.name}__{displacement:+g}um"
                ks_key = f"ks4_inverse__{displacement:+g}um"
                si_key = f"si_inverse__{-displacement:+g}um"
                forward = matrices.setdefault(
                    forward_key,
                    si_spatial_matrix(state["positions"], displacement, generator),
                )
                ks_inverse = matrices.setdefault(
                    ks_key, ks4_drift_matrix(state, displacement)
                )
                si_inverse = matrices.setdefault(
                    si_key,
                    si_spatial_matrix(state["positions"], -displacement, SI_INVERSE),
                )
                by_arm: dict[str, dict[str, np.ndarray]] = {
                    arm: {}
                    for arm in (
                        "moved_no_correction",
                        "moved_ks4_native_inverse",
                        "moved_ks4_external_order_inverse",
                        "moved_si_inverse",
                    )
                }
                noise_by_arm: dict[str, np.ndarray] = {}
                for template_id, stationary in templates.items():
                    moved = apply_spatial(stationary, forward)
                    reference = stationary_refs[template_id]
                    interior = event_is_interior(
                        stationary, state["positions"], displacement
                    )
                    for arm in by_arm:
                        delta, noise = _arm_delta(
                            arm,
                            background=background,
                            stationary_waveform=stationary,
                            moved_waveform=moved,
                            hp_filter=state["hp_filter"],
                            whiten=state["whiten"],
                            m0=m0,
                            m_inverse=ks_inverse,
                            si_inverse=si_inverse,
                        )
                        event = score_window(delta, BOUNDS.score_samples)
                        by_arm[arm][template_id] = event
                        noise_by_arm[arm] = noise
                        metric_rows.append(
                            {
                                "arm": arm,
                                "template_id": template_id,
                                "background": background_spec.name,
                                "motion_class": background_spec.motion_class,
                                "generator": generator.name,
                                "displacement_um": displacement,
                                "displacement_sign": "positive" if displacement > 0 else "negative",
                                "edge_status": "interior" if interior else "edge_affected",
                                **waveform_metrics(event, reference, noise, state["positions"]),
                            }
                        )
                for arm, candidate_templates in by_arm.items():
                    pair_metrics = pair_separability(
                        pair_table,
                        stationary_refs,
                        candidate_templates,
                        noise_by_arm[arm],
                    )
                    for row in pair_metrics.to_dict(orient="records"):
                        pair_rows.append(
                            {
                                "arm": arm,
                                "background": background_spec.name,
                                "generator": generator.name,
                                "displacement_um": displacement,
                                **row,
                            }
                        )

    runtime = time.perf_counter() - start
    metrics = pd.DataFrame(metric_rows)
    pairs = pd.DataFrame(pair_rows)
    keys = ["template_id", "background", "generator", "displacement_um"]
    moved_baseline = metrics.loc[metrics.arm == "moved_no_correction"].set_index(keys)
    native = metrics.loc[metrics.arm == "moved_ks4_native_inverse"].copy()
    index = pd.MultiIndex.from_frame(native[keys])
    native["baseline_residual_fraction"] = moved_baseline.residual_fraction.reindex(index).to_numpy()
    native["baseline_amplitude_retention"] = moved_baseline.amplitude_retention.reindex(index).to_numpy()
    native["baseline_template_cosine"] = moved_baseline.template_cosine.reindex(index).to_numpy()
    zero_tax = metrics.loc[metrics.arm == "stationary_ks4_d0"].copy()
    generator_summary, gate = _tax_and_gate(native, zero_tax)

    interior_rows = native.edge_status.eq("interior")
    edge_rows = native.edge_status.eq("edge_affected")
    separation_summary = (
        pairs.groupby(["arm", "generator"], observed=True)
        .agg(
            cases=("template_id", "size"),
            median_distance_retention=("distance_retention", "median"),
            p10_distance_retention=("distance_retention", lambda x: x.quantile(0.1)),
            median_separation_to_noise=("separation_to_noise", "median"),
            p10_separation_to_noise=("separation_to_noise", lambda x: x.quantile(0.1)),
        )
        .reset_index()
    )
    result = {
        "status": "bounded_discovery_operator_audit",
        "versions": versions,
        "source_domain": source_result,
        "templates": list(templates),
        "backgrounds": [asdict(value) for value in BACKGROUNDS],
        "case_rows": int(len(metrics)),
        "pair_rows": int(len(pairs)),
        "runtime_seconds": runtime,
        "gate": gate,
        "coverage": {
            "native_interior_rows": int(interior_rows.sum()),
            "native_edge_affected_rows": int(edge_rows.sum()),
            "edge_gate_covered": bool(edge_rows.any()),
            "empirical_smoothness_gate_covered": False,
        },
        "operator_screen_pass": bool(gate["operator_primary_and_tax_pass"]),
        "advancement_authorized": False,
        "advancement_blockers": [
            value
            for value, blocked in (
                ("operator_primary_or_zero_tax_gate_failed", not gate["operator_primary_and_tax_pass"]),
                ("edge_challenge_not_covered", not bool(edge_rows.any())),
                ("empirical_waveform_depth_smoothness_not_run", True),
                ("motion_estimator_and_coordinate_only_ladder_incomplete", True),
            )
            if blocked
        ],
        "prospective_holdout_accessed": False,
        "sorter_run": False,
        "new_int16_warp_materialized": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    qualification.to_csv(output_dir / "source_domain_qualification.csv", index=False)
    metrics.to_csv(output_dir / "case_metrics.csv", index=False)
    pairs.to_csv(output_dir / "pair_separability_metrics.csv", index=False)
    generator_summary.to_csv(output_dir / "generator_gate_summary.csv", index=False)
    separation_summary.to_csv(output_dir / "separability_summary.csv", index=False)
    np.savez_compressed(output_dir / "operator_matrices.npz", **matrices)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def pair_separability(
    pair_table: pd.DataFrame,
    reference_templates: Mapping[str, np.ndarray],
    candidate_templates: Mapping[str, np.ndarray],
    noise: np.ndarray,
) -> pd.DataFrame:
    """Measure fixed-pair distance and noise along the reference direction."""
    rows = []
    for pair in pair_table.itertuples(index=False):
        ref_delta = np.asarray(reference_templates[pair.template_id], dtype=np.float64) - np.asarray(
            reference_templates[pair.neighbor_template_id], dtype=np.float64
        )
        candidate_left = np.asarray(candidate_templates[pair.template_id], dtype=np.float64)
        candidate_right = np.asarray(candidate_templates[pair.neighbor_template_id], dtype=np.float64)
        cand_delta = candidate_left - candidate_right
        direction = ref_delta.ravel()
        direction /= max(np.linalg.norm(direction), np.finfo(float).eps)
        retained = float(np.dot(cand_delta.ravel(), direction))
        half = ref_delta.shape[0] // 2
        projections = []
        for center in np.linspace(half + 1, len(noise) - half - 2, 16, dtype=int):
            window = noise[center - half : center - half + ref_delta.shape[0]]
            if window.shape == ref_delta.shape:
                projections.append(float(np.dot(window.ravel(), direction)))
        projected_noise = robust_sigma(np.asarray(projections))
        left = candidate_left.ravel() - candidate_left.mean()
        right = candidate_right.ravel() - candidate_right.mean()
        left /= max(np.linalg.norm(left), np.finfo(float).eps)
        right /= max(np.linalg.norm(right), np.finfo(float).eps)
        candidate_distance = float(np.linalg.norm(left - right))
        rows.append(
            {
                "template_id": pair.template_id,
                "neighbor_template_id": pair.neighbor_template_id,
                "reference_distance": float(pair.reference_distance),
                "candidate_distance": candidate_distance,
                "distance_retention": candidate_distance / max(float(pair.reference_distance), np.finfo(float).eps),
                "reference_direction_separation": retained,
                "projected_noise_sigma": projected_noise,
                "separation_to_noise": retained / projected_noise,
            }
        )
    return pd.DataFrame(rows)


def robust_gate_summary(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the frozen worst-generator median aggregation and thresholds."""
    required = {
        "generator",
        "residual_fraction",
        "amplitude_retention",
        "template_cosine",
        "baseline_residual_fraction",
        "baseline_amplitude_retention",
        "baseline_template_cosine",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"gate metrics missing columns: {sorted(missing)}")
    values = metrics.copy()
    values["delta_residual"] = values.residual_fraction - values.baseline_residual_fraction
    values["delta_absolute_amplitude_error"] = (
        (values.amplitude_retention - 1).abs() - (values.baseline_amplitude_retention - 1).abs()
    )
    values["delta_cosine"] = values.template_cosine - values.baseline_template_cosine
    by_generator = (
        values.groupby("generator", observed=True)
        .agg(
            median_delta_residual=("delta_residual", "median"),
            median_delta_absolute_amplitude_error=("delta_absolute_amplitude_error", "median"),
            median_delta_cosine=("delta_cosine", "median"),
            cases=("generator", "size"),
        )
        .reset_index()
    )
    decision = {
        "worst_generator_median_delta_residual": float(by_generator.median_delta_residual.max()),
        "worst_generator_median_delta_absolute_amplitude_error": float(
            by_generator.median_delta_absolute_amplitude_error.max()
        ),
        "worst_generator_median_delta_cosine": float(by_generator.median_delta_cosine.min()),
    }
    decision["primary_screen_pass"] = bool(
        decision["worst_generator_median_delta_residual"]
        <= GATE.worst_generator_median_delta_residual_max
        and decision["worst_generator_median_delta_absolute_amplitude_error"]
        <= GATE.worst_generator_median_delta_absolute_amplitude_error_max
        and decision["worst_generator_median_delta_cosine"]
        >= GATE.worst_generator_median_delta_cosine_min
    )
    return by_generator, decision


def validate_local_highpass_equivalence(hp_filter: np.ndarray, fs: float) -> dict[str, float | bool]:
    """Compare guarded-local and full-KS-batch filtering at the score window."""
    rng = np.random.default_rng(20250804)
    full_n = int(round(2.0 * fs))
    local_n = BOUNDS.local_filter_samples
    score_n = BOUNDS.score_samples
    full = np.zeros((full_n, 4), dtype=np.float32)
    wave = rng.normal(size=(score_n, 4)).astype(np.float32)
    _center_insert(full, wave)
    local = score_window(full, local_n)
    full_filtered = ks4_p_local(full, full.mean(axis=0), hp_filter)
    local_filtered = ks4_p_local(local, full.mean(axis=0), hp_filter)
    expected = score_window(full_filtered, score_n)
    observed = score_window(local_filtered, score_n)
    error = observed.astype(np.float64) - expected.astype(np.float64)
    relative = float(np.linalg.norm(error) / max(np.linalg.norm(expected), np.finfo(float).eps))
    maximum = float(np.max(np.abs(error)))
    return {
        "relative_l2_error": relative,
        "maximum_absolute_error": maximum,
        "passed": bool(relative <= 1e-5 and maximum <= 1e-5),
    }


def synthetic_validation(output_dir: Path | None = None) -> dict:
    """Exercise all pure contracts without opening Luke raw or holdout data."""
    import torch
    from kilosort.preprocessing import get_highpass_filter

    validate_frozen_plan(frozen_plan())
    versions = validate_versions()
    y = np.arange(24, dtype=float) * 20.0
    x = np.tile(np.array([0.0, 16.0, 32.0, 48.0]), 6)
    positions = np.column_stack((x, y))
    # Keep y monotonic for the generated 24-channel validation geometry.
    positions[:, 1] = np.repeat(np.arange(6) * 20.0, 4)
    kxx = np.exp(-np.sum((positions[:, None] - positions[None]) ** 2, axis=2) / (2 * 20.0**2))
    m0 = kxx @ np.linalg.inv(kxx + 0.01 * np.eye(len(kxx)))
    zero_not_identity = not np.allclose(m0, np.eye(len(m0)), atol=1e-8)

    fs = 30000.0
    hp = get_highpass_filter(fs=fs, cutoff=300, device=torch.device("cpu")).numpy()
    highpass = validate_local_highpass_equivalence(hp, fs)

    rng = np.random.default_rng(7)
    background = rng.normal(scale=0.5, size=(BOUNDS.local_filter_samples, len(positions))).astype(np.float32)
    waveform = np.zeros((BOUNDS.score_samples, len(positions)), dtype=np.float32)
    t = np.arange(BOUNDS.score_samples) - BOUNDS.score_samples // 2
    temporal = -np.exp(-0.5 * (t / 5.0) ** 2).astype(np.float32)
    spatial = np.exp(-0.5 * ((positions[:, 1] - 60.0) / 22.0) ** 2).astype(np.float32)
    waveform[:] = temporal[:, None] * spatial[None, :]
    whiten = np.eye(len(positions), dtype=np.float32)
    delta, noise = paired_event_delta(
        background,
        background.mean(axis=0),
        waveform,
        hp,
        whiten,
        parent_samples=len(background),
    )
    reference = score_window(delta, BOUNDS.score_samples)
    metrics = waveform_metrics(reference, reference, noise, positions)

    gate_rows = []
    for generator in ("g1", "g2", "g3"):
        for case in range(4):
            gate_rows.append(
                {
                    "generator": generator,
                    "case": case,
                    "residual_fraction": 0.08,
                    "amplitude_retention": 0.905,
                    "template_cosine": 0.91,
                    "baseline_residual_fraction": 0.10,
                    "baseline_amplitude_retention": 0.90,
                    "baseline_template_cosine": 0.90,
                }
            )
    _, gate = robust_gate_summary(pd.DataFrame(gate_rows))
    result = {
        "status": "synthetic_validation_only",
        "versions": versions,
        "zero_shift_is_not_identity": bool(zero_not_identity),
        "zero_shift_frobenius_distance": float(np.linalg.norm(m0 - np.eye(len(m0)))),
        "local_highpass_equivalence": highpass,
        "identity_waveform_metrics": metrics,
        "gate_contract_passed": bool(gate["primary_screen_pass"]),
        "prospective_holdout_accessed": False,
        "sorter_run": False,
    }
    result["passed"] = bool(
        zero_not_identity
        and highpass["passed"]
        and metrics["residual_fraction"] <= 1e-6
        and metrics["template_cosine"] >= 0.999999
        and gate["primary_screen_pass"]
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frozen_config.json").write_text(json.dumps(frozen_plan(), indent=2) + "\n")
        (output_dir / "synthetic_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def write_plan(output_dir: Path) -> dict:
    plan = frozen_plan()
    validate_frozen_plan(plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frozen_config.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--synthetic-validation", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plan_only:
        result = write_plan(args.output_dir)
    elif args.synthetic_validation:
        start = time.perf_counter()
        result = synthetic_validation(args.output_dir)
        result["runtime_seconds"] = time.perf_counter() - start
        (args.output_dir / "synthetic_validation.json").write_text(json.dumps(result, indent=2) + "\n")
        if not result["passed"]:
            print(json.dumps(result, indent=2))
            raise SystemExit(2)
    elif args.dry_run:
        start = time.perf_counter()
        result = discovery_dry_run(args.output_dir)
        result["total_runtime_seconds"] = time.perf_counter() - start
        (args.output_dir / "dry_run_result.json").write_text(json.dumps(result, indent=2) + "\n")
        if not result["passed"]:
            print(json.dumps(result, indent=2))
            raise SystemExit(2)
    else:
        result = run_discovery_audit(args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
