"""Option A field prerequisites — the §5.3 qualification receipt, measured.

``docs/luke_two_motion_pipeline_build_instructions.md`` §5.3 requires a receipt
before any motion field may be materialized onto voltage, and §5.6 requires a
test that detects a time-origin mismatch. Neither exists. ``qualify_field`` in
``testing/ladder_motion_estimate.py`` is written to fail closed when support,
reproducibility or error evidence is missing -- and ``_field_diagnostics`` never
computes any of the three, so it fails closed on every field the pipeline can
currently produce. That is correct behaviour, but it means "Option A is blocked"
has only ever been readable from the code, never from a measurement.

This module measures what *can* be measured from artifacts already on disk and
emits the receipt, naming exactly which limbs block. It does not estimate a new
field, does not sort, does not touch voltage, and writes nothing under ``/mnt``.

Three things it establishes that were previously assumed:

* **The time origin.** Every accepted estimator's time axis for Luke 2025-08-04
  imec0 runs 3058.7 → 13530.7 s on a 10,473.55 s recording, because
  SpikeInterface wrote the bins in *acquisition* time while the peaks they were
  computed from are indexed in recording frames from zero. The offset is
  recoverable from two independent sources -- the SpikeGLX ``firstSample`` /
  ``imSampRate`` and the SI ``binary.json`` ``t_starts`` -- and this module
  requires them to agree before reporting a mapping. Prior window-metric work
  applied it correctly, but nothing recorded it, and one artifact on disk
  actively mis-declares it (see :func:`check_declared_time_reference`).
* **Support.** The fraction of the field's own (time, depth) domain that has
  real detected peaks behind it. A field is not applicable where it has no data,
  and ``build_spikeinterface_motion`` already refuses a partially supported
  field rather than extrapolating one.
* **Scale uncertainty.** The four accepted estimators are a lower bound on how
  well the field's gain is known. Decision 0013 records their disagreement as an
  open quantification problem; expressed as the fraction ``qualify_field``
  gates, it is measured here rather than asserted.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from testing.ladder_motion_estimate import FieldGate, qualify_field

SCHEMA = "luke-option-a-field-prerequisites-v1"

#: Estimators decision 0013 accepts for Luke 2025-08-04 imec0.
ACCEPTED_ESTIMATORS = ("ks-motion", "dredge-motion", "medicine", "decentralized-motion")

#: A (time, depth) cell counts as supported when at least this many detected
#: peaks fall in it. Declared, not tuned: the point is that a field cannot be
#: applied where nothing was detected, and one peak is not a raster column.
MIN_PEAKS_PER_CELL = 5

#: Tolerance for agreement between the two independent time-origin sources.
TIME_ORIGIN_TOLERANCE_S = 1e-3

#: The absolute displacement gain of every field on this session is UNMEASURED.
#: Estimator disagreement is not a measurement of it; only comparison against
#: known motion would be, and the D2b-1 envelope that supplied one is retracted.
GAIN_UNMEASURED = "unmeasured"

#: Which qualification requirements are load-bearing for *any* application of a
#: field to voltage, and which are provisional scientific acceptance criteria
#: inherited from the retracted D2b-1 envelope.
#:
#: The distinction matters because they fail for different reasons and can be
#: waived on different terms. An integrity requirement that fails means the
#: operation would be wrong however good the science: a field applied on the
#: wrong clock, extrapolated into a region with no data, or applied with the
#: sign inverted corrupts the recording regardless of how accurate the estimate
#: is. A provisional criterion that fails means the field may not meet a
#: numerical standard that was set by work now retracted, and whose replacement
#: does not exist.
#:
#: Neither `FieldGate` nor `qualify_field` is modified: they are the historical
#: record of that envelope and remain the production gate. This table sits
#: beside them.
INDISPENSABLE = "indispensable_implementation_integrity"
PROVISIONAL = "provisional_scientific_criterion"

REQUIREMENT_CLASS: dict[str, dict[str, str]] = {
    "acquisition_recording_clock_mapping": {
        "class": INDISPENSABLE,
        "why": (
            "A field applied against the wrong time origin misplaces every correction by that "
            "origin. Nothing about estimator accuracy can compensate for it."
        ),
    },
    "supported_application_domain": {
        "class": INDISPENSABLE,
        "why": (
            "Interpolating where no peaks were detected substitutes the regulariser's "
            "extrapolation for a measurement, and writes it into voltage as if it were data."
        ),
    },
    "displacement_polarity": {
        "class": INDISPENSABLE,
        "why": (
            "A sign error doubles the displacement instead of removing it. It is verifiable "
            "independently of accuracy, by a zero-motion identity and a forward/inverse test."
        ),
    },
    "operator_behaviour": {
        "class": INDISPENSABLE,
        "why": (
            "External correction plus KS4 internal correction is double correction; a crop "
            "before correction discards the source of the shift. Both are checkable facts about "
            "the run, not scientific judgements."
        ),
    },
    "max_abs_displacement_um": {
        "class": PROVISIONAL,
        "why": "Range over which the retracted D2b-1 injected-truth calibration claimed cover.",
    },
    "min_support_fraction": {
        "class": PROVISIONAL,
        "why": (
            "A whole-probe, whole-session threshold. The underlying requirement -- do not apply "
            "the field where it has no support -- is indispensable, but *this number over the "
            "entire domain* is a provisional acceptance criterion and can be met locally instead."
        ),
    },
    "min_split_half_correlation": {
        "class": PROVISIONAL,
        "why": "Reproducibility threshold from the retracted envelope; measures precision, not accuracy.",
    },
    "max_estimated_gain_error_fraction": {
        "class": PROVISIONAL,
        "why": (
            "Requires a calibration against known motion. The D2b-1 envelope that supplied one "
            "is retracted, so this criterion is currently NOT EVALUABLE -- the gain is unmeasured, "
            "which is not the same as measured and out of tolerance."
        ),
    },
}


class PrerequisiteRefusal(ValueError):
    """A prerequisite cannot be evaluated honestly. Never caught internally."""


def _reject_mnt(out_root: Path) -> Path:
    resolved = Path(out_root).expanduser().resolve()
    if resolved == Path("/mnt") or str(resolved).startswith("/mnt/"):
        raise PrerequisiteRefusal(f"refusing an output root under /mnt: {resolved}")
    return resolved


# --------------------------------------------------------------------------- #
# time origin
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TimeOrigin:
    """Acquisition-time origin of the recording, from two independent sources."""

    seconds: float
    from_spikeglx_meta: float
    from_spikeinterface_manifest: float
    meta_path: str
    manifest_path: str
    first_sample: int
    sampling_frequency_hz: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "acquisition_time_origin_s": self.seconds,
            "from_spikeglx_meta_s": self.from_spikeglx_meta,
            "from_spikeinterface_manifest_s": self.from_spikeinterface_manifest,
            "sources_agree_within_s": TIME_ORIGIN_TOLERANCE_S,
            "spikeglx_meta": self.meta_path,
            "spikeinterface_manifest": self.manifest_path,
            "first_sample": self.first_sample,
            "sampling_frequency_hz": self.sampling_frequency_hz,
            "mapping": "t_recording_s = t_motion_s - acquisition_time_origin_s",
            "why": (
                "SpikeInterface wrote the motion time bins in acquisition time; the peaks they "
                "were estimated from are indexed in recording frames from zero. Applying the "
                "field without this subtraction misplaces every correction by the origin."
            ),
        }


def read_spikeglx_time_origin(meta_path: Path) -> tuple[float, int, float]:
    text = Path(meta_path).read_text()
    fields = dict(
        line.split("=", 1) for line in text.splitlines() if "=" in line and line.startswith("~") is False
    )
    try:
        first_sample = int(fields["firstSample"])
        fs = float(fields["imSampRate"])
    except KeyError as exc:  # pragma: no cover - a meta without these is unusable
        raise PrerequisiteRefusal(f"{meta_path} lacks {exc.args[0]}") from exc
    return first_sample / fs, first_sample, fs


def read_spikeinterface_time_origin(manifest_path: Path) -> float:
    payload = json.loads(Path(manifest_path).read_text())
    found: list[float] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "t_starts" and isinstance(value, list) and value:
                    found.append(float(value[0]))
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    if not found:
        raise PrerequisiteRefusal(f"{manifest_path} declares no t_starts")
    return found[0]


def verify_time_origin(meta_path: Path, manifest_path: Path) -> TimeOrigin:
    """Recover the acquisition-time origin, requiring two sources to agree.

    One source is a claim; two that agree is a mapping. They come from different
    layers -- the acquisition software's own sample counter and the
    SpikeInterface recording manifest -- so agreement is not circular.
    """
    from_meta, first_sample, fs = read_spikeglx_time_origin(meta_path)
    from_manifest = read_spikeinterface_time_origin(manifest_path)
    if abs(from_meta - from_manifest) > TIME_ORIGIN_TOLERANCE_S:
        raise PrerequisiteRefusal(
            f"time-origin sources disagree: SpikeGLX meta says {from_meta:.6f} s, the "
            f"SpikeInterface manifest says {from_manifest:.6f} s. A field cannot be applied "
            "against an unverified time origin."
        )
    return TimeOrigin(
        seconds=from_manifest,
        from_spikeglx_meta=from_meta,
        from_spikeinterface_manifest=from_manifest,
        meta_path=str(meta_path),
        manifest_path=str(manifest_path),
        first_sample=first_sample,
        sampling_frequency_hz=fs,
    )


def check_declared_time_reference(
    npz_path: Path, origin: TimeOrigin, duration_s: float
) -> dict[str, Any]:
    """Catch an artifact whose declared time reference contradicts its values.

    ``pipeline.motion_coordinates.load_qualified_motion_field`` validates the
    ``time_reference`` *string* and then trusts it. An artifact that declares
    ``selected_recording_start`` while carrying acquisition-time values passes
    that check and is then interpolated against the wrong clock.
    """
    with np.load(npz_path, allow_pickle=False) as values:
        declared = str(np.asarray(values["time_reference"]).reshape(()).item())
        times = np.asarray(values["time_s"], dtype=np.float64).reshape(-1)
    first, last = float(times[0]), float(times[-1])
    consistent_with_recording = first < origin.seconds / 2 and last <= duration_s * 1.01
    consistent_with_acquisition = (
        abs(first - origin.seconds) < max(60.0, 0.02 * duration_s)
    )
    verdict = "consistent"
    if declared == "selected_recording_start" and not consistent_with_recording:
        verdict = (
            "MISDECLARED: declares selected_recording_start but its values are on the "
            "acquisition clock"
            if consistent_with_acquisition
            else "MISDECLARED: declares selected_recording_start but its values are not on it"
        )
    return {
        "path": str(npz_path),
        "declared_time_reference": declared,
        "first_time_s": first,
        "last_time_s": last,
        "recording_duration_s": duration_s,
        "acquisition_time_origin_s": origin.seconds,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# support
# --------------------------------------------------------------------------- #
def support_fraction(
    peaks_path: Path,
    peak_locations_path: Path,
    time_bins_s: np.ndarray,
    depth_bins_um: np.ndarray,
    *,
    fs_hz: float,
    min_peaks_per_cell: int = MIN_PEAKS_PER_CELL,
) -> dict[str, Any]:
    """Fraction of the field's own (time, depth) domain with real peaks behind it.

    Peak sample indices are recording frames, so they are histogrammed against
    the field's time bins **after** those have been mapped to recording time by
    the caller. A cell with no detections did not measure motion; the field's
    value there is an extrapolation of its regulariser.
    """
    peaks = np.load(peaks_path, mmap_mode="r")
    locations = np.load(peak_locations_path, mmap_mode="r")
    seconds = np.asarray(peaks["sample_index"], dtype=np.float64) / float(fs_hz)
    depths = np.asarray(locations["y"], dtype=np.float64)

    time_edges = _edges(np.asarray(time_bins_s, dtype=np.float64))
    depth_edges = _edges(np.asarray(depth_bins_um, dtype=np.float64))
    counts, _, _ = np.histogram2d(seconds, depths, bins=[time_edges, depth_edges])

    supported = counts >= min_peaks_per_cell
    boundary = np.zeros(supported.shape[1], dtype=bool)
    boundary[[0, -1]] = True
    return {
        "n_peaks": int(seconds.size),
        "grid": [int(counts.shape[0]), int(counts.shape[1])],
        "min_peaks_per_cell": int(min_peaks_per_cell),
        "support_fraction": float(supported.mean()),
        "median_peaks_per_cell": float(np.median(counts)),
        "depth_bins_never_supported": int(np.count_nonzero(~supported.any(axis=0))),
        "boundary_depth_support_fraction": float(supported[:, boundary].mean()),
        "time_bins_fully_supported_fraction": float(supported.all(axis=1).mean()),
    }


def _edges(centres: np.ndarray) -> np.ndarray:
    if centres.size < 2:
        raise PrerequisiteRefusal("a bin axis needs at least two centres")
    step = np.diff(centres)
    return np.concatenate(
        [[centres[0] - step[0] / 2], centres[:-1] + step / 2, [centres[-1] + step[-1] / 2]]
    )


# --------------------------------------------------------------------------- #
# scale uncertainty across accepted estimators
# --------------------------------------------------------------------------- #
def rigid_trace(motion: np.ndarray) -> np.ndarray:
    m = np.asarray(motion, dtype=np.float64)
    return m.mean(axis=1) if m.ndim > 1 else m


def window_excursion(
    motion: np.ndarray, time_recording_s: np.ndarray, start_s: float, stop_s: float
) -> float:
    inside = (time_recording_s >= start_s) & (time_recording_s < stop_s)
    if int(np.count_nonzero(inside)) < 3:
        return float("nan")
    trace = rigid_trace(motion)[inside]
    return float(np.percentile(trace, 95) - np.percentile(trace, 5))


def inter_estimator_disagreement(
    fields: dict[str, dict[str, np.ndarray]], windows: list[tuple[float, float]]
) -> dict[str, Any]:
    """How much do the accepted estimators disagree about displacement scale?

    This is **disagreement, not error**. No estimator here has been compared
    against known motion, so the absolute gain of any one field is *unmeasured*:
    the spread says the four cannot all be right, and nothing more. It does not
    licence a statement that some particular field's gain is wrong by any
    amount, and it is not a substitute for the calibration ``qualify_field``'s
    ``estimated_gain_error_fraction`` was meant to consume.
    """
    per_window: dict[str, list[float]] = {name: [] for name in fields}
    for start_s, stop_s in windows:
        for name, field in fields.items():
            per_window[name].append(
                window_excursion(field["motion"], field["time_recording_s"], start_s, stop_s)
            )
    medians = {
        name: float(np.nanmedian(values)) for name, values in per_window.items()
    }
    finite = [v for v in medians.values() if np.isfinite(v) and v > 0]
    ratio = max(finite) / min(finite) if len(finite) > 1 else float("nan")
    return {
        "median_window_rigid_excursion_um": medians,
        "max_over_min_ratio": ratio,
        "absolute_gain_status": GAIN_UNMEASURED,
        "basis": (
            "Per-window rigid excursion (P95-P5 of the depth-averaged displacement) of each "
            "accepted estimator over the same windows on the same recording clock."
        ),
        "note": (
            "Disagreement between estimators, NOT an error measurement. No field here has been "
            "compared against known motion, so every field's absolute gain is unmeasured. The "
            "spread shows the four cannot all be right; it does not say by how much any one is "
            "wrong, and it must not be reported as a measured error. Decision 0013 carries the "
            "disagreement as an open quantification problem."
        ),
    }


# --------------------------------------------------------------------------- #
# local application domain
# --------------------------------------------------------------------------- #
def local_support_profile(
    peaks_path: Path,
    peak_locations_path: Path,
    depth_bins_um: np.ndarray,
    interval_s: tuple[float, float],
    *,
    fs_hz: float,
    time_bin_s: float = 1.0,
    min_peaks_per_cell: int = MIN_PEAKS_PER_CELL,
) -> dict[str, Any]:
    """Per-depth support inside one interval only.

    A whole-session, whole-probe support fraction can be dragged down by times
    and depths that a bounded experiment never touches. What decides whether a
    field may be applied *here* is whether the peaks exist *here*.
    """
    start_s, stop_s = float(interval_s[0]), float(interval_s[1])
    peaks = np.load(peaks_path, mmap_mode="r")
    locations = np.load(peak_locations_path, mmap_mode="r")
    seconds = np.asarray(peaks["sample_index"], dtype=np.float64) / float(fs_hz)
    inside = (seconds >= start_s) & (seconds < stop_s)
    times = seconds[inside]
    depths = np.asarray(locations["y"])[inside]

    depth_edges = _edges(np.asarray(depth_bins_um, dtype=np.float64))
    time_edges = np.arange(start_s, stop_s + 1e-9, time_bin_s)
    counts, _, _ = np.histogram2d(times, depths, bins=[time_edges, depth_edges])
    supported = counts >= min_peaks_per_cell
    per_depth = supported.mean(axis=0)
    return {
        "interval_s": [start_s, stop_s],
        "n_peaks_in_interval": int(times.size),
        "time_bin_s": time_bin_s,
        "min_peaks_per_cell": int(min_peaks_per_cell),
        "depth_bin_centres_um": [float(z) for z in depth_bins_um],
        "depth_bin_edges_um": [float(e) for e in depth_edges],
        "per_depth_support_fraction": [float(v) for v in per_depth],
    }


def usable_application_domain(
    profile: dict[str, Any],
    displacement_um: np.ndarray,
    *,
    min_depth_support: float = 0.95,
    kernel_sigma_um: float = 20.0,
    kernel_reach_sigma: float = 3.0,
) -> dict[str, Any]:
    """The interior depth band that can be corrected without extrapolating.

    Correcting a channel at depth ``y`` by displacement ``d`` reads source data
    near ``y + d``, smeared by the spatial interpolation kernel. So a band is
    usable only when the region it *draws from* is supported too: the supported
    span must be eroded by ``max|d| + kernel_reach_sigma * sigma`` at each end.

    The margin is computed from the field's own displacement inside the
    interval, and the band is frozen from this before any sorting is run. It is
    deliberately a single contiguous interior band: taking whatever scattered
    depths happen to be supported would be cropping to fit.
    """
    centres = np.asarray(profile["depth_bin_centres_um"], dtype=np.float64)
    edges = np.asarray(profile["depth_bin_edges_um"], dtype=np.float64)
    support = np.asarray(profile["per_depth_support_fraction"], dtype=np.float64)
    ok = support >= min_depth_support

    # longest run of consecutive supported depth bins
    best_start = best_len = run_start = run_len = 0
    for index, value in enumerate(ok):
        if value:
            run_len = run_len + 1 if run_len else 1
            run_start = index - run_len + 1
            if run_len > best_len:
                best_start, best_len = run_start, run_len
        else:
            run_len = 0
    if best_len == 0:
        return {"usable": False, "reason": "no depth bin reaches the local support threshold"}

    supported_lo = float(edges[best_start])
    supported_hi = float(edges[best_start + best_len])
    max_abs_displacement = float(np.abs(np.asarray(displacement_um, dtype=np.float64)).max())
    margin = max_abs_displacement + kernel_reach_sigma * kernel_sigma_um
    interior_lo, interior_hi = supported_lo + margin, supported_hi - margin
    usable = interior_hi > interior_lo
    return {
        "usable": bool(usable),
        "supported_band_um": [supported_lo, supported_hi],
        "n_supported_depth_bins": int(best_len),
        "n_depth_bins": int(centres.size),
        "max_abs_displacement_um": max_abs_displacement,
        "kernel_sigma_um": kernel_sigma_um,
        "kernel_reach_sigma": kernel_reach_sigma,
        "required_margin_um": margin,
        "usable_interior_band_um": [interior_lo, interior_hi] if usable else None,
        "min_depth_support": min_depth_support,
        "rule": (
            "The longest contiguous run of depth bins meeting the local support threshold, "
            "eroded at both ends by max|displacement| + reach*sigma so that every corrected "
            "channel draws only from supported depths. Frozen before any sort."
        ),
        "unsupported_depth_bins_excluded": [
            float(z) for z, value in zip(centres, ok) if not value
        ],
    }


def channels_in_band(channel_positions_um: np.ndarray, band_um: list[float]) -> dict[str, Any]:
    y = np.asarray(channel_positions_um, dtype=np.float64)[:, 1]
    inside = (y >= band_um[0]) & (y <= band_um[1])
    return {
        "n_channels_total": int(y.size),
        "n_channels_in_band": int(np.count_nonzero(inside)),
        "band_um": [float(band_um[0]), float(band_um[1])],
        "channel_y_range_um": [float(y.min()), float(y.max())],
    }


# --------------------------------------------------------------------------- #
# split-half reproducibility (precision, not accuracy)
# --------------------------------------------------------------------------- #
def split_half_correlation(
    trace_a: np.ndarray, trace_b: np.ndarray
) -> float:
    """Pearson correlation of two independently estimated displacement traces.

    Reproducibility only. Two halves of the same data estimated by the same
    method agreeing tells you the estimate is stable; it says nothing about
    whether either is the true tissue motion.
    """
    a = np.asarray(trace_a, dtype=np.float64)
    b = np.asarray(trace_b, dtype=np.float64)
    if a.shape != b.shape or a.size < 3:
        raise PrerequisiteRefusal("split-half traces must have equal length >= 3")
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not denominator > 0:
        return float("nan")
    return float(np.dot(a, b) / denominator)


# --------------------------------------------------------------------------- #
# operator integrity: zero-motion identity and displacement polarity
# --------------------------------------------------------------------------- #
def probe_operator_polarity(
    *,
    n_channels: int = 32,
    pitch_um: float = 20.0,
    feature_depth_um: float = 300.0,
    shift_um: float = 60.0,
    sigma_um: float = 20.0,
) -> dict[str, Any]:
    """Establish the sign convention and the zero-motion identity, synthetically.

    Both are §5.6 requirements and neither depends on any real field's accuracy:
    they are facts about what the operator does with a displacement of known
    sign. A sign error does not degrade a correction, it doubles the
    displacement, so this must be settled before anything is applied.

    Returns a structured result; when SpikeInterface is not importable it says
    so rather than guessing, because an unverifiable polarity is a refusal.
    """
    try:
        from probeinterface import Probe
        from spikeinterface.core import NumpyRecording
        from spikeinterface.core.motion import Motion
        from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording
    except ImportError as exc:
        return {"available": False, "reason": f"SpikeInterface not importable here: {exc}"}

    fs, duration_s = 30000.0, 4.0
    depths = np.arange(n_channels) * pitch_um
    positions = np.stack([np.zeros(n_channels), depths], axis=1)
    n_samples = int(duration_s * fs)
    centre = n_samples // 2
    profile = np.exp(-0.5 * ((depths - feature_depth_um) / 30.0) ** 2)
    waveform = -np.exp(-0.5 * (np.arange(-30, 31) / 8.0) ** 2)
    traces = np.zeros((n_samples, n_channels), dtype="float32")
    traces[centre - 30 : centre + 31, :] = np.outer(waveform, profile).astype("float32")

    def recording():
        probe = Probe(ndim=2, si_units="um")
        probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})
        probe.set_device_channel_indices(np.arange(n_channels))
        return NumpyRecording([traces], sampling_frequency=fs).set_probe(probe)

    def corrected(displacement_um: float):
        motion = Motion(
            displacement=[np.full((2, 1), float(displacement_um))],
            temporal_bins_s=[np.array([0.0, duration_s])],
            spatial_bins_um=np.array([depths.mean()]),
            direction="y",
        )
        return InterpolateMotionRecording(
            recording(), motion, border_mode="remove_channels",
            spatial_interpolation_method="kriging", sigma_um=sigma_um,
        )

    def centroid(rec) -> float:
        chunk = rec.get_traces(start_frame=centre - 40, end_frame=centre + 41)
        energy = np.abs(np.asarray(chunk, dtype=float)).max(axis=0)
        y = np.asarray(rec.get_channel_locations(), dtype=float)[:, 1]
        return float((energy * y).sum() / energy.sum()) if energy.sum() > 0 else float("nan")

    source = recording()
    zero = corrected(0.0)
    kept = np.isin(
        np.asarray(source.get_channel_locations())[:, 1],
        np.asarray(zero.get_channel_locations())[:, 1],
    )
    identity = bool(np.allclose(
        np.asarray(source.get_traces(start_frame=centre - 40, end_frame=centre + 41))[:, kept],
        np.asarray(zero.get_traces(start_frame=centre - 40, end_frame=centre + 41)),
        atol=1e-3,
    ))
    baseline = centroid(source)
    forward = centroid(corrected(+shift_um)) - baseline
    inverse = centroid(corrected(-shift_um)) - baseline
    symmetric = bool(abs(forward + inverse) < 0.15 * shift_um)
    resolved = bool(np.isfinite(forward) and abs(forward) >= 0.25 * shift_um)
    return {
        "available": True,
        "zero_motion_identity": identity,
        "applied_displacement_um": float(shift_um),
        "shift_for_positive_displacement_um": float(forward),
        "shift_for_negative_displacement_um": float(inverse),
        "recovered_magnitude_um": float(abs(forward)),
        "forward_inverse_symmetric": symmetric,
        "polarity_resolved": resolved,
        "polarity": (
            "output_depth = source_depth - displacement (a positive displacement moves "
            "the feature DOWN in depth)"
            if resolved and forward < 0 else
            "output_depth = source_depth + displacement (a positive displacement moves "
            "the feature UP in depth)" if resolved else "unresolved: no measurable shift"
        ),
        "passes": bool(identity and symmetric and resolved),
    }


# --------------------------------------------------------------------------- #
# interval nomination
# --------------------------------------------------------------------------- #
def nominate_development_interval(
    fields: dict[str, dict[str, np.ndarray]],
    development_windows: list[tuple[float, float]],
    *,
    window_s: float = 120.0,
) -> dict[str, Any]:
    """Where would external registration have the most to correct?

    Ranked by the **minimum** excursion across accepted estimators, so the
    nomination survives the disagreement between them: a window wins only if
    every estimator agrees there is motion there. Uses motion coordinates only
    -- no sorter output, no candidate result, and no reference to whichever case
    a previous candidate happened to target.
    """
    candidates: list[dict[str, Any]] = []
    for lo, hi in development_windows:
        start = lo
        while start + window_s <= hi:
            stop = start + window_s
            per = {
                name: window_excursion(f["motion"], f["time_recording_s"], start, stop)
                for name, f in fields.items()
            }
            values = [v for v in per.values() if np.isfinite(v)]
            if len(values) == len(fields):
                candidates.append(
                    {
                        "start_s": start,
                        "stop_s": stop,
                        "per_estimator_rigid_excursion_um": per,
                        "min_across_estimators_um": float(min(values)),
                        "median_across_estimators_um": float(np.median(values)),
                    }
                )
            start += window_s
    if not candidates:
        return {"nominated": None, "reason": "no window is covered by every accepted estimator"}
    ranked = sorted(
        candidates, key=lambda c: (-c["min_across_estimators_um"], c["start_s"])
    )
    return {
        "nominated": ranked[0],
        "n_windows_considered": len(candidates),
        "runner_up": ranked[1] if len(ranked) > 1 else None,
        "quietest_considered": min(candidates, key=lambda c: c["min_across_estimators_um"]),
        "selection_rule": (
            "Highest minimum rigid excursion across all accepted estimators, ties by earliest "
            "start. Motion coordinates only; no sorter output is consulted."
        ),
    }


# --------------------------------------------------------------------------- #
# assessment
# --------------------------------------------------------------------------- #
def load_accepted_fields(
    motion_root: Path, origin: TimeOrigin, estimators=ACCEPTED_ESTIMATORS
) -> dict[str, dict[str, np.ndarray]]:
    fields: dict[str, dict[str, np.ndarray]] = {}
    for name in estimators:
        directory = Path(motion_root) / name
        if not directory.exists():
            continue
        motion = np.load(directory / "motion.npy")
        time_bins = np.load(directory / "time_bins.npy").astype(np.float64)
        fields[name] = {
            "motion": motion,
            "time_acquisition_s": time_bins,
            "time_recording_s": time_bins - origin.seconds,
            "depth_bins_um": np.load(directory / "depth_bins.npy").astype(np.float64),
        }
    if not fields:
        raise PrerequisiteRefusal(f"no accepted estimator found under {motion_root}")
    return fields


def assess(
    *,
    motion_root: Path,
    meta_path: Path,
    recording_manifest: Path,
    duration_s: float,
    development_windows: list[tuple[float, float]],
    declared_field_npz: Path | None = None,
    support_estimator: str = "dredge-motion",
    gate: FieldGate | None = None,
) -> dict[str, Any]:
    """Measure every prerequisite that on-disk artifacts can support."""
    gate = gate or FieldGate()
    origin = verify_time_origin(Path(meta_path), Path(recording_manifest))
    fields = load_accepted_fields(Path(motion_root), origin)

    windows = [
        (lo, lo + 120.0)
        for start, stop in development_windows
        for lo in np.arange(start, stop - 120.0 + 1e-9, 120.0)
    ]
    disagreement = inter_estimator_disagreement(fields, windows)

    support: dict[str, Any] | None = None
    peaks = Path(motion_root) / "peaks.npy"
    locations = Path(motion_root) / "peak_locations.npy"
    if support_estimator in fields and peaks.exists() and locations.exists():
        field = fields[support_estimator]
        support = support_fraction(
            peaks, locations, field["time_recording_s"], field["depth_bins_um"],
            fs_hz=origin.sampling_frequency_hz,
        )
        support["estimator"] = support_estimator

    per_estimator: dict[str, Any] = {}
    for name, field in fields.items():
        trace = rigid_trace(field["motion"])
        centred = trace - trace.mean()
        diagnostics = {
            "n_time_bins": int(field["motion"].shape[0]),
            "n_spatial_windows": int(field["motion"].shape[1]) if field["motion"].ndim > 1 else 1,
            "max_abs_displacement_um": float(np.abs(centred).max()),
            "full_session_rigid_excursion_um": float(
                np.percentile(trace, 95) - np.percentile(trace, 5)
            ),
            # the three limbs qualify_field gates on:
            "support_fraction": (
                support["support_fraction"] if support and name == support_estimator else None
            ),
            "split_half_correlation": None,
            "estimated_gain_error_fraction": None,
        }
        per_estimator[name] = {
            "diagnostics": diagnostics,
            "qualification": qualify_field(diagnostics, gate),
        }

    blocking = _blocking(per_estimator, support, disagreement, gate)
    return {
        "schema": SCHEMA,
        "motion_root": str(motion_root),
        "gate": {"digest": gate.digest, **{k: getattr(gate, k) for k in vars(gate)}},
        "time_origin": origin.to_dict(),
        "declared_time_reference_check": (
            check_declared_time_reference(Path(declared_field_npz), origin, duration_s)
            if declared_field_npz and Path(declared_field_npz).exists()
            else None
        ),
        "support": support,
        "inter_estimator_disagreement": disagreement,
        "estimators": per_estimator,
        "blocking_prerequisites": blocking,
        "can_run_bounded_option_a": not blocking,
        "nominated_development_interval": nominate_development_interval(
            fields, development_windows
        ),
    }


def _blocking(
    per_estimator: dict[str, Any],
    support: dict[str, Any] | None,
    scale: dict[str, Any],
    gate: FieldGate,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not any(e["qualification"]["passes"] for e in per_estimator.values()):
        failed = sorted({f for e in per_estimator.values() for f in e["qualification"]["failed"]})
        out.append({
            "prerequisite": "a field that passes qualify_field",
            "requirement_class": PROVISIONAL,
            "status": "blocked",
            "detail": f"no accepted estimator passes; failed checks: {failed}",
        })
    if support is not None and support["support_fraction"] < gate.min_support_fraction:
        out.append({
            "prerequisite": "supported time/depth fraction (whole probe, whole session)",
            "requirement_class": PROVISIONAL,
            "status": "blocked",
            "detail": (
                f"support fraction {support['support_fraction']:.3f} against a gate of "
                f"{gate.min_support_fraction}; {support['depth_bins_never_supported']} depth bins "
                "have no detected peaks at all, and build_spikeinterface_motion refuses a "
                "partially supported field rather than extrapolating it"
            ),
        })
    ratio = scale.get("max_over_min_ratio")
    out.append({
        "prerequisite": "estimated gain error within tolerance",
        "requirement_class": PROVISIONAL,
        "status": "not_evaluable",
        "detail": (
            f"the absolute gain of every field on this session is {GAIN_UNMEASURED}: no field has "
            "been compared against known motion, and the D2b-1 calibration that would supply the "
            "comparison is retracted. The accepted estimators disagree by "
            f"{ratio:.2f}x, which is disagreement between them and NOT a measurement of any "
            "field's error."
        ),
    })
    out.append({
        "prerequisite": "split-half field reproducibility",
        "requirement_class": PROVISIONAL,
        "status": "not_measured",
        "detail": (
            "requires re-estimating the field on two independent halves of the saved peaks; no "
            "whole-session estimate exists, and qualify_field fails closed without it"
        ),
    })
    for item in out:
        item.setdefault("requirement_class", PROVISIONAL)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    args = ap.parse_args(argv)

    config = json.loads(args.config.read_text())
    out_root = _reject_mnt(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    receipt = assess(
        motion_root=Path(config["motion_root"]),
        meta_path=Path(config["spikeglx_meta"]),
        recording_manifest=Path(config["spikeinterface_manifest"]),
        duration_s=float(config["duration_s"]),
        development_windows=[tuple(w) for w in config["development_windows_s"]],
        declared_field_npz=(
            Path(config["declared_field_npz"]) if config.get("declared_field_npz") else None
        ),
    )
    target = out_root / "option_a_field_prerequisites.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    tmp.replace(target)

    print(f"can_run_bounded_option_a: {receipt['can_run_bounded_option_a']}")
    for item in receipt["blocking_prerequisites"]:
        print(f"  [{item['status']}] {item['prerequisite']}: {item['detail']}")
    print(f"receipt: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
