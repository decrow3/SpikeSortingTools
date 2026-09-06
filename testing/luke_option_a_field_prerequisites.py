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


def inter_estimator_scale_spread(
    fields: dict[str, dict[str, np.ndarray]], windows: list[tuple[float, float]]
) -> dict[str, Any]:
    """How well is the field's gain known? Bounded below by estimator spread.

    ``qualify_field`` gates ``estimated_gain_error_fraction`` at 0.30. No
    calibration against known truth is available -- the D2b tolerance envelope is
    retracted -- so this is not that number. It is a *lower bound* on it: four
    accepted estimators applied to one recording cannot all be right, and the
    fraction by which they differ is error somebody's field carries.
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
        "implied_minimum_gain_error_fraction": (
            float(1.0 - min(finite) / max(finite)) if len(finite) > 1 else float("nan")
        ),
        "basis": (
            "Per-window rigid excursion (P95-P5 of the depth-averaged displacement) of each "
            "accepted estimator over the same windows on the same recording clock."
        ),
        "note": (
            "A lower bound, not a calibration. It says the gain is not known to better than "
            "this; it does not say which estimator is right. Decision 0013 carries the "
            "disagreement as an open quantification problem."
        ),
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
    scale = inter_estimator_scale_spread(fields, windows)

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

    blocking = _blocking(per_estimator, support, scale, gate)
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
        "inter_estimator_scale": scale,
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
            "status": "blocked",
            "detail": f"no accepted estimator passes; failed checks: {failed}",
        })
    if support is not None and support["support_fraction"] < gate.min_support_fraction:
        out.append({
            "prerequisite": "supported time/depth fraction",
            "status": "blocked",
            "detail": (
                f"support fraction {support['support_fraction']:.3f} against a gate of "
                f"{gate.min_support_fraction}; {support['depth_bins_never_supported']} depth bins "
                "have no detected peaks at all, and build_spikeinterface_motion refuses a "
                "partially supported field rather than extrapolating it"
            ),
        })
    implied = scale.get("implied_minimum_gain_error_fraction")
    if implied is not None and np.isfinite(implied) and implied > gate.max_estimated_gain_error_fraction:
        out.append({
            "prerequisite": "estimated gain error within tolerance",
            "status": "blocked",
            "detail": (
                f"accepted estimators imply a gain error of at least {implied:.2f} against a gate "
                f"of {gate.max_estimated_gain_error_fraction}; no calibration exists to choose "
                "between them (D2b envelope retracted, decision 0013)"
            ),
        })
    out.append({
        "prerequisite": "split-half field reproducibility",
        "status": "not_measured",
        "detail": (
            "requires re-estimating the field on two independent halves of the saved peaks; no "
            "such estimate exists, and qualify_field fails closed without it"
        ),
    })
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
