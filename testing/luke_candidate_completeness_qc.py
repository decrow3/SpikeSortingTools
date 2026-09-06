"""Baseline-versus-candidate amplitude-completeness QC on one physical interval.

This is the endpoint that motivated the Option B candidate: the bounded audit
selected ``rescue…c37__failure1`` because estimated missing-spike percentage
rose from 0.67% to 43.47% across four consecutive 1,000-spike windows. A
candidate that regroups identities is only interesting if that number moves, so
the same statistic, from the same fitter, is computed for both arms over the
same physical-time interval.

Three rules the prescription and decision 0009 impose, enforced here rather
than remembered:

* **The production amplitude source, and only it.** ``full_st[kept_spikes][:, 2]``
  in sorter-native units. ``amplitudes.npy`` is a different observable and
  microvolt waveform amplitudes are a third; substituting either silently
  changes what "missingness" means. The source string travels in the result.
* **Never pool incompatible cluster amplitude scales.** A family assembled from
  clusters whose amplitude distributions sit at different scales produces a
  bimodal pooled distribution, and a truncated sigmoid fitted to it can report a
  *lower* missing percentage precisely because the fit is being driven by the
  larger mode. That is not recovery. When the contributors' scales differ by
  more than the contract's ratio, the family has no defensible amplitude
  measurement and completeness is reported ``unevaluable``.
* **``unevaluable`` is not a pass.** Neither is "too few fits". The verdict
  vocabulary keeps them distinct from ``fail`` so a reader can tell an endpoint
  that was measured and did not move from one that was never measurable.

Boundary-pinned (50%) fits are censored, not measured: they are counted and
reported but excluded from the median that the gate is applied to, exactly as
``pipeline.truncation.is_saturated`` documents.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from pipeline.truncation import (
    SATURATION_PCT,
    construct_windows,
    fit_amp_cdf,
    is_saturated,
)

COMPLETENESS_SCHEMA = "luke-candidate-completeness-qc-v1"

#: The one amplitude observable production QC fits. Recorded in every result.
PRODUCTION_AMPLITUDE_SOURCE = "full_st[kept_spikes][:, 2]"
PRODUCTION_AMPLITUDE_UNITS = "sorter_native"

VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_UNEVALUABLE = "unevaluable"


class CompletenessRefusal(ValueError):
    """The completeness endpoint refuses to produce a number. Never caught."""


@dataclass(frozen=True)
class CompletenessConfig:
    spikes_per_window: int = 1000
    max_isi_s: float = 10.0
    #: ``historical`` reproduces production exactly: ``construct_windows``
    #: stores inclusive endpoints but the fitter slices ``amps[i0:i1]``, so a
    #: nominal 1,000-spike window fits 999 values. ``exact`` fits
    #: ``amps[i0:i1+1]``. Both are retained; the gate uses whichever the
    #: contract declares, and the other is reported as a sensitivity.
    window_indexing: str = "historical"
    max_family_amplitude_scale_ratio: float = 1.25
    min_finite_interior_windows: int = 2

    def __post_init__(self) -> None:
        if self.window_indexing not in ("historical", "exact"):
            raise CompletenessRefusal(
                f"window_indexing must be 'historical' or 'exact', got {self.window_indexing!r}"
            )
        if self.spikes_per_window < 2:
            raise CompletenessRefusal("spikes_per_window must be at least 2")
        if not self.max_family_amplitude_scale_ratio >= 1.0:
            raise CompletenessRefusal("max_family_amplitude_scale_ratio must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _classify(popt: np.ndarray, missing_pct: float) -> str:
    if not np.all(np.isfinite(popt)) or not np.isfinite(missing_pct):
        return "nonfinite_fit"
    if bool(is_saturated(np.array([missing_pct]))[0]):
        return "boundary_pinned"
    return "finite_interior"


def train_completeness(
    samples: np.ndarray,
    amplitudes: np.ndarray,
    *,
    fs_hz: float,
    interval_s: tuple[float, float],
    config: CompletenessConfig,
    label: str = "",
) -> dict[str, Any]:
    """Fit production's truncation windows over one physical-time interval.

    Windows are constructed from the spikes *inside* the interval, identically
    for both arms, so the two are comparable without pairing windows by ordinal
    number. The interval's uncovered duration is reported rather than filled:
    a stretch with too few spikes to make a window is not 0% missing.
    """
    samples = np.asarray(samples)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    if samples.shape != amplitudes.shape:
        raise CompletenessRefusal(
            f"{label or 'train'}: {samples.size} samples against {amplitudes.size} amplitudes"
        )
    start_s, stop_s = float(interval_s[0]), float(interval_s[1])
    if not stop_s > start_s:
        raise CompletenessRefusal(f"interval must have stop > start, got {interval_s!r}")

    order = np.argsort(samples, kind="stable")
    seconds = samples[order].astype(np.float64) / float(fs_hz)
    inside = (seconds >= start_s) & (seconds < stop_s)
    times = seconds[inside]
    amps = amplitudes[order][inside]

    result: dict[str, Any] = {
        "label": label,
        "interval_s": [start_s, stop_s],
        "interval_duration_s": stop_s - start_s,
        "amplitude_source": PRODUCTION_AMPLITUDE_SOURCE,
        "amplitude_units": PRODUCTION_AMPLITUDE_UNITS,
        "window_indexing": config.window_indexing,
        "n_spikes_in_interval": int(times.size),
        "windows": [],
        "n_windows": 0,
        "n_finite_interior": 0,
        "n_boundary_pinned": 0,
        "n_nonfinite": 0,
        "missing_pct_median": None,
        "covered_duration_s": 0.0,
        "uncovered_duration_s": stop_s - start_s,
        "median_amplitude": float(np.median(amps)) if amps.size else None,
        "status": "no_windows",
    }
    if times.size < config.spikes_per_window:
        result["status"] = "too_few_spikes_for_one_window"
        return result

    window_blocks, _ = construct_windows(times, config.max_isi_s, config.spikes_per_window)
    if len(window_blocks) == 0:
        return result

    rows: list[dict[str, Any]] = []
    covered = 0.0
    for source_row, (i0, i1) in enumerate(np.atleast_2d(window_blocks)):
        i0, i1 = int(i0), int(i1)
        historical = amps[i0:i1]
        exact = amps[i0 : i1 + 1]
        fitted = historical if config.window_indexing == "historical" else exact
        popt, missing_pct = fit_amp_cdf(fitted)
        other_popt, other_missing = fit_amp_cdf(exact if config.window_indexing == "historical" else historical)
        status = _classify(np.asarray(popt), float(missing_pct))
        covered += float(times[i1] - times[i0])
        rows.append(
            {
                "source_row": source_row,
                "i0": i0,
                "i1": i1,
                "historical_count": i1 - i0,
                "nominal_count": i1 - i0 + 1,
                "first_sample_s": float(times[i0]),
                "last_sample_s": float(times[i1]),
                "missing_pct": float(missing_pct),
                "missing_pct_other_indexing": float(other_missing),
                "status": status,
            }
        )

    finite = [r["missing_pct"] for r in rows if r["status"] == "finite_interior"]
    result.update(
        {
            "windows": rows,
            "n_windows": len(rows),
            "n_finite_interior": len(finite),
            "n_boundary_pinned": sum(1 for r in rows if r["status"] == "boundary_pinned"),
            "n_nonfinite": sum(1 for r in rows if r["status"] == "nonfinite_fit"),
            "missing_pct_median": float(np.median(finite)) if finite else None,
            "covered_duration_s": covered,
            "uncovered_duration_s": max(0.0, (stop_s - start_s) - covered),
            "status": (
                "measured"
                if len(finite) >= config.min_finite_interior_windows
                else "insufficient_finite_interior_windows"
            ),
        }
    )
    return result


def family_amplitude_scale_check(
    contributor_medians: dict[int, float], config: CompletenessConfig
) -> dict[str, Any]:
    """Refuse to pool cluster amplitude scales that are not comparable.

    A single-cluster family is trivially compatible. For anything larger the
    ratio of the largest to the smallest contributor median must sit inside the
    contract's tolerance; otherwise the pooled distribution is a mixture and the
    truncated-sigmoid fit on it means nothing, however clean it looks.
    """
    medians = {int(c): float(m) for c, m in contributor_medians.items()}
    finite = {c: m for c, m in medians.items() if np.isfinite(m) and m > 0}
    if len(medians) <= 1:
        return {"contributor_medians": medians, "ratio": 1.0, "compatible": True,
                "reason": "single contributing cluster"}
    if len(finite) != len(medians):
        return {"contributor_medians": medians, "ratio": None, "compatible": False,
                "reason": "a contributing cluster has no usable amplitude median"}
    ratio = max(finite.values()) / min(finite.values())
    compatible = ratio <= config.max_family_amplitude_scale_ratio
    return {
        "contributor_medians": medians,
        "ratio": float(ratio),
        "compatible": bool(compatible),
        "reason": (
            "contributor amplitude scales are comparable"
            if compatible
            else (
                f"contributor amplitude medians differ by {ratio:.2f}x, above the contract's "
                f"{config.max_family_amplitude_scale_ratio}x limit: the pooled distribution is a "
                "mixture and a cleaner fit on it is not recovery"
            )
        ),
    }


def compare_completeness(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    scale_check: dict[str, Any],
    margin_pp: float,
    config: CompletenessConfig,
) -> dict[str, Any]:
    """Apply the contract's completeness gate, or decline to.

    Gate: ``baseline_missing_pct - candidate_missing_pct >= margin_pp``.
    Anything that stops the two numbers from meaning the same thing --
    incompatible pooled scales, too few finite-interior fits under either arm --
    produces ``unevaluable``, which never passes.
    """
    blockers: list[str] = []
    if not scale_check.get("compatible", False):
        blockers.append(f"family amplitude scales: {scale_check.get('reason')}")
    for name, arm in (("baseline", baseline), ("candidate", candidate)):
        if arm["status"] != "measured":
            blockers.append(
                f"{name} arm is {arm['status']} "
                f"({arm['n_finite_interior']} finite-interior fits, "
                f"{config.min_finite_interior_windows} required)"
            )

    if blockers:
        return {
            "verdict": VERDICT_UNEVALUABLE,
            "margin_pp": float(margin_pp),
            "baseline_missing_pct": baseline["missing_pct_median"],
            "candidate_missing_pct": candidate["missing_pct_median"],
            "improvement_pp": None,
            "blockers": blockers,
            "note": "`unevaluable` is not a pass; the gate was not applied.",
        }

    improvement = float(baseline["missing_pct_median"] - candidate["missing_pct_median"])
    return {
        "verdict": VERDICT_PASS if improvement >= margin_pp else VERDICT_FAIL,
        "margin_pp": float(margin_pp),
        "baseline_missing_pct": float(baseline["missing_pct_median"]),
        "candidate_missing_pct": float(candidate["missing_pct_median"]),
        "improvement_pp": improvement,
        "blockers": [],
        "note": (
            f"median over finite-interior windows only; {baseline['n_boundary_pinned']} baseline "
            f"and {candidate['n_boundary_pinned']} candidate windows were censored at "
            f"{SATURATION_PCT}% and excluded from the median."
        ),
    }
