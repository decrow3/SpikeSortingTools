"""Bounded waveform-pair scoring for the BZ/BV identity diagnostic.

This module deliberately does not extract voltage or choose pairs.  It accepts
small waveform arrays prepared by the BV worker.  Events are ``[event,time,
channel]``.  A positive lag means that B is earlier than A by ``lag`` samples,
so A[lag:] is compared with B[:-lag].  Lag and B-to-A gain are fitted on half 1
and held fixed for every half-2 score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


METRICS = (
    "within_a_cosine",
    "within_b_cosine",
    "heldout_cross_cosine",
    "cross_deficit",
    "difference_cosine",
    "heldout_residual_rms",
)


def _events(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.ndim != 3 or min(value.shape) < 1:
        raise ValueError(f"{name} must have shape [event,time,channel]")
    return value


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float).ravel()
    second = np.asarray(second, dtype=float).ravel()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(first @ second / denominator) if denominator else np.nan


def _aligned(first: np.ndarray, second: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        return first[lag:], second[:-lag]
    if lag < 0:
        return first[:lag], second[-lag:]
    return first, second


def _condition(template: np.ndarray, noise_std: np.ndarray, support: np.ndarray) -> np.ndarray:
    selected = np.asarray(template, dtype=float)[:, support]
    selected = selected - selected.mean(axis=0, keepdims=True)
    return selected / noise_std[support][None, :]


def score_waveform_pair(
    a_half1: np.ndarray,
    b_half1: np.ndarray,
    a_half2: np.ndarray,
    b_half2: np.ndarray,
    *,
    noise_std: np.ndarray,
    support: np.ndarray | None = None,
    max_lag: int = 2,
    min_events: int = 4,
    min_supported_channels: int = 4,
) -> dict[str, float | int | str | bool]:
    """Score one preselected pair; never interpret the result as identity truth."""
    arrays = [_events(x, name) for x, name in zip(
        (a_half1, b_half1, a_half2, b_half2),
        ("a_half1", "b_half1", "a_half2", "b_half2"),
    )]
    shape = arrays[0].shape[1:]
    if any(x.shape[1:] != shape for x in arrays):
        raise ValueError("all event arrays must have the same time/channel shape")
    noise_std = np.asarray(noise_std, dtype=float)
    if noise_std.shape != (shape[1],):
        raise ValueError("noise_std must contain one value per channel")
    if support is None:
        support = np.ones(shape[1], dtype=bool)
    support = np.asarray(support, dtype=bool)
    if support.shape != (shape[1],):
        raise ValueError("support must contain one flag per channel")
    valid_noise = np.isfinite(noise_std) & (noise_std > 0)
    finite_channels = np.ones(shape[1], dtype=bool)
    for value in arrays:
        finite_channels &= np.isfinite(value).all(axis=(0, 1))
    used = support & valid_noise & finite_channels
    result: dict[str, float | int | str | bool] = {
        "eligible": False,
        "status": "unresolved",
        "reason": "",
        "supported_channels": int(used.sum()),
        "support_fraction": float(used.mean()),
        "n_a_half1": int(arrays[0].shape[0]),
        "n_b_half1": int(arrays[1].shape[0]),
        "n_a_half2": int(arrays[2].shape[0]),
        "n_b_half2": int(arrays[3].shape[0]),
    }
    if min(x.shape[0] for x in arrays) < min_events:
        result["reason"] = "insufficient_events"
        return result
    if used.sum() < min_supported_channels:
        result["reason"] = "insufficient_finite_support"
        return result

    a1, b1, a2, b2 = [
        _condition(np.median(value, axis=0), noise_std, used) for value in arrays
    ]
    best: tuple[float, int, float] | None = None
    for lag in range(-max_lag, max_lag + 1):
        aa, bb = _aligned(a1, b1, lag)
        denominator = float(np.sum(bb * bb))
        gain = float(np.sum(aa * bb) / denominator) if denominator else np.nan
        cosine = _cosine(aa, gain * bb) if np.isfinite(gain) and gain > 0 else np.nan
        if np.isfinite(cosine) and (best is None or cosine > best[0]):
            best = (cosine, lag, gain)
    if best is None:
        result["reason"] = "no_positive_finite_gain"
        return result

    _, lag, gain = best
    a1c, b1c = _aligned(a1, b1, lag)
    a2c, b2c = _aligned(a2, b2, lag)
    within_a = _cosine(a1c, a2c)
    within_b = _cosine(b1c, b2c)
    heldout_cross = _cosine(a2c, gain * b2c)
    difference_cosine = _cosine(a1c - gain * b1c, a2c - gain * b2c)
    residual_rms = float(np.sqrt(np.mean((a2c - gain * b2c) ** 2)))
    within_reference = min(within_a, within_b)
    deficit = float(within_reference - heldout_cross)
    if within_reference >= 0.90 and deficit <= 0.03:
        label = "limited_waveform_compatibility"
    elif deficit > 0.10 and difference_cosine >= 0.80:
        label = "stable_waveform_difference"
    else:
        label = "inconclusive"
    result.update(
        eligible=True,
        status=label,
        reason="",
        fitted_lag_samples=int(lag),
        fitted_b_to_a_gain=float(gain),
        half1_cross_cosine=float(best[0]),
        within_a_cosine=float(within_a),
        within_b_cosine=float(within_b),
        heldout_cross_cosine=float(heldout_cross),
        cross_deficit=deficit,
        difference_cosine=float(difference_cosine),
        heldout_residual_rms=residual_rms,
    )
    return result


def _resample_blocks(events: np.ndarray, blocks: np.ndarray, sampled: np.ndarray) -> np.ndarray:
    pieces = [events[blocks == block] for block in sampled]
    return np.concatenate(pieces, axis=0)


def temporal_block_bootstrap(
    a_half1: np.ndarray,
    b_half1: np.ndarray,
    a_half2: np.ndarray,
    b_half2: np.ndarray,
    *,
    blocks_a_half1: np.ndarray,
    blocks_b_half1: np.ndarray,
    blocks_a_half2: np.ndarray,
    blocks_b_half2: np.ndarray,
    noise_std: np.ndarray,
    support: np.ndarray | None = None,
    n_bootstrap: int = 200,
    seed: int = 0,
    **score_options: object,
) -> dict[str, np.ndarray]:
    """Bootstrap shared 5-s temporal blocks, never individual events."""
    events = [np.asarray(x) for x in (a_half1, b_half1, a_half2, b_half2)]
    blocks = [np.asarray(x) for x in (
        blocks_a_half1, blocks_b_half1, blocks_a_half2, blocks_b_half2
    )]
    if any(len(e) != len(b) for e, b in zip(events, blocks)):
        raise ValueError("each block vector must match its event array")
    common1 = np.intersect1d(np.unique(blocks[0]), np.unique(blocks[1]))
    common2 = np.intersect1d(np.unique(blocks[2]), np.unique(blocks[3]))
    if not len(common1) or not len(common2):
        raise ValueError("each half needs at least one A/B-shared temporal block")
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in METRICS}
    values.update(fitted_lag_samples=[], fitted_b_to_a_gain=[])
    accepted = 0
    for _ in range(n_bootstrap):
        draw1 = rng.choice(common1, size=len(common1), replace=True)
        draw2 = rng.choice(common2, size=len(common2), replace=True)
        sampled = [
            _resample_blocks(events[0], blocks[0], draw1),
            _resample_blocks(events[1], blocks[1], draw1),
            _resample_blocks(events[2], blocks[2], draw2),
            _resample_blocks(events[3], blocks[3], draw2),
        ]
        score = score_waveform_pair(
            *sampled, noise_std=noise_std, support=support, **score_options
        )
        if not score["eligible"]:
            continue
        accepted += 1
        for metric in values:
            values[metric].append(score[metric])
    result = {key: np.asarray(value) for key, value in values.items()}
    result["accepted_replicates"] = np.asarray(accepted)
    result["requested_replicates"] = np.asarray(n_bootstrap)
    return result


def score_rest_and_episode(
    rest: Mapping[str, object], episode: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Keep rest primary and report episode evidence as a separate eligibility stratum."""
    output: dict[str, object] = {"rest": score_waveform_pair(**rest)}
    if episode is None:
        output["episode"] = {"eligible": False, "status": "unresolved", "reason": "not_supplied"}
    else:
        output["episode"] = score_waveform_pair(**episode)
    return output


def aggregate_parent_votes(
    rows: Sequence[Mapping[str, object]], metric: str = "heldout_cross_cosine"
) -> dict[str, object]:
    """Give every parent one vote after median-combining its incident pairs."""
    by_parent: dict[str, list[float]] = {}
    for row in rows:
        if not bool(row.get("eligible", True)):
            continue
        value = float(row[metric])
        if not np.isfinite(value):
            continue
        for key in ("parent_a", "parent_b"):
            parent = str(row[key])
            by_parent.setdefault(parent, []).append(value)
    parent_values = {key: float(np.median(value)) for key, value in by_parent.items()}
    return {
        "metric": metric,
        "parent_count": len(parent_values),
        "parent_values": parent_values,
        "parent_equal_weight_median": (
            float(np.median(list(parent_values.values()))) if parent_values else np.nan
        ),
    }
