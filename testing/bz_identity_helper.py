"""Bounded BV waveform-pair scoring with frozen training decisions.

Arrays are ``[event,time,channel]``. Positive lag means B is earlier than A,
so ``A[lag:]`` is compared with ``B[:-lag]``. Support, lag, B-to-A gain and
noise weights are selected from rest half 1 and frozen thereafter.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import numpy as np

METRICS = ("within_a_cosine", "within_b_cosine", "heldout_cross_cosine",
           "cross_deficit", "cross_minus_within_gap", "difference_cosine",
           "heldout_residual_rms")


@dataclass(frozen=True)
class FrozenPairState:
    support: np.ndarray
    noise_std: np.ndarray
    lag_samples: int
    b_to_a_gain: float
    half1_cross_cosine: float


def _events(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != 3 or min(value.shape) < 1:
        raise ValueError(f"{name} must have shape [event,time,channel]")
    return value


def _cosine(first, second):
    first, second = np.asarray(first).ravel(), np.asarray(second).ravel()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(first @ second / denominator) if denominator else np.nan


def _aligned(first, second, lag):
    if lag > 0:
        return first[lag:], second[:-lag]
    if lag < 0:
        return first[:lag], second[-lag:]
    return first, second


def _condition(template, state):
    selected = np.asarray(template, dtype=float)[:, state.support]
    selected -= selected.mean(axis=0, keepdims=True)
    return selected / state.noise_std[state.support][None]


def fit_training_state(a_half1, b_half1, *, noise_std, support=None,
                       max_lag=2, min_events=4, min_supported_channels=4):
    """Fit support/lag/gain using training arrays only."""
    a, b = _events(a_half1, "a_half1"), _events(b_half1, "b_half1")
    if a.shape[1:] != b.shape[1:]:
        raise ValueError("training arrays must share time/channel shape")
    if min(len(a), len(b)) < min_events:
        raise ValueError("insufficient_events")
    channels = a.shape[2]
    noise = np.asarray(noise_std, dtype=float)
    if noise.shape != (channels,):
        raise ValueError("noise_std must contain one value per channel")
    declared = np.ones(channels, bool) if support is None else np.asarray(support, bool)
    if declared.shape != (channels,):
        raise ValueError("support must contain one flag per channel")
    training_finite = np.isfinite(a).all((0, 1)) & np.isfinite(b).all((0, 1))
    used = declared & training_finite & np.isfinite(noise) & (noise > 0)
    if used.sum() < min_supported_channels:
        raise ValueError("insufficient_finite_training_support")
    provisional = FrozenPairState(used, noise.copy(), 0, 1.0, np.nan)
    ta = _condition(np.median(a, axis=0), provisional)
    tb = _condition(np.median(b, axis=0), provisional)
    best = None
    for lag in range(-max_lag, max_lag + 1):
        aa, bb = _aligned(ta, tb, lag)
        denominator = float(np.sum(bb * bb))
        gain = float(np.sum(aa * bb) / denominator) if denominator else np.nan
        cosine = _cosine(aa, gain * bb) if np.isfinite(gain) and gain > 0 else np.nan
        if np.isfinite(cosine) and (best is None or cosine > best[0]):
            best = cosine, lag, gain
    if best is None:
        raise ValueError("no_positive_finite_gain")
    return FrozenPairState(used.copy(), noise.copy(), int(best[1]), float(best[2]), float(best[0]))


def score_with_frozen_state(a_half1, b_half1, a_half2, b_half2, *, state, min_events=4):
    """Score all arrays using exactly one pre-fitted state."""
    arrays = [_events(x, name) for x, name in zip(
        (a_half1, b_half1, a_half2, b_half2),
        ("a_half1", "b_half1", "a_half2", "b_half2"))]
    shape = arrays[0].shape[1:]
    if any(x.shape[1:] != shape for x in arrays) or state.support.shape != (shape[1],):
        raise ValueError("arrays and frozen state must share time/channel shape")
    result = {
        "eligible": False, "status": "unresolved", "reason": "",
        "supported_channels": int(state.support.sum()),
        "support_fraction": float(state.support.mean()),
        "n_a_half1": len(arrays[0]), "n_b_half1": len(arrays[1]),
        "n_a_half2": len(arrays[2]), "n_b_half2": len(arrays[3]),
        "fitted_lag_samples": state.lag_samples,
        "fitted_b_to_a_gain": state.b_to_a_gain,
        "half1_cross_cosine": state.half1_cross_cosine,
    }
    if min(map(len, arrays)) < min_events:
        result["reason"] = "insufficient_events"
        return result
    if any(not np.isfinite(x[:, :, state.support]).all() for x in arrays):
        result["reason"] = "missing_frozen_support"
        return result
    a1, b1, a2, b2 = [_condition(np.median(x, axis=0), state) for x in arrays]
    a1, b1 = _aligned(a1, b1, state.lag_samples)
    a2, b2 = _aligned(a2, b2, state.lag_samples)
    within_a, within_b = _cosine(a1, a2), _cosine(b1, b2)
    cross = _cosine(a2, state.b_to_a_gain * b2)
    deficit = float(min(within_a, within_b) - cross)
    difference = _cosine(a1 - state.b_to_a_gain * b1,
                         a2 - state.b_to_a_gain * b2)
    result.update(
        eligible=True, status="descriptive_metrics_only",
        reason="requires_block_interval_for_final_label",
        within_a_cosine=float(within_a), within_b_cosine=float(within_b),
        heldout_cross_cosine=float(cross), cross_deficit=deficit,
        cross_minus_within_gap=float(-deficit), difference_cosine=float(difference),
        heldout_residual_rms=float(np.sqrt(np.mean((a2 - state.b_to_a_gain * b2) ** 2))))
    return result


def score_waveform_pair(a_half1, b_half1, a_half2, b_half2, *, noise_std=None,
                        support=None, frozen_state=None, max_lag=2, min_events=4,
                        min_supported_channels=4):
    """Fit once on half 1 unless a frozen state is supplied, then score."""
    try:
        state = frozen_state or fit_training_state(
            a_half1, b_half1, noise_std=np.asarray(noise_std), support=support,
            max_lag=max_lag, min_events=min_events,
            min_supported_channels=min_supported_channels)
    except ValueError as error:
        return {"eligible": False, "status": "unresolved", "reason": str(error)}
    return score_with_frozen_state(a_half1, b_half1, a_half2, b_half2,
                                   state=state, min_events=min_events)


def _common_domain(events, blocks, min_blocks):
    common1 = np.intersect1d(np.unique(blocks[0]), np.unique(blocks[1]))
    common2 = np.intersect1d(np.unique(blocks[2]), np.unique(blocks[3]))
    if min(len(common1), len(common2)) < min_blocks:
        raise ValueError("insufficient_common_blocks")
    selected = [events[0][np.isin(blocks[0], common1)],
                events[1][np.isin(blocks[1], common1)],
                events[2][np.isin(blocks[2], common2)],
                events[3][np.isin(blocks[3], common2)]]
    return selected, common1, common2


def _resample(events, blocks, draw):
    return np.concatenate([events[blocks == block] for block in draw], axis=0)


def temporal_block_bootstrap(a_half1, b_half1, a_half2, b_half2, *,
                             blocks_a_half1, blocks_b_half1,
                             blocks_a_half2, blocks_b_half2,
                             noise_std, support=None, n_bootstrap=200, seed=0,
                             min_common_blocks=10, frozen_state=None, **options):
    """Point and interval samples share one common-block domain and state."""
    events = [_events(x, "events") for x in (a_half1, b_half1, a_half2, b_half2)]
    blocks = [np.asarray(x) for x in (blocks_a_half1, blocks_b_half1,
                                      blocks_a_half2, blocks_b_half2)]
    if any(len(e) != len(b) for e, b in zip(events, blocks)):
        raise ValueError("each block vector must match its event array")
    selected, common1, common2 = _common_domain(events, blocks, min_common_blocks)
    sb = [blocks[0][np.isin(blocks[0], common1)], blocks[1][np.isin(blocks[1], common1)],
          blocks[2][np.isin(blocks[2], common2)], blocks[3][np.isin(blocks[3], common2)]]
    fit_keys = ("max_lag", "min_events", "min_supported_channels")
    state = frozen_state or fit_training_state(
        selected[0], selected[1], noise_std=noise_std, support=support,
        **{key: options[key] for key in fit_keys if key in options})
    min_events = int(options.get("min_events", 4))
    point = score_with_frozen_state(*selected, state=state, min_events=min_events)
    point.update(common_blocks_half1=len(common1), common_blocks_half2=len(common2))
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in METRICS}
    for _ in range(n_bootstrap):
        draw1 = rng.choice(common1, len(common1), replace=True)
        draw2 = rng.choice(common2, len(common2), replace=True)
        draw = [_resample(selected[0], sb[0], draw1), _resample(selected[1], sb[1], draw1),
                _resample(selected[2], sb[2], draw2), _resample(selected[3], sb[3], draw2)]
        # min_events qualifies the original common-domain estimand above. Every
        # common block contains at least one event from A and B; applying the
        # point gate again to variable-count draws would selectively discard
        # sparse draws and bias the conditional interval.
        score = score_with_frozen_state(*draw, state=state, min_events=1)
        if score["eligible"]:
            for metric in METRICS:
                samples[metric].append(score[metric])
    arrays = {key: np.asarray(value) for key, value in samples.items()}
    return {"point": point, "samples": arrays, "frozen_state": state,
            "common_blocks_half1": common1, "common_blocks_half2": common2,
            "accepted_replicates": len(arrays[METRICS[0]]),
            "requested_replicates": n_bootstrap}


def percentile_intervals(samples: Mapping[str, np.ndarray], alpha=0.05,
                         expected_replicates=None):
    """Intervals only for metrics finite in every expected block draw."""
    intervals = {}
    for key, value in samples.items():
        value = np.asarray(value, dtype=float)
        finite = value[np.isfinite(value)]
        expected = len(value) if expected_replicates is None else int(expected_replicates)
        if expected > 0 and len(value) == expected and len(finite) == expected:
            intervals[key] = (float(np.quantile(finite, alpha / 2)),
                              float(np.quantile(finite, 1 - alpha / 2)))
    return intervals


def qualify_with_intervals(point, intervals, *, bootstrap_accounting=None):
    """Apply BX's original 0.90, +/-0.03, 0.10 and 0.80 margins."""
    if not point.get("eligible"):
        return "unresolved"
    if bootstrap_accounting is None:
        return "unresolved"
    accepted = int(bootstrap_accounting.get("accepted_replicates", -1))
    requested = int(bootstrap_accounting.get("requested_replicates", -1))
    if requested <= 0 or accepted != requested:
        return "unresolved"

    def valid(key):
        if key not in intervals or len(intervals[key]) != 2:
            return False
        lo, hi = map(float, intervals[key])
        return np.isfinite(lo) and np.isfinite(hi) and lo <= hi

    reliability_keys = ("within_a_cosine", "within_b_cosine")
    if not all(valid(key) for key in reliability_keys):
        return "unresolved"
    low = {key: float(intervals[key][0]) for key in reliability_keys}
    reliable = low["within_a_cosine"] >= 0.90 and low["within_b_cosine"] >= 0.90
    if valid("cross_minus_within_gap"):
        gap_low, gap_high = map(float, intervals["cross_minus_within_gap"])
    else:
        gap_low = gap_high = np.nan
    if reliable and gap_low >= -0.03 and gap_high <= 0.03:
        return "limited_waveform_compatibility"
    if not valid("cross_deficit") or not valid("difference_cosine"):
        return "unresolved"
    if (reliable and float(intervals["cross_deficit"][0]) > 0.10
            and float(intervals["difference_cosine"][0]) >= 0.80):
        return "stable_waveform_difference"
    if not valid("cross_minus_within_gap"):
        return "unresolved"
    return "inconclusive"


def score_rest_and_episode(rest: Mapping[str, object], episode: Mapping[str, object] | None = None):
    """Fit on rest half 1 once; never refit state for episodes."""
    state = fit_training_state(rest["a_half1"], rest["b_half1"],
        noise_std=rest["noise_std"], support=rest.get("support"),
        max_lag=int(rest.get("max_lag", 2)), min_events=int(rest.get("min_events", 4)),
        min_supported_channels=int(rest.get("min_supported_channels", 4)))
    rest_score = score_with_frozen_state(rest["a_half1"], rest["b_half1"],
        rest["a_half2"], rest["b_half2"], state=state,
        min_events=int(rest.get("min_events", 4)))
    episode_score = ({"eligible": False, "status": "unresolved", "reason": "not_supplied"}
                     if episode is None else score_with_frozen_state(
                         episode["a_half1"], episode["b_half1"],
                         episode["a_half2"], episode["b_half2"], state=state,
                         min_events=int(episode.get("min_events", rest.get("min_events", 4)))))
    return {"rest": rest_score, "episode": episode_score, "frozen_state": state}


def aggregate_parent_votes(rows: Sequence[Mapping[str, object]], metric="heldout_cross_cosine"):
    """Give each unique incident parent one vote; same-parent pairs add once."""
    by_parent = {}
    for row in rows:
        if not bool(row.get("eligible", True)):
            continue
        value = float(row[metric])
        if not np.isfinite(value):
            continue
        for parent in {str(row["parent_a"]), str(row["parent_b"])}:
            by_parent.setdefault(parent, []).append(value)
    parent_values = {key: float(np.median(value)) for key, value in by_parent.items()}
    return {"metric": metric, "parent_count": len(parent_values),
            "parent_values": parent_values,
            "parent_equal_weight_median": (float(np.median(list(parent_values.values())))
                                             if parent_values else np.nan)}
