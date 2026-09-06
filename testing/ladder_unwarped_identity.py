"""Unwarped motion-aware identity handling (Option B candidate).

This is a **retained-sort replay**, not a re-sort. The accepted recording, the
spike times, the original cluster labels and the production amplitudes are read
back unchanged; nothing here detects, fits templates, or writes into a
production output. What it does is re-group *existing* spike rows into
longitudinal identity families:

1. partition the retained spike rows into overlapping time epochs on a grid
   anchored to the recording clock;
2. summarise each ``(epoch, original cluster)`` observation -- spike rows,
   tissue-frame depth, production amplitude, and a mean waveform expressed on
   the probe's **physical channels**;
3. propose links between observations in adjacent epochs, gated on spatial
   proximity, amplitude ratio, and waveform cosine over a shared physical
   channel neighbourhood;
4. keep only links that are exclusive in *both* directions and unambiguous on
   *both* sides -- anything contested is left separate rather than guessed;
5. union the surviving **cross-cluster** links into families over the original
   cluster ids, and re-check refractory cleanliness on the *exported* train each
   merge produces, pruning merge links until every merged train is clean.

Four properties this module is written to hold, each with a known-answer
fixture in ``testing/test_ladder_unwarped_identity.py``:

* **The input partition is preserved unless a merge is justified.** This is the
  invariant v1 lacked and failed on. Epochs are *observations used to evaluate
  links*; they are not output units. Only an accepted link **between two
  different original clusters** changes the partition. With no such link, the
  exported assignment reproduces the input exactly, up to family renumbering --
  a cluster whose epoch-to-epoch continuity link is refused stays one cluster,
  and a row in no eligible epoch at all stays with its cluster. Splitting a
  cluster is not an operation this module has; adding one would need its own
  gates and its own fixtures.
* **Dedup is by original event identity.** Two distinct spikes may share a
  sample; collapsing on timestamps would delete one of them and hide the
  coincidence from every downstream contamination check. Simultaneous distinct
  rows both survive and both remain visible to the refractory check.
* **Exclusivity runs both ways.** A source observation may claim at most one
  destination and a destination may be claimed by at most one source.
* **Refractory is validated on the exported train, incrementally.** Not on an
  epoch-masked union and not on an anchor: on the rows the export writes. The
  gated quantity is the *increase* a merge causes over the worst contributing
  cluster's own baseline, because that is what "this link joined two neurons"
  means. Preserving an imperfect input cluster is not certifying it -- a
  single-cluster family is never pruned and never declared clean -- and a high
  baseline never licenses a merge on its own: the increment gate supplements
  the depth, amplitude and waveform evidence, it does not replace it.

Motion is never inferred. :class:`MotionDeclaration` has no default: a caller
either supplies a qualified field's displacement or declares its absence
explicitly, and "declared absent" is recorded as a limitation on the run rather
than passed off as zero motion.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


UNWARPED_IDENTITY_SCHEMA = "luke-unwarped-identity-v2"
MANIFEST_NAME = "unwarped_identity_manifest.json"

#: Motion handling is a declared choice, never a fallback.
MOTION_QUALIFIED_FIELD = "qualified_field"
MOTION_DECLARED_ABSENT = "declared_absent"
MOTION_MODES = (MOTION_QUALIFIED_FIELD, MOTION_DECLARED_ABSENT)

#: Why a candidate link was refused. Recorded per rejected pair so a run that
#: links nothing says which gate did it rather than "no links".
REJECTION_REASONS = (
    "self_link_no_output_effect",
    "spatial_distance",
    "amplitude_ratio",
    "waveform_cosine",
    "waveform_unavailable",
    "epoch_pair_refractory_increase",
    "ambiguous_source",
    "ambiguous_destination",
    "source_already_claimed",
    "destination_already_claimed",
    "exported_train_refractory_increase",
)

#: A link between two observations of the *same* original cluster. It is
#: proposed and measured, because its metrics are the evidence that a cluster is
#: continuous, but it can never be accepted and never changes the partition:
#: under the preservation invariant a cluster is already one unit, so there is
#: nothing for a self-link to join. In v1 these were gated like merges, and
#: refusing them fragmented 659 clusters into 2,038 units.
SELF_LINK = "self_link_no_output_effect"


class IdentityRefusal(ValueError):
    """The replay refuses to proceed. Never caught internally."""


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UnwarpedIdentityConfig:
    """Link-gate constants. Mirrors ``candidate.settings.value
    .resolved_configuration.identity_link`` in the delivery contract; the
    runner builds this *from* the contract rather than from a CLI flag."""

    epoch_duration_s: float = 120.0
    epoch_overlap_s: float = 30.0
    epoch_grid_origin_s: float = 0.0
    min_spikes_per_epoch: int = 10
    max_spatial_distance_um: float = 30.0
    max_amplitude_ratio: float = 2.0
    min_waveform_cosine: float = 0.9
    waveform_channel_neighbourhood_um: float = 60.0
    ambiguity_threshold_ratio: float = 0.85
    #: The largest refractory-violation-fraction *increase* a merge may cause
    #: over the worst contributing cluster's own baseline. An absolute cap here
    #: was the v1 defect: applied to units whose baseline already exceeds it, it
    #: is unsatisfiable, and in v1 it fragmented the sort rather than merging it.
    max_refractory_violation_increase: float = 0.01
    refractory_period_ms: float = 1.5

    def __post_init__(self) -> None:
        if not self.epoch_duration_s > self.epoch_overlap_s >= 0:
            raise IdentityRefusal(
                "epoch_duration_s must exceed epoch_overlap_s and neither may be negative; "
                f"got {self.epoch_duration_s} and {self.epoch_overlap_s}"
            )
        if not 0.0 <= self.min_waveform_cosine <= 1.0:
            raise IdentityRefusal("min_waveform_cosine must lie in [0, 1]")
        if not self.max_amplitude_ratio >= 1.0:
            raise IdentityRefusal("max_amplitude_ratio must be >= 1")
        if not 0.0 < self.ambiguity_threshold_ratio <= 1.0:
            raise IdentityRefusal("ambiguity_threshold_ratio must lie in (0, 1]")
        if self.max_refractory_violation_increase < 0:
            raise IdentityRefusal("max_refractory_violation_increase must be >= 0")
        if self.refractory_period_ms <= 0:
            raise IdentityRefusal("refractory_period_ms must be positive")

    @property
    def epoch_step_s(self) -> float:
        return self.epoch_duration_s - self.epoch_overlap_s

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class MotionDeclaration:
    """An explicit statement of what motion this arm applied.

    There is no default and no fallback. ``qualified_field`` requires a
    per-spike displacement the caller obtained from a
    ``qualified-motion-field-v1`` artifact plus that artifact's identity;
    ``declared_absent`` means the arm ran unregistered, is not motion-aware,
    and says so in its own manifest.
    """

    mode: str
    displacement_um: np.ndarray | None = None
    field_identity: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.mode not in MOTION_MODES:
            raise IdentityRefusal(f"motion mode must be one of {MOTION_MODES}, got {self.mode!r}")
        if self.mode == MOTION_QUALIFIED_FIELD:
            if self.displacement_um is None:
                raise IdentityRefusal(
                    "motion mode 'qualified_field' requires a per-spike displacement from an "
                    "identified qualified-motion-field-v1 artifact. Absent motion is never "
                    "silently substituted with zero displacement."
                )
            if not self.field_identity.get("sha256"):
                raise IdentityRefusal(
                    "motion mode 'qualified_field' requires the field's sha256 in field_identity: "
                    "a motion-aware arm must name the motion field it consumed."
                )
        elif self.displacement_um is not None:
            raise IdentityRefusal(
                "motion mode 'declared_absent' must not carry a displacement array; declare "
                "'qualified_field' if a field was actually applied."
            )

    @property
    def is_motion_aware(self) -> bool:
        return self.mode == MOTION_QUALIFIED_FIELD

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "motion_aware": self.is_motion_aware,
            "field_identity": dict(self.field_identity),
            "rationale": self.rationale,
        }


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReplayInput:
    """The retained rows this replay regroups, on the recording clock.

    ``row_id`` is the original 0-based index into the sort's
    ``spike_times.npy``. It is the event identity everything downstream
    deduplicates on; ``sample`` is a *measurement*, and two distinct rows are
    allowed to share one.
    """

    row_id: np.ndarray
    sample: np.ndarray
    cluster: np.ndarray
    depth_um: np.ndarray
    amplitude: np.ndarray
    template: np.ndarray
    template_bank: np.ndarray
    channel_positions_um: np.ndarray
    fs_hz: float

    def __post_init__(self) -> None:
        n = self.row_id.size
        for name in ("sample", "cluster", "depth_um", "amplitude", "template"):
            arr = getattr(self, name)
            if arr.ndim != 1 or arr.size != n:
                raise IdentityRefusal(f"{name} must be a 1-D array of length {n}, got {arr.shape}")
        if np.unique(self.row_id).size != n:
            raise IdentityRefusal("row_id must be unique: it is the original event identity")
        if not np.issubdtype(self.sample.dtype, np.integer):
            raise IdentityRefusal(
                "sample must be integer samples of the recording clock; converting to seconds "
                "before this point loses the clock the export has to preserve"
            )
        if not np.all(np.isfinite(self.depth_um)):
            raise IdentityRefusal(
                "depth_um must be finite real depths. A zero-filled or missing depth array is a "
                "refusal, not a depth of zero."
            )
        if not np.all(np.isfinite(self.amplitude)):
            raise IdentityRefusal("amplitude must be finite")
        if self.template_bank.ndim != 3:
            raise IdentityRefusal(
                "template_bank must be (n_templates, n_samples, n_channels) on the probe's "
                "physical channels"
            )
        n_templates, _, n_channels = self.template_bank.shape
        if self.template.size and (
            int(self.template.min()) < 0 or int(self.template.max()) >= n_templates
        ):
            raise IdentityRefusal("template ids are out of range for template_bank")
        if self.channel_positions_um.shape != (n_channels, 2):
            raise IdentityRefusal(
                f"channel_positions_um must be ({n_channels}, 2) to give the template bank a "
                "physical channel representation"
            )
        if not self.fs_hz > 0:
            raise IdentityRefusal("fs_hz must be positive")

    def seconds(self) -> np.ndarray:
        """Seconds derived from the clock, never stored in place of it."""
        return self.sample.astype(np.float64) / float(self.fs_hz)


# --------------------------------------------------------------------------- #
# refractory
# --------------------------------------------------------------------------- #
def refractory_violation_fraction(
    samples: np.ndarray, fs_hz: float, refractory_period_ms: float
) -> float:
    """Fraction of consecutive intervals shorter than the refractory period.

    Duplicated *timestamps* are preserved, so two distinct rows recorded at the
    same sample contribute an interval of 0 and count as a violation. That is
    the whole point: a merge that stacks simultaneous spikes must be visible
    here rather than deduplicated out of sight.
    """
    samples = np.asarray(samples)
    if samples.size <= 1:
        return 0.0
    ordered = np.sort(samples)
    intervals_s = np.diff(ordered).astype(np.float64) / float(fs_hz)
    threshold_s = refractory_period_ms / 1000.0
    return float(np.count_nonzero(intervals_s < threshold_s) / intervals_s.size)


# --------------------------------------------------------------------------- #
# epoch observations
# --------------------------------------------------------------------------- #
@dataclass
class EpochObservation:
    epoch_idx: int
    cluster_id: int
    start_s: float
    stop_s: float
    num_spikes: int
    firing_rate_hz: float
    mean_observed_depth_um: float
    mean_tissue_depth_um: float
    median_amplitude: float
    refractory_violation_fraction: float
    peak_channel: int
    #: Original row ids, ascending. Not written to CSV; the identity carrier.
    row_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    #: Mean waveform on the probe's physical channels, (n_samples, n_channels).
    mean_waveform: np.ndarray | None = None

    @property
    def key(self) -> tuple[int, int]:
        return (self.epoch_idx, self.cluster_id)

    def to_row(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in asdict(self).items()
            if k not in ("row_ids", "mean_waveform")
        }


CSV_OBSERVATION_FIELDS = (
    "epoch_idx",
    "cluster_id",
    "start_s",
    "stop_s",
    "num_spikes",
    "firing_rate_hz",
    "mean_observed_depth_um",
    "mean_tissue_depth_um",
    "median_amplitude",
    "refractory_violation_fraction",
    "peak_channel",
)


def epoch_bounds(epoch_idx: int, config: UnwarpedIdentityConfig) -> tuple[float, float]:
    start = config.epoch_grid_origin_s + epoch_idx * config.epoch_step_s
    return start, start + config.epoch_duration_s


def epochs_covering(
    interval_s: tuple[float, float], config: UnwarpedIdentityConfig
) -> list[int]:
    """Grid epochs lying wholly inside ``interval_s``.

    The grid is anchored to the recording clock, not to the interval, so the
    same spike lands in the same epoch index whichever bounded run reaches it.
    Only whole epochs are used: a partial epoch would summarise a shorter span
    and silently change every rate and count it feeds into a gate.
    """
    start, stop = float(interval_s[0]), float(interval_s[1])
    if not stop > start:
        raise IdentityRefusal(f"processing interval must have stop > start, got {interval_s!r}")
    step, origin = config.epoch_step_s, config.epoch_grid_origin_s
    first = int(np.ceil((start - origin) / step))
    out: list[int] = []
    idx = first
    while True:
        lo, hi = epoch_bounds(idx, config)
        if lo < start:
            idx += 1
            continue
        if hi > stop:
            break
        out.append(idx)
        idx += 1
    return out


def _mean_waveform(
    template_ids: np.ndarray, template_bank: np.ndarray
) -> tuple[np.ndarray, int]:
    """Count-weighted mean template on the full physical channel set."""
    present, counts = np.unique(template_ids, return_counts=True)
    weights = counts.astype(np.float64) / counts.sum()
    waveform = np.tensordot(weights, template_bank[present], axes=(0, 0))
    peak_to_peak = waveform.max(axis=0) - waveform.min(axis=0)
    return waveform, int(np.argmax(peak_to_peak))


def extract_epoch_observations(
    inputs: ReplayInput,
    epoch_indices: list[int],
    tissue_depth_um: np.ndarray,
    config: UnwarpedIdentityConfig,
) -> list[EpochObservation]:
    """Summarise every ``(epoch, original cluster)`` observation.

    A row inside the overlap belongs to two observations here. That is
    deliberate -- links are proposed between epochs, so the boundary rows have
    to be visible in both -- and is reconciled once, on ``row_id``, in
    :func:`assign_rows_to_families`.
    """
    seconds = inputs.seconds()
    observations: list[EpochObservation] = []
    for epoch_idx in epoch_indices:
        lo, hi = epoch_bounds(epoch_idx, config)
        in_epoch = np.flatnonzero((seconds >= lo) & (seconds < hi))
        if in_epoch.size == 0:
            continue
        # group by cluster with one sort rather than one pass per cluster: on a
        # full probe the latter is (clusters x spikes) and dominates the run
        order = np.argsort(inputs.cluster[in_epoch], kind="stable")
        in_epoch = in_epoch[order]
        clusters_here = inputs.cluster[in_epoch]
        boundaries = np.flatnonzero(np.diff(clusters_here)) + 1
        for rows in np.split(in_epoch, boundaries):
            if rows.size < config.min_spikes_per_epoch:
                continue
            cid = int(inputs.cluster[rows[0]])
            waveform, peak_channel = _mean_waveform(
                inputs.template[rows], inputs.template_bank
            )
            observations.append(
                EpochObservation(
                    epoch_idx=epoch_idx,
                    cluster_id=int(cid),
                    start_s=float(lo),
                    stop_s=float(hi),
                    num_spikes=int(rows.size),
                    firing_rate_hz=float(rows.size / config.epoch_duration_s),
                    mean_observed_depth_um=float(np.mean(inputs.depth_um[rows])),
                    mean_tissue_depth_um=float(np.mean(tissue_depth_um[rows])),
                    median_amplitude=float(np.median(inputs.amplitude[rows])),
                    refractory_violation_fraction=refractory_violation_fraction(
                        inputs.sample[rows], inputs.fs_hz, config.refractory_period_ms
                    ),
                    peak_channel=peak_channel,
                    row_ids=np.sort(inputs.row_id[rows]),
                    mean_waveform=waveform,
                )
            )
    return observations


# --------------------------------------------------------------------------- #
# waveform compatibility on a common physical channel representation
# --------------------------------------------------------------------------- #
def channel_neighbourhoods(
    channel_positions_um: np.ndarray, neighbourhood_um: float
) -> np.ndarray:
    """``(n_channels, n_channels)`` mask of which channels sit near which.

    Precomputed once per run: the shared channel set is asked for on every
    candidate pair, and on a 384-site probe recomputing it from coordinates
    each time is the difference between seconds and hours.
    """
    positions = np.asarray(channel_positions_um, dtype=np.float64)
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    return distances <= neighbourhood_um


def shared_physical_channels(
    peak_a: int,
    peak_b: int,
    channel_positions_um: np.ndarray,
    neighbourhood_um: float,
    neighbourhoods: np.ndarray | None = None,
) -> np.ndarray:
    """Channels physically near either peak, as indices into the full probe.

    Two clusters generally carry their energy on different channel subsets, and
    each sort's own sparse channel list is indexed differently. Comparing them
    on a neighbourhood defined by ``channel_positions_um`` -- micrometres on the
    probe -- is what makes the two waveforms the same kind of object.
    """
    if neighbourhoods is None:
        neighbourhoods = channel_neighbourhoods(channel_positions_um, neighbourhood_um)
    return np.flatnonzero(neighbourhoods[peak_a] | neighbourhoods[peak_b])


def waveform_cosine(
    obs_a: EpochObservation,
    obs_b: EpochObservation,
    channel_positions_um: np.ndarray,
    neighbourhood_um: float,
    neighbourhoods: np.ndarray | None = None,
) -> float | None:
    """Cosine similarity of two mean waveforms on their shared channel set.

    ``None`` when either waveform is missing or has no energy on the shared
    channels: unavailable waveform evidence refuses the link, it does not pass
    it.
    """
    if obs_a.mean_waveform is None or obs_b.mean_waveform is None:
        return None
    channels = shared_physical_channels(
        obs_a.peak_channel, obs_b.peak_channel, channel_positions_um, neighbourhood_um,
        neighbourhoods,
    )
    if channels.size == 0:
        return None
    a = obs_a.mean_waveform[:, channels].ravel()
    b = obs_b.mean_waveform[:, channels].ravel()
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not norm > 0:
        return None
    return float(np.dot(a, b) / norm)


# --------------------------------------------------------------------------- #
# candidate links
# --------------------------------------------------------------------------- #
@dataclass
class CandidateLink:
    epoch_a: int
    cluster_a: int
    epoch_b: int
    cluster_b: int
    spatial_distance_um: float
    amplitude_ratio: float
    waveform_cosine: float
    pair_refractory_fraction: float
    pair_refractory_increase: float
    link_score: float
    is_merge: bool = False
    accepted: bool = False
    rejected_because: str = ""

    @property
    def source(self) -> tuple[int, int]:
        return (self.epoch_a, self.cluster_a)

    @property
    def destination(self) -> tuple[int, int]:
        return (self.epoch_b, self.cluster_b)

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


CSV_LINK_FIELDS = tuple(CandidateLink.__annotations__.keys())


def _link_score(distance_um: float, amplitude_ratio: float, cosine: float) -> float:
    """Higher is better. Bounded in (0, 1] so ambiguity ratios are meaningful."""
    penalty = distance_um / 10.0 + (amplitude_ratio - 1.0) + 10.0 * (1.0 - cosine)
    return 1.0 / (1.0 + penalty)


def build_candidate_links(
    observations: list[EpochObservation],
    inputs: ReplayInput,
    config: UnwarpedIdentityConfig,
    counters: dict[str, int] | None = None,
) -> list[CandidateLink]:
    """Propose and adjudicate links between adjacent-epoch observations.

    Every proposal that is spatially plausible is returned, accepted or not,
    carrying the gate that refused it: a run that links nothing must be able to
    say *why* it linked nothing. Pairs the depth gate rejects are only counted,
    into ``counters`` -- on a full probe there are millions of them and they
    carry no information beyond "far apart".

    Gates are applied cheapest-first and short-circuit, so the waveform cosine
    and the refractory union are computed only for pairs still in contention.
    """
    counters = {} if counters is None else counters
    counters.setdefault("pairs_considered", 0)
    counters.setdefault("spatial_distance", 0)

    by_epoch: dict[int, list[EpochObservation]] = {}
    for obs in observations:
        by_epoch.setdefault(obs.epoch_idx, []).append(obs)

    order = np.argsort(inputs.row_id, kind="stable")
    sorted_rows = inputs.row_id[order]
    sorted_samples = inputs.sample[order]
    neighbourhoods = channel_neighbourhoods(
        inputs.channel_positions_um, config.waveform_channel_neighbourhood_um
    )

    def samples_of(row_ids: np.ndarray) -> np.ndarray:
        return sorted_samples[np.searchsorted(sorted_rows, row_ids)]

    links: list[CandidateLink] = []
    for epoch_a in sorted(by_epoch):
        epoch_b = epoch_a + 1
        if epoch_b not in by_epoch:
            continue
        list_a, list_b = by_epoch[epoch_a], by_epoch[epoch_b]

        depths_a = np.array([o.mean_tissue_depth_um for o in list_a])
        depths_b = np.array([o.mean_tissue_depth_um for o in list_b])
        distances = np.abs(depths_a[:, None] - depths_b[None, :])
        close = distances <= config.max_spatial_distance_um
        counters["pairs_considered"] += int(distances.size)
        counters["spatial_distance"] += int(distances.size - np.count_nonzero(close))

        proposals: list[CandidateLink] = []
        for index_a, index_b in zip(*np.nonzero(close)):
            obs_a, obs_b = list_a[index_a], list_b[index_b]
            distance = float(distances[index_a, index_b])

            amp_hi = max(obs_a.median_amplitude, obs_b.median_amplitude)
            amp_lo = min(obs_a.median_amplitude, obs_b.median_amplitude)
            ratio = float("inf") if amp_lo <= 0 else amp_hi / amp_lo
            cosine: float | None = None
            pair_rvf = float("nan")

            is_merge = obs_a.cluster_id != obs_b.cluster_id
            pair_increase = float("nan")

            # A same-cluster pair is measured but never gated: the two
            # observations are already one output unit, so there is nothing for
            # the link to join and refusing it must not split anything.
            reason = "" if is_merge else SELF_LINK
            if reason:
                cosine = waveform_cosine(
                    obs_a, obs_b, inputs.channel_positions_um,
                    config.waveform_channel_neighbourhood_um, neighbourhoods,
                )
                union_rows = np.union1d(obs_a.row_ids, obs_b.row_ids)
                pair_rvf = refractory_violation_fraction(
                    samples_of(union_rows), inputs.fs_hz, config.refractory_period_ms
                )
                pair_increase = pair_rvf - max(
                    obs_a.refractory_violation_fraction, obs_b.refractory_violation_fraction
                )
            elif ratio > config.max_amplitude_ratio:
                reason = "amplitude_ratio"
            else:
                cosine = waveform_cosine(
                    obs_a, obs_b, inputs.channel_positions_um,
                    config.waveform_channel_neighbourhood_um, neighbourhoods,
                )
                if cosine is None:
                    reason = "waveform_unavailable"
                elif cosine < config.min_waveform_cosine:
                    reason = "waveform_cosine"
                else:
                    # The pair's refractory burden is scored on the union of the
                    # two observations' original rows -- deduplicated by row id,
                    # so rows shared through the epoch overlap are counted once
                    # and two distinct rows at one sample are still counted
                    # twice. What is gated is the *increase* over what the two
                    # clusters already carry apart, not the absolute level.
                    union_rows = np.union1d(obs_a.row_ids, obs_b.row_ids)
                    pair_rvf = refractory_violation_fraction(
                        samples_of(union_rows), inputs.fs_hz, config.refractory_period_ms
                    )
                    pair_increase = pair_rvf - max(
                        obs_a.refractory_violation_fraction,
                        obs_b.refractory_violation_fraction,
                    )
                    if pair_increase > config.max_refractory_violation_increase:
                        reason = "epoch_pair_refractory_increase"

            proposals.append(
                CandidateLink(
                    epoch_a=epoch_a,
                    cluster_a=obs_a.cluster_id,
                    epoch_b=epoch_b,
                    cluster_b=obs_b.cluster_id,
                    spatial_distance_um=distance,
                    amplitude_ratio=float(ratio),
                    waveform_cosine=float("nan") if cosine is None else float(cosine),
                    pair_refractory_fraction=float(pair_rvf),
                    pair_refractory_increase=float(pair_increase),
                    is_merge=is_merge,
                    link_score=(
                        0.0 if reason or cosine is None else _link_score(distance, ratio, cosine)
                    ),
                    rejected_because=reason,
                )
            )

        links.extend(_resolve_exclusive(proposals, config))
    return links


def _contested(
    grouped: dict[tuple[int, int], list[CandidateLink]], config: UnwarpedIdentityConfig
) -> set[tuple[int, int]]:
    """Endpoints whose best two viable links are too close to separate."""
    contested: set[tuple[int, int]] = set()
    for key, group in grouped.items():
        ranked = sorted(group, key=lambda l: l.link_score, reverse=True)
        if len(ranked) < 2 or ranked[0].link_score <= 0:
            continue
        if ranked[1].link_score / ranked[0].link_score >= config.ambiguity_threshold_ratio:
            contested.add(key)
    return contested


def _resolve_exclusive(
    proposals: list[CandidateLink], config: UnwarpedIdentityConfig
) -> list[CandidateLink]:
    """Accept a matching that is exclusive and unambiguous on **both** sides.

    Enforcing exclusivity on the destination alone lets one observation in the
    earlier epoch claim several successors, quietly merging distinct neurons
    into one family. Both directions are enforced, and any endpoint whose top
    two candidates are within ``ambiguity_threshold_ratio`` is dropped from the
    matching entirely -- an ambiguous link leaves both sides separate.
    """
    viable = [l for l in proposals if not l.rejected_because]
    by_source: dict[tuple[int, int], list[CandidateLink]] = {}
    by_destination: dict[tuple[int, int], list[CandidateLink]] = {}
    for link in viable:
        by_source.setdefault(link.source, []).append(link)
        by_destination.setdefault(link.destination, []).append(link)

    ambiguous_sources = _contested(by_source, config)
    ambiguous_destinations = _contested(by_destination, config)

    claimed_sources: set[tuple[int, int]] = set()
    claimed_destinations: set[tuple[int, int]] = set()
    for link in sorted(
        viable,
        key=lambda l: (-l.link_score, l.cluster_a, l.cluster_b),
    ):
        if link.source in ambiguous_sources:
            link.rejected_because = "ambiguous_source"
        elif link.destination in ambiguous_destinations:
            link.rejected_because = "ambiguous_destination"
        elif link.source in claimed_sources:
            link.rejected_because = "source_already_claimed"
        elif link.destination in claimed_destinations:
            link.rejected_because = "destination_already_claimed"
        else:
            link.accepted = True
            claimed_sources.add(link.source)
            claimed_destinations.add(link.destination)
    return proposals


# --------------------------------------------------------------------------- #
# families and row assignment
# --------------------------------------------------------------------------- #
def _components(nodes: list[int], edges: list[tuple[int, int]]) -> dict[int, int]:
    parent = {node: node for node in nodes}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for a, b in edges:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    return {node: find(node) for node in nodes}


def assign_rows_to_families(
    inputs: ReplayInput,
    accepted_merges: list[CandidateLink],
) -> tuple[dict[int, int], dict[int, int], dict[str, Any]]:
    """Give every original row a family, via its cluster. Never via its epoch.

    This is the preservation invariant. The output partition is a partition of
    the **original clusters**: a family is a connected component of clusters
    joined by accepted merge links, and every row of a cluster goes wherever
    that cluster goes. Consequences, all of them deliberate:

    * with no accepted merge, the exported assignment *is* the input partition,
      renumbered;
    * a refused epoch-to-epoch continuity link cannot split a cluster, because
      the partition does not consult epochs at all;
    * a row in no eligible epoch -- below the per-epoch minimum, or outside every
      whole epoch in the interval -- still belongs to its cluster, so there is no
      "unassigned" class to invent an extra fragment for;
    * every row is assigned exactly once, so there is no overlap conflict to
      resolve.

    Returns ``(row_family, family_of_cluster, metadata)``.
    """
    clusters = sorted({int(c) for c in np.unique(inputs.cluster)})
    edges = [(int(l.cluster_a), int(l.cluster_b)) for l in accepted_merges]
    roots = _components(clusters, edges)

    # deterministic family ids, ordered by each component's smallest cluster id
    smallest: dict[int, int] = {}
    for cluster, root in roots.items():
        if root not in smallest or cluster < smallest[root]:
            smallest[root] = cluster
    family_of_root = {
        root: index
        for index, root in enumerate(sorted(smallest, key=lambda r: smallest[r]), start=1)
    }
    family_of_cluster = {c: family_of_root[roots[c]] for c in clusters}

    cluster_array = inputs.cluster.astype(np.int64)
    row_family = {
        int(row): family_of_cluster[int(cluster)]
        for row, cluster in zip(inputs.row_id.tolist(), cluster_array.tolist())
    }
    contributors: dict[int, list[int]] = {}
    for cluster, family in family_of_cluster.items():
        contributors.setdefault(family, []).append(cluster)

    metadata = {
        "num_original_clusters": len(clusters),
        "num_families": len(family_of_root),
        "num_assigned_rows": len(row_family),
        "num_unassigned_rows": int(inputs.row_id.size) - len(row_family),
        "num_families_built_from_a_merge": sum(
            1 for cids in contributors.values() if len(cids) > 1
        ),
        "partition_policy": "preserve_input_partition; only an accepted merge changes it",
        "dedup_key": "original_spike_row_id",
        "input_partition_preserved": all(len(c) == 1 for c in contributors.values()),
    }
    return row_family, family_of_cluster, metadata


def _cluster_samples(inputs: ReplayInput) -> dict[int, np.ndarray]:
    order = np.argsort(inputs.cluster, kind="stable")
    clusters = inputs.cluster[order]
    samples = inputs.sample[order]
    boundaries = np.flatnonzero(np.diff(clusters)) + 1
    return {
        int(part[0]): np.sort(sample_part)
        for part, sample_part in zip(np.split(clusters, boundaries), np.split(samples, boundaries))
        if part.size
    }


def solve_families(
    observations: list[EpochObservation],
    links: list[CandidateLink],
    inputs: ReplayInput,
    config: UnwarpedIdentityConfig,
) -> tuple[dict[int, int], dict[int, int], list[CandidateLink], dict[str, Any]]:
    """Union accepted merges into families, then verify each **merged** train.

    Only merges are prunable and only merges can breach: a single-cluster family
    is exactly its input cluster, so its increase over its own baseline is zero
    by construction. That is the point -- preserving an imperfect cluster is not
    the same as certifying it, and this loop never has to choose between the two.
    Its absolute violation fraction is still reported, descriptively.

    The loop prunes the worst breaching family that *has* a removable link, and
    keeps going while any such family remains. An earlier version returned as
    soon as the single worst family had nothing to prune, which left other
    breaching families with removable links untouched.
    """
    cluster_samples = _cluster_samples(inputs)
    baseline_rvf = {
        cluster: refractory_violation_fraction(
            samples, inputs.fs_hz, config.refractory_period_ms
        )
        for cluster, samples in cluster_samples.items()
    }
    active = [l for l in links if l.accepted and l.is_merge]
    pruned: list[CandidateLink] = []

    while True:
        row_family, family_of_cluster, meta = assign_rows_to_families(inputs, active)
        contributors: dict[int, list[int]] = {}
        for cluster, family in family_of_cluster.items():
            contributors.setdefault(family, []).append(cluster)

        absolute: dict[int, float] = {}
        increase: dict[int, float] = {}
        for family, cids in contributors.items():
            train = np.sort(np.concatenate([cluster_samples[c] for c in cids]))
            rvf = refractory_violation_fraction(
                train, inputs.fs_hz, config.refractory_period_ms
            )
            absolute[family] = rvf
            increase[family] = rvf - max(baseline_rvf[c] for c in cids)

        breaching = {
            f: increase[f]
            for f in contributors
            if increase[f] > config.max_refractory_violation_increase
        }
        prunable = {
            f: v
            for f, v in breaching.items()
            if any(family_of_cluster.get(int(l.cluster_a)) == f for l in active)
        }
        if not prunable:
            meta["exported_train_refractory_fraction"] = {
                str(f): absolute[f] for f in sorted(absolute)
            }
            meta["exported_train_refractory_increase"] = {
                str(f): increase[f] for f in sorted(increase)
            }
            meta["num_pruned_merges"] = len(pruned)
            meta["num_accepted_merges"] = len(active)
            meta["pruned_for_exported_train_refractory"] = [l.to_row() for l in pruned]
            meta["families_breaching_without_a_prunable_link"] = sorted(breaching)
            meta["single_cluster_families_are_preserved_not_certified"] = (
                "A family of one cluster is the input cluster unchanged. Its absolute "
                "refractory violation fraction is reported and may be high; that is a "
                "property of the retained sort, is not corrected here, and is not a claim "
                "that the cluster is one clean neuron."
            )
            return row_family, family_of_cluster, active, meta

        worst = max(prunable, key=lambda f: prunable[f])
        inside = [l for l in active if family_of_cluster.get(int(l.cluster_a)) == worst]
        victim = max(
            inside, key=lambda l: (l.pair_refractory_increase, -l.link_score, l.cluster_b)
        )
        victim.accepted = False
        victim.rejected_because = "exported_train_refractory_increase"
        active.remove(victim)
        pruned.append(victim)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def run_unwarped_identity_replay(
    inputs: ReplayInput,
    *,
    motion: MotionDeclaration,
    config: UnwarpedIdentityConfig,
    processing_interval_s: tuple[float, float],
    output_dir: Path,
) -> dict[str, Any]:
    """Replay the retained rows in ``processing_interval_s`` into families."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if motion.is_motion_aware:
        displacement = np.asarray(motion.displacement_um, dtype=np.float64).reshape(-1)
        if displacement.size != inputs.row_id.size:
            raise IdentityRefusal(
                f"motion displacement has {displacement.size} entries for "
                f"{inputs.row_id.size} spike rows"
            )
        if not np.all(np.isfinite(displacement)):
            raise IdentityRefusal(
                "a qualified motion field left some spikes unsupported; an unsupported spike is "
                "not a spike with zero displacement. Restrict the interval or the arm instead."
            )
        tissue_depth = inputs.depth_um - displacement
    else:
        tissue_depth = inputs.depth_um.copy()

    epoch_indices = epochs_covering(processing_interval_s, config)
    if not epoch_indices:
        raise IdentityRefusal(
            f"no whole {config.epoch_duration_s} s epoch fits inside processing interval "
            f"{processing_interval_s!r} on a grid stepping {config.epoch_step_s} s"
        )

    observations = extract_epoch_observations(inputs, epoch_indices, tissue_depth, config)
    link_counters: dict[str, int] = {}
    links = build_candidate_links(observations, inputs, config, link_counters)
    row_family, family_of_cluster, active_links, meta = solve_families(
        observations, links, inputs, config
    )

    # Families that are exactly one original cluster are the input, preserved.
    contributors: dict[int, set[int]] = {}
    for cluster_id, family in family_of_cluster.items():
        contributors.setdefault(family, set()).add(cluster_id)
    linked_families = sorted(f for f, cids in contributors.items() if len(cids) > 1)

    observations_csv = output_dir / "epoch_observations.csv"
    with observations_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_OBSERVATION_FIELDS))
        writer.writeheader()
        for obs in observations:
            writer.writerow(obs.to_row())

    links_csv = output_dir / "candidate_links.csv"
    with links_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_LINK_FIELDS))
        writer.writeheader()
        for link in links:
            writer.writerow(link.to_row())

    families_csv = output_dir / "family_membership.csv"
    with families_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["family_id", "cluster_id"])
        for cluster_id, family in sorted(family_of_cluster.items(), key=lambda kv: (kv[1], kv[0])):
            writer.writerow([family, cluster_id])

    rejection_counts = {reason: 0 for reason in REJECTION_REASONS}
    # depth-rejected pairs are counted rather than materialised
    rejection_counts["spatial_distance"] = link_counters.get("spatial_distance", 0)
    for link in links:
        if link.rejected_because:
            rejection_counts[link.rejected_because] = (
                rejection_counts.get(link.rejected_because, 0) + 1
            )

    manifest = {
        "schema": UNWARPED_IDENTITY_SCHEMA,
        "execution_mode": "retained_sort_replay",
        "motion": motion.to_dict(),
        "config": config.to_dict(),
        "config_digest": config.digest(),
        "processing_interval_s": [float(processing_interval_s[0]), float(processing_interval_s[1])],
        "epoch_indices": epoch_indices,
        "epoch_span_s": [
            epoch_bounds(epoch_indices[0], config)[0],
            epoch_bounds(epoch_indices[-1], config)[1],
        ],
        "num_input_rows": int(inputs.row_id.size),
        "num_observations": len(observations),
        "num_pairs_considered": link_counters.get("pairs_considered", 0),
        "num_candidate_links": len(links),
        "num_accepted_links": len(active_links),
        "link_rejections": rejection_counts,
        "num_original_clusters": meta["num_original_clusters"],
        "num_families": meta["num_families"],
        "num_families_built_from_a_merge": len(linked_families),
        "families_built_from_a_merge": linked_families,
        "input_partition_preserved": meta["input_partition_preserved"],
        "assignment": meta,
        "output_artifacts": {
            "epoch_observations": observations_csv.name,
            "candidate_links": links_csv.name,
            "family_membership": families_csv.name,
        },
    }
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    return {
        "manifest": manifest,
        "observations": observations,
        "links": links,
        "accepted_links": active_links,
        "row_family": row_family,
        "family_of_cluster": family_of_cluster,
        "families_built_from_a_merge": linked_families,
        "family_contributors": {f: sorted(c) for f, c in contributors.items()},
    }
