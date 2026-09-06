"""Unwarped motion-aware identity handling (Option B candidate).

Leaves accepted raw voltage, spike times, and original cluster IDs unchanged.
Partition spike observations into overlapping time epochs, computes tissue-frame
depths using motion coordinates, generates link candidates across adjacent epochs
gated by spatial proximity, refractory cleanliness, and rate/amplitude compatibility,
solves longitudinal identity tracks, and emits reversible track mappings.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


UNWARPED_IDENTITY_SCHEMA = "luke-unwarped-identity-v1"
MANIFEST_NAME = "unwarped_identity_manifest.json"


@dataclass
class UnwarpedIdentityConfig:
    epoch_duration_s: float = 120.0
    epoch_overlap_s: float = 30.0
    max_spatial_distance_um: float = 30.0
    max_refractory_violation_fraction: float = 0.01
    max_amplitude_ratio: float = 2.0
    ambiguity_threshold_ratio: float = 0.85
    min_spikes_per_epoch: int = 10
    refractory_period_ms: float = 1.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        s = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(s.encode()).hexdigest()


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
    mean_amplitude: float
    refractory_violation_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AcceptedLink:
    epoch_a: int
    cluster_a: int
    epoch_b: int
    cluster_b: int
    spatial_dist_um: float
    amplitude_ratio: float
    union_refractory_rate: float
    link_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_refractory_violation_rate(
    spike_times_s: np.ndarray, refractory_period_ms: float = 1.5
) -> float:
    """Fraction of inter-spike intervals < refractory_period_ms."""
    if spike_times_s.size <= 1:
        return 0.0
    sorted_times = np.sort(spike_times_s)
    isis_s = np.diff(sorted_times)
    refractory_threshold_s = refractory_period_ms / 1000.0
    violations = np.sum(isis_s < refractory_threshold_s)
    return float(violations / len(isis_s))


def extract_epoch_observations(
    spike_times_s: np.ndarray,
    spike_clusters: np.ndarray,
    spike_depths_um: np.ndarray,
    tissue_depths_um: np.ndarray,
    recording_duration_s: float,
    config: UnwarpedIdentityConfig,
    spike_amplitudes: np.ndarray | None = None,
) -> list[EpochObservation]:
    """Partition spikes into overlapping epochs and compute per-cluster summary metrics."""
    times = np.asarray(spike_times_s, dtype=np.float64)
    clusters = np.asarray(spike_clusters, dtype=np.int64)
    obs_depths = np.asarray(spike_depths_um, dtype=np.float64)
    tiss_depths = np.asarray(tissue_depths_um, dtype=np.float64)
    amps = np.abs(np.asarray(spike_amplitudes, dtype=np.float64)) if spike_amplitudes is not None else np.ones_like(times)

    step_s = config.epoch_duration_s - config.epoch_overlap_s
    if step_s <= 0:
        raise ValueError("epoch_duration_s must be strictly greater than epoch_overlap_s")

    num_epochs = max(1, int(np.ceil((max(recording_duration_s, times.max() if times.size else 0) - config.epoch_overlap_s) / step_s)))
    
    unique_clusters = np.unique(clusters)
    observations: list[EpochObservation] = []

    for ep_idx in range(num_epochs):
        ep_start = ep_idx * step_s
        ep_stop = ep_start + config.epoch_duration_s
        mask_ep = (times >= ep_start) & (times < ep_stop)

        if not np.any(mask_ep):
            continue

        times_ep = times[mask_ep]
        clusters_ep = clusters[mask_ep]
        obs_depths_ep = obs_depths[mask_ep]
        tiss_depths_ep = tiss_depths[mask_ep]
        amps_ep = amps[mask_ep]

        for cid in unique_clusters:
            mask_c = clusters_ep == cid
            n_spikes = int(np.sum(mask_c))
            if n_spikes < config.min_spikes_per_epoch:
                continue

            c_times = times_ep[mask_c]
            c_obs_d = obs_depths_ep[mask_c]
            c_tiss_d = tiss_depths_ep[mask_c]
            c_amps = amps_ep[mask_c]

            fr = float(n_spikes / config.epoch_duration_s)
            mean_obs_d = float(np.mean(c_obs_d))
            mean_tiss_d = float(np.mean(c_tiss_d))
            mean_amp = float(np.mean(c_amps)) if c_amps.size else 1.0
            rv_rate = compute_refractory_violation_rate(c_times, config.refractory_period_ms)

            observations.append(
                EpochObservation(
                    epoch_idx=ep_idx,
                    cluster_id=int(cid),
                    start_s=float(ep_start),
                    stop_s=float(ep_stop),
                    num_spikes=n_spikes,
                    firing_rate_hz=fr,
                    mean_observed_depth_um=mean_obs_d,
                    mean_tissue_depth_um=mean_tiss_d,
                    mean_amplitude=mean_amp,
                    refractory_violation_rate=rv_rate,
                )
            )

    return observations


def build_candidate_links(
    observations: list[EpochObservation],
    spike_times_s: np.ndarray,
    spike_clusters: np.ndarray,
    config: UnwarpedIdentityConfig,
) -> list[AcceptedLink]:
    """Build evidence-gated links between observations in adjacent epochs with ambiguity handling."""
    times = np.asarray(spike_times_s, dtype=np.float64)
    clusters = np.asarray(spike_clusters, dtype=np.int64)

    obs_by_epoch: dict[int, list[EpochObservation]] = {}
    for obs in observations:
        obs_by_epoch.setdefault(obs.epoch_idx, []).append(obs)

    accepted_links: list[AcceptedLink] = []
    sorted_epochs = sorted(obs_by_epoch.keys())

    for ep_a, ep_b in zip(sorted_epochs, sorted_epochs[1:]):
        if ep_b != ep_a + 1:
            continue

        list_a = obs_by_epoch[ep_a]
        list_b = obs_by_epoch[ep_b]

        candidate_links_ep: list[AcceptedLink] = []

        for obs_a in list_a:
            for obs_b in list_b:
                dist_um = abs(obs_a.mean_tissue_depth_um - obs_b.mean_tissue_depth_um)
                if dist_um > config.max_spatial_distance_um:
                    continue

                amp_max = max(obs_a.mean_amplitude, obs_b.mean_amplitude)
                amp_min = max(1e-6, min(obs_a.mean_amplitude, obs_b.mean_amplitude))
                amp_ratio = amp_max / amp_min
                if amp_ratio > config.max_amplitude_ratio:
                    continue

                # Union refractory check on spikes in overlapping / adjacent time range
                mask_a = (clusters == obs_a.cluster_id) & (times >= obs_a.start_s) & (times < obs_a.stop_s)
                mask_b = (clusters == obs_b.cluster_id) & (times >= obs_b.start_s) & (times < obs_b.stop_s)
                combined_times = np.unique(np.concatenate([times[mask_a], times[mask_b]]))
                union_rv = compute_refractory_violation_rate(combined_times, config.refractory_period_ms)

                if union_rv > config.max_refractory_violation_fraction:
                    continue

                link_score = 1.0 / (1.0 + dist_um / 10.0 + (amp_ratio - 1.0) + 100.0 * union_rv)

                candidate_links_ep.append(
                    AcceptedLink(
                        epoch_a=ep_a,
                        cluster_a=obs_a.cluster_id,
                        epoch_b=ep_b,
                        cluster_b=obs_b.cluster_id,
                        spatial_dist_um=float(dist_um),
                        amplitude_ratio=float(amp_ratio),
                        union_refractory_rate=float(union_rv),
                        link_score=float(link_score),
                    )
                )

        # Ambiguity resolution: 1-to-1 matching without competing ambiguous links
        if not candidate_links_ep:
            continue

        # Check forward ambiguity (obs_a -> multiple obs_b)
        links_from_a: dict[int, list[AcceptedLink]] = {}
        for link in candidate_links_ep:
            links_from_a.setdefault(link.cluster_a, []).append(link)

        ambiguous_a: set[int] = set()
        for cid_a, c_links in links_from_a.items():
            c_links.sort(key=lambda l: l.link_score, reverse=True)
            if len(c_links) > 1:
                top_score = c_links[0].link_score
                second_score = c_links[1].link_score
                if top_score > 0 and (second_score / top_score) >= config.ambiguity_threshold_ratio:
                    ambiguous_a.add(cid_a)

        # Check backward ambiguity (obs_b <- multiple obs_a)
        links_to_b: dict[int, list[AcceptedLink]] = {}
        for link in candidate_links_ep:
            links_to_b.setdefault(link.cluster_b, []).append(link)

        ambiguous_b: set[int] = set()
        for cid_b, c_links in links_to_b.items():
            c_links.sort(key=lambda l: l.link_score, reverse=True)
            if len(c_links) > 1:
                top_score = c_links[0].link_score
                second_score = c_links[1].link_score
                if top_score > 0 and (second_score / top_score) >= config.ambiguity_threshold_ratio:
                    ambiguous_b.add(cid_b)

        # Keep best non-ambiguous 1-to-1 links
        claimed_b: set[int] = set()
        for link in sorted(candidate_links_ep, key=lambda l: l.link_score, reverse=True):
            if link.cluster_a in ambiguous_a or link.cluster_b in ambiguous_b:
                continue
            if link.cluster_b in claimed_b:
                continue
            accepted_links.append(link)
            claimed_b.add(link.cluster_b)

    return accepted_links


def solve_identity_tracks(
    observations: list[EpochObservation],
    accepted_links: list[AcceptedLink],
    spike_times_s: np.ndarray | None = None,
    spike_clusters: np.ndarray | None = None,
    config: UnwarpedIdentityConfig | None = None,
) -> tuple[dict[tuple[int, int], int], list[AcceptedLink], dict[str, int]]:
    """Solve connected component tracks with complete-track refractory cleanliness pruning."""
    if config is None:
        config = UnwarpedIdentityConfig()

    nodes = {(obs.cluster_id, obs.epoch_idx) for obs in observations}
    active_links = list(accepted_links)
    num_pruned_links = 0

    while True:
        parent = {node: node for node in nodes}

        def find(n: tuple[int, int]) -> tuple[int, int]:
            if parent[n] != n:
                parent[n] = find(parent[n])
            return parent[n]

        def union(n1: tuple[int, int], n2: tuple[int, int]):
            r1, r2 = find(n1), find(n2)
            if r1 != r2:
                parent[r1] = r2

        for link in active_links:
            n_a = (link.cluster_a, link.epoch_a)
            n_b = (link.cluster_b, link.epoch_b)
            if n_a in parent and n_b in parent:
                union(n_a, n_b)

        if spike_times_s is None or spike_clusters is None:
            break

        # Complete-track refractory cleanliness evaluation
        times = np.asarray(spike_times_s, dtype=np.float64)
        clusters = np.asarray(spike_clusters, dtype=np.int64)

        obs_lookup = {(obs.cluster_id, obs.epoch_idx): obs for obs in observations}
        track_components: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for node in nodes:
            r = find(node)
            track_components.setdefault(r, []).append(node)

        dirty_link_to_remove = None
        worst_rv = 0.0

        for root, member_nodes in track_components.items():
            if len(member_nodes) <= 1:
                continue

            track_spikes = []
            for cid, ep_idx in member_nodes:
                obs = obs_lookup.get((cid, ep_idx))
                if obs is None:
                    continue
                mask = (clusters == cid) & (times >= obs.start_s) & (times < obs.stop_s)
                track_spikes.append(times[mask])

            if not track_spikes:
                continue

            all_track_times = np.unique(np.concatenate(track_spikes))
            track_rv = compute_refractory_violation_rate(all_track_times, config.refractory_period_ms)

            if track_rv > config.max_refractory_violation_fraction:
                # Find worst link in this track
                member_set = set(member_nodes)
                track_links = [
                    l for l in active_links
                    if (l.cluster_a, l.epoch_a) in member_set and (l.cluster_b, l.epoch_b) in member_set
                ]
                if track_links:
                    worst_link = max(track_links, key=lambda l: (l.union_refractory_rate, -l.link_score))
                    if track_rv > worst_rv:
                        worst_rv = track_rv
                        dirty_link_to_remove = worst_link

        if dirty_link_to_remove is not None:
            active_links.remove(dirty_link_to_remove)
            num_pruned_links += 1
        else:
            break

    # Assign integer track IDs
    parent = {node: node for node in nodes}
    for link in active_links:
        n_a = (link.cluster_a, link.epoch_a)
        n_b = (link.cluster_b, link.epoch_b)
        if n_a in parent and n_b in parent:
            r1, r2 = find(n_a), find(n_b)
            if r1 != r2:
                parent[r1] = r2

    root_to_track_id: dict[tuple[int, int], int] = {}
    next_track_id = 1
    track_membership: dict[tuple[int, int], int] = {}

    for node in sorted(nodes, key=lambda x: (x[1], x[0])):
        root = find(node)
        if root not in root_to_track_id:
            root_to_track_id[root] = next_track_id
            next_track_id += 1
        track_membership[node] = root_to_track_id[root]

    metadata = {
        "num_active_links": len(active_links),
        "num_pruned_links": num_pruned_links,
        "num_tracks": len(root_to_track_id),
    }

    return track_membership, active_links, metadata


def run_unwarped_identity_pipeline(
    spike_times_s: np.ndarray,
    spike_clusters: np.ndarray,
    spike_depths_um: np.ndarray,
    displacement_um: np.ndarray | None,
    recording_duration_s: float,
    output_dir: Path,
    config: UnwarpedIdentityConfig | None = None,
    spike_amplitudes: np.ndarray | None = None,
) -> dict[str, Any]:
    """Execute Option B unwarped identity pipeline and write artifacts to output_dir."""
    if config is None:
        config = UnwarpedIdentityConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    times = np.asarray(spike_times_s, dtype=np.float64)
    clusters = np.asarray(spike_clusters, dtype=np.int64)
    obs_depths = np.asarray(spike_depths_um, dtype=np.float64)
    amps = np.abs(np.asarray(spike_amplitudes, dtype=np.float64)) if spike_amplitudes is not None else np.ones_like(times)

    if displacement_um is None:
        displacements = np.zeros_like(obs_depths)
    else:
        displacements = np.asarray(displacement_um, dtype=np.float64)

    tissue_depths = obs_depths - displacements

    observations = extract_epoch_observations(
        times, clusters, obs_depths, tissue_depths, recording_duration_s, config, amps
    )
    accepted_links = build_candidate_links(observations, times, clusters, config)
    track_membership, clean_links, prune_meta = solve_identity_tracks(
        observations, accepted_links, times, clusters, config
    )

    # Compute per-spike track assignments
    step_s = config.epoch_duration_s - config.epoch_overlap_s
    spike_tracks = np.copy(clusters)
    for i in range(times.size):
        t = times[i]
        c = clusters[i]
        ep_idx = max(0, int(np.floor(t / step_s)))
        tid = track_membership.get((c, ep_idx))
        if tid is not None:
            spike_tracks[i] = tid

    # Save spike tracks
    tracks_npy_path = output_dir / "spike_tracks.npy"
    np.save(tracks_npy_path, spike_tracks)

    # Write CSV artifacts
    obs_csv = output_dir / "epoch_observations.csv"
    with obs_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(EpochObservation.__annotations__.keys()))
        writer.writeheader()
        for obs in observations:
            writer.writerow(obs.to_dict())

    links_csv = output_dir / "accepted_links.csv"
    with links_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(AcceptedLink.__annotations__.keys()))
        writer.writeheader()
        for link in clean_links:
            writer.writerow(link.to_dict())

    tracks_csv = output_dir / "track_membership.csv"
    with tracks_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cluster_id", "epoch_idx", "track_id"])
        for (cid, ep_idx), tid in sorted(track_membership.items()):
            writer.writerow([cid, ep_idx, tid])

    manifest = {
        "schema": UNWARPED_IDENTITY_SCHEMA,
        "config": config.to_dict(),
        "config_digest": config.digest(),
        "num_observations": len(observations),
        "num_accepted_links": len(clean_links),
        "num_unique_tracks": len(set(track_membership.values())),
        "pruning_metadata": prune_meta,
        "output_artifacts": {
            "spike_tracks_npy": tracks_npy_path.name,
            "epoch_observations": obs_csv.name,
            "accepted_links": links_csv.name,
            "track_membership": tracks_csv.name,
        },
    }

    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return {
        "manifest": manifest,
        "observations": observations,
        "accepted_links": clean_links,
        "track_membership": track_membership,
        "spike_tracks": spike_tracks,
    }
    track_membership = solve_identity_tracks(observations, accepted_links)

    # Write CSV artifacts
    obs_csv = output_dir / "epoch_observations.csv"
    with obs_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(EpochObservation.__annotations__.keys()))
        writer.writeheader()
        for obs in observations:
            writer.writerow(obs.to_dict())

    links_csv = output_dir / "accepted_links.csv"
    with links_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(AcceptedLink.__annotations__.keys()))
        writer.writeheader()
        for link in accepted_links:
            writer.writerow(link.to_dict())

    tracks_csv = output_dir / "track_membership.csv"
    with tracks_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cluster_id", "epoch_idx", "track_id"])
        for (cid, ep_idx), tid in sorted(track_membership.items()):
            writer.writerow([cid, ep_idx, tid])

    manifest = {
        "schema": UNWARPED_IDENTITY_SCHEMA,
        "config": config.to_dict(),
        "config_digest": config.digest(),
        "num_observations": len(observations),
        "num_accepted_links": len(accepted_links),
        "num_unique_tracks": len(set(track_membership.values())),
        "output_artifacts": {
            "epoch_observations": obs_csv.name,
            "accepted_links": links_csv.name,
            "track_membership": tracks_csv.name,
        },
    }

    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return {
        "manifest": manifest,
        "observations": observations,
        "accepted_links": accepted_links,
        "track_membership": track_membership,
    }
