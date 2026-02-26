# %% Copied from Ryans code
#%% Prepend: Load sortings and results for Ryan-style comparison

import numpy as np
import pandas as pd
from pathlib import Path

import spikeinterface.extractors as se

# Set pipeline directories
pipe0 = Path("/mnt/NPX/Luke/20250804/pipeline_results_Luke0804_V2V1_g0_imec0")
pipe1 = Path("/mnt/NPX/Luke/20250804/dredge_pipeline_results_Luke0804_V2V1_g0_imec0")

# Phy/curated output paths
phy0_path = pipe0 / "cur" / "cur_sorter_output"
phy1_path = pipe1 / "cur" / "cur_sorter_output"

# Load sortings
print(f"Loading Sorting 0: {phy0_path}")
sorting0 = se.read_phy(folder_path=str(phy0_path), load_all_cluster_properties=True)
print(f"Loading Sorting 1: {phy1_path}")
sorting1 = se.read_phy(folder_path=str(phy1_path), load_all_cluster_properties=True)

# Load spike clusters and times for both sortings
old_spike_clusters = np.load(phy0_path / "spike_clusters.npy")
old_spike_times = np.load(phy0_path / "spike_times.npy") / float(sorting0.get_sampling_frequency())
new_ks = lambda: None  # Simple namespace
setattr(new_ks, "spike_clusters", np.load(phy1_path / "spike_clusters.npy"))
setattr(new_ks, "spike_times", np.load(phy1_path / "spike_times.npy") / float(sorting1.get_sampling_frequency()))

#%%
# fill old_results and new_results with cids from spike_clusters and spike_counts
old_results = dict(cids=np.unique(old_spike_clusters), spike_counts=np.bincount(old_spike_clusters))
new_results = dict(cids=np.unique(new_ks.spike_clusters), spike_counts=np.bincount(new_ks.spike_clusters))

# # Load results (expects at least: cids, spike_counts)
# def load_results(folder):
#     # Try npz, then csv
#     npz_path = folder / "qc" / "metrics.npz"
#     if npz_path.exists():
#         return dict(np.load(npz_path, allow_pickle=True))
#     csv_path = folder / "qc" / "metrics.csv"
#     if csv_path.exists():
#         df = pd.read_csv(csv_path)
#         return {col: df[col].values for col in df.columns}
#     raise FileNotFoundError(f"No metrics found in {folder}")

# old_results = load_results(pipe0)
# new_results = load_results(pipe1)

# NOTE: Ryan's original script annotates units with SNR and a "responsive" flag.
# For now we explicitly *do not* load/compute those metrics.
#
# def load_responsive(folder):
#     ...
#
# old_responsive = load_responsive(pipe0)
# new_responsive = load_responsive(pipe1)

import logging
logger = logging.getLogger("compare_sortings_ryan")
logger.setLevel(logging.INFO)
print("Loaded all sorting and QC data.\n")


# %% Check for undersplit units by seeing which units from the old sorting have a large fraction of their spikes align with spikes in the new sorting. This creates a map of units in the old sorting to units in the new sorting, which we will print out and inspect to see if any old units have a large fraction of their spikes aligning to the same new unit, which would suggest that the new unit is overmerged/undersplit.

from collections import defaultdict

def find_coincident_spikes(times_a, times_b, tolerance_sec=0.001):
    """
    Find the number of spikes in times_a that have a coincident spike in times_b.
    Two spikes are coincident if they are within tolerance_sec of each other.

    Returns the count of coincident spikes from times_a.
    """
    if len(times_a) == 0 or len(times_b) == 0:
        return 0

    # Sort both arrays
    times_a = np.sort(times_a)
    times_b = np.sort(times_b)

    # Vectorized: find insertion points for all spikes in times_a
    idx = np.searchsorted(times_b, times_a)

    # Check distance to left neighbor (idx-1) and right neighbor (idx)
    # Clamp indices to valid range
    idx_left = np.clip(idx - 1, 0, len(times_b) - 1)
    idx_right = np.clip(idx, 0, len(times_b) - 1)

    # Compute distances to both neighbors
    dist_left = np.abs(times_a - times_b[idx_left])
    dist_right = np.abs(times_a - times_b[idx_right])

    # A spike is coincident if the minimum distance to either neighbor is within tolerance
    min_dist = np.minimum(dist_left, dist_right)
    coincident_count = np.sum(min_dist <= tolerance_sec)

    return coincident_count

# Get old and new spike data
old_cids = np.unique(old_spike_clusters)
new_cids = new_ks.spike_clusters
new_spike_times = new_ks.spike_times
new_cids_unique = np.unique(new_cids)

logger.info(f"Analyzing correspondence between {len(old_cids)} old units and {len(new_cids_unique)} new units")

# Build a mapping: for each old unit, find which new units have >10% coincident spikes
coincidence_threshold = 0.30  # Percent of old spikes that must be coincident to consider it a match
tolerance_sec = .5e-3  # 1 ms tolerance for spike coincidence

# Store results: old_unit -> [(new_unit, coincident_fraction, n_coincident, n_old_spikes), ...]
old_to_new_map = defaultdict(list)

# Also track: new_unit -> [old_units that map to it]
new_to_old_map = defaultdict(list)

from tqdm import tqdm
for old_cid in tqdm(old_cids, desc="Comparing old units to new units"):
    old_mask = old_spike_clusters == old_cid
    old_times = old_spike_times[old_mask]
    n_old_spikes = len(old_times)

    if n_old_spikes < 100:  # Skip units with very few spikes
        continue

    for new_cid in new_cids_unique:
        new_mask = new_cids == new_cid
        new_times = new_spike_times[new_mask]

        if len(new_times) < 100:
            continue

        n_coincident = find_coincident_spikes(old_times, new_times, tolerance_sec)
        coincident_frac = n_coincident / n_old_spikes

        if coincident_frac >= coincidence_threshold:
            old_to_new_map[old_cid].append((new_cid, coincident_frac, n_coincident, n_old_spikes))
            new_to_old_map[new_cid].append((old_cid, coincident_frac, n_coincident, n_old_spikes))

# Print correspondence summary
print("\n" + "="*80)
print("OLD -> NEW UNIT CORRESPONDENCE (>10% coincident spikes within 1ms)")
print("="*80)

for old_cid in sorted(old_to_new_map.keys()):
    matches = old_to_new_map[old_cid]
    matches_sorted = sorted(matches, key=lambda x: -x[1])  # Sort by coincident fraction descending

    # NOTE: SNR/responsive annotations intentionally disabled for now.
    print(f"\nOld unit {old_cid}:")
    for new_cid, frac, n_coinc, n_old in matches_sorted:
        print(f"  -> New unit {new_cid}: {frac*100:.1f}% coincident ({n_coinc}/{n_old} spikes)")

#%%
# Print potential over-merged units (new units with multiple old units mapping to them)
print("\n" + "="*80)
print("POTENTIAL OVER-MERGED UNITS (new units with multiple old units mapping to them)")
print("="*80)

overmerged_candidates = {k: v for k, v in new_to_old_map.items() if len(v) > 1}

if overmerged_candidates:
    for new_cid in sorted(overmerged_candidates.keys()):
        old_matches = overmerged_candidates[new_cid]
        old_matches_sorted = sorted(old_matches, key=lambda x: -x[1])

        n_new_spikes = np.sum(new_cids == new_cid)
        print(f"\nNew unit {new_cid} ({n_new_spikes} spikes) <- {len(old_matches)} old units:")
        for old_cid, frac, n_coinc, n_old in old_matches_sorted:
            print(f"  <- Old unit {old_cid}: {frac*100:.1f}% of old spikes coincident ({n_coinc}/{n_old})")
else:
    print("\nNo new units have multiple old units mapping to them.")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
n_old_with_match = len(old_to_new_map)

print(f"Old units with >{100*coincidence_threshold:.0f}% match to a new unit: {n_old_with_match}/{len(old_cids)}")
print(f"New units with multiple old units mapping: {len(overmerged_candidates)}")

# NOTE: Responsive-unit reporting intentionally disabled for now.
#
# print("\n" + "="*80)
# print("OLD RESPONSIVE UNITS WITH NO MATCH IN NEW SORTING (potential missed detections)")
# print("="*80)
# for old_idx in np.where(old_responsive)[0]:
#     old_cid = old_results['cids'][old_idx]
#     if old_cid not in old_to_new_map:
#         n_spikes = old_results['spike_counts'][old_idx]
#         print(f"  Old unit {old_cid}: {n_spikes} spikes - NO MATCH in new sorting")