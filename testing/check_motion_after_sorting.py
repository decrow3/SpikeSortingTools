#!/usr/bin/env python
"""Standalone motion sanity-check after Kilosort4 sorting.

Goal
----
Use fitted spike positions (and optionally amplitudes) to visualize residual motion
in the *sorted* output. If motion correction is working well, the spike depth
(y-position) distribution should be stable over time.

Inputs
------
- Kilosort4 sorter output folder containing:
  - spike_times.npy (samples)
  - spike_positions.npy (um)
  - amplitudes.npy (optional)
  - ops.npy (optional; used for sampling rate fallback)

Optional overlay
----------------
If a dredge motion estimate folder exists with:
  - motion.npy
  - time_bins.npy
  - depth_bins.npy

we plot the estimated drift trace at a representative depth alongside the
post-sort spike median depth trace.

Example
-------
python check_motion_after_sorting.py \
  --pipeline-dir /mnt/NPX/Luke/20260316/patched_pipeline_results_Luke03162026_V2V1_RH_g0_imec1

python check_motion_after_sorting.py \
  --pipeline-dir /mnt/NPX/Luke/20260316/patched_pipeline_results_Luke03162026_V2V1_RH_g0_imec1 \
  --motion-dir  /mnt/NPX/Luke/20260316/dredge_pipeline_results_Luke03162026_V2V1_RH_g0_imec1/motion/dredge-motion

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


@dataclass(frozen=True)
class MotionOverlay:
    motion: np.ndarray  # (T, D) or similar
    time_bins_s: np.ndarray  # (T,) or (T+1,)
    depth_bins_um: np.ndarray  # (D,) or (D+1,)


def _infer_sweep_motion_dir(pipeline_dir: Path) -> Path | None:
    """Try to infer the corresponding dredge motion folder from a patched pipeline path."""
    # Most of your pipeline naming uses patched_pipeline_results_... vs dredge_pipeline_results_...
    if 'patched_pipeline_results_' in pipeline_dir.name:
        guess = pipeline_dir.with_name(pipeline_dir.name.replace('patched_pipeline_results_', 'dredge_pipeline_results_'))
        cand = guess / 'motion' / 'dredge-motion'
        if cand.exists():
            return cand
    # Also try sibling dredge folder in the same parent
    parent = pipeline_dir.parent
    for sib in parent.iterdir() if parent.exists() else []:
        if sib.is_dir() and sib.name.startswith('dredge_pipeline_results_') and (sib / 'motion' / 'dredge-motion').exists():
            return sib / 'motion' / 'dredge-motion'
    return None


def _load_sampling_frequency(sorter_output_dir: Path, default_fs: float = 30_000.0) -> float:
    """Best-effort sampling frequency from KS output; fallback to default."""
    # 1) params.py often contains sample_rate. Keep it simple: parse ops.npy first.
    ops_p = sorter_output_dir / 'ops.npy'
    if ops_p.exists():
        try:
            ops = np.load(ops_p, allow_pickle=True).item()
            for k in ('fs', 'sample_rate', 'sampling_rate'):
                if k in ops:
                    return float(ops[k])
        except Exception:
            pass

    # 2) spikeinterface_recording.json (one level up)
    rec_json = sorter_output_dir.parent / 'spikeinterface_recording.json'
    if rec_json.exists():
        try:
            obj = json.loads(rec_json.read_text())
            for k in ('sampling_frequency', 'sampling_rate', 'fs'):
                if k in obj:
                    return float(obj[k])
        except Exception:
            pass

    return float(default_fs)


def _load_motion_overlay(motion_dir: Path) -> MotionOverlay:
    motion_p = motion_dir / 'motion.npy'
    time_p = motion_dir / 'time_bins.npy'
    depth_p = motion_dir / 'depth_bins.npy'

    motion = np.load(motion_p)
    time_bins = np.load(time_p)
    depth_bins = np.load(depth_p)

    # time_bins is often in seconds already; if it's in samples that will be obvious (very large)
    time_bins = np.asarray(time_bins).astype(float)
    depth_bins = np.asarray(depth_bins).astype(float)

    return MotionOverlay(motion=motion, time_bins_s=time_bins, depth_bins_um=depth_bins)


def _subsample_indices(n: int, max_n: int, seed: int = 0) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_n, replace=False)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_good_unit_ids(sorter_output_dir: Path) -> set[int]:
    """Read KSLabel TSV and return unit ids labeled 'good'."""
    tsv = sorter_output_dir / 'cluster_KSLabel.tsv'
    if not tsv.exists():
        return set()

    # Avoid pandas dependency; TSV is small.
    good: set[int] = set()
    lines = tsv.read_text().splitlines()
    if not lines:
        return good
    # header: cluster_id\tKSLabel (or similar)
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split('\t')]
        if len(parts) < 2:
            continue
        try:
            cid = int(parts[0])
        except Exception:
            continue
        label = parts[1].lower()
        if label == 'good':
            good.add(cid)
    return good


def _build_unit_index(spike_clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (order, sorted_clusters) for efficient per-unit slicing."""
    order = np.argsort(spike_clusters, kind='mergesort')
    return order, spike_clusters[order]


def _unit_slice(sorted_clusters: np.ndarray, unit_id: int) -> slice:
    left = int(np.searchsorted(sorted_clusters, unit_id, side='left'))
    right = int(np.searchsorted(sorted_clusters, unit_id, side='right'))
    return slice(left, right)


def _select_high_amp_spikes_for_good_units(
    *,
    spike_times_samp: np.ndarray,
    spike_clusters: np.ndarray,
    spike_pos: np.ndarray,
    spike_amp: np.ndarray | None,
    good_unit_ids: set[int],
    fs: float,
    max_good_units: int,
    amp_quantile_per_unit: float,
    per_unit_max_spikes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_s, deflection_um, unit_mean_y_um, unit_id_per_spike) for selected spikes."""
    if not good_unit_ids:
        return (np.array([]), np.array([]), np.array([]), np.array([], dtype=int))

    # Pick top-K good units by spike count (using all spikes)
    u_all, c_all = np.unique(spike_clusters, return_counts=True)
    count_map = {int(u): int(c) for u, c in zip(u_all, c_all)}
    good_sorted = sorted(good_unit_ids, key=lambda u: count_map.get(int(u), 0), reverse=True)
    good_keep = good_sorted[: max_good_units]

    order, sorted_clusters = _build_unit_index(spike_clusters.astype(int))

    rng = np.random.default_rng(int(seed))
    t_out: list[np.ndarray] = []
    d_out: list[np.ndarray] = []
    mean_out: list[np.ndarray] = []
    uid_out: list[np.ndarray] = []

    for uid in good_keep:
        sl = _unit_slice(sorted_clusters, int(uid))
        if sl.start == sl.stop:
            continue
        idx_u = order[sl]

        y_u = spike_pos[idx_u, 1].astype(float)
        if len(y_u) < 200:
            continue
        mean_y = float(np.nanmean(y_u))

        if spike_amp is not None:
            a_u = spike_amp[idx_u].astype(float)
            thr = float(np.nanquantile(a_u, amp_quantile_per_unit))
            keep = idx_u[a_u >= thr]
            if len(keep) == 0:
                continue
            # Prefer top amplitudes; cap per unit
            if len(keep) > per_unit_max_spikes:
                a_keep = spike_amp[keep].astype(float)
                top = np.argsort(a_keep)[-per_unit_max_spikes:]
                keep = keep[top]
        else:
            # Fallback: random subsample spikes for this unit
            keep = idx_u
            if len(keep) > per_unit_max_spikes:
                keep = rng.choice(keep, size=per_unit_max_spikes, replace=False)

        t_u = spike_times_samp[keep].astype(float) / fs
        y_sel = spike_pos[keep, 1].astype(float)
        defl = y_sel - mean_y

        t_out.append(t_u)
        d_out.append(defl)
        mean_out.append(np.full(len(defl), mean_y, dtype=float))
        uid_out.append(np.full(len(defl), int(uid), dtype=int))

    if not t_out:
        return (np.array([]), np.array([]), np.array([]), np.array([], dtype=int))

    t_all = np.concatenate(t_out)
    d_all = np.concatenate(d_out)
    mean_all = np.concatenate(mean_out)
    uid_all = np.concatenate(uid_out)

    o = np.argsort(t_all)
    return t_all[o], d_all[o], mean_all[o], uid_all[o]


def main() -> None:
    ap = argparse.ArgumentParser(description='Check residual motion after sorting using spike positions over time.')
    ap.add_argument('--pipeline-dir', type=Path, required=True, help='Pipeline results folder (contains kilosort4/sorter_output).')
    ap.add_argument('--sorter-output', type=Path, default=None, help='Optional explicit sorter_output folder (overrides pipeline-dir).')
    ap.add_argument('--motion-dir', type=Path, default=None, help='Optional dredge-motion folder (contains motion.npy/time_bins.npy/depth_bins.npy).')
    ap.add_argument('--out-dir', type=Path, default=None, help='Output directory (default: <pipeline-dir>/motion_check_after_sorting).')

    ap.add_argument('--max-spikes', type=int, default=2_000_000, help='Max spikes to use for plots (random subsample).')
    ap.add_argument('--max-good-units', type=int, default=40, help='Max number of good units to include in deflection plot.')
    ap.add_argument('--amp-quantile', type=float, default=0.95, help='Per-unit amplitude quantile for selecting spikes for deflection plot.')
    ap.add_argument('--per-unit-max-spikes', type=int, default=20_000, help='Cap selected spikes per unit for deflection plot.')

    ap.add_argument('--time-bins', type=int, default=300, help='# time bins for 2D histogram.')
    ap.add_argument('--depth-bins', type=int, default=250, help='# depth bins for 2D histogram.')
    ap.add_argument('--seed', type=int, default=0, help='RNG seed for subsampling.')

    args = ap.parse_args()

    pipeline_dir: Path = args.pipeline_dir
    sorter_out = args.sorter_output or (pipeline_dir / 'kilosort4' / 'sorter_output')
    if not sorter_out.exists():
        raise FileNotFoundError(f'Missing sorter_output: {sorter_out}')

    out_dir = args.out_dir or (pipeline_dir / 'motion_check_after_sorting')
    _ensure_dir(out_dir)

    fs = _load_sampling_frequency(sorter_out)

    spike_times_samp = np.load(sorter_out / 'spike_times.npy')
    spike_pos = np.load(sorter_out / 'spike_positions.npy')
    spike_clusters = np.load(sorter_out / 'spike_clusters.npy').astype(int)

    if spike_pos.ndim != 2 or spike_pos.shape[1] < 2:
        raise RuntimeError(f'Unexpected spike_positions shape: {spike_pos.shape}')

    amps_p = sorter_out / 'amplitudes.npy'
    spike_amp = np.load(amps_p) if amps_p.exists() else None

    n_spikes = int(len(spike_times_samp))
    if len(spike_pos) != n_spikes:
        raise RuntimeError(f'Length mismatch: spike_times={n_spikes} spike_positions={len(spike_pos)}')
    if spike_amp is not None and len(spike_amp) != n_spikes:
        spike_amp = None
    if len(spike_clusters) != n_spikes:
        raise RuntimeError(f'Length mismatch: spike_times={n_spikes} spike_clusters={len(spike_clusters)}')

    idx = _subsample_indices(n_spikes, max_n=int(args.max_spikes), seed=int(args.seed))

    t_s = spike_times_samp[idx].astype(float) / fs
    y_um = spike_pos[idx, 1].astype(float)
    x_um = spike_pos[idx, 0].astype(float)
    a = spike_amp[idx].astype(float) if spike_amp is not None else None

    # Basic ranges
    t_min, t_max = float(np.nanmin(t_s)), float(np.nanmax(t_s))
    y_min, y_max = float(np.nanmin(y_um)), float(np.nanmax(y_um))
    dur_min = (t_max - t_min) / 60.0

    # Histogram grid
    t_edges = np.linspace(t_min, t_max, int(args.time_bins) + 1)
    y_edges = np.linspace(y_min, y_max, int(args.depth_bins) + 1)

    # 2D density of spikes (time vs depth)
    H, _, _ = np.histogram2d(t_s, y_um, bins=[t_edges, y_edges])
    H = H.T  # (depth, time)

    t_centers = (t_edges[:-1] + t_edges[1:]) / 2

    # Optional motion overlay
    motion_dir = args.motion_dir
    if motion_dir is None:
        motion_dir = _infer_sweep_motion_dir(pipeline_dir)
    overlay = None
    if motion_dir is not None and motion_dir.exists():
        try:
            overlay = _load_motion_overlay(motion_dir)
        except Exception:
            overlay = None

    # ----------------
    # Figure 1: heatmap + traces
    # ----------------
    fig = plt.figure(figsize=(8.6, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.2], hspace=0.18)

    ax0 = fig.add_subplot(gs[0, 0])
    # avoid LogNorm issues when empty
    vmax = np.nanmax(H) if np.isfinite(H).any() else 1.0
    im = ax0.imshow(
        H,
        origin='lower',
        aspect='auto',
        extent=[t_edges[0] / 60.0, t_edges[-1] / 60.0, y_edges[0], y_edges[-1]],
        norm=LogNorm(vmin=1, vmax=max(vmax, 2)),
        cmap='magma',
    )
    cb = fig.colorbar(im, ax=ax0, shrink=0.85)
    cb.set_label('Spike count/bin (log)')

    ax0.set_ylabel('Spike depth y (µm)')
    ax0.set_title(
        f'Post-sort spike positions over time (subsample={len(idx):,}/{n_spikes:,}, dur≈{dur_min:.1f} min)\n'
        f'{pipeline_dir.name}'
    )

    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1.scatter(t_s / 60.0, x_um, s=0.2, alpha=0.15, color='#333')
    ax1.set_ylabel('Spike x (µm)')
    ax1.set_xlabel('Time (min)')
    ax1.set_ylim(np.nanpercentile(x_um, 1), np.nanpercentile(x_um, 99))

    fig_path_pdf = out_dir / 'fig_postsort_spikepos_depth_time.pdf'
    fig_path_png = out_dir / 'fig_postsort_spikepos_depth_time.png'
    fig.savefig(fig_path_pdf, bbox_inches='tight')
    fig.savefig(fig_path_png, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ----------------
    # Figure 2: good-unit high-amp spike deflection from unit mean depth
    # ----------------
    good_unit_ids = _load_good_unit_ids(sorter_out)
    t_defl_s, defl_um, unit_mean_y, unit_ids = _select_high_amp_spikes_for_good_units(
        spike_times_samp=spike_times_samp,
        spike_clusters=spike_clusters,
        spike_pos=spike_pos,
        spike_amp=spike_amp,
        good_unit_ids=good_unit_ids,
        fs=fs,
        max_good_units=int(args.max_good_units),
        amp_quantile_per_unit=float(args.amp_quantile),
        per_unit_max_spikes=int(args.per_unit_max_spikes),
        seed=int(args.seed),
    )

    if len(t_defl_s):
        # color by unit mean depth (gives a stable mapping without huge legends)
        c = unit_mean_y
        fig = plt.figure(figsize=(8.6, 5.6))
        gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.2], hspace=0.18)

        ax = fig.add_subplot(gs[0, 0])
        sc = ax.scatter(
            t_defl_s / 60.0,
            defl_um,
            c=c,
            s=0.25,
            alpha=0.18,
            cmap='viridis',
            linewidths=0,
            rasterized=True,
        )
        ax.axhline(0, color='#111', lw=0.8, ls='--')
        ax.set_ylabel('Deflection from unit mean y (µm)')
        ax.set_title(
            'Good units: highest-amplitude spikes\n'
            f'deflection per spike vs time (amp≥q{args.amp_quantile:.2f} per unit; max {args.max_good_units} units)'
        )
        cb = fig.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label('Unit mean depth y (µm)')

        # Time-binned mean deflection across all selected spikes
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
        tb = np.linspace(float(t_defl_s.min()), float(t_defl_s.max()), int(args.time_bins) + 1)
        tc = (tb[:-1] + tb[1:]) / 2
        bi = np.clip(np.digitize(t_defl_s, tb) - 1, 0, len(tb) - 2)
        mean_defl = np.full(len(tc), np.nan)
        std_defl = np.full(len(tc), np.nan)
        n_bin = np.zeros(len(tc), dtype=int)
        for b in range(len(tc)):
            m = bi == b
            if m.sum() < 100:
                continue
            vals = defl_um[m]
            mean_defl[b] = float(np.nanmean(vals))
            std_defl[b] = float(np.nanstd(vals))
            n_bin[b] = int(m.sum())

        ax2.plot(tc / 60.0, mean_defl, color='k', lw=1.1)
        ax2.fill_between(tc / 60.0, mean_defl - std_defl, mean_defl + std_defl, color='k', alpha=0.15, linewidth=0)
        ax2.axhline(0, color='#111', lw=0.8, ls='--')
        ax2.set_xlabel('Time (min)')
        ax2.set_ylabel('Mean defl (µm)')

        fig.savefig(out_dir / 'fig_good_units_deflection_time.pdf', bbox_inches='tight')
        fig.savefig(out_dir / 'fig_good_units_deflection_time.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

        # Save per-spike table (useful for downstream checks)
        out_csv = out_dir / 'good_units_deflection_spikes.csv'
        # Keep CSV size manageable
        max_rows = 2_000_000
        if len(t_defl_s) > max_rows:
            keep = _subsample_indices(len(t_defl_s), max_rows, seed=int(args.seed))
            t_csv = t_defl_s[keep]
            d_csv = defl_um[keep]
            my_csv = unit_mean_y[keep]
            uid_csv = unit_ids[keep]
        else:
            t_csv, d_csv, my_csv, uid_csv = t_defl_s, defl_um, unit_mean_y, unit_ids

        # write CSV without pandas
        header = 'time_s,deflection_um,unit_mean_y_um,unit_id\n'
        with out_csv.open('w') as f:
            f.write(header)
            for tt, dd, mm, uu in zip(t_csv, d_csv, my_csv, uid_csv):
                f.write(f'{tt:.6f},{dd:.3f},{mm:.3f},{int(uu)}\n')
    else:
        out_csv = None

    # ----------------
    # Figure 3: motion overlay (if available)
    # ----------------
    if overlay is not None:
        motion = np.asarray(overlay.motion)
        tb = np.asarray(overlay.time_bins_s).astype(float)
        db = np.asarray(overlay.depth_bins_um).astype(float)

        # Make time centers if edges were provided
        if len(tb) == motion.shape[0] + 1:
            t_m = (tb[:-1] + tb[1:]) / 2
        else:
            t_m = tb[: motion.shape[0]]

        if len(db) == motion.shape[1] + 1:
            d_m = (db[:-1] + db[1:]) / 2
        else:
            d_m = db[: motion.shape[1]]

        # Choose a representative depth: median of spike depths
        target_depth = float(np.nanmedian(y_um))
        j = int(np.nanargmin(np.abs(d_m - target_depth))) if len(d_m) else 0
        motion_trace = motion[:, j].astype(float)

        # Align a post-sort trace to the same time base for comparison (roughly)
        # Prefer the high-amp good-unit mean deflection trace if available.
        fig, ax = plt.subplots(figsize=(8.6, 3.6))
        if len(t_defl_s):
            # use the same binning tb/tc from above if possible
            ax.plot(tc, mean_defl, color='k', lw=1.2, label='post-sort mean deflection (µm)')
            ax.axhline(0, color='#111', lw=0.8, ls='--')
            ax.set_ylabel('Mean deflection (µm)')
        else:
            ax.plot(t_centers, np.zeros_like(t_centers), color='k', lw=1.2, label='post-sort (no good-unit defl trace)')
            ax.set_ylabel('Deflection (µm)')
        ax.set_xlabel('Time (s)')

        ax2 = ax.twinx()
        ax2.plot(t_m, motion_trace, color='tab:red', lw=1.0, alpha=0.9, label=f'est. motion @ {d_m[j]:.0f}µm')
        ax2.set_ylabel('Estimated motion (µm)')

        # merged legend (filter out underscore/private labels)
        lines = ax.get_lines() + ax2.get_lines()
        keep = [ln for ln in lines if not str(ln.get_label()).startswith('_')]
        labels = [ln.get_label() for ln in keep]
        ax.legend(keep, labels, loc='upper right', fontsize=8, frameon=False)

        ax.set_title('Post-sort depth trace vs motion estimate (dredge-motion)')
        fig.savefig(out_dir / 'fig_postsort_vs_motion_estimate.pdf', bbox_inches='tight')
        fig.savefig(out_dir / 'fig_postsort_vs_motion_estimate.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

        # Also save the motion heatmap itself
        fig, ax = plt.subplots(figsize=(8.6, 4.2))
        im = ax.imshow(
            motion.T,
            origin='lower',
            aspect='auto',
            extent=[t_m[0] / 60.0, t_m[-1] / 60.0, d_m[0], d_m[-1]],
            cmap='coolwarm',
        )
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label('Estimated motion (µm)')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Depth (µm)')
        ax.set_title('Motion estimate (dredge-motion)')
        fig.savefig(out_dir / 'fig_motion_estimate_heatmap.pdf', bbox_inches='tight')
        fig.savefig(out_dir / 'fig_motion_estimate_heatmap.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

    # Save a small JSON summary
    summary = dict(
        pipeline_dir=str(pipeline_dir),
        sorter_output=str(sorter_out),
        fs=float(fs),
        n_spikes_total=n_spikes,
        n_spikes_plotted=int(len(idx)),
        n_good_units=int(len(_load_good_unit_ids(sorter_out))),
        n_deflection_spikes=int(len(t_defl_s)),
        deflection_amp_quantile=float(args.amp_quantile),
        deflection_max_good_units=int(args.max_good_units),
        deflection_per_unit_max_spikes=int(args.per_unit_max_spikes),
        duration_s=float(t_max - t_min),
        depth_range_um=[float(y_min), float(y_max)],
        motion_dir=str(motion_dir) if motion_dir is not None else None,
        has_motion_overlay=bool(overlay is not None),
    )
    (out_dir / 'motion_check_summary.json').write_text(json.dumps(summary, indent=2))

    print('Saved:')
    print(' ', fig_path_pdf)
    print(' ', fig_path_png)
    if len(t_defl_s):
        print(' ', out_dir / 'fig_good_units_deflection_time.pdf')
        print(' ', out_dir / 'good_units_deflection_spikes.csv')
    if overlay is not None:
        print(' ', out_dir / 'fig_postsort_vs_motion_estimate.pdf')
        print(' ', out_dir / 'fig_motion_estimate_heatmap.pdf')
    print(' ', out_dir / 'motion_check_summary.json')


if __name__ == '__main__':
    main()
