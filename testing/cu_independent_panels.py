#!/usr/bin/env python3
"""Independent colourblind-safe panels from already-saved CU arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "same_constituent": "#0072B2",
    "cross_zero_offset": "#009E73",
    "cross_nonzero_offset": "#D55E00",
    "comparison": "#CC79A7",
    "a": "#E69F00",
    "b": "#009E73",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("packet", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    snippets = np.load(args.packet / "PAIR_SNIPPETS_FULL182.npy", mmap_mode="r")
    rows = pd.read_csv(args.packet / "PAIR_SNIPPET_ROWS.csv")
    metrics = pd.read_csv(args.packet / "PAIR_VOLTAGE_METRICS.csv")
    supports = pd.read_csv(args.packet / "TEMPLATE_PHYSICAL_SUPPORTS.csv")
    refs = np.load(args.packet / "REFERENCE_MEAN_WAVEFORMS.npz", allow_pickle=False)

    choices = [("target", 0), ("target", 32), ("target", 64), ("comparison", 0)]
    fig, axes = plt.subplots(len(choices), 2, figsize=(11, 9), constrained_layout=True)
    for row_ix, (kind, pair_id) in enumerate(choices):
        meta = metrics[(metrics.kind == kind) & (metrics.pair_id == pair_id)].iloc[0]
        loc = rows[(rows.kind == kind) & (rows.pair_id == pair_id)].iloc[0]
        voltage = np.array(
            snippets[int(loc.snippet_index), : int(loc.valid_samples)], dtype=float
        )
        sa = supports[supports.constituent == int(meta.constituent_a)]
        sb = supports[supports.constituent == int(meta.constituent_b)]
        physical = np.union1d(sa.physical_channel_index, sb.physical_channel_index).astype(int)
        y_by_channel = (
            pd.concat([sa, sb])
            .drop_duplicates("physical_channel_index")
            .set_index("physical_channel_index")
            .y_um
        )
        physical = np.array(sorted(physical, key=lambda c: (y_by_channel.loc[c], c)))
        shown = voltage[:, physical].T
        shown -= np.median(shown, axis=1, keepdims=True)
        lim = np.quantile(np.abs(shown), 0.98)
        ax = axes[row_ix, 0]
        ax.imshow(shown, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.axvline(loc.marker_a, color=COLORS["a"], lw=1.5, ls="-")
        ax.axvline(loc.marker_b, color=COLORS["b"], lw=1.5, ls="--")
        ax.set_ylabel("physical support")
        ax.set_xlabel("sample in saved window")
        ax.set_title(f"{kind} {pair_id}: {meta.category}, lag={meta.lag_samples}")

        wa = np.array(refs[f"constituent_{int(meta.constituent_a)}"][:, physical], dtype=float)
        wb = np.array(refs[f"constituent_{int(meta.constituent_b)}"][:, physical], dtype=float)
        wa -= np.median(wa, axis=0, keepdims=True)
        wb -= np.median(wb, axis=0, keepdims=True)
        score = np.ptp(wa, axis=0) + np.ptp(wb, axis=0)
        local_ch = int(np.argmax(score))
        trace = shown[local_ch]
        ax = axes[row_ix, 1]
        ax.plot(trace, color="#666666", lw=0.9, label="saved voltage")
        xa = np.arange(121) + int(loc.marker_a) - 42
        xb = np.arange(121) + int(loc.marker_b) - 42
        ax.plot(
            xa,
            float(meta.two_event_amplitude_a) * wa[:, local_ch],
            color=COLORS["a"],
            lw=1.5,
            ls="-",
            label="A fixed-time component",
        )
        ax.plot(
            xb,
            float(meta.two_event_amplitude_b) * wb[:, local_ch],
            color=COLORS["b"],
            lw=1.5,
            ls="--",
            label="B fixed-time component",
        )
        ax.axvline(loc.marker_a, color=COLORS["a"], lw=0.8, ls=":")
        ax.axvline(loc.marker_b, color=COLORS["b"], lw=0.8, ls=":")
        ax.set_xlim(0, int(loc.valid_samples) - 1)
        ax.set_xlabel("sample in saved window")
        ax.set_ylabel("baseline-centred voltage")
        ax.set_title(
            f"2-vs-1 reduction={meta.two_vs_one_fractional_sse_reduction:.3f}; "
            f"amps={meta.two_event_amplitude_a:.2f}/{meta.two_event_amplitude_b:.2f}"
        )
        if row_ix == 0:
            ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.suptitle("CU independent saved-voltage examples (no new voltage read)")
    fig.savefig(args.output / "CU_INDEPENDENT_SAVED_VOLTAGE_EXAMPLES.png", dpi=180)
    plt.close(fig)

    order = ["same_constituent", "cross_zero_offset", "cross_nonzero_offset", "comparison"]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    rng = np.random.default_rng(20260928)
    for pos, category in enumerate(order):
        vals = metrics.loc[metrics.category == category, "two_vs_one_fractional_sse_reduction"].to_numpy()
        jitter = rng.uniform(-0.12, 0.12, size=vals.size)
        ax.scatter(pos + jitter, vals, s=20, alpha=0.65, color=COLORS[category], marker="o")
        q25, med, q75 = np.quantile(vals, [0.25, 0.5, 0.75])
        ax.plot([pos - 0.25, pos + 0.25], [med, med], color=COLORS[category], lw=2.5, ls="-")
        ax.plot([pos, pos], [q25, q75], color=COLORS[category], lw=2, ls="--")
    ax.set_xticks(range(len(order)), [x.replace("_", "\n") for x in order])
    ax.set_ylabel("fractional SSE reduction from second component")
    ax.set_title("Frozen descriptive residual operator; comparison is not a calibrated null")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(args.output / "CU_INDEPENDENT_RESIDUAL_SUMMARY.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
