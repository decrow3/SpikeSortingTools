#!/usr/bin/env python3
"""Make the corrected DE saved-arm table/figure; no scoring recomputation."""
from pathlib import Path
import hashlib
import json
import time

import matplotlib.pyplot as plt
import pandas as pd

SRC = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/de_common_outcome_scorecard_20260928/host_h1/revision_v2")
OUT = Path(__file__).resolve().parents[1] / "testing/outputs/de_common_scorecard_revision_v2_figure_20260928"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    started = time.process_time(); OUT.mkdir(parents=True, exist_ok=False)
    assoc = pd.read_csv(SRC / "UNIT_RATE_ASSOCIATIONS_CORRECTED.csv")
    counts = pd.read_csv(SRC / "PRIMARY_COUNTS_RATES_SCHEMA_CORRECTED.csv")
    primary = counts[(counts.subset == "all_units")].copy()
    table = primary.merge(assoc, on=["window", "arm"], how="left", validate="many_to_one")
    table.to_csv(OUT / "DE_REVISED_PRIMARY_TABLE.csv", index=False)
    w2 = assoc[assoc.window == "W2"].copy()
    order = w2.sort_values("rho_negative_vs_flat", ascending=False)
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = range(len(order))
    ax.plot(x, order.rho_negative_vs_flat, "o-", color=colors[0], label="own flat")
    ax.plot(x, order.rho_negative_vs_same_row_other_flat, "s--", color=colors[1], label="same-row other-unit flat")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(list(x), order.arm, rotation=30, ha="right")
    ax.set_ylabel("Spearman rho (negative excursion vs flat rate)")
    ax.set_title("DE corrected arm-local rate associations — W2")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(OUT / "DE_REVISED_W2_ASSOCIATIONS.png", dpi=180); plt.close(fig)
    receipt = {
        "status": "complete", "cpu_s": time.process_time() - started,
        "source_validation": json.loads((SRC / "VALIDATION.json").read_text()),
        "source_hashes": {p.name: sha(p) for p in [SRC / "PRIMARY_COUNTS_RATES_SCHEMA_CORRECTED.csv", SRC / "UNIT_RATE_ASSOCIATIONS_CORRECTED.csv"]},
        "semantics": "arm-local; no cross-arm unit joins; same-row control excludes own unit; catalogue subset exposure unavailable; DARTsort good labels unavailable",
        "h5_future_inputs": {
            "S_0": "/mnt/NPX/Luke/DARTsort_motion_experiments/dg_lattice_exploratory_w2_20260928/huklaban5/S_0",
            "S_L": "/mnt/NPX/Luke/DARTsort_motion_experiments/dg_lattice_exploratory_w2_20260928/huklaban5/S_L",
            "status": "not included until COMPLETE.json finals exist",
        },
    }
    (OUT / "COMPLETE.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
