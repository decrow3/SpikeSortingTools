"""Checkpoint C — legacy vs rescue against injected truth on the 8-snippet dev panel.

Plan §5 makes injected hybrid ground truth the primary metric and Checkpoint C
its first use: *"a known-truth score exists for legacy on all 8 development
snippets with the sanity condition met, and a measured drift penalty for both."*
The original drift-penalty and panel results are retracted. This version reruns
both configurations with content-bound inputs and exclusive truth matching;
Checkpoint C remains pending until those outputs exist.

For each frozen development snippet it injects the same fixed multi-unit donor
set (D2b-2 compact cohort, spread across the strip, static) into the snippet's
raw-domain µV, then sorts it under `RESCUE` and `LEGACY_STYLE` and scores both
against the injected trains.

**Headline:** units recovered at accuracy ≥ 0.8 with no split and no merge, per
snippet, per config — the one integer the promotion criteria compare.

    python testing/luke_ladder_checkpoint_c.py

Diagnostic-until-held-out. Outputs to testing/outputs/luke_ladder_checkpoint_c_v2/.
Injected recordings go to /media/huklab/Data; nothing under /mnt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from testing.ladder_l1 import l1_run
from testing.ladder_inject import write_injected_recording
from testing.ladder_sorter import LEGACY_STYLE, RESCUE
from testing.ladder_snippets import load_snippet, snippet_root
from testing.luke_injected_ground_truth_benchmark import InjectionEvent, inject_float32_raw_domain, validate_template

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "testing/outputs/luke_ladder_checkpoint_c_v2"
COHORT = REPO_ROOT / "testing/outputs/luke_d2b2_donor_cohort/donor_templates.npz"
INJECT_ROOT = Path("/media/huklab/Data/ladder_checkpoint_c_v2")
PANEL_MANIFEST = "panel_manifest.json"

# Three D2b-2 donors spanning 73–274 µV (low / mid / high SNR), at start-channels
# that keep their ±16-channel footprints non-overlapping in the 112-channel strip
# — overlapping footprints leak a few % of each train into a neighbour's cluster
# and spuriously trip the split/merge gate.
DONOR_PLAN = [
    ("D08", 2), ("D06", 40), ("D02", 78),
]
RATES_HZ = [5.1, 6.3, 7.7]
SANITY_ACC = 0.8


def _dev_snippets() -> list[Path]:
    root = snippet_root()
    panel = json.loads((root / PANEL_MANIFEST).read_text())
    return [root / s["directory"] for s in panel["snippets"] if s["split"] == "development"]


def _inject(bg_uv: np.ndarray, fs: float) -> tuple[np.ndarray, dict]:
    donors = np.load(COHORT)
    n = bg_uv.shape[0]
    guard = int(round(1.0 * fs))
    templates: dict[str, np.ndarray] = {}
    events: list[InjectionEvent] = []
    truth: dict[str, np.ndarray] = {}
    for i, ((tid, ch), rate) in enumerate(zip(DONOR_PLAN, RATES_HZ)):
        uid = f"inj{i}"
        templates[uid] = validate_template(np.asarray(donors[tid], dtype=np.float32), edge_guard_samples=3)
        step = int(round(fs / rate))
        offset = guard + i * (step // len(DONOR_PLAN))
        train = np.arange(offset, n - guard, step, dtype=np.int64)
        truth[uid] = train
        events += [
            InjectionEvent(event_id=f"{uid}-{j}", template_id=uid, sample_index=int(s),
                           amplitude_scale=1.0, channel_shift=ch)
            for j, s in enumerate(train)
        ]
    injected = inject_float32_raw_domain(bg_uv, templates, events, edge_guard_samples=3)
    return injected, truth


def run(only: list[str] | None = None) -> pd.DataFrame:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for snip_dir in _dev_snippets():
        name = snip_dir.name.split("-")[0]
        if only and name not in only:
            continue
        snip = load_snippet(snip_dir)
        injected, truth = _inject(snip.raw_domain_float32(), snip.fs)
        rec_dir = INJECT_ROOT / name
        write_injected_recording(
            rec_dir, injected, channel_positions=snip.channel_positions, fs=snip.fs,
            gain_uv_per_count=snip.gain_uv_per_count, source_snippet_dir=str(snip_dir),
            name=f"ckc_{name}",
        )
        for cfg in (RESCUE, LEGACY_STYLE):
            sorter = None if cfg.label == "rescue" else cfg
            score = l1_run(rec_dir, sorter=sorter, truth=truth, out_root=OUTPUT / "_l1")["score"]
            prim = score["primary"]
            units = {u["truth_unit"]: u for u in prim["units"]}
            accs = [units.get(f"inj{i}", {}).get("accuracy", 0.0) for i in range(len(DONOR_PLAN))]
            # `n_acc_ge_0.8_raw` is accuracy-only and merge-unfiltered: under the
            # per-cluster scorer (decisions/0014) one merged output cluster can
            # give >=0.8 to two truth trains, so it can over-count. The honest
            # recovery count is `units_recovered` (merge- and split-aware).
            recovered_flags = [
                bool(units.get(f"inj{i}", {}).get("recovered", False))
                for i in range(len(DONOR_PLAN))
            ]
            rows.append({
                "snippet": name,
                "motion_regime": snip.manifest["axes"].get("motion_regime"),
                "snr": snip.manifest["axes"].get("snr"),
                "config": cfg.label,
                "units_recovered": int(prim.get("headline_units_recovered", 0)),
                "n_donors_recovered": int(sum(recovered_flags)),
                "n_acc_ge_0.8_raw": int(sum(a >= SANITY_ACC for a in accs)),
                "median_accuracy": round(float(np.median(accs)), 3),
                "n_truth": len(truth),
                **{f"acc_{tid}": round(accs[i], 2) for i, (tid, _) in enumerate(DONOR_PLAN)},
                **{f"cap_{tid}": units.get(f"inj{i}", {}).get("n_output_units_capturing", 0)
                   for i, (tid, _) in enumerate(DONOR_PLAN)},
                "similar_pairs_per_good": round(score["guardrails"]["similar_pairs_per_good_unit"], 4),
                "edge_spike_frac": round(score["guardrails"]["edge_spike_fraction_40um"], 4),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / "checkpoint_c.csv", index=False)

    man = pd.read_csv(
        REPO_ROOT / "testing/outputs/luke_d2b2_donor_cohort/donor_manifest.csv"
    ).set_index("template_id")
    amps = {tid: float(man.loc[tid, "peak_uv"]) for tid, _ in DONOR_PLAN}

    # `headline_units_recovered` (merge- and split-aware, per decisions/0014) is
    # now the honest headline; per-donor best-cluster accuracy is kept as the
    # continuous readout alongside it.
    acc_win = {}
    for s, sub in df.groupby("snippet"):
        r = sub[sub.config == "rescue"].iloc[0]
        l = sub[sub.config == "legacy_style"].iloc[0]
        acc_win[s] = {
            tid: round(float(r[f"acc_{tid}"] - l[f"acc_{tid}"]), 2) for tid, _ in DONOR_PLAN
        }

    summary = {
        "n_snippets": int(df.snippet.nunique()),
        "donor_amplitudes_uv": amps,
        "per_donor_median_accuracy": {
            cfg: {tid: round(float(sub[f"acc_{tid}"].median()), 3) for tid, _ in DONOR_PLAN}
            for cfg, sub in df.groupby("config")
        },
        "rescue_minus_legacy_accuracy_by_snippet_donor": acc_win,
        "snippets_favouring": {
            "rescue": [s for s, w in acc_win.items() if sum(w.values()) > 0.1],
            "legacy_style": [s for s, w in acc_win.items() if sum(w.values()) < -0.1],
            "tied": [s for s, w in acc_win.items() if abs(sum(w.values())) <= 0.1],
        },
        "headline_units_recovered_total": df.groupby("config")["units_recovered"].sum().to_dict(),
        "donors_recovered_total": df.groupby("config")["n_donors_recovered"].sum().to_dict(),
        "sanity_highest_snr_donor_static": {
            "donor": DONOR_PLAN[-1][0],
            "min_accuracy_by_config": {
                cfg: round(float(sub[f"acc_{DONOR_PLAN[-1][0]}"].min()), 3)
                for cfg, sub in df.groupby("config")
            },
            "passes": bool(
                all(
                    sub[f"acc_{DONOR_PLAN[-1][0]}"].min() >= 0.9
                    for _, sub in df.groupby("config")
                )
            ),
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--only", nargs="+")
    args = ap.parse_args()
    df = run(args.only)
    pd.set_option("display.width", 240)
    print(df.to_string(index=False))
    print(json.dumps(json.loads((OUTPUT / "summary.json").read_text()), indent=2))
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
