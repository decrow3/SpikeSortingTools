"""Where do the deterministic threshold cliffs first diverge?

The static sweep found reproducible collapses beside near-perfect neighbours —
D10 at 12/9 (0.475, 275 FP) against 9/9 (0.994, 0 FP); D14 at 9/8 (0.744, 229 FP)
against 10/8 (0.990, 0 FP); and D14 at 12/7 (0.632, 399 FP). Fifteen independent
re-sorts were bit-identical, so these are cliffs in the response surface, not
variance. Avoiding them by picking a safe cell is not the same as knowing why
they happen — a cliff we cannot explain may move on held-out data.

This localises each collapse to its **first divergent sorter stage**, reading
only retained arrays. No sorting, no GPU, so it runs beside the staircase job.

The stages KS4 exposes, in order
--------------------------------
1. **detection** — `full_st` (n, 3), column 0 sample times: every threshold
   crossing found by the universal templates. Governed by `Th_universal`.
2. **kept** — `kept_spikes`, the boolean mask from detections to final spikes.
3. **assignment** — `full_clu` at detection versus `spike_clusters` at output;
   template learning is governed by `Th_learned`, and `templates.npy` says how
   many templates were learned.

So an injected event that is absent from `full_st` was never detected
(a `Th_universal` effect), while one present in `full_st` but missing from the
matched output cluster was detected and then lost or misassigned (a clustering
effect). That distinction is what "first divergent stage" means here, and it
decides whether a cliff is a detection or a clustering phenomenon.

Run: `python testing/luke_c2_threshold_stage_trace.py`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from testing.ladder_score import truth_digest
from testing.luke_rescue_c2_drift_challenge import PRESPEC as C2_PRESPEC

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = REPO_ROOT / "testing/outputs/luke_c2_threshold_sweep"
DEFAULT_OUTPUT = REPO_ROOT / "testing/outputs/luke_c2_threshold_stage_trace"

SCHEMA = "luke-c2-threshold-stage-trace-v1"
TOL_MS = 0.5

CASES = [
    {"donor": "D10", "fail": "th_12_9", "success": "th_9_9",
     "note": "primary: production cell collapses, recovery candidate does not"},
    {"donor": "D14", "fail": "th_9_8", "success": "th_10_8",
     "note": "primary: collapse between much better neighbours"},
    {"donor": "D14", "fail": "th_12_7", "success": "th_10_8",
     "note": "secondary: the other high-FP failure"},
]


def find_leaves(sweep_root: Path = SWEEP_ROOT) -> dict:
    """Map (donor, cell) -> the curated result and its sorter output."""
    index = {}
    for path in sorted((sweep_root / "runs/_l1").glob("*/cur-*/l1_result.json")):
        result = json.loads(path.read_text())
        donor = Path(result["snippet_dir"]).name.replace("_static", "")
        index[(donor, result["sorter_config"])] = {
            "l1_result": path,
            "sorter_output": path.parent.parent / (
                "sort" if result["sorter_config"] == "rescue"
                else f"sort-{path.parent.name.split('-')[1]}"
            ) / "sorter_output",
            "result": result,
        }
    return index


def truth_train(fs: float) -> np.ndarray:
    """The sweep's static train, regenerated deterministically."""
    return np.arange(
        int(C2_PRESPEC["train"]["guard_s"] * fs),
        int(C2_PRESPEC["background"]["duration_s"] * fs)
        - int(C2_PRESPEC["train"]["guard_s"] * fs),
        int(round(fs / C2_PRESPEC["train"]["rate_hz"])), dtype=np.int64,
    )


def matched(reference: np.ndarray, candidate: np.ndarray, tol: int) -> np.ndarray:
    """Boolean per reference event: is there a candidate spike within tol?"""
    if candidate.size == 0:
        return np.zeros(reference.size, dtype=bool)
    candidate = np.sort(candidate)
    idx = np.searchsorted(candidate, reference)
    best = np.full(reference.size, np.iinfo(np.int64).max, dtype=np.int64)
    for offset in (-1, 0):
        probe = np.clip(idx + offset, 0, candidate.size - 1)
        best = np.minimum(best, np.abs(candidate[probe] - reference))
    return best <= tol


def trace_cell(entry: dict, tol_ms: float = TOL_MS) -> dict:
    """Decompose one (donor, cell) into detection / kept / assignment."""
    out_dir = entry["sorter_output"]
    result = entry["result"]
    fs = float(result["score"]["fs"])
    tol = int(round(tol_ms / 1000.0 * fs))
    train = truth_train(fs)

    contract = result["score"].get("truth_contract")
    if contract and truth_digest({"inj0": train}) != contract["truth_sha256"]:
        raise RuntimeError(
            "regenerated train does not match the contract this cell was scored "
            "against; the trace would describe a different experiment"
        )

    # Detection stages come from the sorter output; the *scored* arrays are the
    # curated ones, and `best_output_unit` is a curated cluster id. Reading
    # `spike_clusters.npy` from sorter_output instead silently compares against
    # pre-merge ids, which do not correspond.
    full_st = np.load(out_dir / "full_st.npy")[:, 0].astype(np.int64)
    kept = np.load(out_dir / "kept_spikes.npy")
    templates = np.load(out_dir / "templates.npy", mmap_mode="r")
    curated = entry["l1_result"].parent / "cur_output"
    final_st = np.load(curated / "spike_times.npy").reshape(-1).astype(np.int64)
    final_clu = np.load(curated / "spike_clusters.npy").reshape(-1)

    detected = matched(train, full_st, tol)
    kept_st = full_st[kept]
    survived = matched(train, kept_st, tol)
    in_final = matched(train, final_st, tol)

    unit = result["score"]["primary"]["units"][0]
    best = unit["best_output_unit"]
    best_st = final_st[final_clu == best] if best is not None else np.array([], dtype=np.int64)
    in_best = matched(train, best_st, tol)

    return {
        "cell": result["sorter_config"],
        "Th_universal": result["sorter_overrides"].get("Th_universal"),
        "Th_learned": result["sorter_overrides"].get("Th_learned"),
        "accuracy": round(unit["accuracy"], 4),
        "n_truth": int(train.size),
        "stage_detection_found": int(detected.sum()),
        "stage_kept_found": int(survived.sum()),
        "stage_final_found": int(in_final.sum()),
        "stage_best_cluster_found": int(in_best.sum()),
        "n_detections_total": int(full_st.size),
        "n_final_total": int(final_st.size),
        "n_templates_learned": int(templates.shape[0]),
        "n_clusters_final": int(np.unique(final_clu).size),
        "best_cluster_size": int(best_st.size),
        "best_cluster_contamination": (
            round(float(1.0 - in_best.sum() / best_st.size), 4) if best_st.size else None
        ),
        "n_output_units_capturing": unit["n_output_units_capturing"],
        "fp": unit["fp"], "fn": unit["fn"],
    }


def first_divergent_stage(fail: dict, success: dict, tolerance_frac: float = 0.02) -> dict:
    """The earliest stage at which the two cells differ materially."""
    n = fail["n_truth"]
    for stage in ("stage_detection_found", "stage_kept_found",
                  "stage_final_found", "stage_best_cluster_found"):
        delta = success[stage] - fail[stage]
        if abs(delta) > tolerance_frac * n:
            return {
                "first_divergent_stage": stage,
                "delta_events": int(delta),
                "fail": fail[stage], "success": success[stage],
                "interpretation": (
                    "detection: the failing cell never found these events "
                    "(a Th_universal effect)" if stage == "stage_detection_found" else
                    "post-detection: the events were found and then lost or "
                    "misassigned (a clustering effect)"
                ),
            }
    return {
        "first_divergent_stage": "none",
        "interpretation": (
            "the injected events survive every stage in both cells; the "
            "difference is contamination of the winning cluster, not lost spikes"
        ),
    }


def run(root=None, sweep_root: Path = SWEEP_ROOT) -> dict:
    root = Path(root or DEFAULT_OUTPUT)
    if str(root.resolve()).startswith("/mnt/"):
        raise ValueError("refusing an output root under /mnt")
    root.mkdir(parents=True, exist_ok=True)
    index = find_leaves(sweep_root)

    traces, comparisons = [], []
    for case in CASES:
        pair = {}
        for role in ("fail", "success"):
            key = (case["donor"], case[role])
            if key not in index:
                raise RuntimeError(f"no retained sweep output for {key}")
            trace = trace_cell(index[key])
            trace.update({"donor": case["donor"], "role": role})
            traces.append(trace)
            pair[role] = trace
        comparisons.append({
            "donor": case["donor"], "fail_cell": case["fail"],
            "success_cell": case["success"], "note": case["note"],
            **first_divergent_stage(pair["fail"], pair["success"]),
            "contamination_fail": pair["fail"]["best_cluster_contamination"],
            "contamination_success": pair["success"]["best_cluster_contamination"],
            "templates_fail": pair["fail"]["n_templates_learned"],
            "templates_success": pair["success"]["n_templates_learned"],
        })

    import pandas as pd
    pd.DataFrame(traces).to_csv(root / "stage_traces.csv", index=False)
    summary = {"schema": SCHEMA, "tol_ms": TOL_MS, "cases": CASES,
               "comparisons": comparisons, "traces": traces}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()
    summary = run(root=args.out_root)
    print(json.dumps({"comparisons": summary["comparisons"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
