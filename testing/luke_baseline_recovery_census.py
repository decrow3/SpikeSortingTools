"""Baseline-only recovery census; no sort, candidate data, or fitter execution.

Keep the original abrupt-transition rule, remove its per-sort nomination cap,
and separately enumerate a broader exploratory rule: two adjacent valid stable
windows followed by two adjacent valid deteriorated windows in the SAME permitted
development interval. Each pair is gap-free; the interval BETWEEN the pairs may
contain gaps or invalid windows and is reported, never fitted across. Pick the
shortest qualifying span per cluster, ties by start then larger change. Exclude
rescue cluster 37. No new hypothesis test or promotion decision is performed.
"""
from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from testing.luke_amplitude_dropout_audit import (
    _structural_reason, classify_failure_run, parse_selection_constants,
    read_attested_windows, window_records,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "testing/outputs/luke_amplitude_dropout_audit/run3_corrected_rule"
OUT = ROOT / "docs/outputs/luke_baseline_recovery_census_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def census(windows, constants, intervals):
    """Return one status row per inventoried cluster and every uncapped abrupt case."""
    grouped = window_records(windows)
    ids = windows[["sort_id", "cluster_id"]].drop_duplicates()
    records, abrupt = [], []
    for sort_id, cid in ids.itertuples(index=False, name=None):
        cid = int(cid)
        rows = grouped.get((sort_id, cid), [])
        record = dict(sort_id=sort_id, cluster_id=cid, n_cached_windows=len(rows))
        if "rescue" in sort_id and cid == 37:
            records.append({**record, "status": "excluded_closed_cluster_37"})
            continue
        pairs = []
        for j in range(len(rows) - 1):
            pair = rows[j:j + 2]
            if _structural_reason(pair, constants) is not None:
                continue
            for di, (lo, hi) in enumerate(intervals):
                if lo <= pair[0]["start_s"] and pair[-1]["end_s"] <= hi:
                    values = [r["missing_pct"] for r in pair]
                    if not all(np.isfinite(v) for v in values):
                        continue
                    pairs.append((j, di, pair, values))
                    break
        stable = [p for p in pairs if max(p[3]) <= constants.reference_max_missing_pct]
        failing = [p for p in pairs if min(p[3]) >= constants.failing_min_missing_pct]
        record.update(n_valid_dev_pairs=len(pairs), n_stable_pairs=len(stable),
                      n_failing_pairs=len(failing))
        best = None
        for fj, di, fp, fv in failing:
            preceding = [p for p in stable if p[1] == di and p[0] + 1 < fj]
            if not preceding:
                continue
            sj, _, sp, sv = max(preceding, key=lambda p: p[2][0]["start_s"])
            delta = float(np.median(fv) - np.median(sv))
            if delta < constants.min_median_difference_pp:
                continue
            start, stop = sp[0]["start_s"], fp[-1]["end_s"]
            candidate = dict(start_s=start, end_s=stop, span_s=stop-start,
                reference_end_s=sp[-1]["end_s"], failing_start_s=fp[0]["start_s"],
                reference_missing_pct=float(np.median(sv)), failing_missing_pct=float(np.median(fv)),
                difference_pp=delta, intervening_windows=fj-sj-2, dev_interval_index=di,
                source_rows=[r["source_row"] for r in sp+fp],
                nominal_samples_in_four_fits=4000,
                historical_samples_in_four_fits=3996,
                fitted_window_duration_s=sum(r["end_s"]-r["start_s"] for r in sp+fp))
            key = lambda p: (p["span_s"], p["start_s"], -p["difference_pp"])
            if best is None or key(candidate) < key(best):
                best = candidate
        for j in range(len(rows)-3):
            run = rows[j:j+4]
            # Restrict before testing amplitude criteria; held-out rankings aren't read.
            if not any(lo <= run[0]["start_s"] and run[-1]["end_s"] <= hi for lo,hi in intervals):
                continue
            reason, metrics = classify_failure_run(run, constants)
            if reason == "qualified":
                abrupt.append(dict(sort_id=sort_id, cluster_id=cid,
                    start_s=run[0]["start_s"], end_s=run[-1]["end_s"], **metrics))
        status = ("qualifying_transition" if best else "no_cached_windows" if not rows else
                  "no_valid_dev_pair" if not pairs else "no_stable_pair" if not stable else
                  "no_deteriorated_pair" if not failing else "no_ordered_pair_in_same_dev_interval")
        records.append({**record, "status": status, **(best or {})})
    return pd.DataFrame(records), pd.DataFrame(abrupt)


def main():
    selection = json.loads((SOURCE / "selection.json").read_text())
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    windows, windows_hash = read_attested_windows(SOURCE / "windows.csv", selection["windows_csv_sha256"])
    constants = parse_selection_constants(selection["selection_constants"])
    intervals = selection["interval_contract"]["development_windows_s"]
    table, abrupt = census(windows, constants, intervals)
    extra_hashes = {}
    for sid, meta in manifest["sorts"].items():
        folder = Path(meta["curated"])
        for name, column in [("cluster_KSLabel.tsv", "KSLabel"), ("cluster_ContamPct.tsv", "ContamPct")]:
            path = folder / name
            raw = path.read_bytes()
            extra_hashes[str(path)] = hashlib.sha256(raw).hexdigest()
            labels = pd.read_csv(io.BytesIO(raw), sep="\t")
            mapping = dict(zip(labels["cluster_id"], labels[column]))
            mask = table.sort_id == sid
            table.loc[mask, column] = table.loc[mask, "cluster_id"].map(mapping)
        table.loc[table.sort_id == sid, "source_voltage_exists"] = any(
            Path(meta["source_recording"]).glob("*.raw"))
        table.loc[table.sort_id == sid, "template_file_exists"] = (folder / "templates.npy").exists()
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "cluster_census.csv", index=False)
    abrupt.to_csv(OUT / "uncapped_original_rule_cases.csv", index=False)
    qualified = table[table.status == "qualifying_transition"].sort_values(
        ["KSLabel", "ContamPct", "span_s", "sort_id", "cluster_id"])
    qualified.to_csv(OUT / "candidate_cases.csv", index=False)
    summary = {"schema": "luke-baseline-recovery-census-v1", "windows_sha256": windows_hash,
        "selection_sha256": digest(SOURCE / "selection.json"), "source_sha256": digest(__file__),
        "extra_input_hashes": extra_hashes, "intervals": intervals,
        "rules": __doc__, "selection_constants": constants.__dict__,
        "by_sort": {sid: {"clusters": len(g), "good": int((g.KSLabel == "good").sum()),
            "status_counts": dict(Counter(g.status)),
            "good_status_counts": dict(Counter(g[g.KSLabel == "good"].status))}
            for sid,g in table.groupby("sort_id")}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary["by_sort"], indent=2))
    print(qualified.drop(columns=["source_rows"]).to_string(index=False))


def inspect_shortlist():
    """Read the three abrupt rescue cases with reported ContamPct <=2%.

    This descriptive shortlist is chosen AFTER the census, not a frozen
    inferential cohort. No claim of mechanistic independence is made.
    """
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    sid = next(s for s in manifest["sorts"] if s.startswith("rescue"))
    folder = Path(manifest["sorts"][sid]["curated"])
    config = json.loads((ROOT / "testing/configs/luke_amplitude_dropout_audit_v1.json").read_text())
    fs = next(s["sampling_frequency_hz"] for s in config["sorts"] if s["sort_id"] == sid)
    names = ["spike_times.npy", "spike_clusters.npy", "full_st.npy", "kept_spikes.npy", "spike_positions.npy"]
    arrays = {n: np.load(folder / n, mmap_mode="r") for n in names}
    times = arrays["spike_times.npy"].reshape(-1)
    clusters = arrays["spike_clusters.npy"].reshape(-1)
    kept = arrays["kept_spikes.npy"].reshape(-1)
    full_indices = np.flatnonzero(kept) if kept.dtype.kind == "b" else kept
    assert len(full_indices) == len(times) == len(clusters)
    candidates = pd.read_csv(OUT / "candidate_cases.csv")
    candidates = candidates[(candidates.sort_id == sid) & (candidates.cluster_id.isin([553,452,36]))]
    evidence = []
    for row in candidates.to_dict("records"):
        cid = int(row["cluster_id"])
        indices = np.flatnonzero(clusters == cid)
        st = times[indices]
        full = arrays["full_st.npy"][full_indices[indices]]
        assert np.array_equal(st, full[:,0].astype(np.int64))
        depths = arrays["spike_positions.npy"][indices,1]
        phases = []
        for phase, lo, hi in [("reference",row["start_s"],row["reference_end_s"]),
                              ("failing",row["failing_start_s"],row["end_s"])]:
            # Original endpoints identify inclusive first/last retained samples.
            use = (st >= round(lo*fs)) & (st <= round(hi*fs))
            samples = st[use]
            if len(samples) != 2000:
                raise ValueError(f"{cid} {phase}: expected two nominal 1,000-spike windows")
            rv = float(np.mean(np.diff(samples)/fs < .0015))
            phases.append(dict(phase=phase,n_spikes=len(samples),rv_fraction_1_5ms=rv,
                median_sorter_amplitude=float(np.median(full[use,2])),
                median_depth_um=float(np.median(depths[use])),
                depth_p10_um=float(np.percentile(depths[use],10)),
                depth_p90_um=float(np.percentile(depths[use],90))))
        evidence.append(dict(cluster_id=cid,phases=phases,
            depth_change_um=phases[1]["median_depth_um"]-phases[0]["median_depth_um"],
            amplitude_drop_fraction=1-phases[1]["median_sorter_amplitude"]/phases[0]["median_sorter_amplitude"],
            selected_input_sha256=hashlib.sha256(st.tobytes()+full.tobytes()+depths.tobytes()).hexdigest()))
    (OUT / "shortlist_evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
    print(json.dumps(evidence,indent=2))


if __name__ == "__main__":
    import sys
    if "--inspect-shortlist" in sys.argv:
        inspect_shortlist()
    else:
        main()
