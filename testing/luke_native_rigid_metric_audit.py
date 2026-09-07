"""Read-only metric audit of the saved native-rigid comparison; runs no sort.

Run from the repository: python testing/luke_native_rigid_metric_audit.py
Output is JSON on stdout, including source hashes for reproducibility.
"""
import collections
import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "testing/outputs/luke_native_rigid_comparison_v1"


def audit(source=SOURCE):
    hashes = {}

    def read(name):
        path = source / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open() as handle:
            return list(csv.DictReader(handle))

    pairs = read("matched_interior_unit_metrics.csv")
    baseline = {r["cluster_id"]: r for r in read("comparison/unit_metrics_baseline.csv")}
    candidate = {r["cluster_id"]: r for r in read("comparison/unit_metrics_candidate.csv")}
    windows = read("comparison/amplitude_windows.csv")
    edges = read("comparison/correspondence_edges.csv")
    rv = [
        float(candidate[r["candidate_cluster"]]["refractory_violation_fraction_1_5ms"])
        - float(baseline[r["baseline_cluster"]]["refractory_violation_fraction_1_5ms"])
        for r in pairs
    ]
    graph = collections.defaultdict(set)
    # Include isolated clusters; connectivity is descriptive, never identity proof.
    for cid in baseline:
        graph["b" + cid]
    for cid in candidate:
        graph["c" + cid]
    for row in edges:
        a, b = "b" + row["baseline_cluster"], "c" + row["candidate_cluster"]
        graph[a].add(b)
        graph[b].add(a)
    seen, components = set(), []
    for node in graph:
        if node in seen:
            continue
        todo, component = [node], []
        seen.add(node)
        while todo:
            current = todo.pop()
            component.append(current)
            for neighbor in graph[current] - seen:
                seen.add(neighbor)
                todo.append(neighbor)
        components.append({
            "baseline_clusters": sum(n.startswith("b") for n in component),
            "candidate_clusters": sum(n.startswith("c") for n in component),
        })
    # Exact counterexamples for the implemented CANDIDATE overlap minus BASELINE
    # overlap score. All extra events in the first example are stipulated true.
    examples = {}
    for name, n_off, n_rigid, shared in (
        ("perfect_rescue", 80, 100, 80),
        ("pure_dropout", 100, 80, 80),
    ):
        examples[name] = shared / n_rigid - shared / n_off
    assert examples["perfect_rescue"] < 0 < examples["pure_dropout"]
    return {
        "schema_version": "native-rigid-metric-audit-v1",
        "source": str(source),
        "source_sha256": hashes,
        "interior_primary_pairs": len(pairs),
        "pairs_with_more_candidate_events": sum(
            int(r["candidate_events"]) > int(r["baseline_events"]) for r in pairs
        ),
        "median_candidate_baseline_event_ratio": statistics.median(
            int(r["candidate_events"]) / int(r["baseline_events"]) for r in pairs
        ),
        "pairs_with_presence_one_in_both": sum(
            float(r["baseline_presence_fraction"]) == float(r["candidate_presence_fraction"]) == 1
            for r in pairs
        ),
        "paired_refractory_delta_median_fraction": statistics.median(rv),
        "paired_refractory_worse_count": sum(v > 0 for v in rv),
        "window_status_counts": {
            arm: dict(collections.Counter(r["status"] for r in windows if r["sort"] == arm))
            for arm in ("baseline", "candidate")
        },
        "graph_components_including_isolates": sorted(
            components, key=lambda c: sum(c.values()), reverse=True
        ),
        "continuity_score_counterexamples": examples,
        "interpretation": "Metric validity audit only; no biological identity or motion efficacy conclusion.",
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
