import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from waveform_preprocessing import center_and_common_median, preprocess_padded_record


PACKET = Path(__file__).resolve().parents[1]
PARENT = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_waveform_contract_targeted_repair_20261002_v2_h5")
CONTRACT_PATH = PACKET / "contracts" / "physical_waveform_evidence.execution_disabled.v4.json"
CLOSURE_PATH = PARENT / "dependencies" / "DEPENDENCY_CLOSURE.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def all_matchings(refs, candidates, tolerance=15, anchor_distance=80):
    edges = {
        (ri, ci)
        for ri, ref in enumerate(refs)
        for ci, candidate in enumerate(candidates)
        if not ref.get("pre_ambiguous") and not candidate.get("pre_ambiguous")
        and abs(ref["frame"] - candidate["frame"]) <= tolerance
        and abs(ref["anchor"] - candidate["anchor"]) <= anchor_distance
    }
    results = []

    def visit(ri, chosen, used_candidates):
        if ri == len(refs):
            results.append(tuple(sorted(chosen)))
            return
        visit(ri + 1, chosen, used_candidates)
        for edge in sorted(edges):
            if edge[0] == ri and edge[1] not in used_candidates:
                visit(ri + 1, chosen + [edge], used_candidates | {edge[1]})

    visit(0, [], set())
    cardinality = max(len(value) for value in results)
    maximal = [value for value in results if len(value) == cardinality]
    cost = min(sum(abs(refs[ri]["frame"] - candidates[ci]["frame"]) for ri, ci in value)
               for value in maximal)
    return [value for value in maximal
            if sum(abs(refs[ri]["frame"] - candidates[ci]["frame"]) for ri, ci in value) == cost]


def classify(refs, candidates):
    solutions = all_matchings(refs, candidates)
    out = {}
    for side, rows in (("REF", refs), ("candidate", candidates)):
        for index, row in enumerate(rows):
            key = (side, row["id"])
            if row.get("pre_ambiguous"):
                out[key] = "ambiguous"
                continue
            assignments = set()
            for solution in solutions:
                if side == "REF":
                    partners = [ci for ri, ci in solution if ri == index]
                    assignments.add(None if not partners else candidates[partners[0]]["id"])
                else:
                    partners = [ri for ri, ci in solution if ci == index]
                    assignments.add(None if not partners else refs[partners[0]]["id"])
            if len(assignments) > 1:
                out[key] = "ambiguous"
            elif next(iter(assignments)) is None:
                out[key] = "lost" if side == "REF" else "gained"
            else:
                out[key] = "retained"
    assert len(out) == len(refs) + len(candidates)
    assert set(out.values()) <= {"retained", "lost", "gained", "ambiguous"}
    return out


def rank_pair(pair, seed):
    payload = "\0".join([
        seed, pair["comparison"], pair["epoch"], "retained", pair["pair_id"],
        str(pair["ref_frame"]), str(pair["candidate_frame"]),
        str(pair["ref_row"]), str(pair["candidate_row"]),
    ]).encode()
    return hashlib.sha256(payload).hexdigest(), pair["pair_id"], pair["ref_row"], pair["candidate_row"]


def test_dependency_closure_and_frozen_impulse():
    closure = json.loads(CLOSURE_PATH.read_text())
    observed = {}
    for relative, expected in closure["members"].items():
        observed[relative] = sha(PARENT / "dependencies" / relative)
        assert observed[relative] == expected
    aggregate = hashlib.sha256(
        json.dumps(observed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert aggregate == closure["aggregate_sha256"]
    assert sha(PACKET / closure["executable_adapter"]["path"]) == closure["executable_adapter"]["sha256"]
    impulse = np.load(PARENT / "dependencies/kilosort/constants/highpass_filter_float32.npy", allow_pickle=False)
    assert impulse.shape == (30122,) and impulse.dtype == np.float32


def test_common_median_reference_known_answer_is_not_average():
    values = torch.tensor([[0, 1, 2], [0, 2, 4], [0, 3, 6], [0, 4, 8]], dtype=torch.float32)
    expected = torch.tensor([[2, 0, -1], [1, 0, 0], [0, 0, 1], [-1, 0, 2]], dtype=torch.float32)
    actual = center_and_common_median(values)
    assert torch.equal(actual, expected)
    average_referenced = (values - values.mean(1, keepdim=True))
    average_referenced -= average_referenced.mean(0, keepdim=True)
    assert not torch.equal(actual, average_referenced)


def test_adapter_matches_bound_kilosort_operation():
    from kilosort.io import BinaryFiltered
    from kilosort.preprocessing import get_highpass_filter

    assert sha(Path(__import__("kilosort.io", fromlist=["x"]).__file__)) == "767b76a07647122b45048ba12cdaf49185f87f3fa33334081d4af099825414bd"
    assert sha(Path(__import__("kilosort.preprocessing", fromlist=["x"]).__file__)) == "2329b288ae068361a937632a53461e6cceeb27e30191337e9b9a0fa7404d763d"
    frozen = torch.from_numpy(np.load(
        PARENT / "dependencies/kilosort/constants/highpass_filter_float32.npy", allow_pickle=False
    ).copy())
    generated = get_highpass_filter(fs=29999.835983263598, cutoff=300.0, device=torch.device("cpu"))
    assert torch.equal(frozen, generated)
    x = (torch.arange(384 * 1145, dtype=torch.float32).reshape(384, 1145) % 997) - 498
    dummy = SimpleNamespace(chan_map=None, invert_sign=False, do_CAR=True, hp_filter=frozen,
                            artifact_threshold=np.inf, whiten_mat=None, dshift=None, device=torch.device("cpu"))
    expected = BinaryFiltered.filter(dummy, x.clone())
    actual = preprocess_padded_record(x.clone(), frozen)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "refs,candidates,expected",
    [
        ([{"id": "r0", "frame": 0, "anchor": 0}, {"id": "r1", "frame": 10, "anchor": 0}],
         [{"id": "c0", "frame": 0, "anchor": 0}],
         {("REF", "r0"): "retained", ("REF", "r1"): "lost", ("candidate", "c0"): "retained"}),
        ([{"id": "r0", "frame": -1, "anchor": 0}, {"id": "r1", "frame": 1, "anchor": 0}],
         [{"id": "c0", "frame": 0, "anchor": 0}],
         {("REF", "r0"): "ambiguous", ("REF", "r1"): "ambiguous", ("candidate", "c0"): "ambiguous"}),
        ([{"id": "r0", "frame": 0, "anchor": 0}],
         [{"id": "c0", "frame": 0, "anchor": 100}],
         {("REF", "r0"): "lost", ("candidate", "c0"): "gained"}),
        ([{"id": "r0", "frame": 0, "anchor": -1, "pre_ambiguous": True}],
         [{"id": "c0", "frame": 0, "anchor": 0}],
         {("REF", "r0"): "ambiguous", ("candidate", "c0"): "gained"}),
    ],
)
def test_event_partition_is_exclusive_and_exhaustive(refs, candidates, expected):
    result = classify(refs, candidates)
    assert result == expected
    counts = {category: sum(value == category for value in result.values())
              for category in ("retained", "lost", "gained", "ambiguous")}
    assert sum(counts.values()) == len(refs) + len(candidates)


def test_atomic_retained_sampling_and_read_caps():
    contract = json.loads(CONTRACT_PATH.read_text())
    seed = contract["sampling"]["seed_utf8"]
    pairs = [{"comparison": "REF_vs_B", "epoch": "early", "pair_id": f"p{i}",
              "ref_frame": i, "candidate_frame": i + 1, "ref_row": i,
              "candidate_row": 1000 + i} for i in range(100)]
    selected = sorted(pairs, key=lambda value: rank_pair(value, seed))[:64]
    sample_rows = list(itertools.chain.from_iterable(
        [(value["pair_id"], "REF"), (value["pair_id"], "candidate")] for value in selected
    ))
    assert len(sample_rows) == 128
    assert all(sum(row[0] == pair_id for row in sample_rows) == 2 for pair_id in {r[0] for r in sample_rows})
    endpoint_reads_per_comparison_epoch = 128 + 128 + 128 + 2 * 64
    endpoint_reads = 3 * 3 * endpoint_reads_per_comparison_epoch
    assert endpoint_reads == contract["sampling"]["maximum_records"] == 4608
    bytes_per_read = 1145 * 384 * 2
    assert bytes_per_read == contract["waveform_extraction"]["raw_logical_bytes_per_record"]
    assert endpoint_reads * bytes_per_read == contract["resources_and_stop_conditions"]["logical_read_bytes_max"]
    assert contract["resources_and_stop_conditions"]["read_accounting"]["extra_partner_reads"] == 0


def test_correlation_alignment_and_low_n_rules_are_frozen():
    contract = json.loads(CONTRACT_PATH.read_text())
    ref_frame, candidate_frame = 100, 115
    ref_window = set(range(ref_frame - 20, ref_frame + 21))
    candidate_window = set(range(candidate_frame - 20, candidate_frame + 21))
    common = sorted(ref_window & candidate_window)
    assert common == list(range(95, 121)) and len(common) == 26
    rule = contract["technical_support"]["retained_pair_correlation"]
    assert ">=2 common channels" in rule["valid"]
    assert ">=26 common frames" in rule["valid"]
    assert "n=0" in rule["low_N"] and "n=1" in rule["low_N"]


def test_contract_remains_disabled_and_prospective():
    contract = json.loads(CONTRACT_PATH.read_text())
    assert contract["execution_enabled"] is False
    assert contract["voltage_access_enabled"] is False
    assert "PROSPECTIVE_BINDING_REQUIRED" in json.dumps(contract["input_identity"])
    assert "RF fitting" in contract["explicit_exclusions"][0]
    assert "holdout" in contract["explicit_exclusions"][1]
