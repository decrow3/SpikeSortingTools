"""Contract tests for configs/first_pipeline_candidate.v1.json and its validator.

The load-bearing behaviour under test is the authoring/execution asymmetry the
plan demands: the contract must be authorable and reviewable while the named
practical failure and the four numerical margins are unset, and it must refuse
to execute while any of them is.
"""

import copy
import csv
import json
from pathlib import Path

import pytest

from testing.first_pipeline_candidate_contract import (
    DEFAULT_CONTRACT,
    EXECUTION_DIGEST_PATHS,
    FREEZE_RECEIPT,
    FREEZE_SCHEMA,
    MANDATORY_REQUIRED_PATHS,
    MODE_AUTHORING,
    MODE_EXECUTION,
    SCHEMA,
    ContractRefusal,
    acceptance_digest,
    canonical_digest,
    check_comparators,
    development_windows,
    freeze_acceptance,
    get_path,
    is_unset,
    load_contract,
    reject_unsafe_out_root,
    required_paths,
    results_present,
    validate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MOTION_METRICS = REPO_ROOT / "testing/outputs/luke_prospective_holdout/candidate_window_motion_metrics_v2.csv"
SEALED_WINDOWS = REPO_ROOT / "testing/outputs/luke_prospective_holdout/sealed_windows_v2.csv"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _write(tmp_path: Path, payload: dict, name: str = "contract.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2))
    return path


@pytest.fixture
def base() -> dict:
    return load_contract(DEFAULT_CONTRACT)


def _set(node: dict, value) -> dict:
    node = dict(node)
    node["state"] = "set"
    node["value"] = value
    node["set_at"] = "2026-09-05T00:00:00+00:00"
    node["set_by"] = "test"
    return node


def _fully_set(payload: dict, *, interval=(3000.0, 3100.0), deps=None) -> dict:
    """A contract with every required field set, as it would look after the
    audit supplied baseline evidence."""
    out = copy.deepcopy(payload)
    out["acceptance"]["practical_failure"] = _set(
        out["acceptance"]["practical_failure"],
        {
            "name": "example_case",
            "sort_id": "rescue_luke0804_v2v1_g0_imec0",
            "cluster_id": 17,
            "interval_s": list(interval),
        },
    )
    for name in ("completeness", "identity", "contamination", "healthy_interval_preservation"):
        out["acceptance"]["margins"][name] = _set(out["acceptance"]["margins"][name], 5.0)
    out["candidate"]["settings"] = _set(
        out["candidate"]["settings"],
        {"intervention_family": "targeted_curation_repair", "execution_mode": "retained_sort_replay"},
    )
    out["candidate"]["dependency_requirements_resolved"] = _set(
        out["candidate"]["dependency_requirements_resolved"], list(deps or [])
    )
    return out


# --------------------------------------------------------------------------- #
# round trip and shape
# --------------------------------------------------------------------------- #
def test_shipped_contract_round_trips(tmp_path, base):
    assert base["schema"] == SCHEMA
    assert base["contract_id"] == "luke0804_imec0_first_pipeline_candidate_v1"
    again = load_contract(_write(tmp_path, base))
    assert again == base
    assert acceptance_digest(again) == acceptance_digest(base)


def test_acceptance_digest_is_key_order_independent(base):
    shuffled = json.loads(json.dumps(base["acceptance"]))
    reordered = dict(reversed(list(shuffled.items())))
    assert canonical_digest(reordered) == canonical_digest(base["acceptance"])


def test_wrong_schema_is_refused(tmp_path, base):
    bad = copy.deepcopy(base)
    bad["schema"] = "something-else"
    with pytest.raises(ContractRefusal, match="schema"):
        load_contract(_write(tmp_path, bad))


def test_knowable_now_fields_are_actually_filled_in(base):
    """Everything the plan says is knowable at authoring time is not a hole."""
    assert base["recording"]["data_dir"]
    assert base["recording"]["stream_id"] == "imec0.ap"
    assert base["recording"]["sampling_frequency_hz"] > 0
    assert base["recording"]["duration_s"] > 0
    assert base["output_root"].startswith("/media/")
    assert base["runtime_budget"]["l2_wall_clock_minutes_max"] == 45
    assert len(development_windows(base)) >= 1
    assert len(base["intervals"]["healthy_control_intervals"]["windows"]) == 3
    assert base["candidate"]["unresolved_implementation_dependencies"]
    comparators = check_comparators(base)
    assert comparators["legacy"]["sort_id"] != comparators["rescue_control"]["sort_id"]


# --------------------------------------------------------------------------- #
# required-but-unset registry
# --------------------------------------------------------------------------- #
def test_every_plan_mandated_field_ships_unset(base):
    for dotted in MANDATORY_REQUIRED_PATHS:
        assert is_unset(get_path(base, dotted), dotted) is True


def test_declared_registry_contains_the_mandated_paths(base):
    declared = required_paths(base)
    for dotted in MANDATORY_REQUIRED_PATHS:
        assert dotted in declared
    # the contract may add its own requirements
    assert "candidate.settings" in declared
    assert "candidate.dependency_requirements_resolved" in declared


def test_dropping_a_mandated_requirement_is_refused(tmp_path, base):
    tampered = copy.deepcopy(base)
    tampered["required_before_execution"] = [
        p for p in tampered["required_before_execution"] if p != "acceptance.margins.identity"
    ]
    with pytest.raises(ContractRefusal, match="never drop"):
        validate(_write(tmp_path, tampered), mode=MODE_AUTHORING, out_root=tmp_path / "out")


def test_empty_or_duplicated_registry_is_refused(tmp_path, base):
    empty = copy.deepcopy(base)
    empty["required_before_execution"] = []
    with pytest.raises(ContractRefusal, match="non-empty"):
        required_paths(empty)
    dup = copy.deepcopy(base)
    dup["required_before_execution"] = dup["required_before_execution"] + [
        "acceptance.margins.identity"
    ]
    with pytest.raises(ContractRefusal, match="duplicate"):
        required_paths(dup)


@pytest.mark.parametrize(
    "node, match",
    [
        ({"state": "set", "value": None}, "is null"),
        ({"state": "unset", "value": 3.0}, "carries a value"),
        ({"state": "maybe", "value": None}, "state must be"),
        ({"value": 1.0}, "'state' and 'value'"),
        (3.0, "'state' and 'value'"),
    ],
)
def test_corrupt_settable_nodes_are_refused(node, match):
    with pytest.raises(ContractRefusal, match=match):
        is_unset(node, "acceptance.margins.identity")


# --------------------------------------------------------------------------- #
# authoring passes, execution refuses
# --------------------------------------------------------------------------- #
def test_authoring_mode_passes_while_everything_is_unset(tmp_path, base):
    report = validate(DEFAULT_CONTRACT, mode=MODE_AUTHORING, out_root=tmp_path / "out")
    assert report.executable is False
    assert set(report.unset_required_fields) == set(base["required_before_execution"])
    for dotted in MANDATORY_REQUIRED_PATHS:
        assert dotted in report.unset_required_fields


def test_execution_mode_refuses_while_any_required_field_is_unset(tmp_path):
    with pytest.raises(ContractRefusal, match="refusing execution") as exc:
        validate(DEFAULT_CONTRACT, mode=MODE_EXECUTION, out_root=tmp_path / "out")
    for dotted in MANDATORY_REQUIRED_PATHS:
        assert dotted in str(exc.value)


@pytest.mark.parametrize("dotted", MANDATORY_REQUIRED_PATHS)
def test_execution_refuses_when_exactly_one_field_is_left_unset(tmp_path, base, dotted):
    payload = _fully_set(base)
    parent, _, leaf = dotted.rpartition(".")
    node = get_path(payload, parent)
    node[leaf] = get_path(base, dotted)  # restore the shipped, unset node
    path = _write(tmp_path, payload)
    with pytest.raises(ContractRefusal, match=dotted.replace(".", r"\.")):
        validate(path, mode=MODE_EXECUTION, out_root=tmp_path / "out")
    # ... but authoring still works, so the contract stays reviewable
    assert validate(path, mode=MODE_AUTHORING, out_root=tmp_path / "out").executable is False


def test_execution_passes_once_set_and_frozen(tmp_path, base):
    path = _write(tmp_path, _fully_set(base))
    out = tmp_path / "out"
    receipt = freeze_acceptance(path, out)
    assert receipt["schema"] == FREEZE_SCHEMA
    report = validate(path, mode=MODE_EXECUTION, out_root=out)
    assert report.executable is True
    assert report.unset_required_fields == ()
    assert report.contract_frozen is True


def test_execution_refuses_when_set_but_never_frozen(tmp_path, base):
    path = _write(tmp_path, _fully_set(base))
    with pytest.raises(ContractRefusal, match="not frozen"):
        validate(path, mode=MODE_EXECUTION, out_root=tmp_path / "out")


def test_bad_mode_is_refused(tmp_path):
    with pytest.raises(ContractRefusal, match="mode must be"):
        validate(DEFAULT_CONTRACT, mode="whenever", out_root=tmp_path / "out")


# --------------------------------------------------------------------------- #
# output-root refusals
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["/mnt", "/mnt/NPX/Luke/20250804/out", "/mnt/anything"])
def test_output_root_under_mnt_is_refused(tmp_path, base, bad):
    payload = copy.deepcopy(base)
    payload["output_root"] = bad
    with pytest.raises(ContractRefusal, match="/mnt"):
        validate(_write(tmp_path, payload), mode=MODE_AUTHORING)


def test_output_root_inside_an_input_directory_is_refused(tmp_path, base):
    payload = copy.deepcopy(base)
    payload["recording"]["data_dir"] = str(tmp_path / "input")
    payload["output_root"] = str(tmp_path / "input" / "out")
    with pytest.raises(ContractRefusal, match="under/over an input directory"):
        validate(_write(tmp_path, payload), mode=MODE_AUTHORING)


def test_output_root_above_an_input_directory_is_refused(tmp_path, base):
    payload = copy.deepcopy(base)
    payload["recording"]["data_dir"] = str(tmp_path / "root" / "input")
    payload["output_root"] = str(tmp_path / "root")
    with pytest.raises(ContractRefusal, match="under/over an input directory"):
        validate(_write(tmp_path, payload), mode=MODE_AUTHORING)


def test_reject_unsafe_out_root_accepts_a_disjoint_local_root(tmp_path):
    assert reject_unsafe_out_root(tmp_path / "out", [tmp_path / "in"]) == (tmp_path / "out").resolve()


def test_shipped_output_root_is_local_and_disjoint(base):
    report = validate(DEFAULT_CONTRACT, mode=MODE_AUTHORING)
    assert not report.out_root.startswith("/mnt")
    assert report.out_root == base["output_root"]


# --------------------------------------------------------------------------- #
# comparator identity refusals
# --------------------------------------------------------------------------- #
def test_missing_comparator_role_is_refused(tmp_path, base):
    payload = copy.deepcopy(base)
    del payload["comparators"]["legacy"]
    with pytest.raises(ContractRefusal, match="missing comparator identity"):
        validate(_write(tmp_path, payload), mode=MODE_AUTHORING, out_root=tmp_path / "out")


@pytest.mark.parametrize("field", ["role", "sort_id", "curated", "qc_dir", "source_recording"])
def test_blank_comparator_field_is_refused(tmp_path, base, field):
    payload = copy.deepcopy(base)
    payload["comparators"]["rescue_control"][field] = "  "
    with pytest.raises(ContractRefusal, match=f"missing comparator identity: rescue_control.{field}"):
        validate(_write(tmp_path, payload), mode=MODE_AUTHORING, out_root=tmp_path / "out")


def test_comparator_without_provenance_is_refused(tmp_path, base):
    payload = copy.deepcopy(base)
    payload["comparators"]["legacy"]["provenance"] = {}
    with pytest.raises(ContractRefusal, match="carries no receipts"):
        validate(_write(tmp_path, payload), mode=MODE_AUTHORING, out_root=tmp_path / "out")


def test_two_comparators_pointing_at_one_sort_are_refused(tmp_path, base):
    payload = copy.deepcopy(base)
    payload["comparators"]["legacy"]["sort_id"] = payload["comparators"]["rescue_control"]["sort_id"]
    with pytest.raises(ContractRefusal, match="distinct sorts"):
        validate(_write(tmp_path, payload), mode=MODE_AUTHORING, out_root=tmp_path / "out")


def test_missing_comparator_identity_is_refused_at_execution_too(tmp_path, base):
    payload = _fully_set(base)
    payload["comparators"]["legacy"]["sort_id"] = ""
    with pytest.raises(ContractRefusal, match="missing comparator identity"):
        validate(_write(tmp_path, payload), mode=MODE_EXECUTION, out_root=tmp_path / "out")


# --------------------------------------------------------------------------- #
# a margin edited after results exist
# --------------------------------------------------------------------------- #
def test_margin_edited_after_results_exist_is_refused(tmp_path, base):
    path = _write(tmp_path, _fully_set(base))
    out = tmp_path / "out"
    freeze_acceptance(path, out)
    (out / "results.csv").write_text("unit,score\n1,0.5\n")

    edited = _fully_set(base)
    edited["acceptance"]["margins"]["completeness"]["value"] = 0.1
    edited_path = _write(tmp_path, edited, name="edited.json")
    with pytest.raises(ContractRefusal, match="edited after results exist"):
        validate(edited_path, mode=MODE_AUTHORING, out_root=out)
    with pytest.raises(ContractRefusal, match="edited after results exist"):
        validate(edited_path, mode=MODE_EXECUTION, out_root=out)


def test_editing_a_margin_before_results_exist_is_allowed_but_unfreezes(tmp_path, base):
    path = _write(tmp_path, _fully_set(base))
    out = tmp_path / "out"
    freeze_acceptance(path, out)

    edited = _fully_set(base)
    edited["acceptance"]["margins"]["identity"]["value"] = 0.9
    edited_path = _write(tmp_path, edited, name="edited.json")
    report = validate(edited_path, mode=MODE_AUTHORING, out_root=out)
    assert report.contract_frozen is False
    with pytest.raises(ContractRefusal, match="not frozen"):
        validate(edited_path, mode=MODE_EXECUTION, out_root=out)


def test_results_without_a_freeze_receipt_are_refused(tmp_path, base):
    out = tmp_path / "out"
    out.mkdir()
    (out / "windows.csv").write_text("a\n1\n")
    path = _write(tmp_path, _fully_set(base))
    with pytest.raises(ContractRefusal, match="cannot be set or frozen after results exist"):
        validate(path, mode=MODE_AUTHORING, out_root=out)


def test_freeze_refuses_after_results_exist(tmp_path, base):
    out = tmp_path / "out"
    out.mkdir()
    (out / "nested").mkdir()
    (out / "nested" / "scores.json").write_text("{}")
    path = _write(tmp_path, _fully_set(base))
    with pytest.raises(ContractRefusal, match="already exist"):
        freeze_acceptance(path, out)


def test_freeze_refuses_while_fields_are_unset(tmp_path):
    with pytest.raises(ContractRefusal, match="cannot freeze acceptance while these fields are unset"):
        freeze_acceptance(DEFAULT_CONTRACT, tmp_path / "out")


def test_freeze_refuses_to_overwrite_a_different_frozen_acceptance(tmp_path, base):
    out = tmp_path / "out"
    freeze_acceptance(_write(tmp_path, _fully_set(base)), out)
    other = _fully_set(base)
    other["acceptance"]["margins"]["contamination"]["value"] = 99.0
    with pytest.raises(ContractRefusal, match="already froze a different acceptance block"):
        freeze_acceptance(_write(tmp_path, other, name="other.json"), out)


def test_refreezing_the_same_acceptance_is_idempotent(tmp_path, base):
    out = tmp_path / "out"
    path = _write(tmp_path, _fully_set(base))
    first = freeze_acceptance(path, out)
    second = freeze_acceptance(path, out)
    assert first["acceptance_digest"] == second["acceptance_digest"]


def test_results_present_ignores_the_receipt_and_tmp_files(tmp_path, base):
    out = tmp_path / "out"
    freeze_acceptance(_write(tmp_path, _fully_set(base)), out)
    (out / "partial.csv.tmp").write_text("x")
    assert results_present(out) == []
    (out / "partial.csv").write_text("x")
    assert results_present(out) == ["partial.csv"]


def test_corrupt_freeze_receipt_is_refused(tmp_path, base):
    out = tmp_path / "out"
    out.mkdir()
    (out / FREEZE_RECEIPT).write_text("{not json")
    path = _write(tmp_path, _fully_set(base))
    with pytest.raises(ContractRefusal, match="not readable JSON"):
        validate(path, mode=MODE_AUTHORING, out_root=out)


# --------------------------------------------------------------------------- #
# provenance under a dirty tree
# --------------------------------------------------------------------------- #
def test_validate_records_commit_and_working_tree_hashes(tmp_path, base):
    report = validate(DEFAULT_CONTRACT, mode=MODE_AUTHORING, out_root=tmp_path / "out")
    prov = report.provenance
    assert len(prov["git_commit"]) == 40
    assert prov["git_status_available"] is True
    # this branch is being authored on a dirty tree; both facts must be recorded
    assert prov["git_tree_dirty"] is True
    assert len(prov["git_status_porcelain_sha256"]) == 64
    assert prov["git_dirty_entry_count"] >= 1
    assert set(prov["source_sha256"]) == {"validator_module", "contract"}
    assert all(len(h) == 64 for h in prov["source_sha256"].values())


def test_freeze_receipt_carries_the_same_provenance(tmp_path, base):
    path = _write(tmp_path, _fully_set(base))
    receipt = freeze_acceptance(path, tmp_path / "out")
    prov = receipt["provenance"]
    assert len(prov["git_commit"]) == 40
    assert prov["git_tree_dirty"] is True
    assert prov["source_sha256"]["contract"] != prov["source_sha256"]["validator_module"]
    assert receipt["required_before_execution"] == base["required_before_execution"]


def test_source_hash_tracks_the_working_tree_not_the_commit(tmp_path, base):
    """Two contracts with the same commit hash to different sources."""
    a = validate(_write(tmp_path, _fully_set(base), name="a.json"), out_root=tmp_path / "out")
    edited = _fully_set(base)
    edited["acceptance"]["margins"]["identity"]["value"] = 7.5
    b = validate(_write(tmp_path, edited, name="b.json"), out_root=tmp_path / "out")
    assert a.provenance["git_commit"] == b.provenance["git_commit"]
    assert a.provenance["source_sha256"]["contract"] != b.provenance["source_sha256"]["contract"]


# --------------------------------------------------------------------------- #
# the selected case must not escape the development windows
# --------------------------------------------------------------------------- #
def test_failure_interval_inside_a_development_window_is_accepted(tmp_path, base):
    path = _write(tmp_path, _fully_set(base, interval=(5300.0, 5400.0)))
    freeze_acceptance(path, tmp_path / "out")
    assert validate(path, mode=MODE_EXECUTION, out_root=tmp_path / "out").executable is True


@pytest.mark.parametrize(
    "interval",
    [
        (1200.0, 1320.0),   # a sealed window
        (1000.0, 1100.0),   # inside a sealed window's buffer
        (3100.0, 3200.0),   # straddles a healthy control interval
        (3150.0, 3200.0),   # inside a healthy control interval
        (5000.0, 5100.0),   # inside the T2 sealed buffer
        (4370.0, 4400.0),   # straddles a development window edge
    ],
)
def test_failure_interval_outside_the_development_windows_is_refused(tmp_path, base, interval):
    path = _write(tmp_path, _fully_set(base, interval=interval))
    with pytest.raises(ContractRefusal, match="not contained in any development window"):
        validate(path, mode=MODE_AUTHORING, out_root=tmp_path / "out")


def test_set_failure_case_without_an_interval_is_refused(tmp_path, base):
    payload = _fully_set(base)
    payload["acceptance"]["practical_failure"]["value"].pop("interval_s")
    with pytest.raises(ContractRefusal, match="must carry an interval_s"):
        validate(_write(tmp_path, payload), mode=MODE_AUTHORING, out_root=tmp_path / "out")


# --------------------------------------------------------------------------- #
# unresolved implementation dependencies
# --------------------------------------------------------------------------- #
def test_execution_refuses_an_unresolved_required_dependency(tmp_path, base):
    payload = _fully_set(base, deps=["thin_candidate_runner"])
    path = _write(tmp_path, payload)
    out = tmp_path / "out"
    freeze_acceptance(path, out)
    with pytest.raises(ContractRefusal, match="thin_candidate_runner"):
        validate(path, mode=MODE_EXECUTION, out_root=out)
    report = validate(path, mode=MODE_AUTHORING, out_root=out)
    assert report.required_dependencies == ("thin_candidate_runner",)
    assert report.unresolved_required_dependencies == ("thin_candidate_runner",)
    assert report.executable is False


def test_unknown_dependency_id_is_refused(tmp_path, base):
    payload = _fully_set(base, deps=["a_thing_nobody_declared"])
    with pytest.raises(ContractRefusal, match="unknown implementation dependency ids"):
        validate(_write(tmp_path, payload), mode=MODE_AUTHORING, out_root=tmp_path / "out")


def test_dependency_catalog_entries_need_id_and_status(tmp_path, base):
    payload = copy.deepcopy(base)
    payload["candidate"]["unresolved_implementation_dependencies"].append({"description": "x"})
    with pytest.raises(ContractRefusal, match="needs an 'id' and a 'status'"):
        validate(_write(tmp_path, payload), mode=MODE_AUTHORING, out_root=tmp_path / "out")


def test_shipped_dependency_catalog_names_the_audit_and_the_runner(base):
    ids = {d["id"] for d in base["candidate"]["unresolved_implementation_dependencies"]}
    assert "amplitude_audit_layers_3_to_5" in ids
    assert "thin_candidate_runner" in ids
    assert "legacy_raw_voltage" in ids
    statuses = {d["id"]: d["status"] for d in base["candidate"]["unresolved_implementation_dependencies"]}
    assert statuses["legacy_raw_voltage"] == "unavailable"
    assert all(s != "resolved" for s in statuses.values())


# --------------------------------------------------------------------------- #
# the shipped intervals are what the stated rules produce
# --------------------------------------------------------------------------- #
def _sealed_from_disk() -> list[tuple[float, float]]:
    with SEALED_WINDOWS.open() as fh:
        return [(float(r["start_s"]), float(r["stop_s"])) for r in csv.DictReader(fh)]


def test_contract_sealed_windows_match_the_sealed_panel_on_disk(base):
    shipped = [tuple(w) for w in base["intervals"]["sealed_panel"]["windows_s"]]
    assert shipped == _sealed_from_disk()


def test_healthy_controls_follow_the_stated_selection_rule(base):
    buffer_s = base["intervals"]["sealed_panel"]["exclusion_buffer_s"]
    excluded = [(a - buffer_s, b + buffer_s) for a, b in _sealed_from_disk()]
    with MOTION_METRICS.open() as fh:
        rows = list(csv.DictReader(fh))
    admissible = [
        r for r in rows
        if all(float(r["stop_s"]) <= a or float(r["start_s"]) >= b for a, b in excluded)
    ]
    expected = []
    for third in ("0", "1", "2"):
        pool = [r for r in admissible if r["time_third"] == third]
        pool.sort(key=lambda r: (float(r["combined_motion_score"]), float(r["start_s"])))
        expected.append((float(pool[0]["start_s"]), float(pool[0]["stop_s"])))
    shipped = [
        (w["start_s"], w["stop_s"]) for w in base["intervals"]["healthy_control_intervals"]["windows"]
    ]
    assert shipped == expected


def test_development_windows_are_the_stated_derivation(base):
    duration = base["recording"]["duration_s"]
    buffer_s = base["intervals"]["sealed_panel"]["exclusion_buffer_s"]
    cuts = [(a - buffer_s, b + buffer_s) for a, b in _sealed_from_disk()]
    cuts += [
        (w["start_s"], w["stop_s"]) for w in base["intervals"]["healthy_control_intervals"]["windows"]
    ]
    cuts.sort()
    expected = []
    cursor = 0.0
    for a, b in cuts:
        a, b = max(a, 0.0), min(b, duration)
        if a > cursor:
            expected.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < duration:
        expected.append((cursor, duration))
    assert [tuple(w) for w in base["intervals"]["development_windows"]["windows_s"]] == expected


def test_no_development_or_control_window_touches_the_sealed_panel(base):
    buffer_s = base["intervals"]["sealed_panel"]["exclusion_buffer_s"]
    excluded = [(a - buffer_s, b + buffer_s) for a, b in _sealed_from_disk()]
    windows = [tuple(w) for w in base["intervals"]["development_windows"]["windows_s"]]
    windows += [
        (w["start_s"], w["stop_s"]) for w in base["intervals"]["healthy_control_intervals"]["windows"]
    ]
    for start, stop in windows:
        for a, b in excluded:
            assert stop <= a or start >= b, f"[{start}, {stop}] intersects the sealed buffer [{a}, {b}]"


@pytest.mark.parametrize(
    "windows_s, match",
    [
        ([[100.0, 100.0]], "stop_s > start_s"),
        ([[200.0, 100.0]], "stop_s > start_s"),
        ([["a", 100.0]], "must be numbers"),
        ([], "non-empty list"),
    ],
)
def test_malformed_interval_lists_are_refused(tmp_path, base, windows_s, match):
    payload = copy.deepcopy(base)
    payload["intervals"]["development_windows"]["windows_s"] = windows_s
    with pytest.raises(ContractRefusal, match=match):
        validate(_write(tmp_path, payload), mode=MODE_AUTHORING, out_root=tmp_path / "out")


# --------------------------------------------------------------------------- #
# the recorded comparator paths are real (listing only; nothing is read)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("role", ["legacy", "rescue_control"])
@pytest.mark.parametrize("key", ["curated", "qc_dir", "source_recording"])
def test_comparator_paths_exist_on_this_host(base, role, key):
    if not Path("/mnt/NPX/Luke/20250804").exists():
        pytest.skip("/mnt/NPX/Luke/20250804 is not mounted on this host")
    assert Path(base["comparators"][role][key]).exists()


# --------------------------------------------------------------------------- #
# review finding 1 -- candidate.settings and the dependency declaration were
# required-but-unset yet absent from the mandatory registry, so a contract could
# drop them from required_before_execution and then freeze and execute with no
# selected candidate and no dependency statement at all.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "dotted", ["candidate.settings", "candidate.dependency_requirements_resolved"]
)
def test_f1_dropping_a_candidate_requirement_is_refused(tmp_path, base, dotted):
    tampered = copy.deepcopy(base)
    tampered["required_before_execution"] = [
        p for p in tampered["required_before_execution"] if p != dotted
    ]
    with pytest.raises(ContractRefusal, match="never drop"):
        validate(_write(tmp_path, tampered), mode=MODE_AUTHORING, out_root=tmp_path / "out")


@pytest.mark.parametrize(
    "dotted", ["candidate.settings", "candidate.dependency_requirements_resolved"]
)
def test_f1_cannot_execute_by_deregistering_an_unset_candidate_field(tmp_path, base, dotted):
    """The whole exploit path: leave the field unset AND drop it from the
    registry, then freeze and execute."""
    payload = _fully_set(base)
    parent, _, leaf = dotted.rpartition(".")
    get_path(payload, parent)[leaf] = get_path(base, dotted)  # back to unset
    payload["required_before_execution"] = [
        p for p in payload["required_before_execution"] if p != dotted
    ]
    path = _write(tmp_path, payload)
    with pytest.raises(ContractRefusal, match="never drop"):
        freeze_acceptance(path, tmp_path / "out")
    with pytest.raises(ContractRefusal, match="never drop"):
        validate(path, mode=MODE_EXECUTION, out_root=tmp_path / "out")


def test_f1_mandatory_registry_covers_every_required_but_unset_field(base):
    assert set(MANDATORY_REQUIRED_PATHS) == set(base["required_before_execution"])


def test_f1_unset_dependency_node_never_means_no_dependencies(tmp_path, base):
    """An unset dependency declaration reports nothing required, but that is a
    missing answer, not `this candidate needs nothing`: execution refuses."""
    payload = _fully_set(base)
    payload["candidate"]["dependency_requirements_resolved"] = copy.deepcopy(
        base["candidate"]["dependency_requirements_resolved"]
    )
    path = _write(tmp_path, payload)
    report = validate(path, mode=MODE_AUTHORING, out_root=tmp_path / "out")
    assert report.required_dependencies == ()
    assert report.executable is False
    with pytest.raises(ContractRefusal, match="candidate.dependency_requirements_resolved"):
        validate(path, mode=MODE_EXECUTION, out_root=tmp_path / "out")


# --------------------------------------------------------------------------- #
# review finding 2 -- the freeze receipt bound only the acceptance subtree, so
# the candidate, comparators, intervals, recording identity and output root
# could all be swapped after freezing while `frozen` stayed true.
# --------------------------------------------------------------------------- #
def _freeze_then_edit(tmp_path, base, mutate, *, with_results: bool):
    path = _write(tmp_path, _fully_set(base))
    out = tmp_path / "out"
    freeze_acceptance(path, out)
    if with_results:
        (out / "scores.csv").write_text("unit,score\n1,0.5\n")
    edited = _fully_set(base)
    mutate(edited)
    return _write(tmp_path, edited, name="edited.json"), out


def _swap_candidate_settings(payload):
    payload["candidate"]["settings"]["value"]["intervention_family"] = "option_b_unwarped_identity"


def _swap_dependency_declaration(payload):
    payload["candidate"]["dependency_requirements_resolved"]["value"] = ["legacy_raw_voltage"]


def _swap_comparator(payload):
    payload["comparators"]["legacy"]["curated"] = "/mnt/NPX/Luke/20250804/somewhere_else"


def _swap_intervals(payload):
    payload["intervals"]["healthy_control_intervals"]["windows"][0]["stop_s"] = 3300.0


def _swap_recording_identity(payload):
    payload["recording"]["raw_ap_sha256_imec0"] = "0" * 64


def _swap_output_root(payload):
    payload["output_root"] = "/media/huklab/Data/NPX/Ryansorting/Luke/somewhere_else"


NON_ACCEPTANCE_EDITS = [
    _swap_candidate_settings,
    _swap_dependency_declaration,
    _swap_comparator,
    _swap_intervals,
    _swap_recording_identity,
    _swap_output_root,
]


@pytest.mark.parametrize("mutate", NON_ACCEPTANCE_EDITS, ids=lambda f: f.__name__)
def test_f2_execution_defining_edit_after_results_is_refused(tmp_path, base, mutate):
    edited_path, out = _freeze_then_edit(tmp_path, base, mutate, with_results=True)
    with pytest.raises(ContractRefusal, match="edited after results exist"):
        validate(edited_path, mode=MODE_AUTHORING, out_root=out)
    with pytest.raises(ContractRefusal, match="edited after results exist"):
        validate(edited_path, mode=MODE_EXECUTION, out_root=out)


@pytest.mark.parametrize("mutate", NON_ACCEPTANCE_EDITS, ids=lambda f: f.__name__)
def test_f2_execution_defining_edit_before_results_unfreezes(tmp_path, base, mutate):
    edited_path, out = _freeze_then_edit(tmp_path, base, mutate, with_results=False)
    assert validate(edited_path, mode=MODE_AUTHORING, out_root=out).contract_frozen is False
    with pytest.raises(ContractRefusal, match="not frozen"):
        validate(edited_path, mode=MODE_EXECUTION, out_root=out)


def test_f2_non_acceptance_edit_is_named_as_such(tmp_path, base):
    edited_path, out = _freeze_then_edit(tmp_path, base, _swap_comparator, with_results=True)
    with pytest.raises(ContractRefusal, match="outside the acceptance block"):
        validate(edited_path, mode=MODE_AUTHORING, out_root=out)


def test_f2_receipt_binds_the_named_execution_defining_paths(tmp_path, base):
    receipt = freeze_acceptance(_write(tmp_path, _fully_set(base)), tmp_path / "out")
    assert receipt["contract_digest_covers"] == list(EXECUTION_DIGEST_PATHS)
    assert receipt["contract_digest"] != receipt["acceptance_digest"]
    for required in ("acceptance", "candidate", "comparators", "intervals", "recording", "output_root"):
        assert required in EXECUTION_DIGEST_PATHS


def test_f2_receipt_without_a_contract_digest_is_refused(tmp_path, base):
    out = tmp_path / "out"
    path = _write(tmp_path, _fully_set(base))
    freeze_acceptance(path, out)
    receipt = json.loads((out / FREEZE_RECEIPT).read_text())
    receipt.pop("contract_digest")
    (out / FREEZE_RECEIPT).write_text(json.dumps(receipt))
    with pytest.raises(ContractRefusal, match="carries no contract_digest"):
        validate(path, mode=MODE_AUTHORING, out_root=out)


def test_f2_prose_edits_after_a_freeze_are_still_allowed(tmp_path, base):
    """The digest is deliberately scoped: wording may be repaired."""
    path = _write(tmp_path, _fully_set(base))
    out = tmp_path / "out"
    freeze_acceptance(path, out)
    (out / "scores.csv").write_text("unit,score\n1,0.5\n")
    edited = _fully_set(base)
    edited["purpose"] = edited["purpose"] + " (typo fixed)"
    edited["notes"].append("a clarifying note added after the run started")
    edited_path = _write(tmp_path, edited, name="edited.json")
    assert validate(edited_path, mode=MODE_EXECUTION, out_root=out).executable is True


# --------------------------------------------------------------------------- #
# review finding 3 -- a required field counted as satisfied purely because its
# value was non-null, so margins could be strings, settings arbitrary data, and
# the failure case could omit its name, sort_id and cluster_id.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["completeness", "identity", "contamination", "healthy_interval_preservation"])
@pytest.mark.parametrize(
    "bad", ["5.0", {}, {"pp": 5.0}, [5.0], True, float("nan"), float("inf")],
    ids=["string", "empty_object", "object", "list", "bool", "nan", "inf"],
)
def test_f3_non_numeric_margin_is_refused(tmp_path, base, name, bad):
    payload = _fully_set(base)
    payload["acceptance"]["margins"][name]["value"] = bad
    path = _write(tmp_path, payload)
    with pytest.raises(ContractRefusal, match="must be a finite number"):
        validate(path, mode=MODE_AUTHORING, out_root=tmp_path / "out")
    with pytest.raises(ContractRefusal, match="must be a finite number"):
        freeze_acceptance(path, tmp_path / "out")


@pytest.mark.parametrize("name", ["completeness", "identity"])
@pytest.mark.parametrize("bad", [0.0, -0.5])
def test_f3_non_positive_minimum_improvement_margin_is_refused(tmp_path, base, name, bad):
    payload = _fully_set(base)
    payload["acceptance"]["margins"][name]["value"] = bad
    with pytest.raises(ContractRefusal, match="strictly positive"):
        validate(_write(tmp_path, payload), mode=MODE_AUTHORING, out_root=tmp_path / "out")


@pytest.mark.parametrize("name", ["contamination", "healthy_interval_preservation"])
def test_f3_negative_tolerance_margin_is_refused_but_zero_is_allowed(tmp_path, base, name):
    payload = _fully_set(base)
    payload["acceptance"]["margins"][name]["value"] = -0.001
    with pytest.raises(ContractRefusal, match="must be >= 0"):
        validate(_write(tmp_path, payload), mode=MODE_AUTHORING, out_root=tmp_path / "out")
    payload["acceptance"]["margins"][name]["value"] = 0.0
    report = validate(_write(tmp_path, payload, name="zero.json"), out_root=tmp_path / "out")
    assert report.unset_required_fields == ()


def test_f3_margin_must_declare_its_unit_direction_and_magnitude_kind(tmp_path, base):
    for key in ("unit", "direction", "comparison", "magnitude_kind", "set_from"):
        payload = copy.deepcopy(base)
        payload["acceptance"]["margins"]["identity"].pop(key)
        with pytest.raises(ContractRefusal, match=f"must declare a non-empty {key!r}"):
            validate(_write(tmp_path, payload), mode=MODE_AUTHORING, out_root=tmp_path / "out")


@pytest.mark.parametrize("key, bad", [("direction", "sideways"), ("magnitude_kind", "vibes")])
def test_f3_unknown_margin_declaration_value_is_refused(tmp_path, base, key, bad):
    payload = copy.deepcopy(base)
    payload["acceptance"]["margins"]["contamination"][key] = bad
    with pytest.raises(ContractRefusal, match=f"{key} must be one of"):
        validate(_write(tmp_path, payload), mode=MODE_AUTHORING, out_root=tmp_path / "out")


def test_f3_shipped_margins_declare_their_magnitude_kinds(base):
    kinds = {n: m["magnitude_kind"] for n, m in base["acceptance"]["margins"].items()}
    assert kinds == {
        "completeness": "minimum_improvement",
        "identity": "minimum_improvement",
        "contamination": "maximum_tolerated_degradation",
        "healthy_interval_preservation": "maximum_tolerated_degradation",
    }


@pytest.mark.parametrize("key", ["name", "sort_id", "cluster_id"])
def test_f3_practical_failure_missing_an_identity_field_is_refused(tmp_path, base, key):
    payload = _fully_set(base)
    payload["acceptance"]["practical_failure"]["value"].pop(key)
    path = _write(tmp_path, payload)
    with pytest.raises(ContractRefusal, match=f"must carry {key!r}"):
        validate(path, mode=MODE_AUTHORING, out_root=tmp_path / "out")
    with pytest.raises(ContractRefusal, match=f"must carry {key!r}"):
        freeze_acceptance(path, tmp_path / "out")


@pytest.mark.parametrize("bad", ["legacy", "", "  ", "some_other_sort"])
def test_f3_practical_failure_sort_id_must_name_a_declared_comparator(tmp_path, base, bad):
    payload = _fully_set(base)
    payload["acceptance"]["practical_failure"]["value"]["sort_id"] = bad
    with pytest.raises(ContractRefusal, match="non-empty string|does not name a declared comparator"):
        validate(_write(tmp_path, payload), mode=MODE_AUTHORING, out_root=tmp_path / "out")


@pytest.mark.parametrize("bad", [17.5, "17", True, None, [17]], ids=["float", "string", "bool", "null", "list"])
def test_f3_practical_failure_cluster_id_must_be_an_integer(tmp_path, base, bad):
    payload = _fully_set(base)
    payload["acceptance"]["practical_failure"]["value"]["cluster_id"] = bad
    with pytest.raises(ContractRefusal, match="cluster_id must be an integer id"):
        validate(_write(tmp_path, payload), mode=MODE_AUTHORING, out_root=tmp_path / "out")


@pytest.mark.parametrize(
    "bad",
    ["anything", 3, [], {}, {"intervention_family": "hand_tuning", "execution_mode": "resort"},
     {"intervention_family": "targeted_curation_repair", "execution_mode": "whatever"},
     {"intervention_family": "targeted_curation_repair"}],
    ids=["string", "int", "empty_list", "empty_object", "unknown_family", "unknown_mode", "no_mode"],
)
def test_f3_candidate_settings_must_be_a_resolved_configuration(tmp_path, base, bad):
    payload = _fully_set(base)
    payload["candidate"]["settings"]["value"] = bad
    path = _write(tmp_path, payload)
    with pytest.raises(ContractRefusal, match="candidate.settings.value"):
        validate(path, mode=MODE_AUTHORING, out_root=tmp_path / "out")
    with pytest.raises(ContractRefusal, match="candidate.settings.value"):
        freeze_acceptance(path, tmp_path / "out")


def test_f3_a_shapeless_but_non_null_contract_never_reports_executable(tmp_path, base):
    """The pre-fix failure in one shot: every required field non-null, none of
    them usable."""
    payload = _fully_set(base)
    for name in payload["acceptance"]["margins"]:
        payload["acceptance"]["margins"][name]["value"] = "as small as possible"
    payload["acceptance"]["practical_failure"]["value"] = {"interval_s": [3000.0, 3100.0]}
    payload["candidate"]["settings"]["value"] = "TBD"
    path = _write(tmp_path, payload)
    with pytest.raises(ContractRefusal):
        validate(path, mode=MODE_EXECUTION, out_root=tmp_path / "out")
    with pytest.raises(ContractRefusal):
        freeze_acceptance(path, tmp_path / "out")
