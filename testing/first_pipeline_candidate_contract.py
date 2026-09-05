"""Validator for the first-version delivery contract.

Contract file: ``configs/first_pipeline_candidate.v1.json``. Plan section:
``docs/pipeline_improvement_plan.md`` -> "Step 1 -- freeze a small first-version
delivery contract".

The point of this module is a single asymmetry the plan states explicitly:

    "Contract omissions block execution, not planning or implementation of
     independent components."

So the contract is *authorable and reviewable* while the named practical
recovery failure and the four numerical margins are still unset -- they cannot
be filled in honestly until the bounded amplitude-completeness audit selects a
case and supplies its BASELINE evidence -- but nothing may *execute* against it
until every one of them is set. :func:`validate` therefore takes a ``mode``:

* ``authoring``  -- report the unset fields, refuse only the things that are
  wrong regardless of when you look (bad output root, missing comparator
  identity, a tampered requirement registry, a set value that does not match
  its declared shape, an execution-defining field edited after results exist,
  a corrupt set/unset node).
* ``execution``  -- all of the above, plus: refuse while any required field is
  unset, refuse an unfrozen or stale freeze receipt, refuse a required
  implementation dependency that is not resolved, and refuse a failure interval
  that does not lie inside a development window.

:data:`MANDATORY_REQUIRED_PATHS` must appear in the contract's own
``required_before_execution`` list; a contract that drops one to make itself
executable is refused in every mode. A set value is checked against its
declared shape (:func:`check_required_value_shapes`) before it can be frozen or
executed: being non-null is not being usable.

The contract must be frozen *before* results exist. :func:`freeze_acceptance`
writes ``acceptance_freeze.json`` into the output root, recording a digest of
the whole execution-defining contract (:data:`EXECUTION_DIGEST_PATHS`) -- not
the acceptance block alone, or results from incompatible configurations could
accumulate under one freeze -- together with the git commit AND the on-disk
working-tree source hashes (the tree is dirty; a commit alone does not identify
what ran).

This module reads the contract, the freeze receipt and directory listings only.
It never sorts, fits, extracts voltage, or touches an input directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "luke-first-pipeline-candidate-v1"
DEFAULT_CONTRACT = REPO_ROOT / "configs/first_pipeline_candidate.v1.json"

FREEZE_RECEIPT = "acceptance_freeze.json"
FREEZE_SCHEMA = "luke-first-pipeline-candidate-contract-freeze-v1"

MODE_AUTHORING = "authoring"
MODE_EXECUTION = "execution"
MODES = (MODE_AUTHORING, MODE_EXECUTION)

#: Requirements a contract may add to but never drop. The first five are the
#: ones the plan itself names. ``candidate.settings`` and
#: ``candidate.dependency_requirements_resolved`` are equally execution-defining
#: -- an execution with no selected candidate configuration, or with no
#: statement of which unresolved implementation dependencies it needs, is not a
#: comparison anybody can interpret -- and neither can be known before Step 2
#: chooses the candidate. So they are mandatory too: an *unset* dependency node
#: never means "this candidate needs nothing", it means the question has not
#: been answered, and execution refuses until it is.
MANDATORY_REQUIRED_PATHS = (
    "acceptance.practical_failure",
    "acceptance.margins.completeness",
    "acceptance.margins.identity",
    "acceptance.margins.contamination",
    "acceptance.margins.healthy_interval_preservation",
    "candidate.settings",
    "candidate.dependency_requirements_resolved",
)

#: The contract subtrees the freeze receipt binds. Freezing the acceptance
#: block alone is not enough: swapping the candidate settings, the dependency
#: declaration, either comparator, the intervals, the recording identity or the
#: output root after freezing would let results produced under one configuration
#: accumulate in an output root receipted for another. Deliberately excluded are
#: the prose fields that carry no execution meaning -- ``status``, ``purpose``,
#: ``non_goals``, ``governing_documents``, ``output_root_rules`` and ``notes``
#: -- so wording may be repaired after a freeze.
EXECUTION_DIGEST_PATHS = (
    "schema",
    "contract_id",
    "recording",
    "comparators",
    "intervals",
    "output_root",
    "candidate",
    "runtime_budget",
    "evaluation",
    "required_before_execution",
    "acceptance",
)

#: Both comparator roles must be identified before anything is comparable.
REQUIRED_COMPARATORS = ("legacy", "rescue_control")
#: A comparator identity that is not a path or a free-text note.
REQUIRED_COMPARATOR_FIELDS = ("role", "sort_id", "curated", "qc_dir", "source_recording")

#: Every margin declares which way improvement runs and what kind of magnitude
#: its number is. A ``minimum_improvement`` margin is the smallest improvement
#: that would justify adoption, so it must be strictly positive -- zero would be
#: exactly the "statistically detectable tiny change" the plan rejects. A
#: ``maximum_tolerated_degradation`` margin is a tolerance, so zero (tolerate
#: nothing) is meaningful and negative is not.
MARGIN_DIRECTIONS = ("increase_is_improvement", "decrease_is_improvement")
MARGIN_MAGNITUDE_KINDS = ("minimum_improvement", "maximum_tolerated_degradation")
MARGIN_DECLARATION_FIELDS = ("unit", "direction", "comparison", "magnitude_kind", "set_from")

#: A resolved candidate configuration names one intervention family and one
#: execution mode; the placeholder in the contract lists the same values.
CANDIDATE_INTERVENTION_FAMILIES = (
    "targeted_curation_repair",
    "option_b_unwarped_identity",
    "option_a_external_voltage_registration",
)
CANDIDATE_EXECUTION_MODES = ("retained_sort_replay", "resort")

STATE_SET = "set"
STATE_UNSET = "unset"


class ContractRefusal(ValueError):
    """The contract refuses to proceed. Never caught internally."""


# --------------------------------------------------------------------------- #
# provenance helpers
# --------------------------------------------------------------------------- #
def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args], text=True, stderr=subprocess.DEVNULL
    )


def git_commit() -> str:
    try:
        return _git("rev-parse", "HEAD").strip()
    except Exception:
        return "unknown"


def git_worktree_state() -> dict[str, Any]:
    """Commit plus a digest of the dirty working tree.

    The tree this contract is authored in is dirty, so the commit alone does
    not identify the source that ran. ``git status --porcelain`` is hashed
    rather than embedded so the receipt stays small but still changes whenever
    the working tree does.
    """
    try:
        porcelain = _git("status", "--porcelain")
    except Exception:
        return {
            "git_commit": git_commit(),
            "git_status_available": False,
            "git_tree_dirty": None,
            "git_status_porcelain_sha256": None,
            "git_dirty_entry_count": None,
        }
    entries = [line for line in porcelain.splitlines() if line.strip()]
    return {
        "git_commit": git_commit(),
        "git_status_available": True,
        "git_tree_dirty": bool(entries),
        "git_status_porcelain_sha256": hashlib.sha256(porcelain.encode()).hexdigest(),
        "git_dirty_entry_count": len(entries),
    }


def sha256_file(path: Path, _buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_buf), b""):
            h.update(chunk)
    return h.hexdigest()


def working_tree_source_hashes(contract_path: Path) -> dict[str, str]:
    """On-disk hashes of the source that is actually about to run."""
    files = {
        "validator_module": Path(__file__),
        "contract": Path(contract_path),
    }
    return {name: sha256_file(p) for name, p in files.items() if p.exists()}


def canonical_digest(node: Any) -> str:
    """Order-independent, whitespace-independent digest of a JSON subtree."""
    return hashlib.sha256(
        json.dumps(node, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# small structural helpers
# --------------------------------------------------------------------------- #
def get_path(payload: dict, dotted: str) -> Any:
    node: Any = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ContractRefusal(f"contract is missing required field {dotted!r}")
        node = node[part]
    return node


def is_unset(node: Any, dotted: str) -> bool:
    """True iff a settable field is still unset.

    Refuses a corrupt node rather than guessing: a settable field is always an
    object carrying ``state`` and ``value``, ``state`` is exactly ``set`` or
    ``unset``, and the two must agree. ``state: set`` with a null value is a
    corrupt null, not a set field, and ``state: unset`` with a value is an
    edit that skipped the freeze.
    """
    if not isinstance(node, dict) or "state" not in node or "value" not in node:
        raise ContractRefusal(
            f"{dotted} must be an object with 'state' and 'value'; got {type(node).__name__}"
        )
    state = node["state"]
    if state not in (STATE_SET, STATE_UNSET):
        raise ContractRefusal(f"{dotted}.state must be {STATE_SET!r} or {STATE_UNSET!r}, got {state!r}")
    value_missing = node["value"] is None
    if state == STATE_SET and value_missing:
        raise ContractRefusal(f"{dotted}.state is {STATE_SET!r} but its value is null")
    if state == STATE_UNSET and not value_missing:
        raise ContractRefusal(f"{dotted}.state is {STATE_UNSET!r} but it carries a value")
    return state == STATE_UNSET


def _resolved(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve()


def reject_unsafe_out_root(out_root: Path, input_paths: list[Path]) -> Path:
    """Refuse an output root under /mnt, or under/over any configured input.

    Same rule as ``testing/luke_amplitude_dropout_audit.py``; repeated rather
    than imported so that module stays free to move.
    """
    resolved = _resolved(out_root)
    if resolved == Path("/mnt") or str(resolved).startswith("/mnt/"):
        raise ContractRefusal(f"refusing an output root under /mnt: {resolved}")
    for p in input_paths:
        if not str(p):
            continue
        try:
            rp = _resolved(p)
        except OSError:
            continue
        if resolved == rp or resolved.is_relative_to(rp) or rp.is_relative_to(resolved):
            raise ContractRefusal(
                f"refusing an output root under/over an input directory: {resolved} vs {rp}"
            )
    return resolved


def results_present(out_root: Path) -> list[str]:
    """Names of result files already in the output root.

    Anything except the freeze receipt itself and in-flight ``.tmp`` files
    counts. An orphaned partial run is a result too: it is never silently
    overwritten, and the acceptance block may not move once it exists.
    """
    root = Path(out_root)
    if not root.exists():
        return []
    found = []
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(root).as_posix()
        if rel == FREEZE_RECEIPT or rel.endswith(".tmp"):
            continue
        found.append(rel)
    return found


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
@dataclass
class ValidationReport:
    schema: str
    mode: str
    contract_path: str
    contract_id: str
    out_root: str
    executable: bool
    unset_required_fields: tuple[str, ...]
    results_present: tuple[str, ...]
    acceptance_digest: str
    contract_digest: str
    contract_frozen: bool
    required_dependencies: tuple[str, ...]
    unresolved_required_dependencies: tuple[str, ...]
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "stage": "validate",
            "mode": self.mode,
            "contract_path": self.contract_path,
            "contract_id": self.contract_id,
            "out_root": self.out_root,
            "executable": self.executable,
            "unset_required_fields": list(self.unset_required_fields),
            "results_present": list(self.results_present),
            "acceptance_digest": self.acceptance_digest,
            "contract_digest": self.contract_digest,
            "contract_frozen": self.contract_frozen,
            "required_implementation_dependencies": list(self.required_dependencies),
            "unresolved_required_dependencies": list(self.unresolved_required_dependencies),
            "provenance": self.provenance,
        }


# --------------------------------------------------------------------------- #
# loading and structural checks
# --------------------------------------------------------------------------- #
def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ContractRefusal(f"contract {path} must be a JSON object")
    schema = payload.get("schema")
    if schema != SCHEMA:
        raise ContractRefusal(f"contract schema {schema!r} != {SCHEMA!r}")
    return payload


def required_paths(payload: dict) -> tuple[str, ...]:
    declared = payload.get("required_before_execution")
    if not isinstance(declared, list) or not declared:
        raise ContractRefusal("contract must declare a non-empty 'required_before_execution' list")
    if not all(isinstance(p, str) and p for p in declared):
        raise ContractRefusal("'required_before_execution' entries must be non-empty dotted paths")
    if len(set(declared)) != len(declared):
        raise ContractRefusal(f"duplicate entries in 'required_before_execution': {declared}")
    missing = [p for p in MANDATORY_REQUIRED_PATHS if p not in declared]
    if missing:
        raise ContractRefusal(
            "'required_before_execution' omits fields the plan requires; a contract may add "
            f"requirements but never drop one: {missing}"
        )
    return tuple(declared)


def check_comparators(payload: dict) -> dict[str, dict]:
    comparators = payload.get("comparators")
    if not isinstance(comparators, dict):
        raise ContractRefusal("contract is missing the 'comparators' block")
    missing_roles = [r for r in REQUIRED_COMPARATORS if r not in comparators]
    if missing_roles:
        raise ContractRefusal(f"missing comparator identity for {missing_roles}")
    for name in REQUIRED_COMPARATORS:
        entry = comparators[name]
        if not isinstance(entry, dict):
            raise ContractRefusal(f"comparator {name!r} must be an object")
        for key in REQUIRED_COMPARATOR_FIELDS:
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ContractRefusal(
                    f"missing comparator identity: {name}.{key} is absent or blank"
                )
        if not isinstance(entry.get("provenance"), dict) or not entry["provenance"]:
            raise ContractRefusal(
                f"missing comparator identity: {name}.provenance carries no receipts"
            )
    ids = [comparators[n]["sort_id"] for n in REQUIRED_COMPARATORS]
    if len(set(ids)) != len(ids):
        raise ContractRefusal(f"comparators must be distinct sorts, got {ids}")
    return {n: comparators[n] for n in REQUIRED_COMPARATORS}


def input_paths(payload: dict) -> list[Path]:
    paths: list[Path] = []
    for entry in check_comparators(payload).values():
        paths += [Path(entry[k]) for k in ("curated", "qc_dir", "source_recording")]
    data_dir = payload.get("recording", {}).get("data_dir")
    if isinstance(data_dir, str) and data_dir:
        paths.append(Path(data_dir))
    return paths


def _interval_list(node: Any, label: str) -> list[tuple[float, float]]:
    if not isinstance(node, list) or not node:
        raise ContractRefusal(f"{label} must be a non-empty list of [start_s, stop_s] pairs")
    out = []
    for item in node:
        if isinstance(item, dict):
            pair = (item.get("start_s"), item.get("stop_s"))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            pair = (item[0], item[1])
        else:
            raise ContractRefusal(f"{label} entry is not an interval: {item!r}")
        start, stop = pair
        if not isinstance(start, (int, float)) or not isinstance(stop, (int, float)):
            raise ContractRefusal(f"{label} interval bounds must be numbers, got {pair!r}")
        if isinstance(start, bool) or isinstance(stop, bool):
            raise ContractRefusal(f"{label} interval bounds must be numbers, got {pair!r}")
        if not (float(stop) > float(start)):
            raise ContractRefusal(f"{label} interval must have stop_s > start_s, got {pair!r}")
        out.append((float(start), float(stop)))
    return out


def development_windows(payload: dict) -> list[tuple[float, float]]:
    node = get_path(payload, "intervals.development_windows.windows_s")
    return _interval_list(node, "intervals.development_windows.windows_s")


def check_failure_interval_in_development_window(payload: dict) -> None:
    """A set failure interval must sit inside one development window.

    The development windows already exclude the sealed panel and its buffer,
    so this is also what keeps a selected case from silently consuming the
    sealed held-out panel.
    """
    node = get_path(payload, "acceptance.practical_failure")
    if is_unset(node, "acceptance.practical_failure"):
        return
    value = node["value"]
    interval = value.get("interval_s") if isinstance(value, dict) else None
    if interval is None:
        raise ContractRefusal(
            "acceptance.practical_failure.value must carry an interval_s once it is set"
        )
    (start, stop), = _interval_list([interval], "acceptance.practical_failure.value.interval_s")
    for w_start, w_stop in development_windows(payload):
        if start >= w_start and stop <= w_stop:
            return
    raise ContractRefusal(
        f"the selected failure interval [{start}, {stop}] is not contained in any development "
        "window; development windows exclude the sealed panel and its buffer"
    )


def check_required_dependencies(payload: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (required ids, required-but-unresolved ids).

    An unset ``candidate.dependency_requirements_resolved`` returns empty
    tuples, which is *not* the same as "this candidate requires nothing": the
    path is in :data:`MANDATORY_REQUIRED_PATHS`, so execution refuses while it
    is unset and this branch is only ever reached while authoring.
    """
    catalog = get_path(payload, "candidate.unresolved_implementation_dependencies")
    if not isinstance(catalog, list):
        raise ContractRefusal("candidate.unresolved_implementation_dependencies must be a list")
    known = {}
    for entry in catalog:
        if not isinstance(entry, dict) or not entry.get("id") or not entry.get("status"):
            raise ContractRefusal(
                f"each implementation dependency needs an 'id' and a 'status': {entry!r}"
            )
        known[entry["id"]] = entry["status"]

    node = get_path(payload, "candidate.dependency_requirements_resolved")
    if is_unset(node, "candidate.dependency_requirements_resolved"):
        return (), ()
    ids = node["value"]
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ContractRefusal(
            "candidate.dependency_requirements_resolved.value must be a list of dependency ids"
        )
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise ContractRefusal(f"unknown implementation dependency ids: {unknown}")
    unresolved = tuple(i for i in ids if known[i] != "resolved")
    return tuple(ids), unresolved


# --------------------------------------------------------------------------- #
# value shapes: a non-null value is not yet a usable value
# --------------------------------------------------------------------------- #
def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractRefusal(f"{label} must be a finite number, got {value!r}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ContractRefusal(f"{label} must be a finite number, got {value!r}")
    return number


def check_margin_declarations(payload: dict) -> dict[str, dict]:
    """Every margin declares its unit, direction, magnitude kind and source.

    These are knowable at authoring time, so they are checked whether or not a
    value has been set: a number without them cannot be interpreted, and there
    is no honest way to check a value against a declaration that is missing.
    """
    margins = get_path(payload, "acceptance.margins")
    if not isinstance(margins, dict) or not margins:
        raise ContractRefusal("acceptance.margins must be a non-empty object")
    for name, node in margins.items():
        label = f"acceptance.margins.{name}"
        if not isinstance(node, dict):
            raise ContractRefusal(f"{label} must be an object")
        for key in MARGIN_DECLARATION_FIELDS:
            value = node.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ContractRefusal(f"{label} must declare a non-empty {key!r}")
        if node["direction"] not in MARGIN_DIRECTIONS:
            raise ContractRefusal(
                f"{label}.direction must be one of {MARGIN_DIRECTIONS}, got {node['direction']!r}"
            )
        if node["magnitude_kind"] not in MARGIN_MAGNITUDE_KINDS:
            raise ContractRefusal(
                f"{label}.magnitude_kind must be one of {MARGIN_MAGNITUDE_KINDS}, "
                f"got {node['magnitude_kind']!r}"
            )
    return margins


def check_margin_value(name: str, node: dict) -> None:
    label = f"acceptance.margins.{name}"
    number = _finite_number(node["value"], f"{label}.value")
    kind = node["magnitude_kind"]
    if kind == "minimum_improvement" and not number > 0:
        raise ContractRefusal(
            f"{label}.value is a minimum_improvement in {node['unit']!r} "
            f"({node['direction']}), so it must be strictly positive; got {number!r}. "
            "A zero or negative margin accepts a statistically detectable tiny change."
        )
    if kind == "maximum_tolerated_degradation" and number < 0:
        raise ContractRefusal(
            f"{label}.value is a maximum_tolerated_degradation in {node['unit']!r}, "
            f"so it must be >= 0; got {number!r}"
        )


def check_practical_failure_value(payload: dict, node: dict) -> None:
    label = "acceptance.practical_failure.value"
    value = node["value"]
    if not isinstance(value, dict):
        raise ContractRefusal(f"{label} must be an object naming the case, got {type(value).__name__}")
    if "interval_s" not in value:
        raise ContractRefusal(f"{label} must carry an interval_s once it is set")
    for key in ("name", "sort_id", "cluster_id"):
        if key not in value:
            raise ContractRefusal(f"{label} must carry {key!r} once it is set")
    for key in ("name", "sort_id"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ContractRefusal(f"{label}.{key} must be a non-empty string, got {value[key]!r}")
    known = {entry["sort_id"] for entry in check_comparators(payload).values()}
    if value["sort_id"] not in known:
        raise ContractRefusal(
            f"{label}.sort_id {value['sort_id']!r} does not name a declared comparator {sorted(known)}"
        )
    cluster_id = value["cluster_id"]
    if isinstance(cluster_id, bool) or not isinstance(cluster_id, int):
        # A cluster id is a value, never an array position, and never a float:
        # 17.0 and "17" both hide which array they were read out of.
        raise ContractRefusal(f"{label}.cluster_id must be an integer id, got {cluster_id!r}")


def check_candidate_settings_value(node: dict) -> None:
    label = "candidate.settings.value"
    value = node["value"]
    if not isinstance(value, dict) or not value:
        raise ContractRefusal(f"{label} must be a non-empty resolved configuration object")
    family = value.get("intervention_family")
    if family not in CANDIDATE_INTERVENTION_FAMILIES:
        raise ContractRefusal(
            f"{label}.intervention_family must be one of {CANDIDATE_INTERVENTION_FAMILIES}, "
            f"got {family!r}"
        )
    mode = value.get("execution_mode")
    if mode not in CANDIDATE_EXECUTION_MODES:
        raise ContractRefusal(
            f"{label}.execution_mode must be one of {CANDIDATE_EXECUTION_MODES}, got {mode!r}"
        )


def check_required_value_shapes(payload: dict) -> None:
    """Check every *set* required value against its declared shape.

    Being non-null is not being usable. A margin that is a string, an empty
    settings object, or a failure case with no cluster id would otherwise
    freeze and report executable.
    """
    margins = check_margin_declarations(payload)
    for name, node in margins.items():
        if not is_unset(node, f"acceptance.margins.{name}"):
            check_margin_value(name, node)

    failure = get_path(payload, "acceptance.practical_failure")
    if not is_unset(failure, "acceptance.practical_failure"):
        check_practical_failure_value(payload, failure)

    settings = get_path(payload, "candidate.settings")
    if not is_unset(settings, "candidate.settings"):
        check_candidate_settings_value(settings)


# --------------------------------------------------------------------------- #
# freeze receipt
# --------------------------------------------------------------------------- #
def acceptance_digest(payload: dict) -> str:
    """Digest of the acceptance block alone, for reporting which half moved."""
    return canonical_digest(get_path(payload, "acceptance"))


def contract_digest(payload: dict) -> str:
    """Digest of the whole execution-defining contract.

    This is what the freeze receipt binds; see :data:`EXECUTION_DIGEST_PATHS`.
    """
    return canonical_digest({p: get_path(payload, p) for p in EXECUTION_DIGEST_PATHS})


def read_freeze_receipt(out_root: Path) -> dict[str, Any] | None:
    path = Path(out_root) / FREEZE_RECEIPT
    if not path.exists():
        return None
    try:
        receipt = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ContractRefusal(f"acceptance freeze receipt {path} is not readable JSON: {exc}")
    if not isinstance(receipt, dict) or receipt.get("schema") != FREEZE_SCHEMA:
        raise ContractRefusal(f"acceptance freeze receipt {path} has the wrong schema")
    for key in ("acceptance_digest", "contract_digest"):
        if not receipt.get(key):
            raise ContractRefusal(f"acceptance freeze receipt {path} carries no {key}")
    return receipt


def check_contract_not_edited_after_results(payload: dict, out_root: Path) -> tuple[list[str], bool]:
    """Refuse an execution-defining edit once results exist.

    The receipt binds :data:`EXECUTION_DIGEST_PATHS`, not the acceptance block
    alone, so swapping the candidate settings, a comparator, the intervals, the
    recording identity or the output root is caught here too. Returns
    ``(results, frozen)``.
    """
    found = results_present(out_root)
    receipt = read_freeze_receipt(out_root)
    digest = contract_digest(payload)
    if found and receipt is None:
        raise ContractRefusal(
            f"{len(found)} result file(s) already exist in {out_root} with no {FREEZE_RECEIPT}; "
            "acceptance margins cannot be set or frozen after results exist"
        )
    if receipt is not None and receipt["contract_digest"] != digest and found:
        moved = (
            "the acceptance block"
            if receipt["acceptance_digest"] != acceptance_digest(payload)
            else "an execution-defining field outside the acceptance block"
        )
        raise ContractRefusal(
            f"{moved} was edited after results exist in the output root "
            f"(frozen {receipt['contract_digest'][:12]}, now {digest[:12]}); "
            f"first result seen: {found[0]}"
        )
    return found, receipt is not None and receipt["contract_digest"] == digest


def freeze_acceptance(contract_path: Path, out_root: Path | None = None) -> dict[str, Any]:
    """Freeze the execution-defining contract into the output root.

    Refuses unless every required field is set, every set value matches its
    declared shape, and no results exist yet: the whole purpose is that the
    margins are fixed from baseline evidence before anybody looks at a
    candidate result.
    """
    payload = load_contract(contract_path)
    declared = required_paths(payload)
    check_comparators(payload)
    root = reject_unsafe_out_root(
        Path(out_root) if out_root is not None else Path(get_path(payload, "output_root")),
        input_paths(payload),
    )
    unset = tuple(p for p in declared if is_unset(get_path(payload, p), p))
    if unset:
        raise ContractRefusal(f"cannot freeze acceptance while these fields are unset: {list(unset)}")
    check_required_value_shapes(payload)
    check_failure_interval_in_development_window(payload)
    check_required_dependencies(payload)
    found = results_present(root)
    if found:
        raise ContractRefusal(
            f"refusing to freeze acceptance: {len(found)} result file(s) already exist in {root} "
            f"(first: {found[0]})"
        )
    existing = read_freeze_receipt(root)
    digest = contract_digest(payload)
    if existing is not None and existing["contract_digest"] != digest:
        raise ContractRefusal(
            f"{root / FREEZE_RECEIPT} already froze a different acceptance block "
            f"({existing['contract_digest'][:12]} != {digest[:12]})"
        )
    receipt = {
        "schema": FREEZE_SCHEMA,
        "contract_id": payload.get("contract_id"),
        "contract_path": str(Path(contract_path).resolve()),
        "contract_digest": digest,
        "contract_digest_covers": list(EXECUTION_DIGEST_PATHS),
        "acceptance_digest": acceptance_digest(payload),
        "required_before_execution": list(declared),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            **git_worktree_state(),
            "source_sha256": working_tree_source_hashes(Path(contract_path)),
        },
    }
    _atomic_write_json(root / FREEZE_RECEIPT, receipt)
    return receipt


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    mode: str = MODE_AUTHORING,
    out_root: Path | None = None,
) -> ValidationReport:
    if mode not in MODES:
        raise ContractRefusal(f"mode must be one of {MODES}, got {mode!r}")
    payload = load_contract(contract_path)
    declared = required_paths(payload)
    check_comparators(payload)
    root = reject_unsafe_out_root(
        Path(out_root) if out_root is not None else Path(get_path(payload, "output_root")),
        input_paths(payload),
    )
    _interval_list(
        get_path(payload, "intervals.healthy_control_intervals.windows"),
        "intervals.healthy_control_intervals.windows",
    )
    development_windows(payload)

    unset = tuple(p for p in declared if is_unset(get_path(payload, p), p))
    check_required_value_shapes(payload)
    check_failure_interval_in_development_window(payload)
    required_deps, unresolved_deps = check_required_dependencies(payload)
    found, frozen = check_contract_not_edited_after_results(payload, root)

    if mode == MODE_EXECUTION:
        if unset:
            raise ContractRefusal(
                "refusing execution: the contract's required fields are still unset. They must be "
                "set from BASELINE evidence before any candidate result is inspected. Unset: "
                f"{list(unset)}"
            )
        if not frozen:
            raise ContractRefusal(
                f"refusing execution: the contract is not frozen in {root} (or was edited since "
                "the freeze). Run `freeze` first so the margins and the rest of the "
                "execution-defining contract are receipted before any candidate result exists."
            )
        if unresolved_deps:
            raise ContractRefusal(
                "refusing execution: implementation dependencies this candidate requires are not "
                f"resolved: {list(unresolved_deps)}"
            )

    return ValidationReport(
        schema=SCHEMA,
        mode=mode,
        contract_path=str(Path(contract_path).resolve()),
        contract_id=str(payload.get("contract_id", "")),
        out_root=str(root),
        executable=(not unset) and frozen and not unresolved_deps,
        unset_required_fields=unset,
        results_present=tuple(found),
        acceptance_digest=acceptance_digest(payload),
        contract_digest=contract_digest(payload),
        contract_frozen=frozen,
        required_dependencies=required_deps,
        unresolved_required_dependencies=unresolved_deps,
        provenance={
            **git_worktree_state(),
            "source_sha256": working_tree_source_hashes(Path(contract_path)),
            "validated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="command", required=True)

    val = sub.add_parser("validate", help="check the contract; --mode execution refuses unset fields")
    val.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    val.add_argument("--mode", choices=MODES, default=MODE_AUTHORING)
    val.add_argument("--out-root", type=Path, default=None)

    frz = sub.add_parser("freeze", help="write the acceptance freeze receipt into the output root")
    frz.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    frz.add_argument("--out-root", type=Path, default=None)

    args = ap.parse_args(argv)
    try:
        if args.command == "validate":
            report = validate(args.contract, mode=args.mode, out_root=args.out_root)
            print(json.dumps(report.to_dict(), indent=2, default=str))
            return 0
        if args.command == "freeze":
            receipt = freeze_acceptance(args.contract, args.out_root)
            print(json.dumps(receipt, indent=2, default=str))
            return 0
    except ContractRefusal as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, indent=2))
        return 1

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
