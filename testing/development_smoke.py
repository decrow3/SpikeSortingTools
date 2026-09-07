"""Bounded engineering-only smoke contract derived from a frozen ladder contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from pipeline.config import fingerprint
from testing.development_ladder import DevelopmentContract


SMOKE_SCHEMA = "longitudinal-development-engineering-smoke-v1"
MAX_SMOKE_DURATION_S = 300.0


def build_smoke_contract(
    parent: DevelopmentContract,
    *,
    candidate_names,
    start_s: float = 0.0,
    duration_s: float = 120.0,
) -> tuple[DevelopmentContract, dict]:
    """Create a short engineering contract that cannot be mistaken for efficacy data."""
    names = tuple(candidate_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("smoke candidate names must be nonempty and unique")
    if not 0 < duration_s <= MAX_SMOKE_DURATION_S:
        raise ValueError(f"smoke duration must be in (0, {MAX_SMOKE_DURATION_S}] seconds")
    recording = parent.raw["recording"]
    if start_s < 0 or start_s + duration_s > recording["total_duration_s"]:
        raise ValueError("smoke interval lies outside the accepted recording")
    by_name = {candidate["name"]: candidate for candidate in parent.candidates}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"unknown smoke candidate name(s): {unknown}")
    raw = copy.deepcopy(parent.raw)
    raw["experiment_id"] = f"{raw['experiment_id']}-engineering-smoke"
    raw["development_status"] = "engineering_only_not_for_scientific_selection"
    raw["decision_statement"] = "If every arm completes and validates its saved settings, then the long-strip launch may be considered; no scientific selection is permitted."
    raw["recording"].update(
        start_s=float(start_s), duration_s=float(duration_s), full_session=False
    )
    raw["candidates"] = [copy.deepcopy(by_name[name]) for name in names]
    identity = {
        "schema_version": SMOKE_SCHEMA,
        "parent_contract_digest": parent.digest,
        "recording": raw["recording"],
        "spatial_contract": raw["spatial_contract"],
        "candidates": raw["candidates"],
        "engineering_only": True,
    }
    smoke = DevelopmentContract(source=parent.source, raw=raw, digest=fingerprint(identity))
    plan = {**identity, "smoke_contract_digest": smoke.digest}
    return smoke, plan


def pin_smoke_plan(plan: dict, path: Path | str) -> dict:
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise RuntimeError("existing smoke plan belongs to another request")
        return plan
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + "\n")
    return plan
