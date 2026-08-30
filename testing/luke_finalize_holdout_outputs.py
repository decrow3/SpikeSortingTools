"""Validate and separate reviewer-facing Luke v2 holdout artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from testing.luke_draw_prospective_holdout_events import (
    MANIFEST,
    METHOD,
    OUTPUT,
    PARENT_SEAL,
    SEALED_WINDOWS,
    sha256_file,
    validate_parent_seal,
)


def finalize(output: Path = OUTPUT) -> dict:
    validate_parent_seal(MANIFEST, PARENT_SEAL, SEALED_WINDOWS)
    summary_path = output / "event_draw_summary_v2.json"
    strata_path = output / "holdout_candidates_v2.csv"
    key_path = output / "holdout_candidate_key_v2.csv"
    deficits_path = output / "holdout_cell_deficits_v2.csv"
    reviewer_path = output / "holdout_reviewer_candidates_v2.csv"

    summary = json.loads(summary_path.read_text())
    if summary.get("complete_draw") is not True or summary.get("events_selected") != 864:
        raise RuntimeError("Full 864-event draw is not complete")
    expected_hashes = {
        strata_path: summary["blinded_candidates_sha256"],
        key_path: summary["candidate_key_sha256"],
        deficits_path: summary["cell_deficits_sha256"],
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"Post-draw hash mismatch for {path}")

    strata = pd.read_csv(strata_path)
    key = pd.read_csv(key_path)
    deficits = pd.read_csv(deficits_path)
    if len(strata) != 864 or len(key) != 864 or strata.candidate_id.nunique() != 864:
        raise RuntimeError("Expected 864 unique candidate IDs in strata and key")
    if set(strata.candidate_id) != set(key.candidate_id):
        raise RuntimeError("Candidate IDs differ between strata and key")
    if len(deficits) != 216 or deficits.selected_count.sum() != 864 or deficits.deficit.sum() != 0:
        raise RuntimeError("Quota/deficit table is not complete")

    reviewer = strata[["candidate_id"]].sort_values("candidate_id")
    reviewer.to_csv(reviewer_path, index=False)
    roles = {
        "version": 2,
        "reviewer_facing": {
            "path": str(reviewer_path),
            "columns": ["candidate_id"],
            "sha256": sha256_file(reviewer_path),
        },
        "internal_stratification_manifest": {
            "path": str(strata_path),
            "sha256": sha256_file(strata_path),
            "not_reviewer_facing": True,
        },
        "sealed_coordinate_key": {
            "path": str(key_path),
            "sha256": sha256_file(key_path),
            "not_reviewer_facing": True,
        },
        "cell_deficits": {
            "path": str(deficits_path),
            "sha256": sha256_file(deficits_path),
            "cells": 216,
            "total_deficit": 0,
        },
        "manifest_sha256": sha256_file(MANIFEST),
        "method_sha256": sha256_file(METHOD),
    }
    roles_path = output / "holdout_output_roles_v2.json"
    roles_path.write_text(json.dumps(roles, indent=2) + "\n")
    return roles


if __name__ == "__main__":
    print(json.dumps(finalize(), indent=2))
