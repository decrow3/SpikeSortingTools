#!/usr/bin/env python3
"""Audit availability of the frozen imec1 lighthouse source packet."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "twenty_candidate_assessment.csv": "639223bbdf1af23c80d8cdcf08b79cbc9baec456a79a982e15a76bfb261f8891",
    "qualification_events_compact.csv": "cd0592c5758e56e9e1a161260c99e5afa39d4839805177bcb94db66913d34f96",
    "frozen_training_templates.npz": "b6d60f71655900312a6e447870b4dc9934313c38d43151880165a326f2f43aa7",
    "families_after_depth_reveal.csv": "c452cc78b1d036f7118aa9a20d850fcbb1c2d9709ac12d457275098b538a55f8",
    "strict_primary_events.csv": "5d0525aed89c1de56f992719ad2605e991c59ab790bfc8ba2580fe526659dc8a",
    "extra_window_summary.json": "b4f188d210eef02cf3e4aa019a4e1bc2af26d998f001b261547709a4a09c45c1",
}
CANDIDATES = {
    "twenty_candidate_assessment.csv": ROOT / "testing/outputs/luke_imec1_twenty_candidate_assessment_v1/twenty_candidate_assessment.csv",
    "qualification_events_compact.csv": ROOT / "testing/outputs/luke_imec1_compact_alignment_pilot_v2/qualification_events_compact.csv",
    "frozen_training_templates.npz": ROOT / "testing/outputs/luke_imec1_compact_alignment_pilot_v2/frozen_training_templates.npz",
    "families_after_depth_reveal.csv": ROOT / "testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3_s300/families_after_depth_reveal.csv",
    "strict_primary_events.csv": ROOT / "testing/outputs/luke_imec1_lighthouse_extra_windows_v3/strict_primary_events.csv",
    "extra_window_summary.json": ROOT / "testing/outputs/luke_imec1_lighthouse_extra_windows_v3/summary.json",
}
ARCHIVE = Path(
    "/mnt/NPX/Luke/luke_imec1_lighthouse_motion_comparison_v1/"
    "saved-array-exports-20260923-v1.zip"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    rows = []
    for name, expected in EXPECTED.items():
        path = CANDIDATES[name]
        actual = sha256(path) if path.is_file() else None
        rows.append(
            {
                "name": name,
                "expected_sha256": expected,
                "canonical_path": str(path),
                "exists": path.is_file(),
                "actual_sha256": actual,
                "hash_matches": actual == expected,
            }
        )

    archive_members = []
    if ARCHIVE.is_file():
        with zipfile.ZipFile(ARCHIVE) as bundle:
            archive_members = sorted(bundle.namelist())
    matching_archive_members = [
        name for name in archive_members if Path(name).name in EXPECTED
    ]
    objects = subprocess.check_output(
        ["git", "rev-list", "--objects", "--all"], cwd=ROOT, text=True
    ).splitlines()
    git_name_matches = [
        line for line in objects if Path(line.split(" ", 1)[-1]).name in EXPECTED
    ]
    status = "available" if all(row["hash_matches"] for row in rows) else "blocked_missing_sources"
    result = {
        "schema": "luke-imec1-lighthouse-input-availability-v1",
        "status": status,
        "voltage_read": False,
        "sort_launched": False,
        "inputs": rows,
        "archive": {
            "path": str(ARCHIVE),
            "exists": ARCHIVE.is_file(),
            "members": len(archive_members),
            "matching_source_members": matching_archive_members,
        },
        "git_object_name_matches": git_name_matches,
        "searched_roots": [
            "/home/huklaban5",
            "/home/huklab",
            "/media/huklaban5",
            "/media/huklab",
            "/mnt/NPX/Luke",
            "/tmp",
            "/var/tmp",
        ],
        "interpretation": (
            "The frozen biological-identity event packet cannot be reconstructed from "
            "the retained hashes alone. Population scorecards and event overlap remain "
            "valid for their stated scope but are not substitutes for lighthouse identity evidence."
        ),
    }
    result_path = args.output / "AUDIT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output / "COMPLETE.json").write_text(
        json.dumps(
            {"status": "complete", "audit_sha256": sha256(result_path)},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
