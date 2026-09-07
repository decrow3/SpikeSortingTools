import hashlib
from pathlib import Path

import pandas as pd
import pytest

from testing.luke_threshold_grid_analysis import EDGES, _interaction_summary, validate_inventory


def _write_inventory(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(f"{digest}  ./{path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n")


def test_edges_complete_two_cells_and_change_one_axis():
    assert len(EDGES) == 8
    for _, _, baseline, candidate in EDGES:
        bu, bl = map(int, baseline.split("/"))
        cu, cl = map(int, candidate.split("/"))
        assert (bu == cu) ^ (bl == cl)
        assert cu <= bu and cl <= bl


def test_inventory_is_exact_and_content_bound(tmp_path):
    (tmp_path / "a").write_bytes(b"one")
    _write_inventory(tmp_path)
    result = validate_inventory(tmp_path)
    assert result["file_count"] == 1
    (tmp_path / "a").write_bytes(b"two")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        validate_inventory(tmp_path)


def test_inventory_rejects_uninventoried_file(tmp_path):
    (tmp_path / "a").write_bytes(b"one")
    _write_inventory(tmp_path)
    (tmp_path / "b").write_bytes(b"two")
    with pytest.raises(RuntimeError, match="exactly cover"):
        validate_inventory(tmp_path)


def test_interaction_summary_matches_difference_of_log_ratios():
    quartets = pd.DataFrame(
        {
            "cell": ["cell_9_10_by_8_9", "cell_9_10_by_8_9"],
            "interaction_log2_rate": [1.0, -0.5],
        }
    )
    spikes = {
        "9/8": 200,
        "9/9": 100,
        "10/8": 100,
        "10/9": 100,
        "10/10": 50,
        "12/9": 50,
        "12/10": 50,
    }
    rows = _interaction_summary(quartets, spikes)
    assert rows[0]["median_interaction_log2_rate"] == 0.25
    assert rows[0]["aggregate_spike_yield_interaction_log2"] == 1.0
    assert rows[1]["strict_path_consistent_quartets"] == 0
