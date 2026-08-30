import pytest

from testing import luke_rigid025_interior_sensitivity as audit


def test_margin_guard_rejects_empty_or_nonpositive_interior(tmp_path):
    for margin in (0.0, -1.0, 470.0):
        with pytest.raises(ValueError):
            audit.run(margin, tmp_path / "unused.csv", tmp_path / "out.json")
