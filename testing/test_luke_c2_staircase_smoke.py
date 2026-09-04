import numpy as np
import pytest

from testing.luke_c2_staircase_smoke import OUTPUT, SMOKE, donor_placement
from testing.luke_c2_staircase_control import STAIRCASE, expected_shift_channels


def test_smoke_covers_the_four_required_cells():
    cells = {(c["arm"], c["sorter"]) for c in SMOKE["cells"]}
    assert cells == {
        ("static", "rescue"),
        ("staircase", "rescue"),
        ("staircase", "rescue_rigid"),
        ("staircase_corrected", "rescue"),
    }


def test_smoke_is_labelled_engineering_only():
    assert "not a C2 result" in SMOKE["status"]
    assert not str(OUTPUT).startswith("/mnt/")


def test_smoke_donor_subset_spans_polarity_and_amplitude():
    import pandas as pd

    from testing.luke_rescue_c2_drift_challenge import DONOR_MANIFEST

    meta = pd.read_csv(DONOR_MANIFEST).set_index("template_id")
    subset = meta.loc[SMOKE["donors"]]
    assert set(subset.polarity) == {"neg", "pos"}          # both polarities
    assert subset.peak_uv.max() / subset.peak_uv.min() > 3  # a real amplitude span
    # D01 fails statically under legacy_style, D10 under rescue: static
    # performance is spanned too, not just clean donors
    assert {"D01", "D10"} <= set(SMOKE["donors"])


def test_placement_refuses_a_donor_too_close_to_the_crop_edge():
    """The excursion must not move the footprint out of the verified region."""
    shift = expected_shift_channels(max(STAIRCASE["levels_um"]))
    n_crop = 40
    geometry = np.column_stack([
        np.tile([0.0, 32.0, 16.0, 48.0], n_crop // 4),
        np.repeat(np.arange(n_crop // 2) * 20.0, 2),
    ])
    template = np.zeros((61, 8), dtype=np.float32)
    template[30, 4] = -150.0
    donors = {"DXX": template}
    meta = {"DXX": {"peak_channel": 4}}

    class _Edge:
        """A placement helper that pins the donor hard against channel 0."""

        @staticmethod
        def place(*args, **kwargs):
            return 1, 5  # base_channel 1: only 1 channel of headroom, < shift

    import testing.luke_c2_staircase_smoke as smoke

    original = smoke.donor_base_channel
    smoke.donor_base_channel = _Edge.place
    try:
        with pytest.raises(RuntimeError, match="crop edge"):
            donor_placement("DXX", donors, meta, geometry, geometry, 8)
    finally:
        smoke.donor_base_channel = original
    assert shift == 4
