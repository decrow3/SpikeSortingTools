import numpy as np
import pytest

from testing.luke_injected_ground_truth_benchmark import validate_template
from testing.luke_rescue_c2_drift_challenge import (
    PRESPEC,
    TP,
    _train,
    _trajectory_fn,
    _resolve_frozen_cohort,
    donor_base_channel,
    prepare_template,
)

FS = 30_000.0


def test_prespec_is_frozen_shape():
    assert PRESPEC["schema"] == "luke-rescue-c2-drift-challenge-v3"
    assert PRESPEC["probe"] == "imec1"
    assert "static" in PRESPEC["trajectories"]
    assert PRESPEC["status"] == "compact_donor_geometry_aware_rerun_pending"
    assert PRESPEC["static_qualification"]["required_sorters"] == [
        "rescue",
        "legacy_style",
    ]


def test_prepare_template_yields_a_sealable_template():
    compact = np.zeros((61, 33), dtype=np.float32)
    compact[25:36, 14:19] = 5.0
    compact[30, 16] = -180.0
    tmpl, peak_col = prepare_template(compact)

    assert tmpl.shape == (TP["time_samples"], 2 * TP["channel_radius"] + 1)
    # passes the sealed primitive's own edge check at the C2 guard width
    validate_template(tmpl, edge_guard_samples=TP["edge_guard_samples"])
    guard = TP["edge_guard_samples"]
    assert np.all(tmpl[:guard] == 0) and np.all(tmpl[-guard:] == 0)
    assert peak_col == TP["channel_radius"]  # peak centred in the crop
    assert np.array_equal(tmpl, compact)  # no second crop/taper changes the donor


def test_c2_v3_forbids_pilot_or_subset_cohorts():
    donors = [f"D{i:02d}" for i in range(1, 15)]
    assert _resolve_frozen_cohort(donors, None) == donors
    with pytest.raises(ValueError, match="all 14"):
        _resolve_frozen_cohort(donors, ["D01", "D02"])
    with pytest.raises(RuntimeError, match="14 compact D-donor"):
        _resolve_frozen_cohort(["T01", *donors[1:]], None)


def test_donor_placement_preserves_four_column_geometry_phase():
    source = np.array(
        [[16, 0], [48, 0], [0, 20], [32, 20],
         [16, 40], [48, 40], [0, 60], [32, 60]],
        dtype=float,
    )
    target = source.copy()
    target[:, 1] += 1_000
    template = np.zeros((61, 5), dtype=np.float32)
    template[30, 2] = -1
    start, peak = donor_base_channel(template, 2, 2, source, target)
    assert (start, peak) == (0, 2)

    target_bad = target.copy()
    target_bad[:, 0] = 0
    with pytest.raises(ValueError, match="no target placement"):
        donor_base_channel(template, 2, 2, source, target_bad)


def test_train_is_regular_and_inside_the_guard():
    train = _train(120.0, FS)
    guard = int(PRESPEC["train"]["guard_s"] * FS)
    assert train[0] >= guard
    assert train[-1] <= int(120.0 * FS) - guard
    step = np.unique(np.diff(train))
    assert len(step) == 1 and step[0] == round(FS / PRESPEC["train"]["rate_hz"])


def test_trajectory_fn_converts_um_to_channels_via_geometry():
    geom = np.column_stack([np.zeros(112), np.linspace(1360.0, 2460.0, 112)])
    fn, meta = _trajectory_fn("rigid_40um", geom, 120.0)
    # 40 µm over a 1100 µm / 111-step strip -> ~4 channels end to end
    end = float(fn(np.array([120.0]))[0])
    assert 3.5 < end < 4.5
    assert meta["total_um"] == 40.0

    static_fn, _ = _trajectory_fn("static", geom, 120.0)
    assert np.all(static_fn(np.linspace(0, 120, 5)) == 0)


def test_trajectory_fn_rejects_unknown():
    geom = np.column_stack([np.zeros(10), np.arange(10) * 10.0])
    with pytest.raises((ValueError, KeyError)):
        _trajectory_fn("does_not_exist", geom, 120.0)
