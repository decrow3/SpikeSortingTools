"""Independent synthetic-only adversarial review of frozen K9 core v1."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


PACKET = Path(__file__).resolve().parents[1] / "joint_model_k9_core_h1_20261012_v1"
REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
SPEC = importlib.util.spec_from_file_location("frozen_joint_model_k9_v1", PACKET / "IMPLEMENTATION.py")
assert SPEC is not None and SPEC.loader is not None
k9 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = k9
SPEC.loader.exec_module(k9)


FS = 30_000.0
ORIGIN = 900_000


def geometry():
    return np.asarray([(x, y) for y in range(0, 240, 40) for x in (0, 32)], dtype=float)


def template(channel=4, scale=1.0):
    value = np.zeros((9, 12), dtype=np.float32)
    waveform = scale * np.asarray([-10, -30, -80, -30, -10], dtype=np.float32)
    value[2:7, channel] = waveform
    value[2:7, channel + 1] = waveform * np.float32(0.4)
    return value


def event(event_id, sample=100, **changes):
    values = dict(
        event_id=event_id,
        biological_unit_id="u1",
        template_id="t1",
        sample_index=sample,
        frame="tissue",
        role="moving_neural",
    )
    values.update(changes)
    return k9.K9Event(**values)


def inject(events, *, background=None, geom=None, templates=None, origin=ORIGIN):
    return k9.inject_k9_precar(
        np.zeros((300, 12), np.float32) if background is None else background,
        geometry() if geom is None else geom,
        {"t1": template()} if templates is None else templates,
        events,
        sampling_frequency_hz=FS,
        session_start_frame=origin,
    )


def peak_channel(values, sample):
    return int(np.argmax(np.max(np.abs(values[sample - 4:sample + 5]), axis=0)))


def test_half_open_state_ownership_at_every_boundary_and_outside():
    samples = np.asarray([0, 39, 40, 79, 80, 119], dtype=np.int64)
    result = k9.assign_half_open_states(
        samples,
        np.asarray([0, 40, 80], dtype=np.int64),
        np.asarray([40, 80, 120], dtype=np.int64),
        np.asarray([-40, 0, 40], dtype=np.int64),
    )
    assert result.tolist() == [-40, -40, 0, 0, 40, 40]
    for outside in (-1, 120):
        with pytest.raises(ValueError, match="outside"):
            k9.assign_half_open_states([outside], [0, 40, 80], [40, 80, 120], [-40, 0, 40])


def test_state_intervals_require_positive_contiguous_exact_lattice_support():
    with pytest.raises(ValueError, match="contiguous"):
        k9.assign_half_open_states([10], [0, 50], [40, 100], [0, 40])
    with pytest.raises(ValueError, match="positive and contiguous"):
        k9.assign_half_open_states([10], [0, 40], [40, 40], [0, 40])
    with pytest.raises(ValueError, match="40-um lattice"):
        k9.assign_half_open_states([10], [0], [40], [20])


def test_tissue_uses_base_plus_true_state_while_probe_uses_base_only():
    out, rows = inject([
        event("tissue", 100, base_shift_um=40, true_state_um=-40),
        event("probe", 200, frame="probe", role="probe_fixed_localized",
              base_shift_um=40, true_state_um=-40),
    ])
    assert [row["placement_shift_um"] for row in rows] == [0, 40]
    assert peak_channel(out, 100) == 4
    assert peak_channel(out, 200) == 6


def test_supplied_state_changes_metadata_only_for_both_frames():
    for frame, role in (("tissue", "field_error_neural"), ("probe", "probe_fixed_localized")):
        first, first_rows = inject([
            event("x", frame=frame, role=role, true_state_um=40, supplied_state_um=-40)
        ])
        second, second_rows = inject([
            event("x", frame=frame, role=role, true_state_um=40, supplied_state_um=80)
        ])
        assert np.array_equal(first, second)
        assert first_rows[0]["field_error_um"] == -80
        assert second_rows[0]["field_error_um"] == 40


def test_only_same_column_exact_lattice_targets_receive_active_support():
    geom = geometry()
    out, _ = inject([event("x", true_state_um=40)], geom=geom)
    assert peak_channel(out, 100) == 6
    # Move the required same-column target off the exact numerical lattice.
    off = geom.copy()
    off[6, 1] += 1e-3
    with pytest.raises(ValueError, match="truncate"):
        inject([event("x", true_state_um=40)], geom=off)


def test_nonzero_spatial_and_full_temporal_support_cannot_be_truncated():
    with pytest.raises(ValueError, match="truncate"):
        inject([event("x", true_state_um=40)], templates={"t1": template(channel=10)})
    # Width 9 means centers 4 and 295 exactly fit [0, 300); adjacent centers fail.
    inject([event("lo", sample=4), event("hi", sample=295)])
    for sample in (3, 296):
        with pytest.raises(ValueError, match="time boundary"):
            inject([event("bad", sample=sample)])


def test_overlap_is_linear_retains_rows_and_collision_receipt():
    events = [
        event("a", biological_unit_id="uA", role="designed_overlap"),
        event("b", biological_unit_id="uB", template_id="t2", role="designed_overlap"),
    ]
    out, rows = inject(events, templates={"t1": template(), "t2": template(scale=0.25)})
    assert np.array_equal(out[96:105], template(scale=1.25))
    assert [row["event_id"] for row in rows] == ["a", "b"]
    assert rows[0]["collision_group"] == rows[1]["collision_group"]
    assert rows[0]["sample_start"] == 96 and rows[0]["sample_stop"] == 105


def test_touching_half_open_temporal_supports_have_distinct_collision_groups():
    _, rows = inject([event("a", sample=100), event("b", sample=109)])
    assert rows[0]["sample_stop"] == rows[1]["sample_start"]
    assert rows[0]["collision_group"] != rows[1]["collision_group"]


def decoy_rows(identities_and_states):
    return [
        {"decoy_group": "d", "biological_unit_id": identity, "true_state_um": state}
        for identity, state in identities_and_states
    ]


def test_complementary_decoys_require_exactly_two_identities_and_disjoint_states():
    good = decoy_rows([("a", -40), ("a", 0), ("b", 40)])
    summary = k9.validate_complementary_decoys(good)
    assert summary["d"]["biological_unit_ids"] == ["a", "b"]
    assert summary["d"]["states_by_unit"] == {"a": {-40, 0}, "b": {40}}
    for bad in (
        decoy_rows([("a", 0)]),
        decoy_rows([("a", 0), ("b", 40), ("c", -40)]),
        decoy_rows([("a", 0), ("b", 0)]),
    ):
        with pytest.raises(ValueError):
            k9.validate_complementary_decoys(bad)


@pytest.mark.parametrize("bad", [True, 1.5, 1 + 0j, "100", object(), 2**63])
def test_local_and_session_clock_dtypes_and_ranges_fail_closed(bad):
    with pytest.raises(ValueError):
        inject([event("bad", sample=bad)])
    with pytest.raises(ValueError):
        inject([event("bad")], origin=bad)


def test_derived_session_clock_must_remain_in_int64_domain():
    with pytest.raises(ValueError, match="int64"):
        inject([event("overflow", sample=100)], origin=np.iinfo(np.int64).max)


@pytest.mark.parametrize("kind", ["complex", "string"])
def test_geometry_rejects_ambiguous_nonreal_or_nonnumeric_dtypes(kind):
    geom = geometry()
    bad = geom.astype(np.complex128) + 1j if kind == "complex" else geom.astype(str)
    with pytest.raises((TypeError, ValueError), match="geometry"):
        inject([event("bad")], geom=bad)


def test_geometry_shape_finiteness_and_uniqueness_fail_closed():
    bad_geometries = [
        geometry()[:, :1],
        np.vstack([geometry()[:-1], geometry()[0]]),
        np.where(np.arange(24).reshape(12, 2) == 0, np.nan, geometry()),
    ]
    for bad in bad_geometries:
        with pytest.raises(ValueError, match="geometry"):
            inject([event("bad")], geom=bad)


def test_background_template_and_event_validation_fail_closed():
    with pytest.raises(TypeError, match="float32"):
        inject([event("x")], background=np.zeros((300, 12), np.float64))
    with pytest.raises(TypeError, match="float32"):
        inject([event("x")], templates={"t1": template().astype(np.float64)})
    with pytest.raises(ValueError, match="temporal edges"):
        bad_template = template(); bad_template[0, 4] = 1
        inject([event("x")], templates={"t1": bad_template})
    with pytest.raises(ValueError, match="duplicate"):
        inject([event("x"), event("x", sample=200)])
    with pytest.raises(ValueError, match="unknown template"):
        inject([event("x", template_id="missing")])
    with pytest.raises(ValueError, match="frame"):
        inject([event("x", frame="field")])
    with pytest.raises(ValueError, match="role"):
        inject([event("x", role="unknown")])


def test_truth_rows_bind_clocks_frames_states_intervals_and_collisions():
    _, rows = inject([
        event("a", sample=100, base_shift_um=40, true_state_um=-40,
              supplied_state_um=0, decoy_group="d"),
        event("b", sample=103, biological_unit_id="u2", role="designed_overlap"),
    ])
    row = rows[0]
    assert row["local_sample"] == 100 and row["session_sample"] == ORIGIN + 100
    assert row["time_s_local"] == 100 / FS
    assert (row["base_shift_um"], row["true_state_um"], row["supplied_state_um"]) == (40, -40, 0)
    assert row["field_error_um"] == 40 and row["placement_shift_um"] == 0
    assert row["frame"] == "tissue" and row["sample_start"] == 96 and row["sample_stop"] == 105
    assert row["collision_group"] == rows[1]["collision_group"]


def test_all_input_arrays_are_nonmutated_and_output_changes_only_support():
    rng = np.random.default_rng(91)
    background = rng.normal(size=(300, 12)).astype(np.float32)
    geom = geometry()
    tmpl = template()
    before = tuple(value.copy() for value in (background, geom, tmpl))
    out, rows = inject([event("x")], background=background, geom=geom, templates={"t1": tmpl})
    assert all(np.array_equal(value, saved) for value, saved in zip((background, geom, tmpl), before))
    mask = np.ones(background.shape[0], dtype=bool)
    mask[rows[0]["sample_start"]:rows[0]["sample_stop"]] = False
    assert np.array_equal(out[mask], background[mask])
    # A later invalid event cannot mutate any caller-owned array either.
    with pytest.raises(ValueError):
        inject([event("ok"), event("bad", sample=1)], background=background, geom=geom,
               templates={"t1": tmpl})
    assert all(np.array_equal(value, saved) for value, saved in zip((background, geom, tmpl), before))
