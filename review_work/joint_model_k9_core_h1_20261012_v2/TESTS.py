import numpy as np
import pytest

from testing.joint_model_k9_benchmark import (
    K9Event,
    assign_half_open_states,
    inject_k9_precar,
    validate_complementary_decoys,
)


FS = 30_000.0
SESSION_START = 900_000


def geometry():
    return np.asarray([(x, y) for y in range(0, 240, 40) for x in (0, 32)], dtype=float)


def template(channel=4, scale=1.0):
    value = np.zeros((9, 12), dtype=np.float32)
    value[2:7, channel] = scale * np.asarray([-10, -30, -80, -30, -10], np.float32)
    value[2:7, channel + 1] = value[2:7, channel] * np.float32(0.4)
    return value


def event(event_id, sample, **changes):
    values = dict(
        event_id=event_id,
        biological_unit_id="u1",
        template_id="t1",
        sample_index=sample,
        frame="tissue",
        role="moving_neural",
    )
    values.update(changes)
    return K9Event(**values)


def peak_channel(delta, sample):
    return int(np.argmax(np.max(np.abs(delta[sample - 4:sample + 5]), axis=0)))


def test_half_open_state_ownership_uses_new_state_at_boundary():
    got = assign_half_open_states(
        np.asarray([0, 99, 100, 199, 200, 299], np.int64),
        np.asarray([0, 100, 200], np.int64),
        np.asarray([100, 200, 300], np.int64),
        np.asarray([0, 40, -40], np.int64),
    )
    assert got.tolist() == [0, 0, 40, 40, -40, -40]


def test_tissue_moves_on_lattice_while_probe_stays_fixed_and_clock_is_bound():
    background = np.zeros((400, 12), dtype=np.float32)
    out, rows = inject_k9_precar(
        background,
        geometry(),
        {"t1": template()},
        [
            event("tissue", 100, true_state_um=40),
            event("probe", 200, frame="probe", role="probe_fixed_localized", true_state_um=40),
        ],
        sampling_frequency_hz=FS,
        session_start_frame=SESSION_START,
    )
    delta = out - background
    assert peak_channel(delta, 100) == 6
    assert peak_channel(delta, 200) == 4
    assert rows[0]["placement_shift_um"] == 40
    assert rows[1]["placement_shift_um"] == 0
    assert [row["session_sample"] for row in rows] == [900_100, 900_200]
    assert np.array_equal(background, np.zeros_like(background))


def test_supplied_field_error_changes_truth_not_voltage():
    background = np.zeros((300, 12), dtype=np.float32)
    common = dict(
        background_uv=background,
        geometry_um=geometry(),
        templates={"t1": template()},
        sampling_frequency_hz=FS,
        session_start_frame=SESSION_START,
    )
    first, rows_first = inject_k9_precar(
        events=[event("a", 100, true_state_um=40, supplied_state_um=40)], **common
    )
    second, rows_second = inject_k9_precar(
        events=[event("a", 100, true_state_um=40, supplied_state_um=80)], **common
    )
    assert np.array_equal(first, second)
    assert rows_first[0]["field_error_um"] == 0
    assert rows_second[0]["field_error_um"] == 40


def test_overlap_adds_linearly_and_retains_both_memberships():
    background = np.zeros((300, 12), dtype=np.float32)
    events = [
        event("a", 100, biological_unit_id="u1", role="designed_overlap"),
        event("b", 100, biological_unit_id="u2", template_id="t2", role="designed_overlap"),
    ]
    out, rows = inject_k9_precar(
        background,
        geometry(),
        {"t1": template(scale=1.0), "t2": template(scale=0.5)},
        events,
        sampling_frequency_hz=FS,
        session_start_frame=SESSION_START,
    )
    expected = template(scale=1.5)
    assert np.array_equal(out[96:105], expected)
    assert len(rows) == 2
    assert rows[0]["collision_group"] == rows[1]["collision_group"]


def test_complementary_decoys_preserve_two_identities_and_reject_shared_state():
    background = np.zeros((400, 12), dtype=np.float32)
    events = [
        event("a0", 100, biological_unit_id="uA", role="decoy_member", decoy_group="d1", true_state_um=0),
        event("a1", 120, biological_unit_id="uA", role="decoy_member", decoy_group="d1", true_state_um=0),
        event("b0", 200, biological_unit_id="uB", role="decoy_member", decoy_group="d1", true_state_um=40),
    ]
    _, rows = inject_k9_precar(
        background, geometry(), {"t1": template()}, events,
        sampling_frequency_hz=FS, session_start_frame=SESSION_START,
    )
    summary = validate_complementary_decoys(rows)
    assert summary["d1"]["biological_unit_ids"] == ["uA", "uB"]
    rows[-1]["true_state_um"] = 0
    with pytest.raises(ValueError, match="not complementary"):
        validate_complementary_decoys(rows)


def test_background_is_byte_identical_outside_event_support():
    rng = np.random.default_rng(3)
    background = rng.normal(size=(300, 12)).astype(np.float32)
    original = background.copy()
    out, rows = inject_k9_precar(
        background, geometry(), {"t1": template()}, [event("a", 100)],
        sampling_frequency_hz=FS, session_start_frame=SESSION_START,
    )
    mask = np.ones(len(background), dtype=bool)
    mask[rows[0]["sample_start"]:rows[0]["sample_stop"]] = False
    assert np.array_equal(out[mask], original[mask])
    assert np.array_equal(background, original)


@pytest.mark.parametrize("bad", [True, 1.5, 1 + 0j, "100", object()])
def test_event_clock_rejects_noninteger_or_ambiguous_values(bad):
    with pytest.raises(ValueError, match="sample_index"):
        inject_k9_precar(
            np.zeros((300, 12), np.float32), geometry(), {"t1": template()},
            [event("bad", bad)], sampling_frequency_hz=FS, session_start_frame=SESSION_START,
        )


@pytest.mark.parametrize("field,value", [("true_state_um", 20), ("supplied_state_um", 60), ("base_shift_um", 1)])
def test_non_lattice_shifts_fail_closed(field, value):
    with pytest.raises(ValueError, match="40-um lattice"):
        inject_k9_precar(
            np.zeros((300, 12), np.float32), geometry(), {"t1": template()},
            [event("bad", 100, **{field: value})],
            sampling_frequency_hz=FS, session_start_frame=SESSION_START,
        )


def test_duplicate_ids_unknown_templates_and_boundaries_fail_closed():
    common = dict(
        background_uv=np.zeros((300, 12), np.float32), geometry_um=geometry(),
        templates={"t1": template()}, sampling_frequency_hz=FS,
        session_start_frame=SESSION_START,
    )
    with pytest.raises(ValueError, match="duplicate"):
        inject_k9_precar(events=[event("x", 100), event("x", 200)], **common)
    with pytest.raises(ValueError, match="unknown template"):
        inject_k9_precar(events=[event("x", 100, template_id="missing")], **common)
    with pytest.raises(ValueError, match="time boundary"):
        inject_k9_precar(events=[event("x", 1)], **common)


def test_spatial_support_truncation_and_wrong_dtypes_fail_closed():
    edge_template = template(channel=10)
    with pytest.raises(ValueError, match="truncate"):
        inject_k9_precar(
            np.zeros((300, 12), np.float32), geometry(), {"edge": edge_template},
            [event("x", 100, template_id="edge", true_state_um=40)],
            sampling_frequency_hz=FS, session_start_frame=SESSION_START,
        )
    with pytest.raises(TypeError, match="float32"):
        inject_k9_precar(
            np.zeros((300, 12), np.float64), geometry(), {"t1": template()},
            [event("x", 100)], sampling_frequency_hz=FS, session_start_frame=SESSION_START,
        )
    with pytest.raises(TypeError, match="templates"):
        inject_k9_precar(
            np.zeros((300, 12), np.float32), geometry(), {"t1": template().astype(np.float64)},
            [event("x", 100)], sampling_frequency_hz=FS, session_start_frame=SESSION_START,
        )


@pytest.mark.parametrize(
    "bad_geometry",
    [geometry().astype(np.complex128) + 1j, geometry().astype(str)],
)
def test_complex_and_string_geometry_fail_closed(bad_geometry):
    with pytest.raises(ValueError, match="real numeric"):
        inject_k9_precar(
            np.zeros((300, 12), np.float32), bad_geometry, {"t1": template()},
            [event("x", 100)], sampling_frequency_hz=FS,
            session_start_frame=SESSION_START,
        )


def test_derived_session_clock_overflow_fails_closed():
    with pytest.raises(ValueError, match="session_sample.*int64"):
        inject_k9_precar(
            np.zeros((300, 12), np.float32), geometry(), {"t1": template()},
            [event("x", 100)], sampling_frequency_hz=FS,
            session_start_frame=np.iinfo(np.int64).max,
        )
