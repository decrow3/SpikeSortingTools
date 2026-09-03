from testing.luke_ladder_panel import REGIME_ORDER, STRIP_ORDER, _cells, _spec


def test_cells_are_16_and_split_8_8():
    cells = _cells()
    assert len(cells) == 16
    dev = [c for c in cells if c["split"] == "development"]
    held = [c for c in cells if c["split"] == "held_out"]
    assert len(dev) == 8 and len(held) == 8


def test_split_rule_is_position_parity_deterministic():
    cells = _cells()
    for c in cells:
        expected = "development" if (c["position"] % 2 == 1) else "held_out"
        assert c["split"] == expected


def test_both_halves_span_every_regime_and_strip():
    cells = _cells()
    for split in ("development", "held_out"):
        half = [c for c in cells if c["split"] == split]
        assert {c["regime"] for c in half} == set(REGIME_ORDER)
        assert {c["strip"] for c in half} == set(STRIP_ORDER)


def test_cells_are_regime_major_strip_minor_then_a_second_quiet():
    cells = _cells()
    assert [c["regime"] for c in cells[:3]] == ["quiet", "quiet", "quiet"]
    assert [c["strip"] for c in cells[:3]] == STRIP_ORDER
    assert cells[15]["regime"] == "quiet" and cells[15]["start_s"] == 3480.0


def test_spec_carries_input_side_selection_basis_only():
    spec = _spec(_cells()[0])
    assert "regime" in spec.selection_basis and "depth strip" in spec.selection_basis
    assert spec.axes["snr"] == "unmeasured"  # filled in only after building
    assert spec.channel_count == 112
    assert spec.duration_s == 120.0


def test_snr_and_artifact_are_emergent_not_selection_inputs():
    # digest must not depend on the axes dict, so filling snr later is safe
    a = _spec(_cells()[0])
    b = _spec(_cells()[0])
    b.axes["snr"] = "high"
    assert a.digest == b.digest
