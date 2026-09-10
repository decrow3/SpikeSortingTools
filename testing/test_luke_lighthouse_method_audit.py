"""Numerical checks for the lighthouse method audit.

Verifies the pieces the audit's conclusions depend on: the depth-shift
operator's exactness and sign, the frozen coincidence rule, and that the
frozen scoring rule recovers a template's own identity at zero offset.
"""

import numpy as np
import pytest

from testing.luke_lighthouse_method_audit_v1 import (
    OFF,
    PATCH_STEP,
    load_geometry,
    near,
    score_against_bank,
    shift_template,
)


@pytest.fixture(scope="module")
def geometry():
    return load_geometry()


def synthetic_template(geo, centre_um=2000.0, width_um=45.0, samples=61):
    """Gaussian depth profile times a biphasic temporal shape."""
    t = np.arange(samples) - samples // 2
    temporal = -np.exp(-(t ** 2) / 8.0) + 0.45 * np.exp(-((t - 6) ** 2) / 18.0)
    spatial = np.exp(-((geo[:, 1] - centre_um) ** 2) / (2 * width_um ** 2))
    return (temporal[:, None] * spatial[None, :]).astype(float)


def energy_centroid(wave, geo, peak_t):
    energy = (wave[peak_t + OFF] ** 2).sum(axis=0)
    return float(energy @ geo[:, 1] / energy.sum())


def test_zero_shift_is_identity(geometry):
    geo = geometry[0]
    w = synthetic_template(geo)
    assert np.allclose(shift_template(w, geo, 0.0), w, atol=1e-12)


def test_lattice_shift_is_an_exact_channel_remap(geometry):
    """A 40 um shift moves each column by exactly one channel, no interpolation."""
    geo = geometry[0]
    w = synthetic_template(geo)
    shifted = shift_template(w, geo, float(PATCH_STEP))
    for x in np.unique(geo[:, 0]):
        idx = np.flatnonzero(geo[:, 0] == x)
        idx = idx[np.argsort(geo[idx, 1])]
        # value at row k after the shift is the original value at row k-1
        assert np.allclose(shifted[:, idx[1:]], w[:, idx[:-1]], atol=1e-10)
        assert np.allclose(shifted[:, idx[0]], 0.0, atol=1e-10)


def test_shift_sign_moves_centroid_upward(geometry):
    """T_new(y) = T_old(y - delta) must raise the energy centroid by delta."""
    geo = geometry[0]
    w = synthetic_template(geo)
    peak_t = int(np.unravel_index(np.abs(w).argmax(), w.shape)[0])
    base = energy_centroid(w, geo, peak_t)
    for delta in (20.0, 40.0, 80.0):
        moved = energy_centroid(shift_template(w, geo, delta), geo, peak_t)
        assert moved - base == pytest.approx(delta, abs=1.5), (delta, moved - base)


def test_interpolated_shift_stays_between_its_lattice_neighbours(geometry):
    geo = geometry[0]
    w = synthetic_template(geo)
    peak_t = int(np.unravel_index(np.abs(w).argmax(), w.shape)[0])
    lo = energy_centroid(shift_template(w, geo, 0.0), geo, peak_t)
    mid = energy_centroid(shift_template(w, geo, 20.0), geo, peak_t)
    hi = energy_centroid(shift_template(w, geo, 40.0), geo, peak_t)
    assert lo < mid < hi


def test_near_matches_the_frozen_tolerance():
    reference = np.array([1.0, 2.0, 3.0])
    probe = np.array([1.0, 2.0 + 2e-4, 2.0 + 5e-4, 10.0])
    assert near(probe, reference).tolist() == [True, True, False, False]
    assert not near(probe, np.array([])).any()


def test_scoring_recovers_self_identity_at_zero_offset(geometry):
    """A template placed on its own lattice node must win against the bank."""
    geo, bases, patches, _ = geometry
    w = synthetic_template(geo, centre_um=float(bases[20]))
    peak_t = int(np.unravel_index(np.abs(w).argmax(), w.shape)[0])
    # Bank of two identities: the template itself and a temporally reversed decoy.
    home = patches[20]
    self_wave = w[peak_t + OFF][:, home]
    decoy = self_wave[::-1]
    flat = np.stack([self_wave.ravel(), decoy.ravel()])
    norms = np.linalg.norm(flat, axis=1)
    best, gains = score_against_bank(w, flat / norms[:, None], norms, peak_t, patches)
    patch = int(np.argmax(best.max(axis=1)))
    assert patch == 20, patch
    assert best[patch].argmax() == 0
    assert best[patch, 0] == pytest.approx(1.0, abs=1e-6)
    assert gains[patch, 0] == pytest.approx(1.0, rel=1e-6)


def test_scoring_follows_the_template_to_a_shifted_patch(geometry):
    """Shifting by two lattice steps must move the winning patch by two."""
    geo, bases, patches, _ = geometry
    w = synthetic_template(geo, centre_um=float(bases[20]))
    peak_t = int(np.unravel_index(np.abs(w).argmax(), w.shape)[0])
    home = patches[20]
    flat = w[peak_t + OFF][:, home].ravel()[None, :]
    norms = np.linalg.norm(flat, axis=1)
    moved = shift_template(w, geo, 2.0 * PATCH_STEP)
    best, _ = score_against_bank(moved, flat / norms[:, None], norms, peak_t, patches)
    patch = int(np.argmax(best.max(axis=1)))
    assert patch == 22, patch
    assert best[patch, 0] == pytest.approx(1.0, abs=1e-6)


def test_cubic_kernel_is_exact_on_lattice_nodes(geometry):
    """The interpolation control must not disturb the exact 0 and 40 um anchors."""
    geo = geometry[0]
    w = synthetic_template(geo)
    assert np.allclose(shift_template(w, geo, 0.0, "cubic"), w, atol=1e-9)
    linear = shift_template(w, geo, 40.0, "linear")
    cubic = shift_template(w, geo, 40.0, "cubic")
    assert np.allclose(linear, cubic, atol=1e-9)


def test_cubic_and_linear_differ_only_mildly_between_nodes(geometry):
    """Half-lattice offsets are interpolated, so kernels may differ - but not much."""
    geo = geometry[0]
    w = synthetic_template(geo)
    linear = shift_template(w, geo, 20.0, "linear")
    cubic = shift_template(w, geo, 20.0, "cubic")
    scale = np.abs(w).max()
    assert 0 < np.abs(linear - cubic).max() < 0.15 * scale
