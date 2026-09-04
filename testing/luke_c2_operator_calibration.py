"""C2 v4 pre-freeze calibration of the injected-motion interpolation operator.

Why this exists
---------------
C2's moving arm is produced by warping the injected voltage through
`InterpolateMotionRecording` (`ladder_motion.warp_array_with_known_motion`);
the static arm is never warped. `moving - static` therefore mixes three things:

1. the sorter's response to **temporal** motion — the quantity C2 wants;
2. the ordinary change in a footprint sampled from a different electrode
   position — physical, and also part of what C2 wants;
3. **numerical resampling error contributed by the operator itself** — an
   artifact of the simulation, which must be bounded before v4 is frozen.

This module measures (3) on the actual frozen cohort, so the v4 prespec can
freeze the operator parameters and state donor-wise acceptance tolerances
instead of inheriting library defaults silently.

Why a fixed-offset sweep is the right probe
-------------------------------------------
Luke's measured rigid speed is 0.2-0.8 um/s (decision 0013). Across one 61-sample
(2.03 ms) template that is < 0.002 um of displacement, so **every spike in the
moving arm is effectively a constant-offset resampling**; what the trajectory
varies is *which* offset each spike gets. A dense constant-offset sweep therefore
characterises the per-spike cost exactly, and the moving arm's expected cost
follows by averaging the sweep over the offsets the trajectory visits
(uniform on [0, D] for a linear ramp of excursion D). A single constant-shift
control samples only one point of that distribution; it is a sensitivity arm,
not a cancellation of the interpolation cost.

What is measured, per donor and offset
--------------------------------------
Metric semantics matter here and are easy to get wrong. After a forward warp the
footprint has *moved*, so any element-wise comparison against the unmoved
original mixes displacement with degradation. Only translation-invariant
measures, and measures taken after the footprint is returned to its original
position, are interpretable as operator error:

* `peak_retention`    - max|warped| / max|original|. Translation-invariant, so
  this **is** a valid forward fidelity measure, and it is the one that maps onto
  detection: a retained peak of 0.5 halves the donor's effective SNR;
* `roundtrip_peak_retention`, `roundtrip_rel_rms` - forward warp then the
  **exact** inverse, compared at the original position. Valid error measures,
  and the **exact-registration round-trip reference**. This is *not* a
  performance ceiling: the exact inverse minimises positional error, not
  amplitude or sorting accuracy, and a partial or estimated correction that
  registers less displacement can retain more amplitude and could sort better;
* `exact_inverse_amplitude_cost` - `roundtrip_peak_retention - peak_retention`.
  Negative means the exact inverse leaves a *smaller* waveform than leaving the
  motion uncorrected, because it adds a second interpolation;
* `snr_after`         - retained peak over the measured background noise of the
  real quiet imec1 strip, i.e. how far attenuation pushes a donor toward the
  detection floor;
* `cosine`, `rel_rms` (forward) - retained as *displacement* diagnostics only.
  They fall as the footprint translates even when the operator is exact, so they
  must not be read as error. `lattice_exactness` is the operator's error check.

Operator validation
-------------------
The target strip has a 20 µm row pitch with a two-column x-stagger, so a shift of
40 µm (two rows) maps every site exactly onto another site. At those
lattice-commensurate offsets the correct answer is known exactly - a channel roll
- and the operator reproduces it to ~1e-6 relative error. That validates the
geometry and sign implementation. It does **not** establish that the operator's
fractional-offset attenuation reproduces the unknown continuous biological
voltage field; the loss at non-commensurate offsets is best described as
**phase-dependent attenuation of the discrete injection/interpolation model**,
which may or may not correspond to what a real neuron displaced by that amount
would produce.

Reads `/mnt` (geometry + background noise only); writes only under
`testing/outputs/`. No sorting, no GPU: this is minutes, not hours.

Run: `python testing/luke_c2_operator_calibration.py [--quick]`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from testing.ladder_inject import channels_per_um, inject_trajectory, static_trajectory
from testing.ladder_motion import signed_um_per_channel, warp_array_with_known_motion
from testing.luke_rescue_c2_drift_challenge import (
    DONOR_GEOMETRY,
    DONOR_MANIFEST,
    DONOR_TEMPLATES,
    PRESPEC,
    _sha256,
    _verify_donor_cohort,
    donor_base_channel,
    load_background,
    prepare_template,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "testing/outputs/luke_c2_operator_calibration"

# The operator C2 v3 used implicitly, by library default. v4 must name it.
REFERENCE_OPERATOR = {
    "spatial_interpolation_method": "kriging",
    "sigma_um": 20.0,
    "border_mode": "force_extrapolate",
    "bin_s": 0.5,
}

CALIB = {
    "schema": "luke-c2-operator-calibration-v1",
    "purpose": (
        "bound the interpolation artifact in the C2 v4 moving arm, and freeze "
        "the operator parameters and donor-wise acceptance tolerances"
    ),
    "donor_cohort": {
        "source": PRESPEC["template_prep"]["source"],
        "templates_sha256": PRESPEC["template_prep"]["templates_sha256"],
        "manifest_sha256": PRESPEC["template_prep"]["manifest_sha256"],
        "source_geometry_sha256": PRESPEC["template_prep"]["source_geometry_sha256"],
    },
    "target": {"probe": PRESPEC["probe"], "background": PRESPEC["background"]},
    # dense enough to resolve the sub-row structure of a 20 um row pitch
    "offset_grid_um": (
        [round(0.5 * i, 1) for i in range(0, 51)] + [40.0, 80.0]
    ),
    # two row pitches: every site maps exactly onto another site, so the correct
    # warped field is a pure channel roll and the operator can be validated
    "lattice_commensurate_um": {"40.0": 4, "80.0": 8},
    "nominal_magnitudes_um": [5.0, 11.0, 22.0],
    "reference_operator": REFERENCE_OPERATOR,
    "operator_grid": [
        {"spatial_interpolation_method": "kriging", "sigma_um": 10.0},
        {"spatial_interpolation_method": "kriging", "sigma_um": 20.0},
        {"spatial_interpolation_method": "kriging", "sigma_um": 40.0},
        {"spatial_interpolation_method": "idw", "sigma_um": 20.0},
        {"spatial_interpolation_method": "kriging", "sigma_um": 20.0,
         "border_mode": "force_zeros"},
    ],
    "window_samples": 601,  # one spike, centred; >> the 61-sample template
    "metrics": [
        "peak_retention", "cosine", "rel_rms", "roundtrip_rel_rms", "snr_after",
    ],
}


# --------------------------------------------------------------------------- #
# geometry-consistent µm trajectories
# --------------------------------------------------------------------------- #
def um_trajectory(offset_um: float, geometry: np.ndarray):
    """A constant-offset trajectory, specified in µm, in the operator's units.

    `warp_array_with_known_motion` takes a channel-index trajectory and
    multiplies by `signed_um_per_channel`. On a monotone strip that factor is
    the exact inverse of `channels_per_um`, so the µm value round-trips exactly
    -- but the double conversion is gratuitous and v4 should specify µm
    directly. `assert_um_channel_inverse` pins the property this relies on.
    """
    return lambda t: np.full_like(
        np.asarray(t, dtype=float), offset_um * channels_per_um(geometry)
    )


def assert_um_channel_inverse(geometry: np.ndarray, tol: float = 1e-9) -> float:
    """`channels_per_um` and `signed_um_per_channel` must be exact inverses."""
    product = channels_per_um(geometry) * signed_um_per_channel(geometry)
    if not abs(abs(product) - 1.0) <= tol:
        raise RuntimeError(
            "µm<->channel conversion is not invertible on this geometry "
            f"(product {product!r}); specify the trajectory in µm directly"
        )
    return float(product)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def lattice_exactness(field: np.ndarray, warped: np.ndarray, channel_shift: int) -> float:
    """Relative error against an exact channel roll (only valid when commensurate)."""
    ideal = np.roll(np.asarray(field, dtype=np.float64), channel_shift, axis=1)
    denominator = float(np.linalg.norm(field))
    return float(np.linalg.norm(warped - ideal) / denominator) if denominator else float("nan")


def fidelity(original: np.ndarray, warped: np.ndarray) -> dict:
    o = np.asarray(original, dtype=np.float64).ravel()
    w = np.asarray(warped, dtype=np.float64).ravel()
    o_norm = float(np.linalg.norm(o))
    peak_o = float(np.abs(o).max())
    return {
        "peak_retention": float(np.abs(w).max() / peak_o) if peak_o else float("nan"),
        "peak_uv_after": float(np.abs(w).max()),
        "cosine": float(o @ w / (o_norm * np.linalg.norm(w))) if o_norm else float("nan"),
        "rel_rms": float(np.linalg.norm(w - o) / o_norm) if o_norm else float("nan"),
    }


# --------------------------------------------------------------------------- #
# per-donor noise-free field on the real target geometry
# --------------------------------------------------------------------------- #
def donor_field(tid: str, donors, donor_meta, source_geometry, geometry, fs: float):
    """Inject one donor spike into a zero background, exactly as C2 places it.

    Noise-free so the operator's contribution is attributable; the placement
    reuses C2's own `prepare_template` / `donor_base_channel`, so this measures
    the operator on the footprints the experiment will actually warp.
    """
    template, peak_col = prepare_template(donors[tid])
    base_channel, target_peak = donor_base_channel(
        template, peak_col, int(donor_meta[tid]["peak_channel"]),
        source_geometry, geometry,
    )
    n_samples = int(CALIB["window_samples"])
    background = np.zeros((n_samples, geometry.shape[0]), dtype=np.float32)
    train = np.array([n_samples // 2], dtype=np.int64)
    field = inject_trajectory(
        background, template, train, fs=fs, base_channel=base_channel,
        trajectory=static_trajectory(), amplitude_scale=PRESPEC["amplitude_scale"],
        template_id=tid, edge_guard_samples=PRESPEC["template_prep"]["edge_guard_samples"],
    )
    return field, base_channel, target_peak


def sweep_donor(field, geometry, fs, offsets_um, operator, noise_uv, roundtrip=True):
    """Constant-offset fidelity curve for one donor field."""
    rows = []
    op = {**REFERENCE_OPERATOR, **operator}
    bin_s = op.pop("bin_s")
    for offset in offsets_um:
        traj = um_trajectory(offset, geometry)
        warped = warp_array_with_known_motion(
            field, geometry, fs=fs, trajectory_fn=traj, sign=-1.0, bin_s=bin_s, **op
        )
        row = {"offset_um": float(offset), **fidelity(field, warped)}
        row["snr_after"] = row["peak_uv_after"] / noise_uv
        if roundtrip:
            restored = warp_array_with_known_motion(
                warped, geometry, fs=fs, trajectory_fn=traj, sign=+1.0,
                bin_s=bin_s, **op
            )
            rt = fidelity(field, restored)
            row["roundtrip_rel_rms"] = rt["rel_rms"]
            row["roundtrip_peak_retention"] = rt["peak_retention"]
            row["exact_inverse_amplitude_cost"] = (
                rt["peak_retention"] - row["peak_retention"]
            )
        shift = CALIB["lattice_commensurate_um"].get(str(float(offset)))
        if shift is not None:
            row["lattice_exactness"] = lattice_exactness(field, warped, shift)
        rows.append(row)
    return rows


def ramp_expectation(curve: list[dict], magnitude_um: float) -> dict:
    """What a linear ramp of `magnitude_um` costs, averaged over its offsets.

    A ramp visits offsets uniformly on [0, D]; per-spike fidelity is the
    constant-offset curve, so the moving arm's expected cost is the mean of the
    curve over that interval, and its *worst* spike is the minimum.
    """
    inside = [r for r in curve if r["offset_um"] <= magnitude_um + 1e-9]
    if len(inside) < 2:
        return {}
    def agg(key, fn):
        return float(fn([r[key] for r in inside if not np.isnan(r[key])]))
    return {
        "magnitude_um": float(magnitude_um),
        "n_offsets": len(inside),
        "mean_peak_retention": agg("peak_retention", np.mean),
        "min_peak_retention": agg("peak_retention", np.min),
        "mean_roundtrip_peak_retention": agg("roundtrip_peak_retention", np.mean),
        "min_roundtrip_peak_retention": agg("roundtrip_peak_retention", np.min),
        "mean_exact_inverse_amplitude_cost": agg("exact_inverse_amplitude_cost", np.mean),
        "mean_roundtrip_rel_rms": agg("roundtrip_rel_rms", np.mean),
        "max_roundtrip_rel_rms": agg("roundtrip_rel_rms", np.max),
        "min_snr_after": agg("snr_after", np.min),
        "mean_snr_after": agg("snr_after", np.mean),
        "displacement_only_mean_cosine": agg("cosine", np.mean),
    }


# --------------------------------------------------------------------------- #
def run(quick: bool = False) -> dict:
    if str(OUTPUT).startswith("/mnt/"):
        raise ValueError("refusing to write the calibration under /mnt")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _verify_donor_cohort()

    bg_uv, geometry, fs, _gain, _start = load_background()
    noise_per_channel = (
        np.median(np.abs(bg_uv - np.median(bg_uv, axis=0)), axis=0) / 0.6745
    )
    noise_uv = float(np.median(noise_per_channel))
    del bg_uv  # only the geometry and the noise level are needed downstream
    inverse_product = assert_um_channel_inverse(geometry)

    import pandas as pd

    donors = np.load(DONOR_TEMPLATES)
    donor_meta = pd.read_csv(DONOR_MANIFEST).set_index("template_id").to_dict("index")
    source_geometry = np.load(DONOR_GEOMETRY)
    tids = sorted(donors.files)
    offsets = CALIB["offset_grid_um"]
    if quick:
        tids = tids[:2]
        offsets = [o for o in offsets if o in (0.0, 5.0, 11.0, 22.0)]

    sweep_rows, expect_rows, grid_rows = [], [], []
    for tid in tids:
        field, base_channel, target_peak = donor_field(
            tid, donors, donor_meta, source_geometry, geometry, fs
        )
        meta = donor_meta[tid]
        curve = sweep_donor(field, geometry, fs, offsets, {}, noise_uv)
        for row in curve:
            sweep_rows.append({
                "template": tid, "peak_uv": meta["peak_uv"],
                "polarity": meta["polarity"], "amplitude_band": meta["amplitude_band"],
                "snr_static": float(meta["peak_uv"]) / noise_uv,
                "base_channel": base_channel, "peak_channel": target_peak, **row,
            })
        for magnitude in CALIB["nominal_magnitudes_um"]:
            expectation = ramp_expectation(curve, magnitude)
            if expectation:
                expect_rows.append({
                    "template": tid, "peak_uv": meta["peak_uv"],
                    "polarity": meta["polarity"],
                    "snr_static": float(meta["peak_uv"]) / noise_uv, **expectation,
                })
        # operator grid, at the nominal magnitudes only
        for operator in CALIB["operator_grid"]:
            for row in sweep_donor(
                field, geometry, fs, CALIB["nominal_magnitudes_um"], operator, noise_uv
            ):
                grid_rows.append({
                    "template": tid, "peak_uv": meta["peak_uv"],
                    "operator": json.dumps(operator, sort_keys=True), **row,
                })

    sweep = pd.DataFrame(sweep_rows)
    expect = pd.DataFrame(expect_rows)
    grid = pd.DataFrame(grid_rows)
    sweep.to_csv(OUTPUT / "offset_sweep.csv", index=False)
    expect.to_csv(OUTPUT / "ramp_expectation.csv", index=False)
    grid.to_csv(OUTPUT / "operator_grid.csv", index=False)

    by_magnitude = {
        str(magnitude): {
            "mean_peak_retention": round(float(sub.mean_peak_retention.mean()), 4),
            "worst_donor_min_peak_retention": round(float(sub.min_peak_retention.min()), 4),
            "worst_donor": str(sub.loc[sub.min_peak_retention.idxmin(), "template"]),
            "mean_roundtrip_peak_retention": round(
                float(sub.mean_roundtrip_peak_retention.mean()), 4),
            "mean_exact_inverse_amplitude_cost": round(
                float(sub.mean_exact_inverse_amplitude_cost.mean()), 4),
            "mean_roundtrip_rel_rms": round(float(sub.mean_roundtrip_rel_rms.mean()), 4),
            "worst_min_snr_after": round(float(sub.min_snr_after.min()), 3),
            "n_below_snr3_static": int((sub.snr_static < 3.0).sum()),
            "n_below_snr3_ramp_min": int((sub.min_snr_after < 3.0).sum()),
            "n_below_snr3_ramp_mean": int((sub.mean_snr_after < 3.0).sum()),
            "cross_above_to_below_snr3_ramp_min": sorted(
                sub.loc[(sub.snr_static >= 3.0) & (sub.min_snr_after < 3.0), "template"]
            ),
            "cross_above_to_below_snr3_ramp_mean": sorted(
                sub.loc[(sub.snr_static >= 3.0) & (sub.mean_snr_after < 3.0), "template"]
            ),
        }
        for magnitude, sub in expect.groupby("magnitude_um")
    } if not expect.empty else {}

    operator_ranking = (
        grid.groupby("operator")[["rel_rms", "roundtrip_rel_rms", "peak_retention"]]
        .mean().round(4).sort_values("roundtrip_rel_rms").reset_index()
        .to_dict("records")
    ) if not grid.empty else []

    lattice = sweep[sweep.get("lattice_exactness").notna()] if "lattice_exactness" in sweep else None
    summary = {
        "calibration": CALIB,
        "operator_validation": {
            "lattice_commensurate_max_rel_err": (
                round(float(lattice.lattice_exactness.max()), 9)
                if lattice is not None and not lattice.empty else None
            ),
            "lattice_commensurate_min_peak_retention": (
                round(float(lattice.peak_retention.min()), 6)
                if lattice is not None and not lattice.empty else None
            ),
            "interpretation": (
                "at 40/80 µm every site maps onto another site, so an exact "
                "operator reproduces a pure channel roll; a max rel err ~1e-6 "
                "validates the geometry and sign implementation. It does not "
                "establish that fractional-offset attenuation reproduces the "
                "true continuous voltage field."
            ),
        },
        "background_noise_uv_mad": round(noise_uv, 3),
        "um_channel_inverse_product": round(inverse_product, 12),
        "n_donors": len(tids),
        "quick": bool(quick),
        "static_snr_by_donor": {
            tid: round(float(donor_meta[tid]["peak_uv"]) / noise_uv, 2) for tid in tids
        },
        "ramp_expectation_by_magnitude": by_magnitude,
        "operator_ranking_by_roundtrip_error": operator_ranking,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--quick", action="store_true", help="2 donors, 4 offsets")
    args = ap.parse_args()
    summary = run(quick=args.quick)
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
