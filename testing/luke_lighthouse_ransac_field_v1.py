"""Cheap cached RANSAC prototype for a lighthouse displacement field d(depth, time)."""

from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.exceptions import UndefinedMetricWarning

ROOT = Path(__file__).resolve().parents[1]
TRACKS = ROOT / "testing/outputs/luke_lighthouse_consensus_reproduction_v1/deduplicated_16_family_tracks.csv"
CANDIDATES = ROOT / "testing/outputs/luke_waveform_only_expansion_v1/candidate_audit.csv"
OUT = ROOT / "testing/outputs/luke_lighthouse_ransac_field_v1"
START_S, STOP_S, BIN_S = 930.0, 1030.0, 5.0
MIN_FAMILIES = 6
RESIDUAL_THRESHOLD_UM = 25.0
DEPTH_CENTER_UM = 2000.0
DEPTH_SCALE_UM = 1000.0
RNG_SEED = 20260909


def fit_ransac(depth_um, displacement_um, residual_threshold_um=RESIDUAL_THRESHOLD_UM):
    depth_um = np.asarray(depth_um, float)
    displacement_um = np.asarray(displacement_um, float)
    x = ((depth_um - DEPTH_CENTER_UM) / DEPTH_SCALE_UM)[:, None]
    minimum = max(3, int(np.ceil(len(x) / 2)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        model = RANSACRegressor(
            estimator=LinearRegression(), min_samples=minimum,
            residual_threshold=residual_threshold_um, max_trials=100,
            random_state=RNG_SEED,
        ).fit(x, displacement_um)
    return (float(model.estimator_.intercept_), float(model.estimator_.coef_[0]),
            np.asarray(model.inlier_mask_, bool))


def family_depths():
    candidates = pd.read_csv(CANDIDATES)
    candidates = candidates[candidates["selected"]].set_index("unit_id")
    result = {}
    for unit_id in candidates.index:
        if int(unit_id) not in (673, 675):
            result[str(int(unit_id))] = float(candidates.loc[unit_id, "seed_centroid_um"])
    result["673/675"] = float(candidates.loc[[673, 675], "seed_centroid_um"].median())
    return result


def cross_validate(wide, depths, threshold):
    rows = []
    for bin_index, row in wide.iterrows():
        observed = row.dropna()
        if len(observed) < MIN_FAMILIES:
            continue
        for family, actual in observed.items():
            train = observed.drop(index=family)
            train_depth = np.array([depths[name] for name in train.index])
            intercept, slope, _ = fit_ransac(train_depth, train.to_numpy(), threshold)
            prediction = intercept + slope * ((depths[family] - DEPTH_CENTER_UM) / DEPTH_SCALE_UM)
            rigid_prediction = float(train.median())
            rows.append({"bin": int(bin_index), "time_s": START_S + (bin_index + 0.5) * BIN_S,
                         "held_out_family": family, "actual_um": actual,
                         "ransac_affine_prediction_um": prediction,
                         "rigid_median_prediction_um": rigid_prediction,
                         "ransac_affine_absolute_error_um": abs(prediction - actual),
                         "rigid_median_absolute_error_um": abs(rigid_prediction - actual)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wide = pd.read_csv(TRACKS).set_index("bin").sort_index()
    depths = family_depths()
    assert set(wide.columns) == set(depths)

    fits = []
    for bin_index, row in wide.iterrows():
        observed = row.dropna()
        if len(observed) < MIN_FAMILIES:
            fits.append({"bin": int(bin_index), "time_s": START_S + (bin_index + 0.5) * BIN_S,
                         "families": len(observed), "fit": False, "intercept_um": np.nan,
                         "slope_um_per_mm": np.nan, "inliers": 0})
            continue
        z = np.array([depths[name] for name in observed.index])
        intercept, slope, inliers = fit_ransac(z, observed.to_numpy())
        fits.append({"bin": int(bin_index), "time_s": START_S + (bin_index + 0.5) * BIN_S,
                     "families": len(observed), "fit": True, "intercept_um": intercept,
                     "slope_um_per_mm": slope, "inliers": int(inliers.sum()),
                     "inlier_fraction": float(inliers.mean())})
    fits = pd.DataFrame(fits).sort_values("bin").reset_index(drop=True)
    fits.to_csv(OUT / "time_bin_ransac_fits.csv", index=False)

    # Piecewise-linear interpolation in time makes the independently fitted
    # affine-in-depth coefficients continuous without adding a smoother.
    times = START_S + (np.arange(len(wide)) + 0.5) * BIN_S
    valid = fits.fit.to_numpy(bool)
    intercept = np.interp(times, fits.time_s[valid], fits.intercept_um[valid])
    slope = np.interp(times, fits.time_s[valid], fits.slope_um_per_mm[valid])
    depth_grid = np.arange(0.0, 3840.0 + 40.0, 40.0)
    field = intercept[:, None] + slope[:, None] * ((depth_grid[None, :] - DEPTH_CENTER_UM) / DEPTH_SCALE_UM)
    np.savez_compressed(OUT / "motion_field.npz", time_s=times, depth_um=depth_grid,
                        displacement_um=field, intercept_um=intercept,
                        slope_um_per_mm=slope, validated=np.array(False))
    rigid = wide.median(axis=1, skipna=True).to_numpy()
    rigid_field = np.broadcast_to(rigid[:, None], field.shape).copy()
    np.savez_compressed(OUT / "rigid_median_baseline_field.npz", time_s=times,
                        depth_um=depth_grid, displacement_um=rigid_field,
                        validated=np.array(False), baseline_only=np.array(True))

    sensitivity = []
    cv_tables = []
    for threshold in (15.0, 25.0, 40.0):
        cv = cross_validate(wide, depths, threshold)
        cv["residual_threshold_um"] = threshold
        cv_tables.append(cv)
        sensitivity.append({
            "residual_threshold_um": threshold, "predictions": len(cv),
            "ransac_affine_median_absolute_error_um": float(cv.ransac_affine_absolute_error_um.median()),
            "rigid_median_median_absolute_error_um": float(cv.rigid_median_absolute_error_um.median()),
            "ransac_affine_mean_absolute_error_um": float(cv.ransac_affine_absolute_error_um.mean()),
            "rigid_median_mean_absolute_error_um": float(cv.rigid_median_absolute_error_um.mean()),
            "fraction_ransac_better": float((cv.ransac_affine_absolute_error_um < cv.rigid_median_absolute_error_um).mean()),
        })
    cv_all = pd.concat(cv_tables, ignore_index=True)
    cv_all.to_csv(OUT / "leave_one_family_out_predictions.csv", index=False)
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(OUT / "ransac_threshold_sensitivity.csv", index=False)

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 8), height_ratios=[2, 0.8], layout="constrained")
    limit = float(np.nanquantile(np.abs(field), 0.98))
    im = ax.imshow(field.T, origin="lower", aspect="auto",
                   extent=[times[0], times[-1], depth_grid[0], depth_grid[-1]],
                   cmap="RdBu_r", vmin=-limit, vmax=limit)
    for family in wide.columns:
        values = wide[family]
        take = values.notna()
        ax.scatter(times[take], np.full(take.sum(), depths[family]), c=values[take],
                   cmap="RdBu_r", vmin=-limit, vmax=limit, s=18, edgecolors="black", linewidths=0.25)
    ax.set(title="Affine-in-depth RANSAC lighthouse field", ylabel="reference depth on probe (um)")
    fig.colorbar(im, ax=ax, label="displacement (um)")
    bx.plot(times, intercept, "o-", color="#2166AC", label="displacement at 2000 um")
    bx2 = bx.twinx()
    bx2.plot(times, slope, "s--", color="#D07A1F", label="depth slope")
    bx.set(xlabel="recording time (s)", ylabel="intercept (um)")
    bx2.set(ylabel="slope (um displacement / mm depth)")
    lines = bx.lines + bx2.lines
    bx.legend(lines, [line.get_label() for line in lines], frameon=False, ncol=2)
    fig.savefig(OUT / "01_ransac_motion_field.png", dpi=180)
    fig.savefig(OUT / "01_ransac_motion_field.pdf")
    plt.close(fig)

    chosen = sensitivity[sensitivity.residual_threshold_um.eq(RESIDUAL_THRESHOLD_UM)].iloc[0]
    summary = {
        "status": "complete_unvalidated_prototype", "model": "per-5s affine-in-depth RANSAC; coefficients linearly interpolated in time",
        "minimum_families_per_fitted_bin": MIN_FAMILIES,
        "fitted_time_bins": int(valid.sum()), "interpolated_time_bins": int((~valid).sum()),
        "residual_threshold_um": RESIDUAL_THRESHOLD_UM,
        "median_inlier_fraction": float(fits.loc[valid, "inlier_fraction"].median()),
        "field_min_um": float(field.min()), "field_max_um": float(field.max()),
        "slope_min_um_per_mm": float(slope.min()), "slope_max_um_per_mm": float(slope.max()),
        "leave_one_family_out": chosen.to_dict(),
        "recommended_field": "none; retain rigid_median_baseline_field.npz as a descriptive control only",
        "interpretation": ("Depth-dependent RANSAC is supported only if it improves held-out error over the rigid median; "
                           "otherwise the 2-D field is a visualization, not a validated estimator."),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
