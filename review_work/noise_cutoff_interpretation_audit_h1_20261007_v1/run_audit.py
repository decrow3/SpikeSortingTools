from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/post_sort_structure_census_h5_20261007_v1/CENSUS_UNITS.csv")
EXPECTED_SHA256 = "7d74b41e37ac4d5606decbc7caf5efd0ac8cf5ad3d9cda86ac6bfff2c38cccc7"
COUNT_ORDER = ["0_to_99", "100_to_999", "1000_to_9999", "ge_10000"]
Q_ORDER = ["displaced_only_q0_fraction_lt_0.10", "q0_dominant_displaced_fraction_lt_0.10", "mixed"]
DEPTH_ORDER = ["inside_lt_100_um", "inside_100_to_lt_200_um", "inside_ge_200_um"]


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def scalar(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def status(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    return pd.Series(np.where(~np.isfinite(x), "undefined", np.where(x < 5, "finite_pass", "finite_fail")), index=values.index)


def count_stratum(n: pd.Series) -> pd.Series:
    return pd.cut(n, [-1, 99, 999, 9999, np.inf], labels=COUNT_ORDER).astype(str)


def q_stratum(d: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(d.q0_fraction < .10, Q_ORDER[0], np.where(d.displaced_fraction < .10, Q_ORDER[1], Q_ORDER[2])), index=d.index)


def depth_stratum(d: pd.DataFrame) -> pd.Series:
    out = np.full(len(d), "outside_scoring_domain", dtype=object)
    inside = d.in_scoring_domain.to_numpy(bool)
    dist = d.distance_to_scoring_edge_um.to_numpy(float)
    out[inside & (dist < 100)] = DEPTH_ORDER[0]
    out[inside & (dist >= 100) & (dist < 200)] = DEPTH_ORDER[1]
    out[inside & (dist >= 200)] = DEPTH_ORDER[2]
    return pd.Series(out, index=d.index)


def summarize(x: pd.DataFrame) -> dict:
    n = len(x)
    finite = x.noise_status != "undefined"
    fail = x.noise_status == "finite_fail"
    passed = x.noise_status == "finite_pass"
    spike_den = int(x.num_spikes.sum())
    return {
        "units": n,
        "spikes": spike_den,
        "finite_pass_units": int(passed.sum()),
        "finite_fail_units": int(fail.sum()),
        "undefined_units": int((~finite).sum()),
        "finite_pass_prevalence_all_units": float(passed.mean()) if n else None,
        "finite_fail_prevalence_all_units": float(fail.mean()) if n else None,
        "undefined_prevalence_all_units": float((~finite).mean()) if n else None,
        "finite_only_failure_rate": float(fail.sum() / finite.sum()) if finite.sum() else None,
        "failure_spikes": int(x.loc[fail, "num_spikes"].sum()),
        "failure_spike_share": float(x.loc[fail, "num_spikes"].sum() / spike_den) if spike_den else None,
    }


def standardized_gap(x: pd.DataFrame, factor: str, order: list[str]) -> dict:
    arms = {}
    total = len(x)
    pooled_counts = x[factor].value_counts()
    common = [s for s in order if all(((x.arm == arm) & (x[factor] == s)).any() for arm in ("REF", "A"))]
    coverage = float(pooled_counts.reindex(common, fill_value=0).sum() / total) if total else 0.0
    weights = pooled_counts.reindex(common, fill_value=0) / pooled_counts.reindex(common, fill_value=0).sum()
    for arm in ("REF", "A"):
        a = x[x.arm == arm]
        fail_rates = a.groupby(factor).noise_failure.mean().reindex(common)
        undef_rates = a.groupby(factor).noise_undefined.mean().reindex(common)
        arms[arm] = {
            "raw_failure_prevalence": float(a.noise_failure.mean()),
            "standardized_failure_prevalence": float((fail_rates * weights).sum()),
            "raw_undefined_prevalence": float(a.noise_undefined.mean()),
            "standardized_undefined_prevalence": float((undef_rates * weights).sum()),
        }
    raw = arms["A"]["raw_failure_prevalence"] - arms["REF"]["raw_failure_prevalence"]
    std = arms["A"]["standardized_failure_prevalence"] - arms["REF"]["standardized_failure_prevalence"]
    reduction = None if raw == 0 else 1 - abs(std) / abs(raw)
    primary = bool(
        raw != 0
        and coverage >= .90
        and reduction is not None
        and reduction >= .50
    )
    return {
        "factor": factor,
        "common_strata": common,
        "common_support_coverage": coverage,
        "pooled_weights": {k: float(v) for k, v in weights.items()},
        "arms": arms,
        "raw_arm_gap_A_minus_REF": raw,
        "standardized_arm_gap_A_minus_REF": std,
        "absolute_gap_reduction_fraction": reduction,
        "meets_frozen_primary_composition_rule": primary,
    }


def main() -> None:
    actual = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"source hash drift: {actual}")
    d = pd.read_csv(SOURCE)
    required = {
        "probe", "arm", "unit_id", "num_spikes", "q0_spikes", "displaced_spikes",
        "in_scoring_domain", "distance_to_scoring_edge_um", "noise_cutoff",
        "noise_cutoff_pass", "q0_noise_cutoff", "q0_noise_cutoff_pass",
        "displaced_noise_cutoff", "displaced_noise_cutoff_pass", "q0_fraction",
        "displaced_fraction", "spike_count_stratum", "q_activity_stratum", "depth_stratum",
    }
    missing_cols = sorted(required - set(d.columns))
    duplicate_keys = int(d.duplicated(["probe", "arm", "unit_id"]).sum())
    unit_id_repeated_rows_without_probe_arm = int(d.duplicated(["unit_id"], keep=False).sum())
    cross_arm_numeric_id_overlaps = {
        probe: int(len(set(g.loc[g.arm == "A", "unit_id"]) & set(g.loc[g.arm == "REF", "unit_id"])))
        for probe, g in d.groupby("probe")
    }
    recomputed_status = status(d.noise_cutoff)
    q0_status = status(d.q0_noise_cutoff)
    displaced_status = status(d.displaced_noise_cutoff)
    checks = {
        "source_sha256_matches": actual == EXPECTED_SHA256,
        "rows": len(d),
        "columns": len(d.columns),
        "missing_required_columns": missing_cols,
        "duplicate_probe_arm_unit_keys": duplicate_keys,
        "unit_id_repeated_rows_if_probe_arm_ignored": unit_id_repeated_rows_without_probe_arm,
        "cross_arm_numeric_unit_id_overlaps_not_identity": cross_arm_numeric_id_overlaps,
        "rows_by_probe_arm": {
            f"{probe}/{arm}": int(len(g)) for (probe, arm), g in d.groupby(["probe", "arm"], sort=True)
        },
        "probe_values": sorted(d.probe.unique().tolist()),
        "arm_values": sorted(d.arm.unique().tolist()),
        "boolean_domain_violations": {
            c: int((~d[c].isin([True, False])).sum())
            for c in ["in_scoring_domain", "noise_cutoff_pass", "q0_noise_cutoff_pass", "displaced_noise_cutoff_pass"]
        },
        "state_count_sum_mismatches": int((d.num_spikes != d.q0_spikes + d.displaced_spikes).sum()),
        "q_fraction_sum_max_abs_error": float(np.max(np.abs(d.q0_fraction + d.displaced_fraction - 1))),
        "saved_full_status_mismatches": int((d.noise_status != recomputed_status).sum()),
        "saved_full_pass_mismatches": int((d.noise_cutoff_pass != (recomputed_status == "finite_pass")).sum()),
        "saved_q0_pass_mismatches": int((d.q0_noise_cutoff_pass != (q0_status == "finite_pass")).sum()),
        "saved_displaced_pass_mismatches": int((d.displaced_noise_cutoff_pass != (displaced_status == "finite_pass")).sum()),
        "saved_count_stratum_mismatches": int((d.spike_count_stratum != count_stratum(d.num_spikes)).sum()),
        "saved_q_activity_stratum_mismatches": int((d.q_activity_stratum != q_stratum(d)).sum()),
        "saved_depth_stratum_mismatches": int((d.depth_stratum != depth_stratum(d)).sum()),
        "full_noise_nonfinite": int(pd.to_numeric(d.noise_cutoff, errors="coerce").isna().sum()),
        "q0_noise_nonfinite": int(pd.to_numeric(d.q0_noise_cutoff, errors="coerce").isna().sum()),
        "displaced_noise_nonfinite": int(pd.to_numeric(d.displaced_noise_cutoff, errors="coerce").isna().sum()),
        "clock_and_exposure_distinct_values_by_probe": {
            probe: {
                "sampling_frequency_hz": sorted(map(float, g.sampling_frequency_hz.unique())),
                "duration_s": sorted(map(float, g.duration_s.unique())),
                "q0_exposure_s": sorted(map(float, g.q0_exposure_s.unique())),
                "displaced_exposure_s": sorted(map(float, g.displaced_exposure_s.unique())),
            }
            for probe, g in d.groupby("probe", sort=True)
        },
        "noise_missingness_by_probe_arm": {
            f"{probe}/{arm}": {
                "full": int(g.noise_cutoff.isna().sum()),
                "q0": int(g.q0_noise_cutoff.isna().sum()),
                "displaced": int(g.displaced_noise_cutoff.isna().sum()),
            }
            for (probe, arm), g in d.groupby(["probe", "arm"], sort=True)
        },
    }
    hard_fail = (
        missing_cols or duplicate_keys or checks["state_count_sum_mismatches"]
        or any(checks["boolean_domain_violations"].values())
        or any(checks[k] for k in checks if k.endswith("_mismatches"))
        or checks["q_fraction_sum_max_abs_error"] > 1e-12
    )
    if hard_fail:
        dump_json(ROOT / "DATA_QUALITY.json", checks)
        raise RuntimeError("frozen data-quality gate failed")

    d["noise_status"] = recomputed_status
    d["noise_failure"] = d.noise_status == "finite_fail"
    d["noise_undefined"] = d.noise_status == "undefined"
    d["q0_noise_status_recomputed"] = q0_status
    d["displaced_noise_status_recomputed"] = displaced_status
    d["state_eligible"] = (
        d.in_scoring_domain
        & (d.q0_spikes >= 100)
        & (d.displaced_spikes >= 100)
        & (q0_status != "undefined")
        & (displaced_status != "undefined")
    )
    both = (q0_status == "finite_fail") & (displaced_status == "finite_fail")
    q0_only = (q0_status == "finite_fail") & (displaced_status == "finite_pass")
    disp_only = (q0_status == "finite_pass") & (displaced_status == "finite_fail")
    d["state_persistence"] = np.where(
        ~d.state_eligible, "state_unavailable",
        np.where(both, "both_fail", np.where(q0_only, "q0_only_fail", np.where(disp_only, "displaced_only_fail", "neither_fail"))),
    )

    primary = d[d.in_scoring_domain].copy()
    headline = []
    strata_rows = []
    state_rows = []
    status_profile = []
    sensitivity = []
    standardization = []
    interpretations = []
    for probe, p in primary.groupby("probe", sort=True):
        by_arm = {}
        for arm, a in p.groupby("arm", sort=True):
            by_arm[arm] = summarize(a)
            headline.append({"probe": probe, "arm": arm, **summarize(a)})
            for factor, order in (("spike_count_stratum", COUNT_ORDER), ("q_activity_stratum", Q_ORDER), ("depth_stratum", DEPTH_ORDER)):
                for value in order:
                    z = a[a[factor] == value]
                    strata_rows.append({"probe": probe, "arm": arm, "factor": factor, "stratum": value, **summarize(z)})
            eligible = a[a.state_eligible]
            full_fails = a[a.noise_failure]
            for cls in ["both_fail", "q0_only_fail", "displaced_only_fail", "neither_fail", "state_unavailable"]:
                n = int((a.state_persistence == cls).sum())
                n_elig = int((eligible.state_persistence == cls).sum())
                n_full_fail = int((full_fails.state_persistence == cls).sum())
                state_rows.append({
                    "probe": probe, "arm": arm, "state_persistence": cls,
                    "units": n, "all_primary_units": len(a),
                    "fraction_all_primary_units": n / len(a) if len(a) else None,
                    "state_eligible_units": len(eligible),
                    "fraction_state_eligible_units": n_elig / len(eligible) if len(eligible) else None,
                    "full_session_finite_fail_units": len(full_fails),
                    "fraction_full_session_failures": n_full_fail / len(full_fails) if len(full_fails) else None,
                })
            high = a[a.num_spikes >= 1000]
            sensitivity.append({"probe": probe, "arm": arm, "sensitivity": "exclude_low_count_lt_1000", **summarize(high)})
            for noise_state in ["finite_pass", "finite_fail", "undefined"]:
                z = a[a.noise_status == noise_state]
                status_profile.append({
                    "probe": probe, "arm": arm, "noise_status": noise_state,
                    "units": len(z),
                    "median_spikes": float(z.num_spikes.median()) if len(z) else None,
                    "q10_spikes": float(z.num_spikes.quantile(.10)) if len(z) else None,
                    "q90_spikes": float(z.num_spikes.quantile(.90)) if len(z) else None,
                    "low_count_lt_1000_units": int((z.num_spikes < 1000).sum()),
                    "mixed_q_units": int((z.q_activity_stratum == "mixed").sum()),
                    "ordinary_interior_units": int((z.depth_stratum == "inside_ge_200_um").sum()),
                })
        raw_gap = by_arm["A"]["finite_fail_prevalence_all_units"] - by_arm["REF"]["finite_fail_prevalence_all_units"]
        for factor, order in (("spike_count_stratum", COUNT_ORDER), ("q_activity_stratum", Q_ORDER), ("depth_stratum", DEPTH_ORDER)):
            standardization.append({"probe": probe, **standardized_gap(p, factor, order)})
        mixed = {arm: summarize(p[(p.arm == arm) & (p.q_activity_stratum == "mixed")]) for arm in ("REF", "A")}
        interior = {arm: summarize(p[(p.arm == arm) & (p.depth_stratum == "inside_ge_200_um")]) for arm in ("REF", "A")}
        mixed_gap = mixed["A"]["finite_fail_prevalence_all_units"] - mixed["REF"]["finite_fail_prevalence_all_units"]
        interior_gap = interior["A"]["finite_fail_prevalence_all_units"] - interior["REF"]["finite_fail_prevalence_all_units"]
        interpretations.append({
            "probe": probe,
            "headline_gap_A_minus_REF": raw_gap,
            "mixed_q_gap_A_minus_REF": mixed_gap,
            "mixed_q_retains_headline_sign_and_half_magnitude": bool(raw_gap != 0 and np.sign(mixed_gap) == np.sign(raw_gap) and abs(mixed_gap) >= .5 * abs(raw_gap)),
            "ordinary_interior_gap_A_minus_REF": interior_gap,
            "ordinary_interior_retains_headline_sign_and_half_magnitude": bool(raw_gap != 0 and np.sign(interior_gap) == np.sign(raw_gap) and abs(interior_gap) >= .5 * abs(raw_gap)),
        })

    pd.DataFrame(headline).to_csv(ROOT / "HEADLINE.csv", index=False)
    pd.DataFrame(strata_rows).to_csv(ROOT / "STRATIFIED_COUNTS.csv", index=False)
    pd.DataFrame(state_rows).to_csv(ROOT / "STATE_PERSISTENCE.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(ROOT / "SENSITIVITY.csv", index=False)
    pd.DataFrame(status_profile).to_csv(ROOT / "STATUS_PROFILE.csv", index=False)
    dump_json(ROOT / "STANDARDIZATION.json", standardization)
    dump_json(ROOT / "INTERPRETATION_RULE_RESULTS.json", interpretations)
    dump_json(ROOT / "DATA_QUALITY.json", checks)


if __name__ == "__main__":
    main()
