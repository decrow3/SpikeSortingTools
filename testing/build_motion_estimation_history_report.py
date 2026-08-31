"""Build the canonical portable report artifact for the estimation survey."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path("testing/outputs/motion_estimation_history_survey")
ARTIFACT = OUTPUT / "artifact.json"


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.replace({np.nan: None}).to_json(orient="records"))


runs = pd.read_csv(OUTPUT / "run_inventory.csv")
methods = pd.read_csv(OUTPUT / "method_metrics.csv")
agreement = pd.read_csv(OUTPUT / "cross_method_agreement.csv")
jumps = pd.read_csv(OUTPUT / "motion_jump_peak_context.csv")

dredge_dc = agreement[
    agreement.left_method.eq("decentralized") & agreement.right_method.eq("dredge")
].copy()
dredge_dc["run"] = (
    dredge_dc.session.astype(str)
    + " "
    + dredge_dc.probe.astype(str)
    + " "
    + dredge_dc.pipeline_name.str.extract(r"^(motiontest1|motiontest|branchingtest0|dredgetest)", expand=False).fillna("full")
)
dredge_dc["support"] = np.where(
    dredge_dc.absolute_rigid_correlation >= 0.65,
    "≥0.65 consensus",
    np.where(dredge_dc.absolute_rigid_correlation < 0.35, "<0.35 disagreement", "partial"),
)
agreement_dataset = dredge_dc[
    ["run", "session", "probe", "absolute_rigid_correlation", "median_absolute_difference_um", "support"]
].sort_values("absolute_rigid_correlation")

method_summary = (
    methods.groupby("method")
    .agg(
        fields=("method", "size"),
        median_rigid_excursion_um=("rigid_excursion_p95_p5_um", "median"),
        median_nonrigid_spread_um=("median_nonrigid_spread_um", "median"),
        median_p99_step_um=("p99_abs_step_um", "median"),
        median_largest_to_p99_step=("max_to_p99_step_ratio", "median"),
    )
    .reset_index()
)

large_jumps = jumps[jumps.max_step_um.abs() >= 50].copy()
jump_rows = []
for method in ["kilosort_style", "decentralized", "dredge", "medicine"]:
    group = large_jumps[large_jumps.method.eq(method)]
    jump_rows.append(
        {
            "method": method,
            "large_top_events": int(len(group)),
            "fields": int(group.motion_dir.nunique()),
            "median_coherent_depth_fraction": float(group.coherent_depth_fraction.median()) if len(group) else None,
            "peak_rate_burst_fraction": float((group.event_to_context_peak_rate_ratio > 1.5).mean()) if len(group) else 0.0,
            "amplitude_tail_burst_fraction": float((group.event_to_context_amp_p99_ratio > 1.5).mean()) if len(group) else 0.0,
            "near_baseline_rate_fraction": float(group.event_to_context_peak_rate_ratio.between(0.7, 1.3).mean()) if len(group) else None,
        }
    )
jump_summary = pd.DataFrame(jump_rows)
jump_long = jump_summary.melt(
    id_vars=["method", "large_top_events", "fields"],
    value_vars=[
        "median_coherent_depth_fraction",
        "peak_rate_burst_fraction",
        "near_baseline_rate_fraction",
    ],
    var_name="signature",
    value_name="fraction",
)
jump_long["signature"] = jump_long.signature.map(
    {
        "median_coherent_depth_fraction": "depth coherence",
        "peak_rate_burst_fraction": "peak-rate burst",
        "near_baseline_rate_fraction": "near-baseline peak rate",
    }
)

dredge = methods[methods.method.eq("dredge")].copy()
dredge["severity"] = pd.cut(
    dredge.rigid_excursion_p95_p5_um,
    bins=[-np.inf, 10, 20, 40, np.inf],
    labels=["<10 µm", "10–20 µm", "20–40 µm", ">40 µm"],
)
severity = (
    dredge.groupby(["subject", "severity"], observed=False)
    .size()
    .rename("fields")
    .reset_index()
)

band_fields = pd.read_csv(
    "/mnt/NPX/Luke/20250804/dredge_pipeline_results_Luke0804_V2V1_g0_imec1/"
    "motion_estimator_band_ablation/field_summary.csv"
)
band_fields["detected_peaks"] = [244805, 198019]
band_fields["heldout_raster_correlation"] = [0.2191935564, -0.1734362926]

factorial_root = Path("testing/outputs/luke_motion_input_factorial")
factorial_summary = pd.read_csv(factorial_root / "factorial_field_summary.csv")
factorial_agreement = pd.read_csv(factorial_root / "factorial_agreement.csv")
regime_evidence = pd.read_csv(factorial_root / "regime_evidence.csv")
cross_probe_full = factorial_agreement[
    factorial_agreement.scope.eq("cross_probe")
    & factorial_agreement.condition.eq("full")
    & factorial_agreement.left_estimator.isin(
        ["dredge_300_200_cpu", "decentralized_300_200_numpy"]
    )
].copy()
cross_probe_full["estimator"] = cross_probe_full.left_estimator.map(
    {
        "dredge_300_200_cpu": "DREDGE",
        "decentralized_300_200_numpy": "decentralized",
    }
)
perturbation_evidence = factorial_agreement[
    factorial_agreement.scope.eq("condition_vs_full")
    & factorial_agreement.probe.eq("imec1")
    & factorial_agreement.left_estimator.isin(
        ["dredge_300_200_cpu", "decentralized_300_200_numpy"]
    )
    & factorial_agreement.condition.isin(
        [
            "random_quarter",
            "high_amplitude_half",
            "exclude_synchronous",
            "exclude_bursty_seconds",
            "exclude_dominant_channel",
        ]
    )
].copy()
perturbation_evidence["estimator"] = perturbation_evidence.left_estimator.map(
    {
        "dredge_300_200_cpu": "DREDGE",
        "decentralized_300_200_numpy": "decentralized",
    }
)
perturbation_evidence["condition"] = perturbation_evidence.condition.str.replace(
    "_", " ", regex=False
)
direct_scale = pd.read_csv(
    "testing/outputs/luke_direct_motion_scale_audit/scale_summary.csv"
)
threshold_root = Path("testing/outputs/luke_motion_detection_threshold_factorial")
threshold_summary = pd.read_csv(threshold_root / "threshold_field_summary.csv")
threshold_agreement = pd.read_csv(threshold_root / "threshold_agreement.csv")
threshold_cross_estimator = pd.read_csv(
    threshold_root / "threshold_cross_estimator_agreement.csv"
)
threshold_evidence = pd.read_csv(threshold_root / "threshold_regime_evidence.csv")
threshold_chart = threshold_agreement[threshold_agreement.threshold.eq(7.0)].copy()
threshold_chart["estimator_label"] = threshold_chart.estimator.map(
    {
        "dredge_300_200_cpu": "DREDGE",
        "decentralized_300_200_numpy": "decentralized",
    }
)
threshold_chart = threshold_chart.merge(
    threshold_evidence[["regime", "threshold7_retained_fraction"]], on="regime", how="left"
)

definitions = pd.DataFrame(
    [
        {
            "signature": "Supported estimate",
            "operational definition": "Adequate time-depth peak support plus split-half/parameter stability and an independent method, probe, or held-out observable that agrees in direction and approximate scale.",
            "interpretation": "Plausible tissue displacement; still not proof that voltage correction helps.",
        },
        {
            "signature": "Whole-depth discontinuity",
            "operational definition": "One-bin jump coherent across most depth bins but absent or much smaller in independent estimators.",
            "interpretation": "Algorithmic registration jump is more likely than instantaneous tissue translation.",
        },
        {
            "signature": "Local depth-edge jump",
            "operational definition": "Large step confined to a small fraction of depth bins, often near a spatial boundary.",
            "interpretation": "Weak local support, activity turnover, or optimizer edge behavior; not a global motion event.",
        },
        {
            "signature": "Noise-coupled estimate",
            "operational definition": "Estimate excursion coincides with a peak-rate burst, unusually synchronous channels, amplitude-tail shift, or reference/filter transition.",
            "interpretation": "Candidate hallucination from altered observations; requires noise-masked re-estimation.",
        },
        {
            "signature": "Correction-worthy motion",
            "operational definition": "Supported estimate plus replicated uncorrected waveform/unit-family degradation that scales with motion and improves under a preservation-safe correction.",
            "interpretation": "Only this regime justifies enabling a transform for that session.",
        },
    ]
)

factor_evidence = pd.DataFrame(
    [
        {
            "factor": "Peak threshold",
            "historical evidence": "A historical-equivalent 5/6/7 sweep on Luke imec1 retains only 31–37% of threshold-5 peaks at threshold 7. Rapid and noise-plus-motion rigid trajectories remain r=0.95–0.99 to threshold 5; decentralized dropout falls to r=0.70.",
            "current conclusion": "Supported rigid motion is not an artifact of the threshold-5 background population. Threshold sensitivity is instead a useful failure flag during support dropout; nonrigid structure is generally less stable.",
            "next test": "Persist exact noise levels and repeat a fresh 5/7 detection/localization comparison in July Luke and severe Rocky sessions.",
        },
        {
            "factor": "Number of peaks",
            "historical evidence": "Random 50% and 25% subsets preserve supported rapid/noise-plus-motion rigid fields; quarter density degrades the support-dropout field despite a similar absolute count to another stable window.",
            "current conclusion": "Temporal/depth support geometry matters more than a universal peak-count cutoff.",
            "next test": "Peak-count dose response using deterministic subsamples from one fixed peak population.",
        },
        {
            "factor": "Noise exclusion / bad channels",
            "historical evidence": "Channel 191 contributes almost no detections in selected windows. Synchrony and dominant-channel masks preserve supported rigid trajectories but can alter nonrigid residuals, especially when noise coincides with motion.",
            "current conclusion": "Simple peak exclusion does not explain supported rigid motion; local residuals are more noise-sensitive. Voltage-level interpolation policy remains open.",
            "next test": "Rerun detection/localization under exclude, interpolate, and retain-with-zero-weight preprocessing on the same windows.",
        },
        {
            "factor": "Estimator",
            "historical evidence": "DREDGE/decentralized often agree; Kilosort adds large coherent jumps; MEDiCINe is strongly smoothed by its saved configuration.",
            "current conclusion": "Algorithm choice materially changes failure signature and apparent scale.",
            "next test": "Use estimator consensus and split halves rather than selecting one algorithm by appearance.",
        },
        {
            "factor": "Filtering",
            "historical evidence": "300–3000 versus 300–6000 Hz changed peak count 23.6%, while DREDGE rigid/residual correlations remained 0.85/0.90.",
            "current conclusion": "Broad field direction is robust in the tested window; absolute scale remains uncertain.",
            "next test": "Repeat in quiet, rapid-motion, and sustained-noise windows.",
        },
        {
            "factor": "Rapid motion / signal change",
            "historical evidence": "The rapid and noise-plus-motion windows retain r≈0.97–0.99 DREDGE/decentralized agreement and r≈0.82–0.97 cross-probe rigid agreement. Support dropout fails both gates and produces DREDGE-only tail spread.",
            "current conclusion": "Method, exclusion, and cross-probe robustness distinguish supported rapid motion from support-driven hallucination.",
            "next test": "Event-aligned held-out waveform/raster and cross-probe comparison.",
        },
    ]
)

policy = pd.DataFrame(
    [
        {"stage": "Estimate", "default": "On as sidecar", "gate": "Always save provenance, support, confidence, and diagnostic traces."},
        {"stage": "Coordinate-only use", "default": "Off", "gate": "Enable per session after independent estimate validation and held-out coordinate improvement."},
        {"stage": "Voltage correction", "default": "Off", "gate": "Require preservation-safe application plus replicated downstream benefit over no correction."},
        {"stage": "Kilosort internal correction", "default": "Off for baseline", "gate": "Reconsider only after discontinuity control and matched no-motion comparison."},
    ]
)

generated_at = datetime.now(timezone.utc).isoformat()


def source(source_id: str, label: str, description: str, files: list[str], filters: list[str]):
    csv_files = [file for file in files if file.endswith(".csv")]
    sql = (
        "SELECT * FROM read_csv_auto('" + csv_files[0] + "')"
        if csv_files
        else "SELECT * FROM read_json_auto('" + files[0] + "')"
    )
    return {
        "id": source_id,
        "label": label,
        "query": {
            "engine": "Python",
            "description": description,
            "sql": sql,
            "language": "python",
            "filters": filters,
            "metric_definitions": [],
            "tables_used": files,
        },
    }


sources = [
    source(
        "history-survey",
        "Historical estimation cache survey",
        "Bounded deterministic profiling of cached peaks and saved displacement arrays; no motion correction applied.",
        [
            "testing/outputs/motion_estimation_history_survey/run_inventory.csv",
            "testing/outputs/motion_estimation_history_survey/method_metrics.csv",
            "testing/outputs/motion_estimation_history_survey/cross_method_agreement.csv",
        ],
        ["pipeline result motion caches", "estimation only", "bounded peak samples"],
    ),
    source(
        "jump-context",
        "Peak context around estimate jumps",
        "Exact cached-peak windows around the three largest separated steps in each saved field.",
        ["testing/outputs/motion_estimation_history_survey/motion_jump_peak_context.csv"],
        ["top three steps per field", "30 second separation", "no voltage resampling"],
    ),
    source(
        "band-ablation",
        "Luke estimator filtering ablation",
        "Matched DREDGE estimation on 300–3000 Hz and 300–6000 Hz peak populations with held-out peak halves.",
        ["motion_estimator_band_ablation/field_summary.csv", "motion_estimator_band_ablation/heldout_raster_summary.csv"],
        ["Luke 2025-08-04 imec1", "8160–8220 s", "detect threshold 5"],
    ),
    source(
        "correction-decision",
        "Luke motion reintroduction decision",
        "Prespecified voltage-preservation, bounded sorting, depth-strip, and supported-crop evidence.",
        ["testing/outputs/luke_motion_reintroduction_decision/decision.json"],
        ["Luke 2025-08-04", "matched no-motion controls"],
    ),
    source(
        "input-factorial",
        "Luke estimation-only input factorial",
        "Matched DREDGE, decentralized, and iterative-template estimates across five prespecified regimes and deterministic input perturbations; no motion field applied.",
        [
            "testing/outputs/luke_motion_input_factorial/factorial_field_summary.csv",
            "testing/outputs/luke_motion_input_factorial/factorial_agreement.csv",
            "testing/outputs/luke_motion_input_factorial/regime_evidence.csv",
        ],
        ["Luke 2025-08-04", "120 second windows", "estimation only", "185 completed runs"],
    ),
    source(
        "direct-scale",
        "Direct amplitude-depth raster scale audit",
        "Independent raster-matching implementation on deterministic peak halves; shares the pre-registration peak source with DREDGE.",
        [
            "testing/outputs/luke_direct_motion_scale_audit/scale_summary.csv",
            "testing/outputs/luke_direct_motion_scale_audit/gain_scores.csv",
        ],
        ["Luke 2025-08-04 imec1", "60 second peak-raster pairs", "no voltage resampling"],
    ),
    source(
        "threshold-factorial",
        "Luke historical-equivalent detection-threshold audit",
        "Threshold-5 historical localized peaks with per-channel accepted-amplitude boundaries inferred over the full session; exact nested 6/7 subsets, estimation only.",
        [
            "testing/outputs/luke_motion_detection_threshold_factorial/threshold_field_summary.csv",
            "testing/outputs/luke_motion_detection_threshold_factorial/threshold_agreement.csv",
            "testing/outputs/luke_motion_detection_threshold_factorial/threshold_cross_estimator_agreement.csv",
            "testing/outputs/luke_motion_detection_threshold_factorial/threshold_regime_evidence.csv",
        ],
        ["Luke 2025-08-04 imec1", "five 120 second windows", "thresholds 5/6/7", "30 estimates", "no correction"],
    ),
]

charts = [
    {
        "id": "agreement-chart",
        "title": "DREDGE and decentralized rigid-trace agreement",
        "subtitle": "Matched caches; ≥0.65 is a screening threshold for reproducibility, not ground truth.",
        "type": "bar",
        "dataset": "agreement",
        "sourceId": "history-survey",
        "intent": "comparison",
        "question": "When do two independent estimators recover a similar rigid trajectory?",
        "rationale": "Horizontal bars keep repeated cache labels readable and expose mixed session behavior.",
        "comparisonContext": "Eleven Luke caches across four recording dates.",
        "encodings": {
            "x": {"field": "run", "type": "nominal", "title": "Cache"},
            "y": {"field": "absolute_rigid_correlation", "type": "quantitative", "title": "Absolute correlation"},
            "color": {"field": "support", "type": "nominal", "title": "Support class"},
            "tooltip": [
                {"field": "run", "type": "nominal", "title": "Cache"},
                {"field": "absolute_rigid_correlation", "type": "quantitative", "title": "|r|", "format": ".3f"},
                {"field": "median_absolute_difference_um", "type": "quantitative", "title": "Median difference (µm)", "format": ".2f"},
            ],
        },
        "layout": "full",
        "palette": {"kind": "categorical"},
        "legend": {"position": "bottom"},
        "settings": {"showGrid": True, "showValues": True},
        "surface": {"surface": "export", "viewMode": "both"},
    },
    {
        "id": "jump-signatures-chart",
        "title": "Input and spatial signatures around large estimate steps",
        "subtitle": "Top-event sample; fractions describe signatures among selected ≥50 µm steps, not event prevalence.",
        "type": "bar",
        "dataset": "jump_signatures",
        "sourceId": "jump-context",
        "intent": "comparison",
        "question": "Do the largest estimate steps coincide with peak bursts or coherent motion across depth?",
        "rationale": "Grouped fractions contrast algorithm-specific failure signatures on one scale.",
        "comparisonContext": "Three largest separated steps per saved field.",
        "encodings": {
            "x": {"field": "method", "type": "nominal", "title": "Estimator"},
            "y": {"field": "fraction", "type": "quantitative", "title": "Fraction"},
            "color": {"field": "signature", "type": "nominal", "title": "Signature"},
            "tooltip": [
                {"field": "method", "type": "nominal", "title": "Estimator"},
                {"field": "signature", "type": "nominal", "title": "Signature"},
                {"field": "fraction", "type": "quantitative", "title": "Fraction", "format": ".1%"},
                {"field": "large_top_events", "type": "quantitative", "title": "Large selected events"},
            ],
        },
        "valueFormat": "percent",
        "layout": "full",
        "palette": {"kind": "categorical"},
        "legend": {"position": "bottom"},
        "settings": {"groupMode": "grouped", "showGrid": True, "showValues": True},
        "surface": {"surface": "export", "viewMode": "both"},
    },
    {
        "id": "severity-chart",
        "title": "Saved DREDGE rigid-excursion bands",
        "subtitle": "Cache counts are repeated analyses, not independent sessions; most Rocky fields lack a second estimator.",
        "type": "bar",
        "dataset": "severity",
        "sourceId": "history-survey",
        "intent": "composition",
        "question": "How large are the saved rigid excursions across subjects?",
        "rationale": "Stacked counts show descriptive scale without implying validated correction need.",
        "comparisonContext": "P95–p5 rigid excursion in each saved DREDGE field.",
        "encodings": {
            "x": {"field": "subject", "type": "nominal", "title": "Subject/archive"},
            "y": {"field": "fields", "type": "quantitative", "title": "Saved fields"},
            "color": {"field": "severity", "type": "nominal", "title": "Rigid excursion"},
        },
        "layout": "full",
        "palette": {"kind": "categorical"},
        "legend": {"position": "bottom"},
        "settings": {"groupMode": "stacked", "showGrid": True, "showValues": True},
        "surface": {"surface": "export", "viewMode": "both"},
    },
    {
        "id": "cross-probe-regime-chart",
        "title": "Cross-probe rigid support separates motion from support dropout",
        "subtitle": "Full-input matched 120-second windows; cross-probe agreement is corroborating evidence, not biological ground truth.",
        "type": "bar",
        "dataset": "cross_probe_full",
        "sourceId": "input-factorial",
        "intent": "comparison",
        "question": "Which estimated rigid trajectories recur on both simultaneously recorded probes?",
        "rationale": "Grouped bars compare two estimator families at the same window and scale.",
        "comparisonContext": "Five prespecified Luke regimes on imec0 and imec1.",
        "encodings": {
            "x": {"field": "regime", "type": "nominal", "title": "Regime"},
            "y": {"field": "rigid_correlation", "type": "quantitative", "title": "imec0–imec1 rigid correlation"},
            "color": {"field": "estimator", "type": "nominal", "title": "Estimator"},
            "tooltip": [
                {"field": "regime", "type": "nominal", "title": "Regime"},
                {"field": "estimator", "type": "nominal", "title": "Estimator"},
                {"field": "rigid_correlation", "type": "quantitative", "title": "Rigid r", "format": ".3f"},
                {"field": "nonrigid_correlation", "type": "quantitative", "title": "Residual r", "format": ".3f"},
            ],
        },
        "layout": "full",
        "palette": {"kind": "categorical"},
        "legend": {"position": "bottom"},
        "settings": {"groupMode": "grouped", "showGrid": True, "showValues": True},
        "surface": {"surface": "export", "viewMode": "both"},
    },
    {
        "id": "threshold-rigid-chart",
        "title": "Supported rigid trajectories survive threshold 7",
        "subtitle": "Threshold 7 retains 31–37% of threshold-5 peaks; correlation is within estimator against its threshold-5 field.",
        "type": "bar",
        "dataset": "threshold_chart",
        "sourceId": "threshold-factorial",
        "intent": "comparison",
        "question": "Does a channel-calibrated threshold increase change each estimated rigid trajectory?",
        "rationale": "Grouped bars expose estimator-specific instability in the same prespecified windows.",
        "comparisonContext": "Historical-equivalent thresholds 5 and 7 on Luke imec1.",
        "encodings": {
            "x": {"field": "regime", "type": "nominal", "title": "Regime"},
            "y": {"field": "rigid_correlation_vs_threshold5", "type": "quantitative", "title": "Rigid r: threshold 7 vs 5"},
            "color": {"field": "estimator_label", "type": "nominal", "title": "Estimator"},
            "tooltip": [
                {"field": "regime", "type": "nominal", "title": "Regime"},
                {"field": "estimator_label", "type": "nominal", "title": "Estimator"},
                {"field": "rigid_correlation_vs_threshold5", "type": "quantitative", "title": "Rigid r", "format": ".3f"},
                {"field": "nonrigid_correlation_vs_threshold5", "type": "quantitative", "title": "Residual r", "format": ".3f"},
                {"field": "threshold7_retained_fraction", "type": "quantitative", "title": "Peaks retained", "format": ".1%"},
            ],
        },
        "layout": "full",
        "palette": {"kind": "categorical"},
        "legend": {"position": "bottom"},
        "settings": {"groupMode": "grouped", "showGrid": True, "showValues": True},
        "surface": {"surface": "export", "viewMode": "both"},
    },
]


def table(table_id: str, title: str, subtitle: str, dataset: str, source_id: str, columns: list[dict], sort_field: str):
    return {
        "id": table_id,
        "title": title,
        "subtitle": subtitle,
        "dataset": dataset,
        "sourceId": source_id,
        "layout": "full",
        "density": "spacious",
        "defaultSort": {"field": sort_field, "direction": "asc"},
        "columns": columns,
    }


tables = [
    table(
        "definitions-table",
        "Estimation signatures",
        "Operational definitions separate plausibility, failure screening, and correction-worthiness.",
        "definitions",
        "history-survey",
        [
            {"field": "signature", "label": "Signature", "type": "string"},
            {"field": "operational definition", "label": "Operational definition", "type": "string"},
            {"field": "interpretation", "label": "Interpretation", "type": "string"},
        ],
        "signature",
    ),
    table(
        "method-table",
        "Estimator field summaries",
        "Medians across saved fields; configurations and session mix differ by estimator.",
        "method_summary",
        "history-survey",
        [
            {"field": "method", "label": "Estimator", "type": "string"},
            {"field": "fields", "label": "Fields", "type": "number", "format": ".0f"},
            {"field": "median_rigid_excursion_um", "label": "Rigid excursion (µm)", "type": "number", "format": ".1f"},
            {"field": "median_nonrigid_spread_um", "label": "Nonrigid spread (µm)", "type": "number", "format": ".1f"},
            {"field": "median_p99_step_um", "label": "P99 step (µm)", "type": "number", "format": ".1f"},
            {"field": "median_largest_to_p99_step", "label": "Largest/P99 step", "type": "number", "format": ".1f"},
        ],
        "method",
    ),
    table(
        "factor-table",
        "What the archive can and cannot say about causal factors",
        "Threshold sensitivity is now controlled on Luke imec1; voltage-level bad-channel policy still requires a fresh sweep.",
        "factor_evidence",
        "history-survey",
        [
            {"field": "factor", "label": "Factor", "type": "string"},
            {"field": "historical evidence", "label": "Historical evidence", "type": "string"},
            {"field": "current conclusion", "label": "Current conclusion", "type": "string"},
            {"field": "next test", "label": "Next test", "type": "string"},
        ],
        "factor",
    ),
    table(
        "band-table",
        "Filtering sensitivity in the controlled Luke window",
        "Matched 60-second estimates with detect threshold 5 and split peak halves.",
        "band_fields",
        "band-ablation",
        [
            {"field": "estimator_band", "label": "Estimator band", "type": "string"},
            {"field": "detected_peaks", "label": "Detected peaks", "type": "number", "format": ",.0f"},
            {"field": "rigid_excursion_p95_p5_um", "label": "Rigid excursion (µm)", "type": "number", "format": ".2f"},
            {"field": "median_nonrigid_spread_um", "label": "Nonrigid spread (µm)", "type": "number", "format": ".2f"},
            {"field": "rigid_correlation_to_other_band", "label": "Rigid cross-band r", "type": "number", "format": ".3f"},
            {"field": "residual_correlation_to_other_band", "label": "Residual cross-band r", "type": "number", "format": ".3f"},
            {"field": "heldout_raster_correlation", "label": "Held-out raster r", "type": "number", "format": ".3f"},
        ],
        "estimator_band",
    ),
    table(
        "policy-table",
        "Recommended pipeline policy",
        "Estimation remains available while all transforms require explicit evidence gates.",
        "policy",
        "correction-decision",
        [
            {"field": "stage", "label": "Stage", "type": "string"},
            {"field": "default", "label": "Default", "type": "string"},
            {"field": "gate", "label": "Evidence gate", "type": "string"},
        ],
        "stage",
    ),
    table(
        "regime-evidence-table",
        "Controlled estimation regimes",
        "Classification requires method, input-perturbation, and simultaneous-probe evidence; no correction was applied.",
        "regime_evidence",
        "input-factorial",
        [
            {"field": "regime", "label": "Regime", "type": "string"},
            {"field": "classification", "label": "Classification", "type": "string"},
            {"field": "dredge_rigid_excursion_range_um", "label": "DREDGE rigid range (µm)", "type": "string"},
            {"field": "within_probe_dredge_dc_r_range", "label": "Within-probe DREDGE/DC r", "type": "string"},
            {"field": "cross_probe_dredge_r", "label": "Cross-probe rigid r", "type": "number", "format": ".3f"},
            {"field": "cross_probe_dredge_residual_r", "label": "Cross-probe residual r", "type": "number", "format": ".3f"},
            {"field": "interpretation", "label": "Interpretation", "type": "string"},
        ],
        "regime",
    ),
    table(
        "perturbation-table",
        "Input perturbation robustness on imec1",
        "Correlation to the full-input field; high-amplitude-half is a post-detection proxy, not a true detect-threshold rerun.",
        "perturbation_evidence",
        "input-factorial",
        [
            {"field": "regime", "label": "Regime", "type": "string"},
            {"field": "estimator", "label": "Estimator", "type": "string"},
            {"field": "condition", "label": "Condition", "type": "string"},
            {"field": "rigid_correlation", "label": "Rigid r to full", "type": "number", "format": ".3f"},
            {"field": "nonrigid_correlation", "label": "Residual r to full", "type": "number", "format": ".3f"},
        ],
        "regime",
    ),
    table(
        "direct-scale-table",
        "Direct peak-raster scale remeasurement",
        "Qualified time-pair fits share the original peak source but use an independent spatial matching implementation.",
        "direct_scale",
        "direct-scale",
        [
            {"field": "raster_spec", "label": "Raster specification", "type": "string"},
            {"field": "qualified_pairs", "label": "Qualified pairs", "type": "number", "format": ".0f"},
            {"field": "slope", "label": "Observed/DREDGE slope", "type": "number", "format": ".3f"},
            {"field": "slope_ci95_low", "label": "Slope CI low", "type": "number", "format": ".3f"},
            {"field": "slope_ci95_high", "label": "Slope CI high", "type": "number", "format": ".3f"},
            {"field": "correlation", "label": "Correlation", "type": "number", "format": ".3f"},
            {"field": "median_absolute_error_um", "label": "Median error (µm)", "type": "number", "format": ".2f"},
        ],
        "raster_spec",
    ),
    table(
        "threshold-evidence-table",
        "Detection-threshold robustness by regime",
        "Threshold 7 is an exact nested amplitude subset of the historical threshold-5 cache using inferred per-channel acceptance boundaries.",
        "threshold_evidence",
        "threshold-factorial",
        [
            {"field": "regime", "label": "Regime", "type": "string"},
            {"field": "threshold7_retained_fraction", "label": "Threshold-5 peaks retained", "type": "number", "format": ".1%"},
            {"field": "dredge_rigid_r_threshold7_vs5", "label": "DREDGE rigid r", "type": "number", "format": ".3f"},
            {"field": "decentralized_rigid_r_threshold7_vs5", "label": "Decentralized rigid r", "type": "number", "format": ".3f"},
            {"field": "cross_estimator_rigid_r_threshold7", "label": "DREDGE/DC rigid r at 7", "type": "number", "format": ".3f"},
            {"field": "interpretation", "label": "Interpretation", "type": "string"},
        ],
        "regime",
    ),
]

blocks = [
    {
        "id": "title",
        "type": "markdown",
        "layout": "full",
        "body": "# When motion estimates are believable in our Neuropixels recordings",
    },
    {
        "id": "technical-summary",
        "type": "markdown",
        "layout": "full",
        "body": (
            "## Technical summary\n\n"
            "**Motion is identifiable in some recordings, but a large saved field is not sufficient evidence that it is real or correction-worthy.** "
            "DREDGE and decentralized rigid traces agree at |r| ≥ 0.65 in 7 of 11 matched caches, concentrated in the 2025-08-04 and 2025-08-05 recordings; July sessions are mixed. "
            "A new 185-run estimation-only factorial identifies two supported 120-second regimes: rapid motion and noise-plus-motion retain within-probe DREDGE/decentralized r≈0.97–0.99 and cross-probe rigid r≈0.82–0.97. In contrast, support dropout has weak method agreement, cross-probe r near zero or negative, and DREDGE-only nonrigid tails.\n\n"
            "**Absolute peak count is not the main validity threshold; support geometry and estimator family matter.** "
            "Random quarter populations preserve supported rigid fields, while a similar count is less stable during support dropout. In a separate 30-run historical-equivalent threshold sweep, threshold 7 retains only 31–37% of peaks yet preserves rapid and noise-plus-motion rigid trajectories at r≈0.95–0.99; decentralized dropout falls to r≈0.70. Threshold sensitivity therefore flags weak support rather than defining a universal count cutoff. Voltage-level bad-channel effects remain open.\n\n"
            "**DREDGE scale is directionally informative but not calibrated as literal micrometers.** Direct peak-raster matching agrees in direction on 96–100% of qualified pairs, while fitted observed/DREDGE slopes vary roughly 0.5–0.8 across primary raster specifications.\n\n"
            "**Keep estimation available as a diagnostic sidecar and keep correction off by default.** "
            "Correction should be enabled only per session after independent estimate validation and replicated evidence that uncorrected waveform or unit-family continuity degrades with motion and improves under a preservation-safe transform."
        ),
    },
    {
        "id": "definitions-heading",
        "type": "markdown",
        "layout": "full",
        "body": "## Success means independent support, not a smooth-looking trace\n\nThe survey uses an evidence ladder: peak support, internal stability, parameter robustness, independent agreement, and held-out observables. Correction-worthiness is a separate downstream criterion.",
    },
    {"id": "definitions", "type": "table", "layout": "full", "tableId": "definitions-table", "sourceId": "history-survey"},
    {
        "id": "controlled-regimes-heading",
        "type": "markdown",
        "layout": "full",
        "sourceId": "input-factorial",
        "body": "## Controlled windows expose supported motion and a support-dropout failure regime\n\nFive non-overlapping 120-second windows were selected from peak rate, synchrony, amplitude/channel concentration, depth support, and saved DREDGE/decentralized dynamics. Across 185 completed re-estimates, rapid motion and noise-plus-motion survive method, count, amplitude-composition, noise-exclusion, and simultaneous-probe checks. Support dropout fails method and cross-probe checks even though DREDGE alone can produce large local tail spreads.",
    },
    {"id": "controlled-regimes", "type": "table", "layout": "full", "tableId": "regime-evidence-table", "sourceId": "input-factorial"},
    {"id": "controlled-cross-probe", "type": "chart", "layout": "full", "chartId": "cross-probe-regime-chart", "sourceId": "input-factorial"},
    {
        "id": "controlled-perturbations-heading",
        "type": "markdown",
        "layout": "full",
        "sourceId": "input-factorial",
        "body": "## Noise coincident with motion biases residual structure more than the rigid trajectory\n\nIn the noise-plus-motion window, synchronous-peak removal changes 3–5% of peaks and dominant-channel removal changes 2–3%, yet DREDGE/decentralized rigid trajectories remain correlated at ≥0.98 with their full-input estimates. Residual DREDGE agreement after dominant-channel removal falls to 0.67 on imec1. The supported common displacement therefore survives, while exact local nonrigid structure should be treated as noise-sensitive.",
    },
    {"id": "controlled-perturbations", "type": "table", "layout": "full", "tableId": "perturbation-table", "sourceId": "input-factorial"},
    {
        "id": "threshold-heading",
        "type": "markdown",
        "layout": "full",
        "sourceId": "threshold-factorial",
        "body": "## Raising the detection threshold preserves supported rigid motion and exposes weak support\n\nThreshold 7 retains roughly one third of the historical threshold-5 detections. Despite that reduction, rapid-motion and noise-plus-motion rigid trajectories remain r≈0.95–0.99 to their threshold-5 versions and DREDGE/decentralized agreement remains r≈0.97–0.98. Quiet rigid drift is also stable but only 6 µm. During support dropout, decentralized stability falls to r=0.70 and cross-probe support remains absent; thresholding does not rescue the field. Residual nonrigid structure is more threshold-sensitive than the rigid component.",
    },
    {"id": "threshold-chart", "type": "chart", "layout": "full", "chartId": "threshold-rigid-chart", "sourceId": "threshold-factorial"},
    {"id": "threshold-evidence", "type": "table", "layout": "full", "tableId": "threshold-evidence-table", "sourceId": "threshold-factorial"},
    {
        "id": "agreement-heading",
        "type": "markdown",
        "layout": "full",
        "body": "## DREDGE–decentralized agreement identifies supported and mixed regimes\n\nAgreement is strongest in the August 4 short-window variants and remains useful in the August 4 full cache and one August 5 probe. July 30 imec1 is the clearest disagreement case. Repeated variants from one date are sensitivity evidence, not independent session replications.",
    },
    {"id": "agreement", "type": "chart", "layout": "full", "chartId": "agreement-chart", "sourceId": "history-survey"},
    {
        "id": "algorithm-heading",
        "type": "markdown",
        "layout": "full",
        "body": "## Algorithms fail in recognizably different ways\n\nHistorical MEDiCINe is heavily smoothed by two depth bins and a 50-second kernel. Decentralized fields are larger and temporally variable. Kilosort-style fields are spatially coherent but prone to isolated large jumps. DREDGE is usually smoother in time, but large local depth-edge steps remain common screening events.",
    },
    {"id": "methods", "type": "table", "layout": "full", "tableId": "method-table", "sourceId": "history-survey"},
    {
        "id": "jump-heading",
        "type": "markdown",
        "layout": "full",
        "body": "## Large DREDGE steps usually look local, not like whole-probe rapid motion\n\nAmong the selected ≥50 µm DREDGE steps, the median event is coherent across only 10% of depth bins; 18% coincide with a >1.5× peak-rate burst, none with a >1.5× amplitude-tail burst, and 55% occur near baseline peak rate. Kilosort's large jumps are instead coherent across essentially the full depth range. This distinguishes local registration instability from global motion-like discontinuities, but neither signature alone establishes ground truth.",
    },
    {"id": "jumps", "type": "chart", "layout": "full", "chartId": "jump-signatures-chart", "sourceId": "jump-context"},
    {
        "id": "factors-heading",
        "type": "markdown",
        "layout": "full",
        "body": "## Peak count and threshold matter through support quality, not a universal cutoff\n\nAcross saved DREDGE fields, higher peak rate is associated with smaller high-percentile steps, but this observational association is confounded. Controlled random-quarter and threshold-7 populations show that supported rigid fields can survive losing two thirds to three quarters of detections. Instability is concentrated in dropout and residual nonrigid structure, so temporal/depth coverage and independent agreement are more useful gates than an absolute count.",
    },
    {"id": "factors", "type": "table", "layout": "full", "tableId": "factor-table", "sourceId": "history-survey"},
    {
        "id": "filter-heading",
        "type": "markdown",
        "layout": "full",
        "body": "## Filtering changes the observations more than the broad field\n\nThe controlled band ablation is positive robustness evidence: a 23.6% peak-count change preserves the field direction and residual structure. Weak held-out raster correlation prevents treating either estimate as calibrated displacement.",
    },
    {"id": "filter", "type": "table", "layout": "full", "tableId": "band-table", "sourceId": "band-ablation"},
    {
        "id": "direct-scale-heading",
        "type": "markdown",
        "layout": "full",
        "sourceId": "direct-scale",
        "body": "## The estimated direction is reproducible, but the micrometer scale is not literal\n\nAn independent amplitude-depth raster matcher recovers the same direction on 96–100% of qualified 60-second pairs. Primary observed-versus-DREDGE slopes range from about 0.5 to 0.8 across raster representations, with broad confidence intervals. DREDGE therefore supplies useful relative trajectories and severity bands, but correction gains should not assume a perfectly calibrated 1× physical displacement.",
    },
    {"id": "direct-scale", "type": "table", "layout": "full", "tableId": "direct-scale-table", "sourceId": "direct-scale"},
    {
        "id": "severity-heading",
        "type": "markdown",
        "layout": "full",
        "body": "## Estimated scale is often biologically meaningful but does not imply a correction requirement\n\nMany Luke caches contain 10–40 µm rigid excursions—roughly sub-pitch to two-site shifts on the 20 µm geometry. Rocky includes several >40 µm caches, but those fields mostly lack independent estimator support. Severity should trigger validation and continuity analysis, not automatic resampling.",
    },
    {"id": "severity", "type": "chart", "layout": "full", "chartId": "severity-chart", "sourceId": "history-survey"},
    {
        "id": "methodology",
        "type": "markdown",
        "layout": "full",
        "body": "## Scope and methodology\n\nThe survey inventories 88 historical `motion` directories and 102 saved fields. Exact peak counts and durations are combined with deterministic bounded peak blocks, field-shape metrics, matched estimator correlations, and exact peak windows around 306 selected jumps. A separate Luke factorial adds 185 completed estimates across five prespecified windows, two probes, three algorithms, random count subsets, amplitude composition, synchrony, burst-second, and channel exclusions. A 30-run imec1 threshold audit adds matched thresholds 5/6/7. No voltage correction is applied. The controlled band ablation supplies filtering evidence; the direct raster audit supplies relative-scale evidence; the prior correction decision is used only to set pipeline policy.",
    },
    {
        "id": "limitations",
        "type": "markdown",
        "layout": "full",
        "body": "## Limits and robustness\n\nThe archive contains repeated caches from the same recordings, mixed software/configuration eras, incomplete threshold/filter/bad-channel provenance, and only 11 matched historical multi-method caches from four Luke dates. The threshold audit forms exact nested amplitude subsets, but the original per-channel noise values were not saved; their threshold-5 boundaries are inferred from the full-session minimum accepted amplitudes. It therefore tests raising 5→6→7 well, but cannot reconstruct threshold 4 or independently rerun localization. Peak exclusion cannot undo voltage-level interpolation. Estimators and the direct raster tracker share detected peaks, so their agreement is correlated evidence rather than independent biological ground truth. Cross-probe agreement can reflect real mechanics or shared electrical contamination. Rocky severity remains descriptive until replicated with independent estimators or held-out observables.",
    },
    {
        "id": "policy-heading",
        "type": "markdown",
        "layout": "full",
        "body": "## Estimation should be routine; correction should remain conditional\n\nLuke 2025-08-04 proves that plausible motion information can coexist with harmful voltage resampling. The supported-crop confirmation still failed to beat no motion. That rejects a universal correction default, not estimation or motion-aware tracking as a class.",
    },
    {"id": "policy", "type": "table", "layout": "full", "tableId": "policy-table", "sourceId": "correction-decision"},
    {
        "id": "next-steps",
        "type": "markdown",
        "layout": "full",
        "body": (
            "## Recommended next steps\n\n"
            "1. Persist exact channel noise levels and repeat fresh threshold-5/7 detection and localization in the mixed July Luke and severe Rocky sessions.\n"
            "2. Compare bad-channel exclusion, interpolation, and zero-weight handling before detection/localization, including the neighborhood around the selected dominant channel.\n"
            "3. Add DREDGE confidence/support maps and held-out waveform-family or independent mechanical/LFP anchors to the supported rapid and noise-plus-motion windows.\n"
            "4. Apply the same regime protocol to Rocky >40 µm fields and the mixed July Luke sessions before generalizing across animals.\n"
            "5. Save support/confidence maps and full provenance with every future field; advance to any correction only when uncorrected continuity degradation is demonstrated."
        ),
    },
    {
        "id": "questions",
        "type": "markdown",
        "layout": "full",
        "body": "## Further questions\n\n- Do July 24/30 disagreements reflect different motion, lower time-depth support, or a preprocessing/configuration change?\n- Are Rocky >40 µm fields reproduced by a second estimator and by held-out waveforms or LFP landmarks?\n- Do depth-local DREDGE jumps track firing-rate turnover within the affected band rather than global noise?\n- What uncorrected continuity loss, expressed relative to site pitch and waveform footprint, predicts an actual benefit from correction?",
    },
]

artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "When motion estimates are believable in our Neuropixels recordings",
        "description": "Historical estimation-regime survey separating estimator plausibility from correction benefit.",
        "generatedAt": generated_at,
        "blocks": blocks,
        "cards": [],
        "charts": charts,
        "tables": tables,
        "sources": sources,
    },
    "snapshot": {
        "version": 1,
        "status": "ready",
        "generatedAt": generated_at,
        "datasets": {
            "agreement": records(agreement_dataset),
            "method_summary": records(method_summary),
            "jump_signatures": records(jump_long),
            "severity": records(severity),
            "definitions": records(definitions),
            "factor_evidence": records(factor_evidence),
            "band_fields": records(band_fields),
            "policy": records(policy),
            "regime_evidence": records(regime_evidence),
            "cross_probe_full": records(cross_probe_full),
            "perturbation_evidence": records(
                perturbation_evidence[
                    [
                        "regime",
                        "probe",
                        "condition",
                        "estimator",
                        "rigid_correlation",
                        "nonrigid_correlation",
                    ]
                ]
            ),
            "direct_scale": records(direct_scale),
            "threshold_chart": records(
                threshold_chart[
                    [
                        "regime",
                        "estimator_label",
                        "rigid_correlation_vs_threshold5",
                        "nonrigid_correlation_vs_threshold5",
                        "threshold7_retained_fraction",
                    ]
                ]
            ),
            "threshold_evidence": records(threshold_evidence),
        },
    },
    "sources": [{"id": item["id"], "label": item["label"]} for item in sources],
}

OUTPUT.mkdir(parents=True, exist_ok=True)
ARTIFACT.write_text(json.dumps(artifact, indent=2) + "\n")
print(ARTIFACT)
