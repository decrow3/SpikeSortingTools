"""Build the executed companion notebook for the historical estimation survey."""

from pathlib import Path

import nbformat as nbf


OUTPUT = Path("testing/outputs/motion_estimation_history_survey")
NOTEBOOK = OUTPUT / "motion_estimation_history_survey.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
notebook["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
notebook["cells"] = [
    markdown(
        """
# Historical motion-estimation regime survey

## tl;dr

- The saved archive supports **real, repeatable rigid motion in some Luke recordings**, but not a universal assumption that every large field is correct. DREDGE and decentralized rigid traces reach absolute correlation ≥0.65 in 7 of 11 matched caches, concentrated in the 2025-08-04 and 2025-08-05 sessions; 2025-07-24 and 2025-07-30 are mixed or discordant.
- **Kilosort-style estimates are the clearest hallucination/discontinuity risk** in these caches: they broadly track DREDGE, but often add coherent whole-depth jumps of 100–281 µm that are not reproduced at the same scale by the other estimators.
- **Peak count helps conditioning but is not a validity test.** Across saved DREDGE fields, higher peak rate is associated with smaller 99th-percentile steps, yet many large local jumps occur without a peak-rate or amplitude burst. A large number of peaks can still be dominated by unstable activity or artifacts.
- Filtering from 300–3000 Hz to 300–6000 Hz changed the Luke peak population by 24%, but the DREDGE rigid and residual fields remained correlated at 0.85 and 0.90. This is positive robustness evidence, while held-out raster agreement remained too weak to validate absolute scale.
- The present evidence supports **estimation as an always-available diagnostic, not motion correction as a default transform**. Correction should remain off until an estimate passes independent validation and uncorrected waveform/unit continuity demonstrably worsens with motion in a replicated session.
"""
    ),
    markdown(
        """
## Context & Methods

This notebook surveys cached estimation outputs only. It does not apply displacement fields, resample voltage, or use unit yield to score estimator quality. Downstream correction evidence is used only for the final pipeline-default decision.

### Key assumptions

- Saved `peaks.npy` arrays are time-sorted and use approximately 30 kHz sample indices.
- Cross-method agreement supports reproducibility, not ground-truth accuracy.
- Historical peak threshold, noise exclusion, and preprocessing parameters are treated as unknown unless an explicit manifest exists.
- Repeated test caches from one session are not independent biological replications.
"""
    ),
    code(
        """
from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display, Image

candidate_roots = [Path.cwd(), *Path.cwd().parents]
REPO_ROOT = next(
    root for root in candidate_roots
    if (root / "testing/outputs/motion_estimation_history_survey/run_inventory.csv").exists()
)
OUTPUT = REPO_ROOT / "testing/outputs/motion_estimation_history_survey"
runs = pd.read_csv(OUTPUT / "run_inventory.csv")
methods = pd.read_csv(OUTPUT / "method_metrics.csv")
agreement = pd.read_csv(OUTPUT / "cross_method_agreement.csv")
jumps = pd.read_csv(OUTPUT / "motion_jump_peak_context.csv")
manifest = json.loads((OUTPUT / "manifest.json").read_text())
manifest
"""
    ),
    markdown("## Data\n\nThe survey contains exact cache counts and bounded deterministic peak profiles."),
    code(
        """
inventory_summary = pd.DataFrame({
    "measure": ["motion directories", "saved fields", "matched multi-method caches", "caches with peaks", "caches with localized peaks"],
    "value": [len(runs), len(methods), int((runs.method_count >= 2).sum()), int(runs.has_peaks.sum()), int(runs.has_peak_locations.sum())],
})
display(inventory_summary)
display(runs.estimate_support_class.value_counts().rename_axis("support class").to_frame("caches"))
"""
    ),
    markdown(
        """
## Results

### DREDGE and decentralized estimates provide the strongest available consensus

The comparison below uses matched caches estimated from the same detected/localized peaks. MEDiCINe is intentionally not required for consensus because its historical configuration used only two depth bins and a 50-second temporal kernel, making it a much smoother observable.
"""
    ),
    code(
        """
dredge_dc = agreement[(agreement.left_method == "decentralized") & (agreement.right_method == "dredge")].copy()
dredge_dc = dredge_dc[["session", "pipeline_name", "rigid_correlation", "median_absolute_difference_um", "right_on_left_slope"]]
dredge_dc = dredge_dc.sort_values("rigid_correlation", ascending=False)
display(dredge_dc.round(3))
print("Caches at |r| >= 0.65:", int((dredge_dc.rigid_correlation.abs() >= 0.65).sum()), "/", len(dredge_dc))
"""
    ),
    code(
        """
display(Image(filename=str(OUTPUT / "cross_method_agreement_distribution.png")))
"""
    ),
    markdown(
        """
### Algorithms fail differently

Kilosort-style fields are spatially coherent but temporally discontinuous; DREDGE's largest steps are usually confined to a small subset of depth bins; decentralized estimates are more variable and larger in scale; historical MEDiCINe fields are strongly smoothed. These are algorithm signatures, not a single quality ranking.
"""
    ),
    code(
        """
method_summary = methods.groupby("method").agg(
    fields=("method", "size"),
    median_rigid_excursion_um=("rigid_excursion_p95_p5_um", "median"),
    median_nonrigid_spread_um=("median_nonrigid_spread_um", "median"),
    median_p99_step_um=("p99_abs_step_um", "median"),
    median_largest_to_p99_step=("max_to_p99_step_ratio", "median"),
).round(2)
display(method_summary)
display(Image(filename=str(OUTPUT / "abrupt_estimate_flags_by_method.png")))
"""
    ),
    markdown(
        """
### Large estimate steps are usually not simple amplitude-noise bursts

For each saved field, the three largest temporally separated steps were matched to exact cached-peak windows. “Large” below means an absolute displacement step of at least 50 µm among those selected events. Because this is a top-event sample, percentages describe failure signatures, not their population incidence.
"""
    ),
    code(
        """
large = jumps[jumps.max_step_um.abs() >= 50].copy()
jump_summary = []
for method, group in large.groupby("method"):
    jump_summary.append({
        "method": method,
        "large_top_events": len(group),
        "fields": group.motion_dir.nunique(),
        "median_depth_fraction": group.coherent_depth_fraction.median(),
        "peak_rate_burst_fraction": (group.event_to_context_peak_rate_ratio > 1.5).mean(),
        "amplitude_tail_burst_fraction": (group.event_to_context_amp_p99_ratio > 1.5).mean(),
        "near_baseline_rate_fraction": group.event_to_context_peak_rate_ratio.between(0.7, 1.3).mean(),
    })
display(pd.DataFrame(jump_summary).set_index("method").round(3))
"""
    ),
    code(
        """
display(Image(filename=str(OUTPUT / "peak_support_vs_estimator_stability.png")))
"""
    ),
    markdown(
        """
### Peak volume appears stabilizing, but threshold and noise-exclusion effects remain unidentified

The historical cache survey shows a negative Spearman association between peak rate and the DREDGE 99th-percentile step. This is compatible with better statistical support at higher peak counts, but it is confounded by animal, session, probe, preprocessing version, and duplicate caches. The old pipeline did not consistently save `detect_threshold`, radius, excluded channels, or filtering parameters, so the archive cannot answer a causal threshold question.
"""
    ),
    code(
        """
factor_rows = []
for method, group in methods.groupby("method"):
    factor_rows.append({
        "method": method,
        "fields": len(group),
        "spearman_peak_rate_vs_p99_step": group.peak_rate_hz.corr(group.p99_abs_step_um, method="spearman"),
        "spearman_rate_variability_vs_isolated_jump": group.peak_rate_cv_10s.corr(group.max_to_p99_step_ratio, method="spearman"),
    })
display(pd.DataFrame(factor_rows).set_index("method").round(3))
"""
    ),
    markdown(
        """
### The controlled filtering comparison is robust in direction but not absolute scale

In the 2025-08-04 imec1 60-second band ablation, the 300–3000 Hz branch produced 244,805 peaks versus 198,019 for 300–6000 Hz. Despite that 23.6% increase, the two DREDGE estimates had rigid correlation 0.852 and residual correlation 0.896. The narrower branch increased rigid excursion from 9.57 to 10.23 µm and nonrigid spread from 29.18 to 33.72 µm. However, held-out raster correlations were only +0.219 and −0.173, so robustness to filtering does not establish calibrated displacement.
"""
    ),
    code(
        """
band_root = Path("/mnt/NPX/Luke/20250804/dredge_pipeline_results_Luke0804_V2V1_g0_imec1/motion_estimator_band_ablation")
band_fields = pd.read_csv(band_root / "field_summary.csv")
band_raster = pd.read_csv(band_root / "heldout_raster_summary.csv")
display(band_fields.round(3))
display(band_raster.round(3))
"""
    ),
    markdown(
        """
### Estimated severity is common, but correction-worthiness is not implied

Using DREDGE's p95–p5 rigid excursion as a descriptive scale, many Luke caches fall between 10 and 40 µm; none exceed 40 µm. Rocky includes several >40 µm caches, but most lack an independent estimator and therefore cannot establish that those fields are real. These cache counts are repeated analyses, not independent sessions.
"""
    ),
    code(
        """
dredge = methods[methods.method == "dredge"].copy()
bands = [-np.inf, 10, 20, 40, np.inf]
labels = ["<10 µm", "10–20 µm", "20–40 µm", ">40 µm"]
dredge["rigid_excursion_band"] = pd.cut(dredge.rigid_excursion_p95_p5_um, bins=bands, labels=labels)
display(pd.crosstab(dredge.subject, dredge.rigid_excursion_band))
"""
    ),
    markdown(
        """
## Takeaways

1. Treat a motion estimate as supported only when it survives an **independence ladder**: adequate peak support → split-half stability → nearby parameter/filter stability → cross-method or cross-probe agreement → held-out raster/waveform evidence.
2. Flag coherent whole-depth single-bin jumps, depth-edge-only jumps, cap/boundary plateaus, and fields coupled to peak-rate/common-mode bursts. None is automatically “motion” or “artifact”; each specifies the next check.
3. Add explicit provenance to every future estimate: detection threshold/radius, peak count by time-depth bin, excluded/interpolated channels, filter/reference graph, estimator parameters, software versions, and support/confidence outputs.
4. Keep motion correction **off by default**. Run estimation as a diagnostic sidecar, and enable correction only for a prespecified session when the estimate is independently supported and the uncorrected data show replicated motion-linked waveform or unit-family degradation that the candidate correction improves.
5. The next decisive experiment is a factorial estimation-only sweep over peak threshold, bad-channel handling, and noise masking on quiet, rapid-motion, sustained-noise, and noise-plus-motion windows—with split peaks and held-out observables frozen before estimation.
"""
    ),
]

OUTPUT.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, NOTEBOOK)
print(NOTEBOOK)
