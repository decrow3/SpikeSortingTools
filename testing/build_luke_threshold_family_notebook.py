"""Build and execute the reader-facing threshold-family diagnostic notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis_dir = args.analysis_dir.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.cells = [
        nbf.v4.new_markdown_cell(
            """## tl;dr

Across 50 cycle-consistent families and four learned-threshold transitions, lowering `Th_learned` adds 23–28% of the candidate cohort's spike mass. Roughly 81–88% of added spikes fall below the matched baseline unit's 25th amplitude percentile, but 35–38% are near-coincident with another candidate unit versus a 7–9% circular-shift null. Added fraction strongly tracks refractory worsening and amplitude-CV worsening, but not increasing unit-level coincidence. The added spikes almost never occupy baseline-empty 300 s bins, so this cohort provides no evidence of longitudinal recovery. The observed signature is best described as **low-amplitude, duplicate-prone expansion of recognizable spike trains**, not clean recovery and not heterogeneous waveform broadening."""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

The cohort contains the 50 reciprocal-primary, cycle-consistent unit families with more than 2,000 curated spikes in every one of seven threshold arms. The four comparisons lower only `Th_learned`: `12/10→12/9`, `10/10→10/9`, `10/9→10/8`, and `9/9→9/8`.

### Key Assumptions

- Correspondence is spike-train identity evidence, not biological ground truth.
- Core events are exclusive chronological matches within 0.5 ms; candidate-only events are the remainder.
- Chance-aware coincidence marks spikes within 0.5 ms and 75 µm of another unit, then subtracts a deterministic cluster-wise circular-shift null.
- Amplitude uses the production sorter-native `full_st[kept_spikes][:, 2]` measure.
- Detection-template composition is a cheap within-candidate proxy and is not raw-waveform correlation."""
        ),
        nbf.v4.new_code_cell(
            f"""from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

analysis_dir = Path({str(analysis_dir)!r})
metrics = pd.read_csv(analysis_dir / 'family_transition_metrics.csv')
summary = pd.read_csv(analysis_dir / 'transition_summary.csv')
correlations = pd.read_csv(analysis_dir / 'added_fraction_correlations.csv')
classes = pd.read_csv(analysis_dir / 'signature_classification.csv')
receipt = json.loads(Path({str(analysis_dir.parent / 'luke_threshold_family_diagnostic_v3.receipt.json')!r}).read_text())

assert receipt['state'] == 'complete' and receipt['returncode'] == 0
assert len(metrics) == 200 and metrics.family_id.nunique() == 50
assert metrics.transition.nunique() == 4
assert int(metrics.candidate_only_spikes.sum()) == 2_984_983
print('Validated:', len(metrics), 'family-transition rows; added spikes:', f"{{metrics.candidate_only_spikes.sum():,}}")"""
        ),
        nbf.v4.new_markdown_cell("## Added spikes are predominantly low-amplitude but coincidence-heavy"),
        nbf.v4.new_code_cell(
            """show = summary[[
    'transition', 'candidate_only_spikes', 'aggregate_added_spike_fraction',
    'added_near_other_baseline_unit_fraction',
    'added_candidate_coincidence_observed_fraction',
    'added_candidate_coincidence_null_fraction',
    'added_amplitude_below_baseline_q25_fraction',
    'added_in_baseline_empty_bins_fraction',
]].copy()
display(show.style.format({
    'candidate_only_spikes': '{:,.0f}',
    'aggregate_added_spike_fraction': '{:.1%}',
    'added_near_other_baseline_unit_fraction': '{:.1%}',
    'added_candidate_coincidence_observed_fraction': '{:.1%}',
    'added_candidate_coincidence_null_fraction': '{:.1%}',
    'added_amplitude_below_baseline_q25_fraction': '{:.1%}',
    'added_in_baseline_empty_bins_fraction': '{:.3%}',
}))

plot = show.set_index('transition')[
    ['added_near_other_baseline_unit_fraction',
     'added_candidate_coincidence_observed_fraction',
     'added_candidate_coincidence_null_fraction',
     'added_amplitude_below_baseline_q25_fraction']
]
ax = plot.plot.bar(figsize=(11, 5), color=['#355C7D', '#C06C84', '#B5B5B5', '#D6B35A'])
ax.set_ylabel('Fraction of candidate-only spikes')
ax.set_xlabel('Learned-threshold transition')
ax.set_title('Candidate-only spike signatures')
ax.set_ylim(0, 1)
ax.grid(axis='y', color='#DDDDDD', linewidth=0.6)
ax.legend(['Near another baseline unit', 'Observed candidate coincidence', 'Shift-null coincidence', 'Below baseline amplitude Q25'], frameon=False, ncol=2)
plt.xticks(rotation=0)
plt.show()"""
        ),
        nbf.v4.new_markdown_cell("## More added spikes track refractory and amplitude-CV degradation"),
        nbf.v4.new_code_cell(
            """pooled = correlations[correlations.transition == 'pooled_within_transition_centered'].copy()
display(pooled[['outcome', 'n', 'spearman_rho']].style.format({'spearman_rho': '{:+.2f}'}))

outcomes = [
    ('refractory_change', 'RV fraction change'),
    ('unit_coincidence_excess_change', 'Chance-aware coincidence change'),
    ('amplitude_cv_change', 'Amplitude CV change'),
    ('presence_change', 'Presence fraction change'),
]
colors = {'12/10->12/9':'#355C7D', '10/10->10/9':'#C06C84', '10/9->10/8':'#F67280', '9/9->9/8':'#F8B195'}
fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
for ax, (field, label) in zip(axes.flat, outcomes):
    for transition, frame in metrics.groupby('transition', sort=False):
        ax.scatter(frame.added_spike_fraction, frame[field], s=22, alpha=.72, color=colors[transition], label=transition)
    ax.axhline(0, color='#333333', linewidth=.8)
    ax.set_xlabel('Added candidate spike fraction')
    ax.set_ylabel(label)
    ax.grid(color='#DDDDDD', linewidth=.6)
axes.flat[0].legend(frameon=False, fontsize=8)
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## Limitations, classification, and robustness

The strict descriptive rule finds no recovery-compatible family-transition observations: none has a material longitudinal gain. However, presence is ceiling-limited—47–50 of the 50 baseline families are already present in every 300 s bin—so the result is specifically **no demonstrated longitudinal recovery in this high-count cohort**, not proof that recovery is impossible elsewhere.

The added spikes are temporally distributed similarly, though not identically, to baseline/core spikes (median Jensen–Shannon divergence 0.056–0.071). Their detection-template IDs are usually drawn from the same within-candidate template set (median core-seen fraction 100%; median divergence 0.027–0.055). This argues against gross waveform heterogeneity, but raw waveform similarity was not available cheaply enough for per-event attribution.

The aggregate coincidence calculations reconcile exactly to the saved threshold-grid outputs. Completeness remains selected: only 21–23 of 50 families per transition are measurable in both arms, and added fraction has only weak association with completeness change."""
        ),
        nbf.v4.new_code_cell(
            """display(classes.pivot(index='transition', columns='signature', values='families').fillna(0).astype(int))
display(summary[['transition', 'median_added_vs_baseline_time_js_divergence', 'median_template_core_seen_fraction', 'median_template_js_divergence']].style.format({
    'median_added_vs_baseline_time_js_divergence': '{:.3f}',
    'median_template_core_seen_fraction': '{:.1%}',
    'median_template_js_divergence': '{:.3f}',
}))"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. Lower learned thresholds expand recognizable spike trains rather than merely creating tiny or unmatched clusters.
2. The additions are mostly low-amplitude and template-consistent, so they superficially resemble sensitivity recovery.
3. They do not add meaningful longitudinal support in this cohort and are heavily enriched for cross-unit near-coincidence.
4. Added fraction monotonically tracks refractory worsening across all four backgrounds and tracks amplitude-CV worsening overall.
5. Unit-level coincidence worsening is widespread but not proportional to added fraction, suggesting a broad duplicate-assignment effect rather than one confined to the largest gainers.

The threshold-axis conclusion is therefore stable: retain `12/9`. Lower `Th_learned` increases permissiveness without demonstrated biological recovery, and this bounded diagnostic does not identify a substantial clean-recovery subgroup."""
        ),
    ]
    client = NotebookClient(nb, timeout=180, kernel_name="python3", allow_errors=False)
    client.execute(cwd=str(Path.cwd()))
    nbf.write(nb, output)
    print(output)


if __name__ == "__main__":
    main()
