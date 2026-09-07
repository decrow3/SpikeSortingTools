"""Build and execute the reproducible threshold-grid analysis notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ANALYSIS_ROOT = Path(
    "/media/huklab/Data/SpikeSortingTools/luke0804_hindsight_ladder_v1/threshold_grid_analysis_v1"
)


def build(output: Path) -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Luke threshold-grid interaction analysis

## tl;dr

- Lowering the learned threshold increases total spike yield by 31.5–34.2% at universal thresholds 9, 10, and 12, so the large 9/8 yield is not a unique nonlinear interaction.
- Every learned-threshold reduction increases chance-aware coincident-spike excess by 1.46–2.26 percentage points, above the prospective 1-point guardrail.
- Amplitude-completeness coverage is only 2.0–4.8% on every edge, so that endpoint cannot rank settings.
- Strict path-consistent unit matching supports zero quartets in the lower cell and two in the upper cell. Unit-level interactions are therefore not identified.
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

The analysis compares the two completed 2×2 cells using the full 10,473.55-second session and the fixed 1600–2180 µm scoring interior. Each edge changes exactly one Kilosort threshold. Pairwise correspondence uses reciprocal primary spike-train matches; the stricter quartet diagnostic additionally requires degree-one correspondence and agreement around both paths through a cell.

### Key Assumptions

Group 1, Group 2, and Group 3 are comparable only because their recording request/content hashes, geometry, clock, production lock, curation profile, and evaluation definitions were validated as identical. Results are descriptive for this session and do not establish biological identity or causal generalization.
"""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ANALYSIS_ROOT = Path('/media/huklab/Data/SpikeSortingTools/luke0804_hindsight_ladder_v1/threshold_grid_analysis_v1')
summary = json.loads((ANALYSIS_ROOT / 'summary.json').read_text())
pairs = pd.read_csv(ANALYSIS_ROOT / 'pairwise_summary.csv')
quartets = pd.read_csv(ANALYSIS_ROOT / 'strict_consistent_quartets.csv')
provenance = json.loads((ANALYSIS_ROOT / 'provenance_validation.json').read_text())
assert provenance['inventory']['file_count'] == 50
assert pairs.shape[0] == 8
assert set(pairs.coverage_status) == {'infeasible_insufficient_coverage'}
print('analysis digest:', summary['analysis_digest'])
print('validated handoff files:', provenance['inventory']['file_count'])
"""
        ),
        nbf.v4.new_markdown_cell("## Data\n\nThe edge table retains yield, correspondence, coverage, and guardrail changes. Positive changes are candidate-minus-baseline; every candidate lowers exactly one threshold."),
        nbf.v4.new_code_cell(
            """display(pairs[['edge', 'baseline', 'candidate', 'spike_change_pct', 'primary_matches',
               'median_jaccard', 'measurable_units', 'measurable_fraction',
               'refractory_change', 'coincidence_excess_change',
               'edge_unit_fraction_change', 'edge_spike_fraction_change']].round(4))"""
        ),
        nbf.v4.new_markdown_cell("## Results\n\nStrict degree-one matching is summarized separately from the reciprocal primary matches because split/merge ambiguity is itself decision-relevant."),
        nbf.v4.new_code_cell(
            """rate_rows = []
for comparison_dir in sorted((ANALYSIS_ROOT / 'comparisons').iterdir()):
    primary = pd.read_csv(comparison_dir / 'primary_matches.csv')
    edges = pd.read_csv(comparison_dir / 'correspondence_edges.csv')
    baseline = pd.read_csv(comparison_dir / 'unit_metrics_baseline.csv')
    candidate = pd.read_csv(comparison_dir / 'unit_metrics_candidate.csv')
    baseline_degree = edges.groupby('baseline_cluster').candidate_cluster.nunique()
    candidate_degree = edges.groupby('candidate_cluster').baseline_cluster.nunique()
    strict = primary[primary.interior_primary.astype(bool)].copy()
    strict = strict[
        strict.baseline_cluster.map(baseline_degree).eq(1)
        & strict.candidate_cluster.map(candidate_degree).eq(1)
    ]
    strict = strict.merge(
        baseline[['cluster_id', 'mean_rate_hz']], left_on='baseline_cluster', right_on='cluster_id'
    ).merge(
        candidate[['cluster_id', 'mean_rate_hz']], left_on='candidate_cluster',
        right_on='cluster_id', suffixes=('_baseline', '_candidate')
    )
    effect = np.log2(strict.mean_rate_hz_candidate / strict.mean_rate_hz_baseline)
    rate_rows.append({
        'edge': comparison_dir.name,
        'strict_pairs': len(effect),
        'median_log2_rate_change': effect.median(),
        'q25': effect.quantile(.25),
        'q75': effect.quantile(.75),
        'fraction_higher': (effect > 0).mean() if len(effect) else np.nan,
    })
rate_summary = pd.DataFrame(rate_rows)
display(rate_summary.round(4))
display(pd.DataFrame(summary['interaction_cells']).round(4))
"""
        ),
        nbf.v4.new_code_cell(
            """labels = [f'{row.baseline}→{row.candidate}' for row in pairs.itertuples()]
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
axes[0].barh(labels, pairs.spike_change_pct, color='#3b82a0', edgecolor='#263238')
axes[0].axvline(0, color='#263238', linewidth=1)
axes[0].set_title('Total spike-yield change')
axes[0].set_xlabel('candidate minus baseline (%)')
axes[0].invert_yaxis()
axes[1].barh(labels, 100 * pairs.coincidence_excess_change, color='#d18b2c', edgecolor='#263238')
axes[1].axvline(1.0, color='#263238', linestyle='--', label='prospective +1 pp limit')
axes[1].axvline(0, color='#263238', linewidth=1)
axes[1].set_title('Chance-aware coincident-spike excess')
axes[1].set_xlabel('candidate minus baseline (percentage points)')
axes[1].invert_yaxis()
axes[1].legend(frameon=False, loc='lower right')
fig.suptitle('Threshold-edge outcomes, full session')
fig.tight_layout()
plt.show()
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. The learned threshold is the dominant yield lever. Its effect is repeatable across universal-threshold settings, not peculiar to 9/8.
2. The extra yield arrives with a repeatable duplicate/coincidence guardrail breach and extensive correspondence ambiguity. It should not be interpreted as recovered biological signal without further evidence.
3. Universal-threshold reductions add only −0.1% to +1.7% spikes, while 12→10 increases boundary burden. There is no compelling efficacy case for lowering it.
4. Retain 12/9 motion-off as the provisional threshold setting. Motion correction remains a separate unresolved question because its amplitude endpoint also failed coverage.
"""
        ),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(notebook, timeout=300, kernel_name="python3")
    executed = client.execute(cwd=str(ROOT if (ROOT := Path.cwd()).exists() else Path.cwd()))
    nbf.write(executed, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
