"""Build and execute the reader-facing Luke native-rigid comparison notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


def build(output_root: Path) -> Path:
    output_root = output_root.resolve()
    notebook_path = output_root / "native_rigid_vs_motion_off.ipynb"
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Native rigid versus motion off\n\n"
            "## tl;dr\n\n"
            "**The motion decision is unresolved.** Native rigid has fewer clusters and spikes "
            "and a higher aggregate refractory-violation median. Amplitude coverage is inadequate, "
            "and subtracting directional overlap fractions cannot measure continuity improvement. "
            "The original closure verdict is withdrawn; saved motion fields and validated families "
            "need examination before a full-run decision."
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "This notebook reads the checksum-validated compact Group 1 handoff and the locally "
            "recomputed comparison artifacts. The comparison spans the full 10,473-second recording, "
            "with the contracted halo-supported strip and 60-contact scoring interior. Correspondence "
            "uses exclusive 0.5-ms event matching.\n\n"
            "### Key Assumptions\n\n"
            "- Correspondence identifies similar spike trains, not biological ground truth.\n"
            "- Amplitude completeness is non-rankable when fewer than 50% of eligible baseline units "
            "are measurable.\n"
            "- `amplitude_cv_range` replays SpikeInterface 0.102.1 using sorter-native `full_st` "
            "amplitudes; sliding-RP contamination is stored as a fraction."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n\n"
            f"ROOT = Path({str(output_root)!r})\n"
            "summary = json.loads((ROOT / 'summary.json').read_text())\n"
            "assert summary['schema_version'] == 'luke-native-rigid-comparison-v2'\n"
            "comparison = ROOT / 'comparison'\n"
            "guardrails = pd.read_csv(comparison / 'guardrail_summary.csv')\n"
            "matched = pd.read_csv(ROOT / 'matched_interior_unit_metrics.csv')\n"
            "summary['decision']"
        ),
        nbf.v4.new_markdown_cell("## Data\n\n### 1. Verify the comparison population and coverage"),
        nbf.v4.new_code_cell(
            "pd.DataFrame([{\n"
            "    'arm': '12/9 motion off',\n"
            "    'units': summary['comparison_summary']['baseline_units'],\n"
            "    'spikes': summary['comparison_summary']['baseline_spikes'],\n"
            "}, {\n"
            "    'arm': '12/9 native rigid',\n"
            "    'units': summary['comparison_summary']['candidate_units'],\n"
            "    'spikes': summary['comparison_summary']['candidate_spikes'],\n"
            "}])"
        ),
        nbf.v4.new_code_cell(
            "pd.DataFrame([summary['coverage_summary']]).T.rename(columns={0: 'value'})"
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### 2. Compare efficacy and guardrails"),
        nbf.v4.new_code_cell(
            "s = summary['comparison_summary']\n"
            "headline = pd.DataFrame([\n"
            "    {'metric': 'Unit change', 'value': s['candidate_units'] / s['baseline_units'] - 1},\n"
            "    {'metric': 'Spike change', 'value': s['candidate_spikes'] / s['baseline_spikes'] - 1},\n"
            "    {'metric': 'Overlap asymmetry (not efficacy)', 'value': summary['overlap_asymmetry']},\n"
            "]).set_index('metric')\n"
            "headline.style.format({'value': '{:+.1%}'})"
        ),
        nbf.v4.new_code_cell(
            "available = guardrails[guardrails.available].copy()\n"
            "available['change_pp'] = 100 * available['candidate_minus_baseline']\n"
            "available[['metric', 'baseline', 'candidate', 'change_pp']]"
        ),
        nbf.v4.new_code_cell(
            "paired_fields = [\n"
            "    ('sliding_rp_contamination_change', 'Sliding-RP contamination'),\n"
            "    ('amplitude_cv_range_change', 'Amplitude-CV range'),\n"
            "    ('amplitude_cv_change', 'Global amplitude CV'),\n"
            "]\n"
            "rows = []\n"
            "for field, label in paired_fields:\n"
            "    values = matched[field].dropna()\n"
            "    rows.append({'metric': label, 'matched_n': len(values), "
            "'median_change': values.median(), 'fraction_worse': (values > 0).mean()})\n"
            "paired_summary = pd.DataFrame(rows)\n"
            "paired_summary"
        ),
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))\n"
            "arms = ['Motion off', 'Native rigid']\n"
            "axes[0].bar(arms, [s['baseline_units'], s['candidate_units']], color=['#4C78A8', '#F58518'])\n"
            "axes[0].set_title('Curated units')\n"
            "axes[0].set_ylabel('Units')\n"
            "axes[1].bar(arms, np.array([s['baseline_spikes'], s['candidate_spikes']]) / 1e6, color=['#4C78A8', '#F58518'])\n"
            "axes[1].set_title('Curated spike mass')\n"
            "axes[1].set_ylabel('Spikes (millions)')\n"
            "g = available.set_index('metric')['change_pp']\n"
            "axes[2].barh(range(len(g)), g, color=['#E45756' if v > 0 else '#54A24B' for v in g])\n"
            "axes[2].set_yticks(range(len(g)), [x.replace('_', ' ') for x in g.index])\n"
            "axes[2].axvline(0, color='black', linewidth=.8)\n"
            "axes[2].set_title('Rigid minus off guardrails')\n"
            "axes[2].set_xlabel('Percentage points')\n"
            "fig.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "- Native rigid has **66/230 fewer clusters (28.7%)** and **688,167/7,772,209 fewer spikes (8.9%)**. "
            "Cluster-count loss is not measured neuron loss.\n"
            "- There are **40** interior primary matches. Directional overlap fractions differ by "
            "**−6.8 percentage points**, but extra genuine candidate events can cause this sign. "
            "In **27/40** pairs rigid contains more events; the median count ratio is **1.052**.\n"
            "- Only **4/141 eligible baseline units (2.8%)** support the amplitude endpoint, far below "
            "the prospective 50% coverage gate. The observed missingness estimate is therefore not rankable.\n"
            "- Conventional refractory violations worsen by **0.75 percentage points**. Coincidence "
            "excess improves by **0.23 points**. Aggregate guardrails and matched-survivor metrics "
            "do not establish either biological superiority or equivalence.\n"
            "- **Motion diagnostics remain required.** The original automatic closure is withdrawn. "
            "The separate motion-conditioned screen and method-selection report document current "
            "evidence and the missing native motion-field audit."
        ),
    ]
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, notebook_path)
    executed = NotebookClient(nb, timeout=600, kernel_name="python3").execute(
        cwd=str(output_root)
    )
    nbf.write(executed, notebook_path)
    return notebook_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.output_root))


if __name__ == "__main__":
    main()
