"""Generate the reproducible Luke motion-input-factorial notebook."""

from pathlib import Path

import nbformat as nbf


OUTPUT = Path("testing/outputs/luke_motion_input_factorial")
NOTEBOOK = OUTPUT / "luke_motion_input_factorial.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
notebook["cells"] = [
    markdown(
        """
# Luke motion-estimation input factorial

This notebook audits when motion is identifiable from cached threshold-5 peaks. It never applies a motion field and never reads sorter labels.

Main result: rapid motion and noise-plus-motion pass method, perturbation, and simultaneous-probe checks; support dropout does not. Noise masks preserve the supported rigid trajectory but can change the local nonrigid residual.
"""
    ),
    code(
        """
from pathlib import Path
import pandas as pd
from IPython.display import display, Image

repo = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'testing').is_dir()), None)
if repo is None:
    raise RuntimeError('Could not locate repository root')
root = repo / 'testing/outputs/luke_motion_input_factorial'
windows_root = repo / 'testing/outputs/luke_motion_regime_windows'
summary = pd.read_csv(root / 'factorial_field_summary.csv')
agreement = pd.read_csv(root / 'factorial_agreement.csv')
evidence = pd.read_csv(root / 'regime_evidence.csv')
windows = pd.read_csv(windows_root / 'selected_windows.csv')
summary.shape, agreement.shape
"""
    ),
    markdown("## Prespecified discovery regimes"),
    code(
        """
display(windows[['regime', 'start_s', 'duration_s', 'selection_basis']])
display(Image(filename=str(windows_root / 'selected_regimes.png')))
"""
    ),
    markdown("## Evidence classification"),
    code("display(evidence)"),
    code("display(Image(filename=str(root / 'regime_rigid_traces.png')))"),
    markdown(
        """
The rigid trajectories are centered independently for display. Similar shape across DREDGE/decentralized and probes is a reproducibility check, not proof that both probes contain identical biological motion.
"""
    ),
    markdown("## Input perturbation robustness"),
    code("display(Image(filename=str(root / 'input_perturbation_robustness.png')))"),
    code(
        """
condition_vs_full = agreement[
    agreement.scope.eq('condition_vs_full')
    & agreement.probe.eq('imec1')
    & agreement.left_estimator.isin(['dredge_300_200_cpu', 'decentralized_300_200_numpy'])
]
display(condition_vs_full.pivot_table(
    index=['regime', 'condition'],
    columns='left_estimator',
    values=['rigid_correlation', 'nonrigid_correlation'],
).round(3))
"""
    ),
    markdown("## Simultaneous-probe support"),
    code("display(Image(filename=str(root / 'cross_probe_regime_support.png')))"),
    code(
        """
cross_probe = agreement[agreement.scope.eq('cross_probe') & agreement.condition.eq('full')]
display(cross_probe[['regime', 'left_estimator', 'rigid_correlation', 'nonrigid_correlation']].round(3))
"""
    ),
    markdown("## Peak-count and exclusion doses"),
    code(
        """
selection = summary[
    summary.estimator.eq('dredge_300_200_cpu')
    & summary.probe.eq('imec1')
    & summary.condition.isin([
        'full', 'random_half', 'random_quarter', 'high_amplitude_half',
        'exclude_synchronous', 'exclude_bursty_seconds', 'exclude_dominant_channel'
    ])
][['regime', 'condition', 'selected_peaks', 'retained_fraction',
   'rigid_excursion_p95_p5_um', 'median_nonrigid_spread_um']]
display(selection.sort_values(['regime', 'condition']).round(3))
"""
    ),
    markdown("## Relative physical-scale check"),
    code(
        """
direct_scale = pd.read_csv(repo / 'testing/outputs/luke_direct_motion_scale_audit/scale_summary.csv')
display(direct_scale.round(3))
"""
    ),
    markdown(
        """
## Interpretation limits

- These runs start from historical threshold-5 localized peaks. The high-amplitude half is not a true detection-threshold rerun.
- Peak exclusion does not undo upstream voltage interpolation or change localization.
- DREDGE, decentralized, iterative-template, and the direct raster audit share event observations; agreement is correlated evidence.
- Cross-probe agreement can reflect shared mechanical motion or shared electrical contamination.
- A correction decision still requires independent evidence that uncorrected waveform/unit-family continuity degrades and that a safe transform improves it.
"""
    ),
]

OUTPUT.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, NOTEBOOK)
print(NOTEBOOK)
