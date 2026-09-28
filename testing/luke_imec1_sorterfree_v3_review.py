#!/usr/bin/env python
"""Cached corrected review renderer for the completed imec1 v3 discovery."""
from pathlib import Path

import numpy as np
import pandas as pd

from testing.luke_imec1_dots_sorterfree_waveform_discovery import plot_candidate_review

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3"
OUTPUT = ROOT / "testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3_review"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    families = pd.read_csv(SOURCE / "families_after_depth_reveal.csv")
    # The shared v2 renderer prints this legacy column. Display the v3 global
    # supported rival while retaining both original columns in the source CSV.
    families["nearest_rival_cosine"] = families.global_rival_cosine
    with np.load(SOURCE / "family_templates.npz") as saved:
        templates = np.asarray(saved["waveforms"])
    seed = pd.read_csv(SOURCE / "seed_cluster_events.csv")
    heldout = pd.read_csv(SOURCE / "heldout_events.csv")
    plot_candidate_review(OUTPUT, families, templates, seed, heldout, fs=29999.835983263598)
    (OUTPUT / "README.md").write_text(
        "Cached rerender of completed v3 evidence. Candidate pages display the global "
        "supported rival cosine; no extraction, matching, selection, or depth calculation "
        "was changed. Numerical source remains the immutable v3 output directory.\n"
    )


if __name__ == "__main__":
    main()
