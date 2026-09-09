# Cached expansion of waveform-only lighthouse candidates

The user requested more cells after reviewing the five-cell peak-scatter overlays. The cheapest direct check reused the247-identity cached whole-probe score pass rather than reading voltage or launching a sort.

Script: `testing/luke_waveform_only_expansion_v1.py`. Output: `testing/outputs/luke_waveform_only_expansion_v1/`.

Selection used only930–940s. Unchanged waveform seed gates require repeatability≥0.85,SNR≥5,local energy fraction≥0.35,far-peak ratio≤0.8 and global closest-rival cosine<0.95. Candidate-level recovery permits cosine≥0.80 and gain0.35–3, including ambiguous winners, requiring at least5 and20% of cached seed events recovered within0.3ms. There are17 candidates: original5;8 additional candidates meeting recovery with identity margin≥0.025;4 additional candidates requiring ambiguity to meet recovery. Selection was saved before reading held-out events for the expanded cohort. No depth quotas, motion agreement or peak background were used.

Original strict event decisions remain unchanged: filled blue marks retain original accepted status (cosine≥0.86,margin≥0.025). Open blue marks show additional exploratory support at cosine0.80–0.86 with sufficient margin. Orange crosses preserve margin failures at cosine≥0.80. The old five-cell outputs are untouched. This is17 candidate templates, not17 certified independent identities; lookalikes and duplicate families remain possible.

Deliverables:01_candidate_overview.pdf (3 pages, original peak background),02_all_17_candidate_overlays.pdf (17 pages, original and compensated backgrounds), per-cell PNGs, candidate_audit.csv,heldout_counts.csv,overlay_events.csv,settings and provenance manifest. All peak scatter points are rendered without subsampling. Only strict same-patch segments with gaps≤2s are connected. Additional evidence is shown as points without imposing a trajectory.
