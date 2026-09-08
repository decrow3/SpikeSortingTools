# Shallow lighthouse qualification before transition tracking

**Unit 154 passed the frozen, bounded quiet/injection gate; unit 80 failed sensitivity and must not be used by the new translated tracker.** Neither outcome certifies biological identity or motion accuracy. Unit 80's failure is not evidence that it is non-neural or that false neighboring identities dominated its matches.

The prior fixed matcher retained only one rival per target after an unshifted similarity cutoff, despite many nearby labeled templates. The new audit removes that similarity pruning, reconstructs compensated broadband training waveforms, and includes independently detected unlabeled waveform families. It compares every retained identity/shift hypothesis with common weights and separates other-identity margins from other-shift margins.

| Frozen qualification measure | Unit 80, 220 µm | Unit 154, 620 µm |
|---|---:|---:|
| Nearby labeled identities | 43 | 25 |
| Retained independent waveform families | 12 | 12 |
| Identity/shift hypotheses | 198 | 158 |
| Quiet held-out reference events | 160 | 35 |
| Independently detected reference-event recall | 53.75% | 60.00% |
| Accepted quiet events unmatched to provisional labels | 0% | 0% |
| Real-background injected target recovery | 65.33% | 92.67% |
| Injected target recovery with unique correct shift | 65.33% | 92.67% |
| Tested rival injections accepted as target | 0% | 0% |
| Worst tested rival identity's false-target fraction | 0% | 0% |
| Frozen gate outcome | Fail; no tracking | Eligible for bounded transition audit |

Training uses 4080–4100 s with interleaved independent-half templates. Holdout uses 4100–4110 s. Injections add independent-half waveforms to real held-out voltage backgrounds at exact −80, −40, 0, +40, +80 µm shifts, with target amplitude gains 0.75, 1, and 1.25. Match gates were fixed before execution: cosine ≥0.8, gain 0.4–2.5, identity margin ≥0.03; unique shift additionally requires shift margin ≥0.03. Passing requires ≥90% target injection recovery and ≥90% unique correct-shift recovery, plus quiet support/recall and ≤1% pooled and worst-rival false-target fractions.

Unit 80 misses the injection sensitivity gate substantially. Common weights over a spatially expanded search can reduce sensitivity; weak/noisy injected events and conservative competing families can also cause rejection. The present result does not isolate which explanation dominates. With zero accepted rival controls, it would be incorrect to describe this failure as demonstrated identity confusion. Conversely, zero observed errors on finitely many rival injections does not establish a zero biological false-positive rate.

For unit 154, only 35 quiet reference events support the 60% recall estimate. The independent family clusters are conservative waveform competitors, not a complete local cell inventory. Duplicate or overlapping provisional labels and unmodeled transition-specific waveforms remain possible. Exact 40 µm translations do not calibrate fractional movement or guarantee recovery through the disputed transition.

The authorized next step is **unit 154 alone, 4240–4260 s**, using the frozen bank and detector support, no motion-field input to matching, explicit missing/mixed-shift bins, and only post-selection DREDGE comparisons. A clean accepted population can corroborate that location; missing events cannot establish zero motion. Unit 80 remains excluded.

Artifacts: `testing/outputs/luke_shallow_identity_audit_v3/qualification.csv`, full quiet/injection score matrices, per-identity/shift controls, and hashed settings. The managed audit completed with exit 0. No transition or session-wide tracking was run by the qualification script.
