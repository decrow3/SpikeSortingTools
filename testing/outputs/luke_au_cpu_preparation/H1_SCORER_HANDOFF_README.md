# AZ h1 scorer/generator handoff

Status: **scorer qualified; donor-dependent assets pending.** No arm output was
seen when the scorer was frozen. This packet does not authorize a sort or a new
voltage extraction.

The scorer performs full-train donor-to-label association before episode/rest
splitting, then global exclusive matching at inclusive ±12 samples using the
actual 29,999.759166666667 Hz clock. It preserves pre-exclusivity duplicate
candidates, ambiguity, label reuse/false-merge evidence and seeded circular-
shift controls. Native output label names are not treated as identities.

`luke_au_cpu_preparation.py` supplies the deterministic independent 5 Hz,
3 ms-refractory generator primitive and exact-lattice helpers. It is not a
sealed donor bank: templates, donor IDs, relocation placements, event arrays
and arm outputs remain pending a successful authorized donor extraction and
the coordinator's complete scientific gates.

The intended template domain is post-ibllikecmr standardized float32 with no
spatial whitening. A future worker must inject in that same domain; this is not
an equivalence claim for the accepted spatially whitened template bank. The
injection is applied exactly once before arm branching; a second ibllikecmr or
spatial-whitening pass is forbidden, and waveform units plus shared-source
identity must be validated before GPU work. Total
measured 384-channel energy remains the support denominator. AP191 is retained
as its actual processed trace, never zero-filled or extrapolated; only the 383
ordinary good channels carry the ≤1e-5 ordinary-path equivalence claim.

Attempts 1–3 consumed 778 s extraction process wall and both additional retry
slots. No fourth launch occurred. `future_extraction_preflight_design.json` is
prospective only and adds explicit 30,000-frame/50 MB request guards before
voltage access.

The h5 handoff schema was cross-checked read-only. The current packet satisfies
only scorer/generator-code fields. Donors, immutable generated trains, operator
transition checks, new-hybrid stage lineage and launch permission remain false
or pending. The h5 example's 30,000 Hz placeholder must be replaced by the
recorded 29,999.759166666667 Hz in any eventual real handoff.
