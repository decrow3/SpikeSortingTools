# Contract-to-code correction audit

## Corrected decision

The original run and score output remain immutable. The primary H−F result
remains **INCONCLUSIVE** at the unchanged ±0.05 rule: its interval is above zero
but not wholly greater than `+0.05`. The intended H/L same-window quality
guardrails all pass. No progression is supported because the primary criterion
was missed, not because H had 461 arm-local cluster labels versus F's 972.

H rho is `0.7421635214982477`, near L's `0.7525836443406859` and above F's
`0.6761670705417997`. The observed continuity-proxy difference persisted in this
run but missed the frozen criterion. This is not a causal claim, an across-run
replication claim, biological identity evidence, or a neuron-loss statement.

## Trace from instruction to execution

1. The pre-run proposal said the existing F/L active and eligible populations
   were descriptive and not attributable (`em_full_vs_w3_equivalence.../FINAL_AUDIT.md:16-18`).
   Its frozen prediction again stated active/eligible counts were descriptive,
   not acceptance endpoints (`FINAL_AUDIT.md:58-67`).
2. The independent integrated review described the scientific contract as
   containing **L-based yield and three short-interval guardrails**
   (`em_harmonized_w3_control_integrated_review.../INTEGRATED_REVIEW.md:16-20`).
3. The producer's own implementation review bound only the L reference values
   (`em_harmonized_w3_control_implementation_20261003_v1_h5/IMPLEMENTATION_REVIEW.md:12-13`).
4. The first divergent contract line is
   `config/harmonized_w3_scorecard.v1.json:68`, which says all comparisons
   against both F and L must pass. Lines 69-75 apply comparator assigned-label
   yield, while line 98 simultaneously labels assigned and eligible counts
   descriptive. This is an internal contradiction, not a documented redesign.
5. The executed scorer makes that divergence operative at
   `source/harmonized_scorecard.py:199-201`: it creates F and L guardrails and
   aggregates both with `all(...)`. The generic yield calculation is at lines
   88-116. Lines 202-207 separately make any inconclusive primary result
   non-advancing.
6. The coordinator authorization binds the config hash but contains no semantic
   decision to replace L-based guardrails with F+L or to promote descriptive
   populations into acceptance endpoints
   (`em_harmonized_w3_control_coordinator_release_20261003_v3/COORDINATOR_AUTHORIZATION.json:4-15`).
   The release receipt likewise records hashes/resources only
   (`RELEASE_RECEIPT.json:31-56`). No explicit authorized superseding decision
   was found.

Therefore the F whole-population yield gate was unintended. Later reviews that
called the frozen guardrails unchanged did not explicitly resolve this
contradiction and do not constitute an authorized supersession.

## Correction to the prior H1 verdict

The prior additive final-review packet is preserved. Its primary classification
at `FINAL_REVIEW.md:11-17` remains valid. Its statement at lines 19-24 that
quality advancement independently failed because H had 461 labels versus F's
972 is corrected: that F denominator was not the intended same-window quality
gate. The correct existing-data check is H versus L:

- yield: `(461−478)/478 = -0.03556485355648536`, within 5%, pass;
- negative-excursion short-interval difference: `+0.0017409786778535804`, pass;
- outside-mask-flat difference: `+0.0005043072555330572`, pass;
- operational-remainder difference: `-0.0005476230828800818`, pass.

All use the already frozen 0.05 yield and 0.01 short-interval thresholds. No
rescore, threshold change, new endpoint, or new hypothesis is introduced.

## Implementation checks

- Done: original instruction -> traced proposal and independent pre-run review;
  both preserve descriptive population semantics and identify L guardrails.
- Done: first divergence -> config v1 line 68; execution at scorer lines 199-201,
  with the comparator-yield helper at lines 88-116.
- Done: override search -> coordinator authorization, release receipt, H1 GO,
  and v1/v2/v3 implementation/review records inspected; no explicit authorized
  semantic override found.
- Done: correction arithmetic -> H/L relative yield and all three short-interval
  differences recomputed from the sealed result; every intended guardrail passes.
- Done: unchanged primary -> H−F point estimate, 2,000-draw interval, and
  classification remain exactly as executed.
- Not done: rescore or new execution -> prohibited and unnecessary.
- Can establish: the F yield aggregation deviated from the intended contract;
  correcting that attribution leaves an inconclusive primary and no progression.
- Cannot establish: causal training-context benefit, across-run persistence,
  biological identity/purity, neuron loss, or support for long/full/production.
