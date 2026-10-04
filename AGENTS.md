# Running spike sorts

Never launch a spike sort whose lifetime depends on a chat process, agent tool
session, or interactive terminal remaining open. This includes downstream jobs
that wait for the sort and then run curation, QC, or comparisons.

- Use an independent job manager, such as a systemd user service or a batch
  scheduler. Backgrounding a command in an agent shell is not sufficient proof
  that it will survive chat shutdown.
- Confirm that the job survives launcher disconnection before entrusting it
  with a full-session sort. Use a cheap dummy job to verify a new launch method.
- Persist the launch command, resolved settings, job identifier, stdout/stderr,
  and final exit status outside the chat. Check the actual job/process state
  when reporting progress; a stale log or status file is not proof of liveness.
- Check checkpoint/resume behavior before launching. Reuse of completed stages
  is not checkpointing within a sort. State explicitly when an interruption
  requires restarting the whole sort; do not describe that as resumable.
- Preserve failed-run evidence and investigate unexpected termination before
  restarting. Honor user cancellation and run-specific holds.

Background: [Luke full-session interruption investigation](docs/luke_full_session_interruption_20260906.md).

# Standing publication authorization for Luke motion/sorting work

The user has given standing authorization to publish verified, compact
scientific outputs produced while pursuing the Luke spike-sorting improvements
under `/mnt/NPX/Luke/DARTsort_motion_experiments/`. Do not pause for a new
per-packet or exact-destination approval when that publication is an ordinary
handoff step in an already authorized analysis.

- Verify hashes and write `COMPLETE.json` last.
- Never overwrite a conflicting existing packet; compare it and report.
- Do not publish raw voltage, large scratch/cache files, or another host's held
  payload unless the user separately authorizes that material.
- This standing publication authority does not authorize a new spike sort,
  voltage modification, deletion, scientific parameter change, or budget
  expansion. Those remain governed by their run-specific instructions.

# Coordination scope and experiment-contract handoffs

Do not carry temporary experiment limits into future handoffs as global project
restrictions. Handoffs should state the project goal, current evidence, next
decision, ownership, and applicable standing user instructions. Keep byte,
runtime, window, arm, and launch limits inside the relevant experiment contract;
link that contract only when executing or reviewing that experiment.

Mark completed narrow assignments and their restrictions as historical. A later
experiment does not renew, inherit, or imply human approval merely because its
scope differs. Preserve experiment contracts and approval history, and obtain
the authorization required for the current experiment. The automatic-review
denial of A recording access is a real, narrowly scoped unresolved tool block
and must be disclosed when relevant; it is not a general project hold. Existing
explicit standing exclusions remain unchanged.

# Delivery ladder and task sizing

User direction (2026-10-01): debug on snippets for speed, validate the surviving
approach on representative medium windows, then test the surviving candidate on
long/full development data. Target delivery in 5–8 days when evidence supports
progression; reforecast to 10–14 days when a material repair or outage requires
it.

Every task must name its milestone, the decision it can change, the cheapest
adequate test, and its completion condition. Snippet tests must preserve required
padding, geometry, time origin, and training context; they do not establish
longitudinal performance.

After checking dependencies, reuse unchanged reviewed dependencies and valid
intermediates. Require one independent review for load-bearing deltas; do not
repeat serial review ceremonies for unchanged code. Keep experiment budgets in
their experiment contracts, and preserve all existing exclusions.

# RF evaluation hold for Luke spike-sorting work

Do not automatically start or repeat per-arm, per-pair, or development RF
fitting, rescoring, or comparison loops. Preserve the completed RF results and
keep the 20-trial outer holdout sealed. RF may return once as a final-candidate
non-inferiority harm check only after a separate instruction freezes its primary
endpoint, minimum meaningful effect, support/power assumptions, pass/fail rule,
95% confidence-interval plan, and resulting pipeline decision.

Rate correlation is a continuity proxy and short-interval counts are a
contamination proxy; neither proves biological identity or purity. Do not call
historical W2 thresholds preregistered retroactively. Prefer a concrete W3 or
independent-window replication with a precommitted decision rule over more
four-pair diagnostics or RF loops.

# Lighthouse motion validation

The preferred method for finding new lighthouse candidates is
[waveform-only, whole-probe discovery](docs/lighthouse_candidate_discovery.md).
Exclude absolute depth from candidate ranking and identity matching while
retaining relative multichannel waveform geometry. Compare identities across
the available probe, without original-depth proximity gates or motion priors.
Start with cached templates and scores; select using the seed interval before
reviewing later support. Preserve strict, lower-score, ambiguous, and unmatched
evidence separately. Plot absolute waveform depths over peak depth/time scatters
after matching. This promotes a candidate-finding workflow, not every candidate
to a verified cell or motion ground truth.

Start scientific work with the cheapest direct check of the premise: inspect
measurements, look for replication across cells, then check obvious artifacts.
Do not introduce a new estimator when a direct visualization or simple statistic
can answer the current question. Establish that lighthouse measurements show
shared movement before adding consensus, interpolation, smoothing, or precision
machinery. Runtime optimization comes after evidence of correctness.

Whenever discussing a new plan or idea, proactively suggest a simpler, cheaper,
quicker alternative when one is available. State what it could establish and
what would still require the larger approach. Prefer reuse of cached evidence
for the first check. This is a decision aid, not a requirement to discard useful
existing analyses or to demand perfect evidence before making progress.

For new lighthouse validation, trace cells through depth rather than accepting
only a template at its original spatial position. Search independently of the
motion estimate being evaluated, compare competing identities, and preserve
unmatched events, depth ambiguity, and gaps. Move waveform measurement support
with the depth hypothesis. Keep historical fixed-template tracks as controls;
their dropout or apparent stationarity does not establish motion-estimator error.

See [depth-aware lighthouse policy](docs/luke_depth_aware_lighthouse_policy_20260908.md)
for the bounded implementation and interpretation requirements.


<!-- BEGIN coordinator implementation-first review -->
# Instructions for work in this folder

These instructions apply to this folder and its descendants. Explicit user
instructions take precedence. This file adapts the Claude
`implementation-first-review` skill into standing agent instructions; it does
not require that skill to be installed or invoked.

## Implementation-first review

Apply this workflow when interpreting experimental or analysis results,
comparing pipeline arms or conditions, reviewing another agent's or person's
findings, or proposing follow-up work that depends on a result. Apply it equally
to your own analyses and earlier claims.

**Check implementation before interpreting science.** Before accepting a
conclusion or building another analysis on it, check whether a code,
configuration, or measurement error could produce the result. A completed job,
verified hash, passing test suite, or internally consistent summary does not
by itself validate the scientific interpretation.

For a progress-only request, verify current processes, receipts, and artifacts.
Distinguish execution status from scientific validity. Attribute an unreviewed
verdict to its report or producing task rather than presenting it as your own
validated conclusion. If you interpret the result or recommend a next step,
perform the applicable checks below and include the report block.

### Ten check-questions

1. **What actually ran?** Read the executed source, commit, uncommitted changes,
   frozen source snapshots where applicable, and effective configuration saved
   by the run. Trace parameters to the stage and code that consume them; names
   and defaults are insufficient.
2. **Did the arms differ only in the intended factor?** List other differences
   in inputs, preprocessing, channels, intermediate state, template banks,
   whitening, refitted models, and downstream reruns before attributing an
   effect.
3. **Are axes, frames, and coordinates correct?** Verify column semantics and
   reference frames. Distinguish measured fields from model- or template-derived
   fields, observed from registered coordinates, and physical from padded or
   virtual sites.
4. **Are clocks and time bases correct?** Verify the actual sampling rate,
   recording and crop origins, timestamp version, per-chunk versus per-event
   times, and inclusive versus exclusive sample radii and interval boundaries.
5. **Are there silent caps, filters, rounding, or defaults?** Inspect top-N
   truncation, quantization, hard-coded constants, missing-metadata fallbacks,
   clamp bounds, and priors, including their units and practical effect.
6. **Are matching and counting semantics appropriate?** Check exact versus
   tolerant matching, greedy versus maximum-cardinality matching, exclusive
   versus all-neighbour counting, and whether the null uses the same statistic
   and support as the observation.
7. **Do states and domains mean what the report says?** Verify the physical
   meaning of rest, episode, baseline, and displaced states. Check adjacency
   across excluded gaps, common spatial and temporal support, eligibility, and
   numerator/denominator definitions across arms.
8. **Is the validation circular?** Check whether a gate tests actual data or
   merely transformed saved coordinates, whether a fixture compares a helper
   against itself, and whether thresholds or selections were chosen using the
   outcomes they are meant to evaluate.
9. **Was the claimed state reproduced, and is provenance sufficient?**
   Distinguish historical reproduction from prospective regeneration. Confirm
   that intermediate membership, labels, models, and other inputs needed for
   attribution were actually saved and belong to the claimed run.
10. **What is the defensible scope?** State which windows, conditions,
    populations, probes, sorters, and historical or prospective settings support
    the claim, and what remains unestablished. Continuity proxies do not prove
    biological identity, and short-interval proxies do not prove contamination
    or distinguish bad merges from repeated detections of the same spike.

Start with proportionate, cheap checks: executed-source reads, effective-config
diffs, saved-array inspections, independent known-answer fixtures, and positive
and negative controls. Identify which questions are applicable and unresolved;
do not claim that reading this checklist completes them. Recomputing a saved
ratio validates arithmetic, not the construction of its numerator or denominator.

Name expensive or unavailable checks as prerequisites rather than assuming
their outcomes. This workflow does not authorize new sorts, voltage processing
or transfers, parameter searches, RF or holdout access, remote-task messages,
or budget expansion. Respect the user's existing authorizations and holds.

### Required report block

Attach this block to every conclusion or recommendation drawn from a result,
including conclusions in chat, review notes, and handoffs. Keep it proportionate
and cite the executed source or evidence actually inspected.

```text
Implementation checks
- Done: <check> -> <outcome> (file:line or artifact/hash where relevant)
- Not done: <check> -> <why / what it would take>
- Can establish: <scoped claim>
- Cannot establish: <what remains open>
```

If no implementation checks were possible, say so explicitly. Do not turn
another task's checklist into a claim of independent review. Put implementation
checks first in any proposed follow-up, before mechanism explanations or new
experiments.

### Working practices

- **Freeze first.** Fix predictions, acceptance criteria, and interpretation
  rules before inspecting outcomes. Label later changes as redesigns and retain
  the original rule and result. Resolve conflicting contract versions visibly.
- **Preserve failures.** Keep failed and flawed versions with clear labels.
  Never overwrite historical evidence or tune a failed check until it passes.
  Use a new output namespace for corrected analyses.
- **Record corrections.** When a claim changes, record the correction and point
  to the original claim and the evidence responsible. Apply this to your own
  reviews as well as other agents' work.
- **Independent review.** Load-bearing claims need a second reader checking the
  executed source and artifacts, not just the producer's summary. Use an
  authorized reviewer or existing independent review. If none is available,
  record that gap; do not imply the review occurred or message another task
  without authorization.
- **Report from artifacts.** Read counts, hashes, and verdicts from authoritative
  outputs. Check current process state when reporting liveness; stale logs and
  planned actions are not evidence that work is running or complete.

## Worked examples and provenance

Read [implementation-first-review-patterns.md](implementation-first-review-patterns.md)
when applying this workflow, especially before proposing an attribution test.
It is a local copy of the Claude skill's worked examples; treat examples as
historical cases, not evidence about the current run.

The longer project history is in
[implementation-first-method.md](implementation-first-method.md).

Adapted on 2026-09-30 from:

- `C:/Users/Declan/.claude/skills/implementation-first-review/SKILL.md`
- `C:/Users/Declan/.claude/skills/implementation-first-review/patterns.md`

The local instructions and examples are self-contained. The Claude source files
are provenance references and are not modified by this adaptation.
<!-- END coordinator implementation-first review -->
