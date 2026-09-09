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
