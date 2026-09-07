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
