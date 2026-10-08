# Candidate2 v7 terminal pre-read failure

The single authorized v7 start failed closed in `ExecStartPre`. The external monitor became ready and captured four samples; the target then rejected its loaded-unit authorization because systemd decorates `ExecStart`/`ExecStartPre` with runtime fields after activation. The reviewed executable path, `argv[]`, `ignore_errors`, contract/source bindings, unit bytes, dependency membership, resource thresholds, and uncapped lifecycle did not change.

There is no `PREFLIGHT.json`, `JOB.json`, or output namespace. The recording was not opened, zero voltage bytes were read, and sorter GPU work did not start. The v7 namespace is terminal and will not be retried.

The narrow successor repair is to compare the static execution identity exactly while excluding only systemd's runtime annotations (`start_time`, `stop_time`, `pid`, `code`, `status`). It needs a fresh namespace, regression tests for active/inactive equality plus changed-path/argv/ignore-errors rejection, and independent review.

Implementation checks
- Done: actual service log and terminal monitor receipt identify the failure as the loaded-unit equality guard before receipt creation.
- Done: direct before/after comparison shows differences only in runtime annotation suffixes of target `ExecStartPre` and monitor `ExecStart`.
- Done: target is failed, monitor is cleanly inactive, output/preflight/JOB are absent, and v7 has no retry.
- Not done: no successor repair, authorization, or second start has been created.
- Can establish: v7 failed safely because stable command identity was conflated with runtime display state.
- Cannot establish: sort memory behavior, completion, or scientific Candidate2 performance.
