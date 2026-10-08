# Sparse covariance/W smoke dispatch result

Status: **not executed — normal tool review denied the install/start action**.

The immutable enabled packet and H1 independent GO packet both verified. The
immediate mutable-host recheck passed: all frozen hashes matched, the exact
metadata-only preflight saw CUDA and one RTX A5000, recording metadata was
readable without opening content, and the real unit/output/reservation paths
were absent.

The subsequent single requested action was installation of the exact reviewed
static user service followed by daemon reload and exactly one nonblocking start.
The normal execution tool rejected that request before the command ran. Its
reason is preserved verbatim in `TOOL_DENIAL.json`. No retry or alternate route
was attempted.

Post-denial inspection confirms:

- service file absent;
- unit `not-found`, `inactive/dead`, PID 0, restart count 0;
- run root, native-results, launch-evidence, and reservation absent;
- recording content not opened and read bytes 0.

Consequently there is no covariance, W, batch journal, support/state exposure,
artifact inventory, or scientific result. Kilosort remains non-resumable, but
no run began and nothing requires restart or cleanup.

## Implementation checks

- Done: enabled and independent-review packet manifests -> every entry passed.
- Done: config/source/helper/tests/launcher/service/resolved-Python/Kilosort
  module hashes -> exact reviewed values.
- Done: immediate metadata-only preflight -> CUDA/RTX A5000 present,
  `recording_content_opened=false`, all prospective paths absent.
- Done: normal tool dispatch request -> rejected before execution; exact denial
  and authorization references persisted; no retry.
- Done: post-denial state -> no installed/loaded/running unit, no namespace,
  reservation, artifacts, voltage read, or process.
- Not done: service start, recording read, real C/W, rank/condition, W finiteness,
  stop boundary, artifact ceiling, training, detection, RF, or holdout.
- Can establish: technical preconditions passed, but platform review blocked the
  only authorized dispatch path without changing host or recording state.
- Cannot establish: real covariance/W validity, successful smoke completion, or
  any scientific interpretation.
