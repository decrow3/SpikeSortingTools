# BA — approved bounded extraction infrastructure recovery

Status: **APPROVED** by the user's explicit “Approve BA and resume extraction”,
recorded by the hub at 2026-09-27 03:05 UTC (2026-09-26 20:05 PDT).

AZ's initial extraction and two additional relaunches remain charged. BA allows
at most three more infrastructure relaunches for the same donor extraction,
within the unchanged cumulative 7,200 s extraction wall allowance, 20 GB RAM,
two numerical CPU threads, one reader, output/storage limits and scientific
gates. It does not authorize a new scientific arm, extra GPU work or a larger
resource budget.

The first three service windows are charged at the exact saved systemd total of
778 s. Because older small interactive setup/test wall was not individually
metered, BA prelaunch conservatively adds 60 s rather than silently calling it
zero. Thus 838 s are charged before the first BA service and 6,362 s remain.
Each BA attempt is capped at the lesser of 3,600 s and the remaining allowance.

Before full measurement, the service must:

1. validate manifest frame bounds and predicted dimensions/bytes without
   opening voltage;
2. enforce named `start_frame`/`end_frame`, at most 30,000 frames and at most
   50,000,000 float32 bytes per explicit audit request;
3. run real bounded one-second reads under a hard process/memory guard;
4. reproduce ordinary preprocessing on all 383 good channels within max
   absolute difference 1e-5;
5. retain AP191 only as its real finite, nonzero processed trace, with no
   padding or extrapolation; and
6. confirm the post-ibllikecmr standardized, pre-spatial-whitening donor domain.
   Injection occurs once at that stage, while normal downstream DARTsort
   operations—including configured internal whitening—remain unchanged and
   identical across arms.

A failed preflight consumes a relaunch. A repeated failure requires evidenced
repair. Preserve all prior and new failed artifacts. No unit-test mock alone may
open the full measurement. Scientific donor/operator failures stop dependent
work; thresholds, donor count and support gates are not weakened. Successful
20–30-donor qualification and handoff remain authorized by AZ.
