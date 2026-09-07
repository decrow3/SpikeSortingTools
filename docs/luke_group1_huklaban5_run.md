# Group 1 execution on huklaban5

Group 1 owns the frozen 12/9 motion axis: motion off reference, native rigid,
and native nonrigid. The three arms execute serially on the single RTX A5000.
Group 2 threshold arms are running independently on huklaban1 and are excluded
from this controller.

The persistent-manager dummy completed with retained exit status. The real-data
120-second smoke completed all three Group 1 sorts, identical curation, QC, and
export in 5 minutes 23 seconds. Applied settings resolved to 0, 1, and 6 motion
blocks respectively, all with `do_CAR=True` and 12/9 thresholds. Smoke results
are engineering-only and are not used to rank candidates.

The remaining controller runs only: full-duration strip preparation, the three
long-strip arms, and manifest finalization. Each stage has its own durable job
receipt. Failure stops the controller without retry. Kilosort has no within-sort
checkpoint; interruption restarts only the affected arm, while completed arms
and downstream stages remain reusable after investigation.

Outputs are rooted at
`/media/huklaban5/Data/SpikeSortingTools/luke0804_hindsight_ladder_v1/group1`.
The controller requires the data mount, at least 500 GiB free, the exact parent
contract, the completed manager and smoke receipts, and production CUDA before
writing `launch.json` or starting preparation.

The first long-run attempt was killed during strip preparation at 02:55 PDT on
2026-09-07. `systemd-oomd` reported user-slice memory pressure of 79.45%, above
its 50% threshold for more than 20 seconds. No sorter arm had started. Its full
output tree and stale SIGKILL receipts are retained under `failed_attempts`.
The replacement launch uses one preparation worker and
`ManagedOOMPreference=avoid`; both deviations are explicit in the plan and the
systemd unit. A Kilosort interruption still requires restarting the affected
arm from its beginning.

After all three long-strip manifests complete, Group 1 benchmarking runs two
generic comparisons sequentially: native rigid versus motion off, then native
nonrigid versus motion off. The controller is
`testing/luke_group1_benchmark_job.py`; it validates the completed group receipt
and all three sort identities before launch, records a separate managed receipt
for each comparison, and stops without retry on failure. Comparison artifacts
are written below `long/arms/comparisons`, while controller receipts and compact
reports are written below `benchmarks`. The evaluator reports coverage and
Pareto inputs but never manufactures a composite rank.
