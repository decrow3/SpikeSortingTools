# Minimum-training support-mask covariance/W fresh relaunch candidate

Status: **review-only; not installed, not authorized by a fresh H1 GO, and not
launched**.

## Delivery contract

- Global milestone: snippet-stage numerical feasibility of the pointwise
  support mask at Kilosort's native covariance/whitening boundary.
- Decision changed: whether the unchanged 12-batch masked covariance is finite,
  full-rank and conditioned at or below `1e8`, and yields finite hash-bound W.
- Cheapest adequate test: the frozen covariance schedule only, stopping before
  training/detection.
- Completion condition: batches `0,25,...,275` journaled, at most 554,084,352
  padded input bytes, durable C validation and finite W, retained namespace at
  or below 33,554,432 bytes, intentional pre-training stop and terminal unit
  evidence.

The prior v1 namespace and its denied launch decisions are consumed historical
contracts. This candidate preserves the recording, crop `[208498882,226498783)`,
29,999.835983263598 Hz clock, geometry, motion field, preprocessing, 12-batch
schedule, 24.0001312141065 interior seconds, covariance gate, W construction,
stop boundary, no-restart/non-resumable semantics, and exclusions.

The only candidate source delta is the exact run root suffix `v1 -> v2` in the
source policy. Config changes are the corresponding run/evidence/results paths
and source hash. The service changes only its config path/hash and fresh-root
condition. Launcher, helper, tests, and all bound Kilosort modules are unchanged.

Fresh prospective paths:

- config `configs/en_minimum_training_support_mask.covariance_smoke.enabled.review_pending.v2.json`;
- service `testing/en_minimum_training_support_mask_covariance_smoke_enabled_v2.service`;
- run root `/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/en_minimum_training_support_mask_covariance_smoke_20261001_v2`.

Runtime/CPU/RAM/GPU-utilization caps remain unspecified exactly as in the prior
reviewed contract; CUDA on the reviewed RTX A5000 is required. The user's new
standing authorization covers that documented resource condition, but this
packet still requires independent H1 implementation review and a fresh GO.

Prior denial evidence:
`en_minimum_training_support_mask_covariance_smoke_result_20261002_v2_h5`,
manifest `51e5203054bb6b03d2c7671a712e199daf6703cf5ac8095d022036e0e1d907c6`.

## Required H1 review

Inspect the one-line source-policy root change, exact config/service rebinding,
unchanged schedule and bounds, and all frozen hashes. Re-run the 34 focused
candidate/helper tests after installing the reviewed source/config/service
byte-for-byte, then issue a fresh decision. No live installation or launch is
authorized by this packet.

## Implementation checks

- Done: source comparison -> only exact run-root suffix changed.
- Done: config/service comparison -> only corresponding fresh namespace,
  source hash, config path/hash and fresh-root condition changed.
- Done: unchanged live candidate/helper suite -> 34/34 pass; candidate delta is
  supplied for independent review before installation.
- Done: prospective config/service/run-root paths -> absent.
- Not done: candidate installation/full-validator execution, CUDA preflight,
  voltage, C/W, training, detection, RF, or holdout.
- Can establish: a collision-free, scientifically unchanged candidate packet is
  ready for independent H1 implementation review.
- Cannot establish: installed-source binding, fresh GO, real covariance/W
  validity, successful execution, or scientific benefit.
