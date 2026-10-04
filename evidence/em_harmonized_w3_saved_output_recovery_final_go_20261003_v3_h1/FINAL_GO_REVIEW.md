# H1 final GO — saved-output-only harmonized W3 recovery

## Decision

**GO_TO_COORDINATOR_RELEASE for exactly one CPU-only score-only attempt.**

This is not a launch. A separate coordinator release and immediate fail-closed
preflight remain required. The recovery may consume only the already completed,
hash-bound H/F/L saved sortings and exact field/mask/catalogue inputs. It may not
read recording voltage or preprocessing caches, use a GPU, rerun sorting, alter
the frozen scorer, overwrite the original failed score namespace, retry, or
promote any outcome automatically.

H5 repaired the sole blocker from the preceding H1 review by recopying the
byte-identical `COMPLETE.json` after every packet member. Its SHA-256 remains
`d4f858bb43cc8cf18a190fe464d6ed1769d065291b12cf596f47967c175a6a30`;
the packet manifest remains
`72f9f1bf719f91713015f2e6ac671f34c14998e6d26715caed6f37f34b636e0e`.
The completion mtime is now `1791094101383668800` ns, strictly later than the
latest other member at `1791093526217652800` ns. All manifest members rehash.

## Release conditions

- Bind the exact recovery packet manifest and this H1 GO receipt.
- Reverify H/F/L and field/mask/catalogue hashes, H receipt and sort marker,
  original failed-score preservation, and fresh recovery output absence.
- Use the frozen disabled launch template as one managed systemd job with
  `Restart=no`, CPU-only `CUDA_VISIBLE_DEVICES=`, `RuntimeMaxSec=900`,
  `MemoryMax=32G`, `CPUQuota=400%`, one attempt, and zero retries.
- Require the score worker to seal a fresh output with `COMPLETE.json` last, or
  preserve one failure. Do not clean or overwrite either outcome.
- Do not access RF, the outer holdout, recording voltage, raw waveforms, or any
  additional arm/seed; do not initiate long/full/production work.
- Submit any successful result to independent final review before scientific or
  pipeline interpretation. That review must verify point estimates, joint
  bootstrap pairing and valid draws, domain exposures and segment-safe
  denominators, yield/short-interval guardrails, result sealing, and the frozen
  interpretation rules.

## Implementation checks

- Done: reseal byte identity -> COMPLETE hash is unchanged before/after.
- Done: packet integrity -> all 18 manifest members rehashed successfully;
  manifest hash is unchanged.
- Done: terminal ordering -> COMPLETE is strictly newer than every member.
- Done: prior delta review -> scorer closure and scientific contract unchanged;
  real-input preflight and independent H1 domain recomputation pass.
- Not done: current release-state checks -> intentionally deferred to the
  coordinator's immediate preflight.
- Not done: score/result -> no scoring was performed by this review.
- Can establish: the exact reviewed packet is eligible for a separate release
  of one saved-output-only CPU scoring attempt.
- Cannot establish: future input availability, execution success, scientific
  result, biological identity/purity, or eligibility for further progression.
