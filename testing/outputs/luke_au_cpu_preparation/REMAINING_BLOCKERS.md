# AW remaining blockers after AV handoff validation

Verdict: **the handoff is valid, but the frozen full-probe worker is not ready.**

| Check | Result |
|---|---:|
| Bundle files passing size and SHA-256 | 21/21 |
| Exact D2L-v1 authority | pass (`85062a37…c2d9`) |
| Local adapter filtering/resampling | neither |
| D2L matcher | `drifty`, 7,500 samples / 0.25 s |
| W2 chunk centres | 1,360 |
| Observed-crop templates passing >99% interior energy | 7/654 |
| Templates passing every occupied exact state | 375/654 |
| Passing both crop screens | 7/654 |
| Frozen donor target | 30 |
| Worker ready | no |

The shallow AP202--AP383 bank cannot identify energy on AP0--AP201. Even as a
crop-only design it supplies only seven templates passing both necessary
screens, so it cannot reach the frozen 30-donor target. No threshold or donor
count was changed.

The frozen `quality` label and `isolation_score` are also absent and not
uniquely recoverable. Stage the exact small S/W2 `qc-phy` unit table referenced
by the receipt rather than inventing replacements.

If strict full-probe qualification remains required, the smallest extraction is
a read-only 900--1240 s waveform/template pass using accepted S labels/times:
one approximately 7.83 GB raw read, exact 384-channel `ibllikecmr`, a 15.67 GB
temporary RAM cache, and only the 30 selected templates persisted. Estimated
CPU wall time is 20--35 minutes; persistent output is under 0.2 GB. This performs
no sort and uses no GPU. It has not been launched.

The synchronized regular 5 Hz train is scorer-fixture-only. The eventual
population uses independent per-unit seeded trains, mean 5 Hz, minimum 3 ms
refractory, base seed 20260926.

AV continuity observations are exploratory. The chronological split, fixed
left/right builder self-review and unpaired representative templates are the
accepted corrections; T6 supplies no unit-quality inference.
