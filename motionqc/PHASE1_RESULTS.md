# motionqc phase-1 validation

Focused tests: **16 passed**.

Frozen Luke0804/Bacon reproduction output:
`testing/outputs/motionqc_phase1_20260925/`.

| Target | Result |
|---|---|
| Stage-1 M accepted set | PASS, 30/30 exact |
| Stage-1 M best shifts | PASS, exact |
| S corrected episode pool | PASS, 45 episodes |
| AE canonical mask | PASS, 422 intervals, 2,263.5 s, SHA-256 `86425e8a6627a10ac8932ba01b799cb7be7141f731ebb4ecc01e303326b9da55` |
| Hub T8 episodes | MISMATCH, 9/10 in −160 to −240 µm |
| Hub T8 null | PASS, 60/60 at 0 µm |
| Q.2 matched quiet increment | PASS, 6.30724 µm stitched vs 6.06488 µm per-window |
| Bacon p50 | PASS, zero episode candidates, 20/20 null, 4.22 µm rigid P5–P95 |
| Luke p10 | PASS, five accepted episodes |

The T8 mismatch is preserved rather than tuned away. Running the supplied hub
`t8_episode_shift.py` logic on the supplied NPX peak export and portable native
field also gives −80 µm for the final 2429.000–2430.250 s episode; the stated
target says all ten scored episodes should lie from −160 to −240 µm.

Reports:

- `testing/outputs/motionqc_phase1_20260925/bacon_p50/REPORT.md`
- `testing/outputs/motionqc_phase1_20260925/luke_p10/REPORT.md`
