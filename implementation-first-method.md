# Implementation-first review: method note

**Status:** written by Claude (hourly review) on 2026-09-28 at the user's request. The principle was set by the user on 2026-09-27. It comes from the Luke0804 motion program, but it's meant to be reusable across datasets and sessions.

## The rule

When a new result arrives, **suspect the implementation before the science.** Before drawing a conclusion from it, or designing a new analysis on top of it, check whether a code, configuration or measurement error could produce it. Record what was checked and what wasn't, next to the result. A conclusion stands only after the checks.

The rule applies with equal force to:
- every agent's results;
- the hub's own analyses;
- the reviewer's own earlier claims. Several of the cases below are the reviewer's.

## Why

In this program, implementation and measurement defects changed an interpretation more often than new data did. Indirect metrics kept moving in the expected direction for the wrong reason. Each defect below was load-bearing: it would have changed a decision, redirected an experiment, or produced a false "pass".

## Checklist

Run it before interpreting any result, and before any block builds on one.

1. **Executed source and effective config.** Read the code path that actually ran (commit, dirty files) and the *effective* config written by the run, not source defaults or parameter names. Confirm which stage a parameter controls. Matching chunks aren't construction chunks.
2. **Arms differ only in the intended factor.** List every other difference between arms, such as bank, whitening state, refit feature weights or refinement run, before attributing an effect.
3. **Axes, frames and coordinates.**
   - Column meanings: z_abs vs y, and depth vs radial.
   - Registered vs observed frame, and padded registered sites vs physical channels.
   - Whether a saved "channel" is a measured peak or a template-derived label.
4. **Clocks and time bases.**
   - The actual sampling rate (29,999.76 Hz, not 30,000).
   - Window and source-frame origins, and matching vs final timestamps.
   - Chunk-centre vs per-event time.
   - Inclusive vs exclusive radii in samples.
5. **Silent caps, filters, rounding and defaults.** Top-N truncation, rounding to whole pitches, hard-coded thresholds, default values standing in for missing metadata, and clamp bounds and priors whose units weren't checked.
6. **Matching and correspondence semantics.**
   - Exact-key vs tolerant matching.
   - Greedy vs maximum-cardinality matching.
   - Exclusive vs all-neighbour counting.
   - Whether the null matches the statistic it's compared with.
   - Global vs state-restricted pairing.
7. **State and domain definitions.**
   - What "rest", "episode" and "displaced" actually cover: catalogue cores vs mask vs field deviation.
   - Adjacency that bridges excluded gaps.
   - Common support between arms, and whether denominators match.
8. **Circularity.**
   - A gate that transforms saved coordinates instead of testing the real data.
   - A fixture that checks a helper against itself.
   - A threshold chosen on the same events used to judge success.
   - Selection that depends on the outcome.
9. **Reproduction and provenance.**
   - Whether the claimed historical state was reproduced (Gate H) or only regenerated prospectively.
   - Whether the intermediate data needed for attribution was saved at all.
10. **Proportionality.** Say what the result can and can't establish, and give the scope: window, state, arm and prospective vs historical.

## How it's applied

- **Results** report their checks: done, not done, and why.
- **Every block starts with a proportionate implementation step** before any outcome. Not every block needs a separate independent gate chain; use judgement about what's load-bearing: source and config confirmation, synthetic fixtures with known answers, and a positive control. BF.0, BH.1 and CS.0 are examples.
- **Frozen readings.** Prediction and interpretation rules are fixed before outcomes are seen. A later redesign is labelled as such, and the original failure is kept.
- **Independent review.** A second reader checks load-bearing claims against the executed source: the coordinator, the second machine's agent, or a second Claude session. Any reviewer's corrections are logged as corrections, never silently absorbed.
- **Failures are preserved.** A failed or flawed version is kept and labelled. It isn't overwritten or re-tuned to pass.

## Case log (Luke0804, 2026-09-25 to 2026-09-28)

| # | What looked true | Actual cause | Effect on interpretation |
|---|---|---|---|
| 1 | Static units "stay home" during episodes, so there was no motion | Fragmentation and reassignment confound sorter-based tests (user) | "Activity, not motion" withdrawn; label-free T8 shift test adopted |
| 2 | T8 counts 10/10 and 11/12 | Summary overstated; real counts 9/10 and 10/12 | Counts corrected; conclusion unchanged |
| 3 | Censor-mask hashes differ across machines | CRLF vs LF serialization | Canonical LF hashing |
| 4 | D1 loses about 3 pp of spikes at rest (M1) | Exact (sample, channel) key matching; only 22% of events matched even for S | The rest "spike loss" was a metric artifact |
| 5 | Coherence (M4) deficit from motion correction | The evaluator rounded displacement to whole 40 µm pitches (quantized reference) | Explains only 5–19% of the deficit; the rest is real |
| 6 | S and D2L have equal duplication (98/100) | Duplicate count capped at the top-100 pairs | Capped summaries, not totals |
| 7 | The nonrigid inverse works | `uncorrect_s` added the registered coordinate twice | Library defect; not active for rigid fields |
| 8 | Motion correction "fixes fragmentation" (M3 falls 60–90%) | M3 depends on coordinates and is many-to-many | No identity recovery shown; claim withdrawn |
| 9 | No 2–10 Hz jitter anywhere | Observed and null time support differed in the spectral helper | Full-session absence claim qualified |
| 10 | "1 s chunking is not the problem" (D1 ≈ D4) | D1 vs D4 varied only the *matching* chunk; templates were still registered at 1 s chunk centres | Conclusion withdrawn; became the BF/BW tests |
| 11 | D2L loses interior episode events (BG) | Saved `channels` are interpolated-template labels, and correspondence used them spatially | Most of the excess vanished with time-only matching |
| 12 | Episode stratification by BE depth | BE used off-probe y instead of z_abs | Stratification repaired |
| 13 | BC centred cosine | Advanced-indexing axis error | Outside-support agreement corrected (0.18 → 0.29) |
| 14 | BF.2 would test template registration causally | Construction labels and membership weren't saved (`save_intermediate_labels=false`) | Blocked; provenance requirement added |
| 15 | BW.2 short-chunk design | The draft changed *matching* chunks, not construction grab chunks | Caught before launch |
| 16 | Merge realignment makes duplicates escape dedup | In the reviewed CE/CL captured states, the current merge-realignment step moved 1/6,250 (W2) and 0/5,685 (W3) close pairs out of the dedup radius; offsets span about 3 samples | That specific step is ruled out as the source of separation in those states. Other alignment mechanisms aren't ruled out (narrowed at the coordinator's review, 2026-09-28) |
| 17 | Ghost clusters merged through scale-tolerant distance | Boundary 1/3 means 0.75–1.33×. How strongly the prior (scale_var 1e-4) pulls toward 1 depends on the actual weighted norms and dots, which weren't checked | Scale-tolerance explanation withdrawn; the prior's practical strength remains unverified |
| 18 | Refractory violations mark bad merges | The same spike found twice also looks like a violation (user) | Good and bad merges aren't separable by ISI alone |
| 19 | The CS fixture dedup radius is 8 samples | Hard-coded 30,000 Hz; the actual rate gives 7 | Fixture corrected |
| 20 | A QDA secondary gate is a third force-merge treatment | QDA ∨ (force ∧ QDA) = QDA, the same as no-force | Degenerate design withdrawn |
| 21 | The DD lattice gate passed | Both gates transformed saved peak coordinates (z − q), not the remapped voltage | Rerun on voltage; the gate then failed and was preserved |
| 22 | Rest-state metrics are at rest | Catalogue cores (median 1.75 s) leave 9–15% of "rest" displaced by > 80 µm | Field-deviation states adopted (DC/DE) |
| 23 | The ISI drop comes from "another pass" | REMATCH0 also changed the bank, feature weights and refinement run | Native-bank control: 1.18% (accepted) → 0.91% (native-bank rematch) → 0.67% (regenerated bank). This is **not** an additive causal decomposition: the native-bank control also changes continuous features and adaptive steps |
| 24 | Various budget totals | Charges were combined across hosts | Per-host accounting enforced |
| 25 | The lattice hybrid arm was favoured by construction over D2L_h (reviewer claim) | Checking the executed START/worker/adapter showed D2L_h consumed the same exact-lattice states | Reviewer concern withdrawn. Check the executed consumer, not the design text, before alleging an asymmetry |

## Recurring patterns

- **Parameter names mislead.** Always read the executed stage: matching vs construction chunks, `amplitude_scaling_boundary` semantics, template-derived `channels`.
- **Defaults hide missing data.** A crop offset defaulting to 0, and a nominal 30 kHz clock.
- **Metrics built for one arm break across arms.** Exact keys, per-arm template coordinates, and denominators that differ between arms.
- **"Pass" can be circular.** Coordinate-transform gates and self-referential fixtures.
- **State labels drift from physical states.** Catalogue cores vs actual displacement.

## Pointers

- Principle in the question sheet: standing context in [agent-questions-20260925.md](agent-questions-20260925.md).
- Handoff rule 6 in [hub-handoff.md](hub-handoff.md).
- Code reviews: [code-review-20260926.md](code-review-20260926.md), `weightbearing-code-review-20260926.md`, [code-review-BC-BD-BE-20260927.md](code-review-BC-BD-BE-20260927.md).
