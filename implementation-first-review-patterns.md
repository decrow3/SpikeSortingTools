# Recurring implementation failure patterns

These are generic patterns distilled from a multi-agent spike-sorting and motion-correction analysis (2026-09). Each one changed an interpretation before it was caught.

## Parameter names mislead
- A "chunk length" setting controlled one pipeline stage (matching) while another stage (template construction) used a different, larger chunk. An experiment that varied the first was read as testing both.
- A "scaling boundary" of 1/3 meant a 0.75–1.33× range, not "down to one third". A strong prior also pulled the scale toward 1 anyway.
- **Check:** trace the parameter to the line that consumes it.

## Derived fields that look measured
- A per-event "channel" was the main channel of a model template, not the voltage peak. Spatial matching across methods on that field produced spurious "lost events".
- **Check:** find where the field is written, and what it's computed from.

## Wrong axis or column
- A depth analysis used the off-probe distance column instead of depth.
- **Check:** assert column semantics against the geometry, for example with a known displacement.

## Defaults hiding missing data
- A crop offset defaulted to 0 when metadata was missing. A fixture hard-coded a nominal sampling rate, which changed an inclusive radius from 7 to 8 samples.
- **Check:** fail loudly on missing metadata, and read the true rate from the data.

## Metrics that break across arms
- Exact-key accounting matched only 22% of events even for the reference, so a small difference between arms was a keying artifact, not spike loss.
- A pair-count metric used each arm's own coordinates, so a coordinate change alone could "fix" fragmentation.
- A duplicate metric silently counted only the top 100 pairs.
- **Check:** run every metric on a no-effect control, and look for caps.

## Circular passes
- A gate "validated" a remapped signal by transforming saved coordinates, never touching the remapped data.
- A fixture compared a helper with itself.
- A secondary acceptance test was already part of the union being gated, which made the new "treatment" identical to an existing arm (A ∨ (B ∧ A) = A).
- **Check:** ask what independent evidence the pass rests on.

## State labels drifting from physical states
- "Rest" was defined as "outside catalogued event cores", but the cores were short. 9–15% of "rest" was in fact displaced.
- **Check:** define states from the physical variable, and keep curated labels separate.

## More than one thing changed
- A "one more pass" arm also changed the template bank, the fitted feature weights and the downstream rerun. A control that held the bank fixed moved the result about halfway. Even so, that isn't an additive decomposition, because the control itself changed adaptive steps.
- **Check:** list every difference between arms, and add the control that isolates the claimed factor.

## Intermediate state not saved
- A causal test couldn't run because construction-time membership and intermediate labels weren't saved.
- **Check:** before designing an attribution test, confirm that the inputs it needs exist. Add provenance capture to future runs.

## Summary drift
- A human-written summary overstated counts (10/10 when the output said 9/10).
- **Check:** quote numbers from the artifact, not from memory or earlier summaries.
