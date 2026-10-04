# Frozen invocation boundary

There is intentionally no executable command in this v1 packet. The accepted
managed entry is:

`python -m testing.qualification_managed_launcher --contract <contract> --request <request> --output <fresh-output>`

but its worker accepts only `phase1_controls` and `phase2_existing_b`. Supplying
the two new arms would be rejected before evaluation. Presenting that command as
post-pair-ready would be a false-success risk.

The only permitted follow-up implementation is a narrow Phase-3 composition
entry with this order:

1. Validate exact contract/source/Phase-1/pair-completion identities and claim a
   single-use start token before any arm load.
2. Rebind REF and validate it against the accepted Phase-1 receipt.
3. Rebind existing B, repaired B, and REF repeat from exact current-host bytes;
   validate clock, row ancestry, spatial identity, QC identity, and domains.
4. Call the accepted measurement module unchanged for `R(existing)`,
   `R(repaired)`, paired-unit `DeltaR`, and the strict lower95 gate.
5. Call accepted diagnostic paths unchanged for repeatability, fixed cells,
   conditional support, and guardrails. Waveform stays `UNMEASURED` unless a
   separately reviewed capture receipt is supplied.
6. Write the compact report, manifest, and `COMPLETE.json` last. Execution
   success and scientific decision remain separate fields.

Any changed estimator, threshold, cohort, matcher, clock, waveform method, or
interpretation is outside this preparation and requires a new freeze/review.
