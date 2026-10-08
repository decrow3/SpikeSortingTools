# Transactional DARTsort peel checkpoint design

The current checkpoint uses progress scalars as intent rather than commit records. The repair changes the invariant: **payload extents are durable first; one append-only commit row publishes the chunk second.** Resume derives both progress and rollback extents from the last valid row.

Proposed HDF5 dataset:

```text
chunk_commit_log: int64[N, 5]
  0 chunk_index
  1 chunk_start_samples
  2 committed_n_spikes
  3 committed_n_residuals
  4 committed_external_residual_bytes (-1 when absent)
```

On a new file, create an empty extendible log. After computing a chunk, append/flush every event and residual payload, flush any external residual file, flush the HDF5 payload, append exactly one commit row, flush again, then update legacy progress scalars as derived compatibility metadata. Returning before the commit row means the chunk is uncommitted even if some arrays grew.

On resume, validate the full log against the frozen chunk schedule and monotone extents. Let its last row define the next chunk and committed extents. Reject if any required dataset is shorter than its committed extent. Truncate longer variable datasets and residual storage to the committed extents. This rollback is safe because rows beyond the commit are, by definition, unpublished partial work. Only then schedule the next chunk.

The log removes ambiguity between zero-event chunks and missing event writes: zero-event commits advance chunk identity while preserving the previous event extent. It also avoids inferring progress from `len(times_samples)`, which cannot distinguish a complete append from a partial multi-dataset append. Candidate-3 should set `save_intermediate_labels=true` so stage-level fast-forward has an explicit label checkpoint; fitted model folders and training-sample/random-state receipts remain separately required.

Legacy behavior is deliberately conservative. A partial legacy peel file with only `last_chunk_index/start` is rejected for candidate-3 resume. A completed historical output remains readable as a result, but is not evidence that the new transactional checkpoint was exercised. External raw residual files require byte-extent recording and truncation; if that path cannot be exercised reliably, candidate-3 must forbid it rather than claiming support.

This design still does not make clustering/refinement internally checkpointed. Candidate-3's full contract must list which stages are transactional, which restart from the previous stage checkpoint, and the maximum recomputation after interruption. “Fully checkpointed” should mean every long stage has either a verified internal checkpoint or a bounded, explicitly named restart unit—not merely that completed pipeline stages are reusable.
