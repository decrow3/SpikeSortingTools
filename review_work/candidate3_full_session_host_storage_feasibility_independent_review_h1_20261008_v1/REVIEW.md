# Independent review: candidate-3 full-session host/storage feasibility

## Verdict

`PASS_SCOPED_HOST_STORAGE_FEASIBILITY_WITH_PLACEMENT_REBIND_REQUIRED`

Reviewed packet:

- `/mnt/NPX/Luke/DARTsort_motion_experiments/candidate3_full_session_host_storage_feasibility_h1_20261008_v1`
- manifest SHA-256 `fcfa3167f03ae0c084740537fab09cb2a366229b44330eb840c6bd3537e4dd40`
- complete SHA-256 `acdc0301c7a1357e5586841b61b4f23a0a42ebb8592327d92e6f69cb256607f9`

The packet seal is valid: every manifested member matches its recorded size and
SHA-256, and `COMPLETE.json` binds the exact manifest.

The main conclusion is supported within its metadata-only scope. H5 is the
simpler conditional topology because the accepted Arm-A materialization is
recorded as H5-local, while H1 exposes only the original REF binary over
read-only CIFS. Neither host is launch-ready. H5 requires a fresh resource,
mount, oomd, competing-job, source and wrapper preflight; H1 additionally
requires an accepted on-the-fly remap/prewhitening implementation, padded real
window equality and bounded CIFS throughput evidence.

## Findings

1. **Accepted input identity is correctly transcribed as a metadata binding.**
   `INPUT_BINDINGS.json` rehashes to `7c045a88...` and records half-open frames
   `[0, 314204894)`, 29,999.835983263598 Hz, 384 channels, `int16`, corrected-A
   path `/media/huklaban5/Data/en_rounded_field_ks129_20260929_v1/arm_a_v1/full/recording/traces_cached_seg0.raw`, manifest-recorded binary SHA-256
   `672938e9...`, and content SHA-256 `90b6a07a...`. It explicitly says no binary
   was opened or hashed. Accordingly, these are authoritative accepted
   identity bindings, not evidence of current path readability or a fresh
   payload hash. The packet's fresh-preflight conditions preserve that limit.

2. **The byte arithmetic is exact.** `314204894 * 384 * 2` is
   241,309,358,592 bytes; float32 is 482,618,717,184 bytes; float32 plus the
   120,000,000,000-byte planning floor is 602,618,717,184 bytes. All reported
   H1 and H5 subtractions reproduce exactly. The packet correctly labels 120 GB
   as a floor rather than a safe cap and leaves caches, scratch, logs,
   publication, and preserved failed stages unbounded.

3. **DARTsort's core no-copy claim is source-supported.** At clean DARTsort
   commit `edcfe1b5...`, the cited source hashes match. `preprocessing="none"`
   returns the passed recording unchanged; `copy_recording_to_tmpdir="no"`, or
   `"if_preprocessing"` with preprocessing none, makes `ds_will_copy_recording`
   false; with `work_in_tmpdir=false`, `ds_all_to_workdir` returns the recording
   directly. The startup sanity check samples 25 one-second chunks rather than
   materializing the recording. This establishes the core capability only; it
   does not validate the absent lazy prewhitening wrapper or its storage/runtime
   behavior.

4. **The request-length boundary blocker is real and appropriately scoped.**
   Exact installed Kilosort `io.py` SHA-256 `767b76a0...` centers each requested
   tensor, optionally applies median CAR, constructs `fft_highpass(...,
   NT=X.shape[1])`, filters, handles artifacts, and only then whitens. A wrapper
   that independently filters arbitrary requests can therefore change edge and
   FFT semantics unless request size, padding/overlap, cropping, channel/CAR
   order, artifact behavior and transition context are frozen and checked
   against accepted samples. The packet does not claim that equality exists.

5. **H1 capacity is current enough for a conditional arithmetic result, but
   placement is not yet exact.** The review observed 488,013,488,128 free bytes
   on the root filesystem shortly after the packet's 488,032,186,368-byte
   snapshot. `/mnt` is read-only CIFS and `/media/huklab/Data` is read-only.
   In this execution namespace `/` is broadly read-only while the project and
   `/tmp` are writable bind paths on the same ext4 filesystem. Thus "root
   ext4" is a capacity class, not yet a verified exact candidate-3 output path.
   A future contract must name and write-probe a permitted persistent directory
   and show that its filesystem accounting is the one used here. This is a
   nonblocking correction because H1 is already explicitly not launch-ready.

6. **H5 evidence is correctly stale and now has a concrete oomd caveat.** The
   sealed October 4 readiness receipt rehashes to `a06d125f...` and reports
   904,844,582,912 run-filesystem free bytes, 191,968,174,080 available memory,
   20 CPUs and one visible GPU without computation. It belongs to another
   candidate and does not bind a current candidate-3 target or prove that the
   run filesystem is the accepted input's filesystem. The packet already
   requires the exact run mount to be rebound. The independent candidate-2
   terminal packet confirms a later parent-cgroup `systemd-oomd` kill at 71.13%
   pressure, with no retained kill-time peak/composition, so available-memory
   arithmetic is not a safety proof. "Colocated" must be read as H5-local, not
   as proven same-filesystem placement, until the fresh preflight names the
   mount and target.

7. **The two-option status and topology are defensible.** H5 is conditionally
   preferred, not ready; H1 is conditional capacity only, not ready. The
   minimum topology—lazy context-correct float input, no DARTsort recopy,
   locally monitored output/scratch/cache/failure retention, and compact shared
   publication—follows from the inspected implementation and placement facts.
   It becomes invalid under the packet's stated input/path, wrapper-equality,
   throughput, capacity, oomd, competing-job, source/environment or safeguard
   changes. No scientific-performance or runtime claim follows.

No blocking factual, arithmetic, or scope defect was found. The two wording
qualifications above—accepted hashes are recorded bindings rather than fresh
payload hashes, and H1/H5 exact writable mount targets remain to be rebound—must
remain visible in any launch contract.

## Implementation checks

- Done: immutable packet membership and completion binding -> all sizes and
  hashes match the manifest and `COMPLETE.json`.
- Done: accepted input metadata and ARM-A completion binding -> path, clock,
  shape, dtype, size and recorded hashes agree; payload was not opened or
  freshly hashed.
- Done: exact DARTsort source and configuration branches -> lazy input can pass
  without a full recording copy under the stated settings.
- Done: exact Kilosort filter source -> FFT length follows each request and
  validates the stated padding/context blocker.
- Done: storage arithmetic and current H1 mount/free-space metadata -> exact
  arithmetic reproduced; exact persistent writable target remains unbound.
- Done: sealed H5 resource receipt and later oomd terminal evidence -> October 4
  capacity is historical only and current memory safety is unestablished.
- Not done: voltage equality, throughput, current H5 state, payload hash,
  benchmark, transfer, materialization, service launch or sort -> excluded by
  the frozen metadata-only review.
- Can establish: DARTsort core can avoid a 482.6 GB float32 copy; H5 is the
  simpler conditional data-local option; H1 is only a conditional fallback;
  neither is launch-ready.
- Cannot establish: wrapper equivalence, current accepted-path readability,
  exact safe storage/memory/runtime envelope, same-filesystem H5 placement,
  H1 I/O sufficiency, full-session completion or scientific performance.
