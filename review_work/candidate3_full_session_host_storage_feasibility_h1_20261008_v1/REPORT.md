# Candidate-3 full-session host/storage feasibility

Verdict: `H5_CONDITIONALLY_PREFERRED_H1_CONDITIONAL_NOT_LAUNCH_READY`.

The exact accepted input is the 241,309,358,592-byte H5-local Arm-A int16 binary at `/media/huklaban5/Data/en_rounded_field_ks129_20260929_v1/arm_a_v1/full/recording/traces_cached_seg0.raw` (file SHA `672938e9...`, content SHA `90b6a07a...`). H1 does not mount it; H1 sees only the original REF binary over read-only CIFS.

DARTsort itself can consume a lazy SpikeInterface float recording without creating the 482,618,717,184-byte float32 copy: `preprocessing="none"`, `copy_recording_to_tmpdir="no"` (or `if_preprocessing`) and `work_in_tmpdir=false` make `ds_all_to_workdir` return the recording unchanged. This is a core capability, not current run readiness. The required lazy prewhitening wrapper is absent. Kilosort's FFT high-pass uses each request's block length, so chunk size, overlap/padding, crop, CAR/centering and transition context must be frozen and checked against accepted samples; a naïve per-request filter is not proven equivalent.

## Host options

**H5 — conditionally preferred.** It is the only host with the accepted input already local. A 2026-10-04 receipt reported 904,844,582,912 bytes free on its run volume, 191,968,174,080 bytes available memory and a visible A5000. Against the current 120 GB output planning floor, that leaves 784.8 GB without materialization. Those values are stale, the exact candidate-3 environment is not frozen there, and candidate-2 was recently killed by systemd-oomd. A fresh input/path, local free-space, memory/oomd, GPU/job and source/environment preflight is mandatory.

**H1 — conditional fallback, not launch-ready.** Root ext4 currently has 488,032,186,368 bytes free; `/media/huklab/Data` has 184,387,878,912 bytes free but is mounted read-only, and `/mnt` is read-only CIFS. Streaming plus the 120 GB output floor fits root arithmetically and leaves 368.0 GB, but 120 GB is not a safe upper bound and excludes scratch, caches and preserved failed stages. Materialization plus the floor would require 602,618,717,184 bytes, exceeding root by 114,586,530,816 bytes. H1 additionally needs the reviewed on-the-fly remap/prewhitening wrapper, real padded-window equality and a bounded CIFS throughput check.

## Minimum safe topology

Prefer H5 after fresh preflight. Read the accepted local int16 binary through a context-correct lazy float32 wrapper; configure DARTsort for no second preprocessing and no recording copy; colocate output, scratch, cache and failed-stage retention on one monitored local run volume; publish compact evidence only after completion. H1 is fallback only after wrapper/equality/I/O closure, with all writable run state on root ext4. Neither option should materialize the full float32 recording.

## Implementation checks

- Done: accepted input identity -> exact path, frames, clock, channels, dtype, file/content hashes and metadata packet hashes inspected.
- Done: materialization premise -> exact float32 byte count and DARTsort no-copy branches traced in executed checkout source (`main.py:141-160`; `main_util.py:121-148`).
- Done: boundary semantics -> Kilosort filter order and request-length FFT dependency traced (`kilosort/io.py:952-985`).
- Done: H1 placement -> current filesystem type, mount mode and free bytes inspected; only root ext4 is a viable writable large target.
- Done: H5 option -> existing sealed data-local input and October 4 resource receipt inspected; staleness and recent oomd evidence retained.
- Not done: voltage equality or I/O throughput -> prohibited by this metadata-only task and requires a separately frozen bounded check.
- Not done: safe aggregate output upper bound -> 120 GB is only a planning floor; representative-window sizing and failed-stage retention must define the future cap.
- Can establish: full float32 materialization is unnecessary at the DARTsort core boundary; H5 is the simpler data-local topology; neither host is launch-ready today.
- Cannot establish: current H5 availability, wrapper equivalence, adequate throughput, full-session runtime/storage sufficiency, scientific performance or launch readiness.
