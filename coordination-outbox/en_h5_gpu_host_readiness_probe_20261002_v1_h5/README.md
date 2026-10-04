# H5 GPU and host readiness probe

Verdict: `PASS_GPU_HOST_READY_NOW`.

The RTX A5000, driver 535.274.02, proposed PyTorch 2.6.0+cu124 runtime, CUDA allocation, and tiny synchronized kernel all pass at host level. The GPU was at 0% utilization with 24,120 MiB free; no conflicting scientific GPU job was present. All proposed fresh namespaces were absent and all relevant filesystems exceeded the frozen 200 GiB free-space threshold.

The initial sandboxed `nvidia-smi` failure was caused by platform device isolation: NVIDIA character devices are hidden inside the command sandbox. Repeating the same checks through the approved host-level read-only boundary showed the driver, devices, permissions, libraries, persistence daemon, and proposed runtime are healthy. This is neither a driver outage nor a Python/package/configuration failure.

Shared project storage is the narrowest filesystem at about 229.5 GiB free, so free space must be rechecked immediately before any future launch.

Phase-1 previously completed successfully. This probe neither changes that result nor performs waveform-method repair. GPU readiness alone does not authorize trained sorting: the saved-time validation repair, waveform/statistic decision, enabled contract, and independent one-start gate remain separate prerequisites.

Implementation checks
- Done: authoritative host driver/device state, runtime identities, tiny CUDA kernel, device permissions, persistence daemon, competing jobs, service access, disk capacity, and namespace freshness.
- Not done: no driver/service change, sort/training, voltage, RF, holdout, waveform repair, or trained-launch authorization.
- Can establish: this host and the proposed runtime were GPU-ready at the recorded time.
- Cannot establish: future availability, trained-path correctness, sort completion, scientific efficacy, identity, purity, recovery, or advancement.
