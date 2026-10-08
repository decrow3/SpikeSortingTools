# Candidate 3 H1 actual-runtime and storage probe

- Milestone: candidate-3 host readiness.
- Decision changed: whether H1 remains a viable execution host or H5 must be selected by default.
- Cheapest adequate test: run one four-byte CUDA allocation in the exact prospective DARTsort environment outside the task sandbox; inspect device identity, running services/processes, input path locality, and filesystem capacity without reading voltage payload or writing large data.
- Inputs: `/home/huklab/Documents/DARTsort/.venv`, H1 device/runtime state, H1 REF path metadata, expected H5 Arm-A path metadata, and mounted-filesystem capacity.
- Outputs: this immutable receipt only.
- Resource bound: one float32 CUDA element (4 bytes); metadata/stat/service queries only; no voltage data read; no persistent GPU process.
- Stop condition: stop after environment allocation and metadata/capacity queries, or on any allocation/access error.
- Exclusions: no sort, no service creation, no driver/device modification, no reboot, no input transfer, no output allocation, no RF/holdout access.

The contract was frozen before the approved actual-runtime command was executed. The earlier sandbox CUDA failure is retained as a separate environment observation.
