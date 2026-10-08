# Result

Verdict: `H1_GPU_RUNTIME_GO_STORAGE_AND_INPUT_ROUTE_UNRESOLVED`.

The exact prospective DARTsort interpreter successfully imported the editable checkout and allocated one four-byte float32 tensor on H1's NVIDIA RTX A5000 through the approved actual execution context. This corrects the host-level interpretation of the earlier task-sandbox result: the sandbox lacked NVIDIA device exposure, but H1 itself has a working CUDA path.

H1 is therefore a viable GPU candidate, not an automatic rejection. Host selection is still unresolved because the accepted Arm-A materialization is H5-local and absent on H1. H1 sees the 241,309,358,592-byte REF source over read-only CIFS. A real H1 run would need a frozen, reviewed streaming reconstruction path or a separately authorized input route.

Storage is the tighter H1 constraint. `/media/huklab/Data` had 184,387,878,912 bytes free and root had 488,675,999,744 bytes free. The planning minimum for output is about 120 GB; a whole-recording float32 prewhitening materialization would be 482,618,717,184 bytes. Neither existing local target can safely hold both. The real contract must avoid full materialization or bind a different approved writable target. No throughput benchmark was run, so I/O performance remains unknown.

No Candidate-3 service/process was active. Historical services were observed but do not define the prospective runner.

Implementation checks
- Done: exact runtime -> DARTsort `.venv` Python, editable import, commit, lock hash, Torch/CUDA versions, device identity, and a four-byte CUDA allocation were observed in the approved actual context.
- Done: environment comparison -> task sandbox failure and actual-context success are explicitly separated.
- Done: input locality -> H1 REF exists over read-only CIFS; accepted H5 Arm-A path is absent on H1.
- Done: capacity arithmetic -> current free bytes were compared with the planning output floor and full float32 materialization size.
- Done: current job state -> actual user-service, process, and GPU-process views were checked; no Candidate-3 job was found.
- Not done: sustained I/O throughput -> no voltage payload was read; requires a separately frozen bounded benchmark if host choice depends on it.
- Not done: real adapter correctness -> requires the reviewed real snippet contract and execution.
- Can establish: H1 has a working exact-environment CUDA path and remains a candidate host, subject to input-route and storage design.
- Cannot establish: real Arm-A equivalence, adequate sustained I/O, full-session readiness, runtime, or superiority of H1 versus H5.
