# Candidate2 stable systemd identity and complete preflight test

This fresh v8 derivative preserves v7's terminal pre-read failure. It replaces unstable human-formatted `systemctl show` command strings with systemd's typed user-bus API: `Manager.GetUnit`, followed by `Service.ExecStart` and `ExecStartPre` with signature `a(sasbttttuii)`. Authorization compares only the stable typed execution identity—executable path, exact argv array, and `ignore_errors`—for every command in exact order and cardinality. Empty arrays remain explicit; malformed signatures, malformed rows, and query failures reject.

All other manager scalars, dependency membership, unit/source/environment/resource bindings, GPU/process gates, sorter settings, monitor behavior, and the absence of memory/runtime caps remain strict. The preflight also repairs an independent line-local defect: systemctl stdout no longer overwrites the `Path` object later used to create the receipt parent.

Before production deployment, the actual complete preflight main passed under temporary realistic user services captured inactive and then run active. It traversed source/input/resource/GPU/process/systemd/monitor/Kilosort/environment checks and serialized a valid receipt. Changed executable path, argv boundary/cardinality, and `ignore_errors` each rejected. No recording voltage was opened or read and no sorter work ran.

Independent focused review returned GO. A final no-start deployed capture matched the reviewed typed identity (the only textual delta was dependency set ordering, which the reviewed comparator canonicalizes), and exact typed authorization is sealed for one v8 attempt with no retry. At packet seal time the production services remained inactive with PID 0 and the target disabled.
