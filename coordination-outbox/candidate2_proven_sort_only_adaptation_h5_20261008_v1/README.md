# Candidate2 proven sort-only adaptation

This is a minimal adaptation of the October 6 imec1 bounded-pread sort-only runner that completed one 241 GB sort in 14,168.845 seconds. It retains cache-dropping exact-target full-content validation, whole-file `DONTNEED` plus pressure settle/recheck, and the exact-target reusable bounded reader.

The intended change is Candidate2 science: the runner binds the accepted imec0 cache/clock/384-channel extent, consumes the complete saved 46-parameter native-rigid request, calls the reviewed `RESCUE_RIGID` implementation with those exact parameters, and requires saved effective `nblocks=1`. It never falls back to REF defaults or the Arm-A remap.

The ordinary systemd service uses one explicit source path, persistent stdout/stderr, an 800% CPU quota and an eight-hour timeout, with no MemoryHigh or MemoryMax. No nested bootstrap or mandatory external monitor exists. The literal final entrypoint/config passed a metadata/import/path boundary check and a changed-channel negative without reading recording content.
