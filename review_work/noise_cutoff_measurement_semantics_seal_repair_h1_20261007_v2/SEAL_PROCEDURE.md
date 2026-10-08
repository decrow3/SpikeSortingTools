# Seal-repair procedure

This packet changes only the publication chronology/provenance boundary identified by the independent v1 review.

- V1 and its independent review are immutable inputs and are not copied over or edited.
- All v2 members are created before `MANIFEST.json`.
- `MANIFEST.json` contains byte size and SHA-256 for every member and excludes only itself and `COMPLETE.json`.
- `COMPLETE.json` binds the manifest SHA-256 and is created and published last.
- A separate bounded reviewer must read the published v2 namespace, validate every binding, and record each observed `st_mtime_ns` plus hashes. It must reject unless COMPLETE is strictly later than both MANIFEST and every member.

Acceptance of this repair does not strengthen or alter the scientific result. It establishes only that the v2 filesystem publication exhibits the requested observed seal chronology and content bindings.
