# No-mask REF-repeat heterogeneous launcher candidate

This packet implements the previously missing serial wrapper for the first
prospective 600-second pair:

1. `REF384_repeat` through the exact historical no-mask path.
2. `repaired_B384` through the support-aware clock-v5 trained entry.

Only immutable execution completion state is observed between arms. Scientific
outcomes cannot influence whether the second arm runs. Failures and completed
first-arm evidence are preserved without writing a pair COMPLETE.

Current state: **execution disabled and ready for independent H1 review**.
Neither the proposed service nor any scientific job was installed or started.

The full-voltage identity service has already completed successfully with an
exact 241,309,358,592-byte digest match and zero retries. The clock-v5 repair
has a scoped H5 GO. These satisfy the promotion prerequisites but do not replace
the required independent launcher review or enabled one-start contract.
