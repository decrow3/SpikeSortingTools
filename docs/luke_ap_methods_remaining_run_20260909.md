# Remaining 8σ motion comparisons

Authorized continuation of the 930–1030 s sweep: four MEDiCINe screens, then four DREDGE screens. Original 32 sealed fits and failed 8σ DREDGE evidence remain unchanged. Offset conventions and frozen lighthouse observations remain unchanged.

Independent service: `luke-ap-methods-remaining-v1.service`; persistent launch, logs and receipt in `testing/outputs/luke_ap_methods_remaining_v1_job`. Per-arm commands, input hash checks, receipts and successful checksums in `testing/outputs/luke_ap_methods_remaining_v1`. This launch mechanism was previously verified with a dummy job; the current service was checked active after launcher exit. Completed arms can be reused after checksum validation, but an interrupted fit must restart from the beginning.

Separate failure reproduction: `luke-dredge-8sigma-failure-audit-v1.service`, outputs in `testing/outputs/luke_dredge_8sigma_failure_audit_v1`. The original strict assertion is retained while correlation curves, old shifts and old correlations are saved for mismatches. Installed DREDGE selects CUDA automatically; the bounded helper reconstructs on CPU. Cross-device near ties are a hypothesis to test, not an established explanation at launch.
