import json
from pathlib import Path

import numpy as np

from testing.first_medium_evaluator import load_arm_from_spec


CONFIG = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_evaluator_spatial_provenance_candidate_20261002_v1_h5/config/en_first_medium_four_arm_evaluator.spatial_provenance.execution_disabled.v1.json")
PACKET = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_saved_output_qualification_contract_20261002_v1_h1")
request = json.loads(CONFIG.read_text())
spec = next(row for row in request["arms"] if row["arm_id"] == "REF384")
spec = json.loads(json.dumps(spec))
spec["provenance"].pop("spatial_identity", None)
spec["spatial_provenance"]["source_generation"] = {
    "function": "kilosort.postprocessing.compute_spike_positions",
    "dependency": {
        "schema": "content-addressed-python-source-root-v1",
        "root": str(PACKET / "dependencies/kilosort/d1e5a4465d387c66a7655db442ecbf8740d2cae0032c300b0456432b0080c614"),
        "digest_sha256": "d1e5a4465d387c66a7655db442ecbf8740d2cae0032c300b0456432b0080c614",
        "files": {
            "io.py": "767b76a07647122b45048ba12cdaf49185f87f3fa33334081d4af099825414bd",
            "postprocessing.py": "d20da9026de10eb83a02466cdb233c85ae4215c15af7d5656c3593c3809a0a55",
        },
    },
}
arm = load_arm_from_spec(
    spec,
    sampling_frequency_hz=float(request["evaluation_config"]["sampling_frequency_hz"]),
    spatial_region=request["spatial_region"],
)
receipt = arm.sort["spatial_provenance"]
summary = {
    "arm_id": arm.arm_id,
    "event_rows": int(len(arm.sort["st"])),
    "cluster_count": int(len(np.unique(arm.sort["cl"]))),
    "local_frame_min": int(np.min(arm.sort["st"])),
    "local_frame_max": int(np.max(arm.sort["st"])),
    "curated_identity_digest": arm.sort["identity_digest"],
    "qc_request_digest": arm.qc["request_digest"],
    "clock_normalization": arm.provenance["clock_normalization"],
    "spatial_identity_sha256": receipt["spatial_identity_sha256"],
    "spike_positions_sha256": receipt["sha256"],
    "ops_sha256": receipt["ops_sha256"],
    "event_lineage_sha256": receipt["event_lineage_sha256"],
    "geometry_float64_sha256": receipt["geometry_float64_sha256"],
    "shape": receipt["shape"],
    "reference_frame": receipt["reference_frame"],
    "row_semantics": receipt["spatial_identity_binding"]["row_semantics"],
    "source_dependency": receipt["source_dependency"],
}
print(json.dumps(summary, indent=2, sort_keys=True))
