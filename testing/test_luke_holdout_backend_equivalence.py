import json

import pytest

from testing.luke_draw_prospective_holdout_events import MANIFEST, OUTPUT
from testing.luke_holdout_backend_equivalence import protocol_payload

# See test_luke_finalize_holdout_outputs.py: the sealed v2 holdout artifacts
# are untracked generated research evidence.
requires_sealed_holdout = pytest.mark.skipif(
    not MANIFEST.is_file(),
    reason=f"sealed holdout artifacts not present under {OUTPUT}",
)


@requires_sealed_holdout
def test_representative_chunks_cover_probes_thirds_and_motion_strata():
    chunks = protocol_payload()["chunks"]
    assert len(chunks) == 6
    assert {(row["probe"], row["time_third"]) for row in chunks} == {
        (probe, third) for probe in ("imec0", "imec1") for third in (1, 2, 3)
    }
    assert [row["motion_stratum"] for row in chunks].count("relative_quiet") == 3
    assert [row["motion_stratum"] for row in chunks].count("high_motion") == 3
    json.dumps(protocol_payload(), allow_nan=False)
