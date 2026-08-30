import json

from testing.luke_holdout_backend_equivalence import protocol_payload


def test_representative_chunks_cover_probes_thirds_and_motion_strata():
    chunks = protocol_payload()["chunks"]
    assert len(chunks) == 6
    assert {(row["probe"], row["time_third"]) for row in chunks} == {
        (probe, third) for probe in ("imec0", "imec1") for third in (1, 2, 3)
    }
    assert [row["motion_stratum"] for row in chunks].count("relative_quiet") == 3
    assert [row["motion_stratum"] for row in chunks].count("high_motion") == 3
    json.dumps(protocol_payload(), allow_nan=False)
