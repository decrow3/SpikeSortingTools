from pathlib import Path

from testing.luke_finalize_holdout_outputs import finalize


def test_finalized_reviewer_artifact_is_opaque():
    roles = finalize()
    reviewer = Path(roles["reviewer_facing"]["path"])
    assert reviewer.read_text().splitlines()[0] == "candidate_id"
    assert len(reviewer.read_text().splitlines()) == 865
    assert roles["internal_stratification_manifest"]["not_reviewer_facing"] is True
    assert roles["sealed_coordinate_key"]["not_reviewer_facing"] is True
