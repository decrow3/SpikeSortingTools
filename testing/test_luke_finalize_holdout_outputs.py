from pathlib import Path

from testing.luke_finalize_holdout_outputs import finalize

import pytest

from testing.luke_draw_prospective_holdout_events import MANIFEST, OUTPUT

# These assertions read the sealed v2 holdout artifacts, which are generated
# research evidence under testing/outputs/ and are deliberately not tracked.
# On a fresh clone they are absent; regenerate them with
# ``python testing/luke_draw_prospective_holdout_events.py`` before running this.
pytestmark = pytest.mark.skipif(
    not MANIFEST.is_file(),
    reason=f"sealed holdout artifacts not present under {OUTPUT}",
)



def test_finalized_reviewer_artifact_is_opaque():
    roles = finalize()
    reviewer = Path(roles["reviewer_facing"]["path"])
    assert reviewer.read_text().splitlines()[0] == "candidate_id"
    assert len(reviewer.read_text().splitlines()) == 865
    assert roles["internal_stratification_manifest"]["not_reviewer_facing"] is True
    assert roles["sealed_coordinate_key"]["not_reviewer_facing"] is True
