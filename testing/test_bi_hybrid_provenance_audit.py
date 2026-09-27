from testing.bi_hybrid_provenance_audit import fixture_bundle, validate_fixture


def test_proposed_hybrid_provenance_fixture():
    got = validate_fixture(fixture_bundle())
    assert got == {
        "pass": True,
        "donors": 1,
        "source_events": 2,
        "memberships": 2,
        "foreign_keys_valid": True,
        "local_absolute_clock_valid": True,
        "template_hash_lineage_valid": True,
    }
