from testing.em2g_publish_compact import PRODUCTS


def test_compact_packet_excludes_voltage_and_large_scratch_products():
    assert "RECEIPT.json" in PRODUCTS
    assert all(not name.endswith(".npz") for name in PRODUCTS)
    assert all("PAIR_SCORES" not in name for name in PRODUCTS)
    assert all("voltage" not in name.lower() for name in PRODUCTS)
