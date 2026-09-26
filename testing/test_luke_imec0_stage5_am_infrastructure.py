from pathlib import Path


def test_am3_controller_delegates_slow_fit_to_medicine_environment():
    source = Path("testing/luke_imec0_stage5_am.py").read_text()
    assert '"am3-slow-fit"' in source
    assert '[str(stage1.MEDPY),str(Path(__file__).resolve()),"am3-slow-fit"]' in source
    assert 'elif a.phase=="am3-slow-fit":am3_slow_fit()' in source
    assert "AM.3 slow fit returned without a receipt" in source
