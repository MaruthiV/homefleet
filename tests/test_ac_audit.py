from pathlib import Path
import importlib.util
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEEDER = ROOT / 'data/raw/smartds/feeder/p1uhs0_1247--p1udt2142/Master_static.dss'


@pytest.mark.skipif(not FEEDER.exists() or importlib.util.find_spec('opendssdirect') is None, reason='optional feeder data and solver')
def test_ac_audit_distinguishes_disconnected_nodes_from_low_voltage():
    spec = importlib.util.spec_from_file_location('ac_check', ROOT / 'scripts/ac_check.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.solve_series(np.ones((2, 2)), np.ones((2, 2)), .3)
    assert result.conv.all()
    assert (result.disconnected_phase_nodes == 18).all()
    assert (result.all_phase_vmin == 0).all()
    assert (result.vmin > 0).all()
    assert np.isfinite(result.max_line_loading).all()
