import os
import numpy as np
import pytest
from homefleet.control.dp import dp_plan, backward
from homefleet.sim.battery import make_fleet
from homefleet.sim.tariff import Tariff


@pytest.mark.skipif(backward is None, reason='optional numba accelerator not installed')
@pytest.mark.parametrize('risk,idle', [(0., None), (5., None), (0., np.array([22.5, 45.]))])
def test_compiled_solver_matches_numpy(monkeypatch, risk, idle):
    rng = np.random.default_rng(123)
    p = make_fleet(['gen2_single', 'gen2_double'])
    H = 24
    args = (np.array([12.5, 25.]), rng.uniform(0, 5, (H, 2)), rng.uniform(0, 2, (H, 2)),
            rng.uniform(-100, 5000, H), np.full((H, 2), 10.), p, np.ones(2), np.ones(2),
            Tariff(.09, .05, .04), .25)
    kwargs = dict(S=32, lam_r=risk, idle_target=idle, idle_penalty_per_kwh=.02,
                  need=np.full((H, 2), 15.), hazard=np.full((H, 2), .01))
    monkeypatch.setenv('HOMEFLEET_NUMPY_DP', '1')
    expected = dp_plan(*args, **kwargs)
    monkeypatch.delenv('HOMEFLEET_NUMPY_DP')
    actual = dp_plan(*args, **kwargs)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
