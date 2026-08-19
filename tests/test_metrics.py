import numpy as np
from homefleet.eval.metrics import cvar, ride_through_rate, margin_per_home_week, bootstrap_ci


def test_cvar_and_rate():
    x = np.arange(100, dtype=float)
    assert cvar(x, 0.05) == np.mean(np.arange(5))
    assert ride_through_rate(3, 100) == 0.97
    assert margin_per_home_week(np.ones(4 * 24 * 7), 1, 0.25) == 4 * 24 * 7
    lo, hi = bootstrap_ci(np.random.default_rng(0).normal(size=200), np.mean)
    assert lo < 0 < hi
