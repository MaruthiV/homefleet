import numpy as np
from homefleet.sim.tariff import Tariff, firm_margin


def test_margin_components():
    t = Tariff(p_ret=0.09, c_tdsp=0.05, beta=0.04)
    # 2 kw load, no pv, battery idle, price 100 $/mwh, 15 min
    m = firm_margin(np.array([2.0]), np.array([0.0]), np.array([0.0]), np.array([100.0]), t, 0.25)
    assert np.isclose(m[0], 0.25 * (0.09 * 2 - 0.15 * 2))
    # discharge 2 kw into own load: saves lambda + tdsp
    m2 = firm_margin(np.array([2.0]), np.array([0.0]), np.array([2.0]), np.array([100.0]), t, 0.25)
    assert np.isclose(m2[0] - m[0], 0.25 * 0.15 * 2)
    # extra export only earns lambda
    m3 = firm_margin(np.array([2.0]), np.array([0.0]), np.array([4.0]), np.array([100.0]), t, 0.25)
    assert np.isclose(m3[0] - m2[0], 0.25 * 0.10 * 2)


def test_islanded_no_grid_flows():
    t = Tariff(0.09, 0.05, 0.04)
    m = firm_margin(np.array([3.0]), np.array([1.0]), np.array([0.0]), np.array([5000.0]), t, 0.25, islanded=np.array([True]))
    assert np.isclose(m[0], 0.25 * (0.09 * 3 - 0.04 * 1))
