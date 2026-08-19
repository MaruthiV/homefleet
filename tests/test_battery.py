import numpy as np
from homefleet.sim.battery import make_fleet, clip_power, step_energy, envelope


def test_roundtrip_efficiency():
    f = make_fleet(["gen2_single"])
    e = np.array([12.5])
    soh = np.ones(1)
    av = np.ones(1)
    floor = np.array([2.5])
    pc, pd_ = clip_power(e, np.array([10.0]), np.array([0.0]), f, soh, av, floor, 0.25)
    e1 = step_energy(e, pc, pd_, f, 0.25)
    pc2, pd2 = clip_power(e1, np.array([0.0]), np.array([10.0]), f, soh, av, floor, 0.25)
    e2 = step_energy(e1, pc2, pd2, f, 0.25)
    # charge 2.5 kwh in, get 0.9*2.5 back out
    assert np.isclose(e1[0], 12.5 + 0.25 * 10 * f.eta_c)
    assert np.isclose(e1[0] - e2[0], 0.25 * 10 / f.eta_d)


def test_floor_and_inverter_limits():
    f = make_fleet(["core"])
    e = np.array([5.0])
    soh = np.ones(1)
    av = np.ones(1)
    floor = np.array([3.92])
    pc, pd_ = clip_power(e, np.array([0.0]), np.array([50.0]), f, soh, av, floor, 0.25)
    # can't go below floor and can't exceed 20 kw
    assert pd_[0] <= 20.0 + 1e-9
    assert step_energy(e, pc, pd_, f, 0.25)[0] >= 3.92 - 1e-9
    pc_max, pd_max = envelope(e, f, soh, av, floor, 0.25)
    assert np.isclose(pd_max[0], pd_[0])


def test_unavailable_home_cannot_move():
    f = make_fleet(["gen2_double"])
    pc, pd_ = clip_power(np.array([25.0]), np.array([5.0]), np.array([5.0]), f, np.ones(1), np.zeros(1), np.array([5.0]), 0.25)
    assert pc[0] == 0 and pd_[0] == 0
