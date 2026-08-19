import numpy as np
from homefleet.sim.degradation import rainflow_cycles, cycle_based, energy_throughput


def test_rainflow_simple():
    soc = np.array([0.5, 1.0, 0.2, 0.9, 0.2, 1.0])
    cyc = rainflow_cycles(soc)
    assert sum(c for _, c in cyc) > 0
    assert cycle_based(soc) > 0
    assert np.isclose(energy_throughput(1000.0, 25.0), 2.71e-5 * 40)
