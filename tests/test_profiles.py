import numpy as np
import pandas as pd
from homefleet.sim.profiles import map_profile_to_year, forward_positive_energy, slot_quantile, year_index


def test_map_profile_keeps_clock_alignment():
    idx18 = year_index(2018)
    s = pd.Series(np.arange(len(idx18), dtype=float), index=idx18)
    m = map_profile_to_year(s, 2021)
    assert len(m) == len(year_index(2021))
    # same local clock on a plain day maps to the same value
    assert m.loc["2021-06-10 12:00"] == s.loc["2018-06-10 12:00"]
    assert not m.isna().any()


def test_forward_energy():
    e = forward_positive_energy([2, -1, 4, 0], steps=2, dt=0.25)
    assert np.allclose(e, [0.5, 1.0, 1.0, 0.0])


def test_slot_quantile_shape():
    idx = year_index(2018)
    v = np.tile(np.arange(96), len(idx) // 96)
    q = slot_quantile(v, idx, q=0.9, window_slots=0)
    assert q.shape == (96,) and q[10] == 10
