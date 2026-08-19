import numpy as np
import pandas as pd
from homefleet.sim.outage import event_windows, draw_outages, episodes


def test_event_windows():
    f = np.array([0, 0, 0.01, 0.02, 0, 0, 0.5, 0])
    assert event_windows(f, 0.002) == [(2, 4), (6, 7)]


def test_quantile_matching_monotone():
    idx = pd.date_range("2024-07-08", periods=8, freq="15min", tz="America/Chicago")
    panel = pd.DataFrame({"48201": [0, 0.2, 0.8, 0.8, 0.3, 0, 0, 0]}, index=idx)
    out = draw_outages(panel, ["48201"] * 1000, np.random.default_rng(0))
    share = out.mean(axis=1)
    # share of homes out tracks county fraction, and a home out at 0.2 is also out at 0.8
    assert abs(share[2] - 0.8) < 0.05 and abs(share[1] - 0.2) < 0.05
    assert np.all(out[2][out[1]])
    eps = episodes(out[:, 0])
    assert all(e > s for s, e in eps)
