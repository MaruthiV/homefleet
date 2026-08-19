import numpy as np
from homefleet.control.baselines import hazard_step_probs, SlotMedianForecaster


def test_hazard_step_probs_sums_to_cum():
    p = hazard_step_probs([0.01, 0.02, 0.03, 0.05, 0.08], (1, 3, 6, 12, 24), 96, 0.25)
    assert np.isclose(p.sum(), 0.08, atol=1e-9)
    assert p[:4].sum() == 0.01 and (p >= 0).all()


def test_slot_median_forecaster_causal():
    slots = np.tile(np.arange(96), 40)
    x = np.tile(np.arange(96, dtype=float), 40)[:, None]
    f = SlotMedianForecaster(x, slots, days=7)
    f.update_to(96 * 30)
    fc = f.forecast(96 * 30, 96)
    assert np.allclose(fc[:, 0], np.arange(96))
