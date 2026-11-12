from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from homefleet.control.baselines import SlotMedianForecaster, da_aware_price_forecast
from homefleet.control.qp_layer import priority_disaggregate
from homefleet.env import HomeFleetEnv
from homefleet.sim.battery import make_fleet, clip_power, step_energy
from homefleet.sim.profiles import slot_of_day, slot_groups
from homefleet.sim.hazard import build_features


def tiny_data():
    idx = pd.date_range('2024-01-01', periods=96 * 32, freq='15min', tz='America/Chicago')
    T = len(idx)
    d = SimpleNamespace(index=idx, T=T, N_pool=2, slot=slot_of_day(idx), groups=slot_groups(idx),
                        load=np.full((T, 2), 2.), pv=np.zeros((T, 2)), price_rt=np.full(T, 50.),
                        price_da=np.full(T, 45.), home_fips=np.array(['48201'] * 2),
                        frac=pd.DataFrame({'48201': np.zeros(T)}, index=idx),
                        alert=pd.DataFrame({'48201': np.zeros(T, dtype=bool)}, index=idx),
                        weather=np.zeros((T, 1, 7)), hazard=None)
    d.county_idx = lambda h: np.zeros(len(h), dtype=int)
    return d


def test_future_changes_do_not_change_slot_forecast():
    slots = np.tile(np.arange(96), 40)
    x = np.full((len(slots), 1), 2.)
    x[28 * 96:] = 1e6
    f = SlotMedianForecaster(x, slots, start=28 * 96)
    assert np.all(f.forecast(28 * 96, 96) == 2.)
    f.update_to(28 * 96 + 1)
    assert f.seen == 28 * 96 + 1


def test_unpublished_next_day_prices_are_hidden_across_dst():
    idx = pd.date_range('2024-03-10', periods=96 * 3, freq='15min', tz='America/Chicago')
    k = idx.get_loc(pd.Timestamp('2024-03-10 13:00', tz=idx.tz))
    da = np.full(len(idx), 4000.)
    result = da_aware_price_forecast(k, 96, da, np.full(96, 50.), lambda t: idx[t].hour, index=idx)
    tomorrow = idx[k:k + 96].date != idx[k].date()
    assert (result[tomorrow] == 50.).all()
    assert (result[~tomorrow] == 4000.).all()


def test_reserve_and_observation_ignore_future_loads_and_rt_prices():
    d = tiny_data()
    env = HomeFleetEnv(d)
    start = 28 * 96
    a = env.reset(start, 96, [0, 1], ['gen2_single'] * 2)
    need = env.need_lut.copy()
    d.load[start:] = 1000.
    d.price_rt[start:] = 1e6
    b = env.reset(start, 96, [0, 1], ['gen2_single'] * 2)
    assert np.array_equal(need, env.need_lut)
    assert np.array_equal(a, b)


@pytest.mark.parametrize('preset', ['gen2_single', 'gen2_double', 'core'])
def test_guardrail_charges_even_for_idle_or_discharge_proposal(preset):
    p = make_fleet([preset])
    floor = .9 * p.energy_kwh
    initial = p.soc_min * p.energy_kwh
    steps = int(np.ceil((floor[0] - initial[0]) / (.25 * p.eta_c * p.p_charge_kw[0])))
    for target in (0., 10.):
        e = initial.copy()
        for _ in range(steps):
            pc, pd_, _ = priority_disaggregate(target, e, p, np.ones(1), np.ones(1), floor, .25)
            assert not ((pc > 0) & (pd_ > 0)).any()
            assert (pc <= p.p_charge_kw).all()
            e = step_energy(e, pc, pd_, p, .25)
        assert np.allclose(e, floor)


def test_unavailable_battery_cannot_satisfy_raised_floor():
    p = make_fleet(['gen2_single'])
    e = np.array([2.5])
    pc, pd_ = clip_power(e, [0.], [10.], p, np.ones(1), np.zeros(1), np.array([22.5]), .25)
    assert pc[0] == pd_[0] == 0


def test_hazard_features_use_completed_hours_only():
    idx = pd.date_range('2020-01-01', periods=48, freq='1h', tz='America/Chicago')
    w = pd.DataFrame({k: np.ones(48) for k in ['wind_gusts_10m', 'precipitation', 'temperature_2m', 'surface_pressure']}, index=idx)
    frac = pd.Series(0., index=idx)
    x, _ = build_features(w, frac, {})
    w.iloc[24:] = 1000.
    frac.iloc[24:] = 1.
    y, _ = build_features(w, frac, {})
    pd.testing.assert_frame_equal(x.iloc[:25], y.iloc[:25])


def test_hazard_threshold_cannot_lower_the_shared_operating_minimum(monkeypatch):
    import homefleet.control.baselines as baselines
    data = tiny_data()
    data.load[:] = 0.
    data.hazard = np.full((data.T, 1, 5), .8)
    data.hazard_h = [1, 3, 6, 12, 24]
    env = HomeFleetEnv(data)
    env.reset(28 * 96, 4, [0], ['gen2_single'])
    observed = []
    def plan(e0, load, pv, price, floors, *args, **kwargs):
        observed.append(floors.copy())
        return np.zeros_like(load), np.zeros_like(load)
    monkeypatch.setattr(baselines, 'dp_plan', plan)
    baselines.run_controller(env, mode='chance', eps_cc=.1)
    minimum = .1 * env.params.energy_kwh * env.soh
    assert observed and all((floors >= minimum - 1e-9).all() for floors in observed)
