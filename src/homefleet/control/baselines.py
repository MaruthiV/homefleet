from datetime import timedelta
import numpy as np
from homefleet.control.dp import dp_plan

HOUR_STEPS = 4


class SlotMedianForecaster:
    # only observations strictly before start initialize the forecast
    def __init__(self, series_TxN, slots, days=28, start=0):
        self.x = np.asarray(series_TxN, dtype=float)
        self.slots = slots
        self.days = days
        self.buf = {s: [] for s in range(96)}
        for t in range(max(0, start - days * 96), start):
            self.buf[self.slots[t]].append(self.x[t])
        self.med = np.stack([np.median(np.stack(self.buf[s]), axis=0) if self.buf[s] else np.zeros(self.x.shape[1:]) for s in range(96)])
        self.seen = start

    def update_to(self, t):
        while self.seen < t:
            s = self.slots[self.seen]
            self.buf[s].append(self.x[self.seen])
            if len(self.buf[s]) > self.days:
                self.buf[s].pop(0)
            self.med[s] = np.median(np.stack(self.buf[s]), axis=0)
            self.seen += 1

    def forecast(self, t, H):
        positions = np.minimum(t + np.arange(H), len(self.slots) - 1)
        return self.med[self.slots[positions]]


def da_aware_price_forecast(k, H, price_da, price_rt_slot_med, hour_of_day, index=None):
    # dam price known for today (and tomorrow after 14:00), else trailing slot median of rt
    T = len(price_da)
    positions = np.minimum(k + np.arange(H), T - 1)
    day0 = k // 96 if index is None else index[k].date()
    days = positions // 96 if index is None else index[positions].date
    tomorrow = day0 + 1 if index is None else day0 + timedelta(days=1)
    known = (days == day0) | ((days == tomorrow) & (hour_of_day(k) >= 14))
    return np.where(known, price_da[positions], price_rt_slot_med)


def hazard_step_probs(hz_row, horizons_h, H, dt):
    # cumulative p(event within k hours) -> per-step onset probability over the horizon
    cum = np.r_[0.0, np.maximum.accumulate(np.asarray(hz_row, dtype=float))]
    edges = np.r_[0, np.array(horizons_h) / dt].astype(int)
    p = np.zeros(H)
    for j in range(len(horizons_h)):
        a, b = edges[j], min(edges[j + 1], H)
        if b > a:
            p[a:b] = (cum[j + 1] - cum[j]) / (edges[j + 1] - edges[j])
    if edges[-1] < H:
        p[edges[-1]:] = p[edges[-1] - 1]
    return np.clip(p, 0, 1)


def run_controller(env, mode, H=96, resolve_every=HOUR_STEPS, S=40, lam_r=0.0, eps_cc=0.05, fixed_frac=0.20, deg_cost_per_kwh=0.0, price_fc="da",
                   idle_target_frac=None, idle_penalty_per_kwh=0.02, oracle_hazard=False):
    # mode in {fixed, storm_watch, az_tier, risk_priced, chance, oracle}; oracle uses realized prices/loads
    # idle_target_frac: base-style hold-high preference (e.g. 0.9) on top of any floor
    d = env.d
    hod = lambda t: d.index[t].hour
    fc_load = SlotMedianForecaster(d.load[:, env.h], d.slot, start=env.t0)
    fc_pv = SlotMedianForecaster(d.pv[:, env.h], d.slot, start=env.t0)
    fc_price = SlotMedianForecaster(d.price_rt[:, None], d.slot, start=env.t0)
    onset = None
    if oracle_hazard:
        # realized county onsets replace the hazard forecast, same objective
        fr = d.frac.to_numpy()[env.t0:env.t0 + env.T][:, env.cidx] >= 0.01
        onset = fr.copy()
        onset[1:] &= ~fr[:-1]
    E = env.params.energy_kwh * env.soh
    emin = env.emin()
    plan_p = None
    for k in range(env.T):
        if k % resolve_every == 0:
            Hk = min(H, env.T - k)
            if mode == "oracle":
                load_f, pv_f, price_f = env.load[k:k + Hk], env.pv[k:k + Hk], env.price[k:k + Hk]
            else:
                t = env.t0 + k
                fc_load.update_to(t); fc_pv.update_to(t); fc_price.update_to(t)
                load_f, pv_f = fc_load.forecast(t, Hk), fc_pv.forecast(t, Hk)
                pm = fc_price.forecast(t, Hk)[:, 0]
                price_f = da_aware_price_forecast(t, Hk, d.price_da, pm, hod, index=d.index) if price_fc == "da" else pm
            need = None; hz = None
            minimum = np.maximum(emin, env.cfg.get('floor_hard_min', .10) * E)
            floors = np.broadcast_to(fixed_frac * E if mode in ('fixed', 'storm_watch') else minimum, (Hk, env.N)).copy()
            if mode == "storm_watch" and env.alert[k].any():
                floors = np.where(env.alert[k][None, :], E[None, :], floors)
            if mode == "az_tier":
                floors = np.stack([env.tier_reserve_now(min(k + h, env.T - 1)) for h in range(Hk)])
            if mode in ("risk_priced", "chance"):
                need = np.stack([env.need_now(min(k + h, env.T - 1)) + emin for h in range(Hk)])
                if oracle_hazard:
                    hz = onset[k:k + Hk].astype(float)
                else:
                    if env.hz is None:
                        raise ValueError("hazard missing in data bundle")
                    hz = np.stack([hazard_step_probs(env.hz[k][c], d.hazard_h, Hk, env.dt) for c in range(env.hz.shape[1])], axis=1)
                if mode == "chance":
                    cum = np.stack([np.maximum.accumulate(np.cumsum(hz[:, c])) for c in range(hz.shape[1])], axis=1)
                    capped = np.minimum(np.maximum(need, minimum[None, :]), 0.9 * E[None, :])
                    floors = np.where(cum >= eps_cc, capped, floors)
                    hz = None; need = None
            if mode == "oracle":
                floors = np.full((Hk, env.N), emin)
            avail_now = env.avail[k].astype(float)
            idle = None if idle_target_frac is None else idle_target_frac * E
            _, plan_p = dp_plan(env.e, load_f, pv_f, price_f, floors, env.params, env.soh, avail_now, env.tariff, env.dt, S=S,
                                lam_r=lam_r if mode == "risk_priced" else 0.0, need=need, hazard=hz, deg_cost_per_kwh=deg_cost_per_kwh,
                                idle_target=idle, idle_penalty_per_kwh=idle_penalty_per_kwh)
            plan_floor = floors
            j0 = k
        p = plan_p[k - j0]
        pc = np.maximum(-p, 0.0)
        pd_ = np.maximum(p, 0.0)
        floor = plan_floor[min(k - j0, len(plan_floor) - 1)]
        if mode == 'storm_watch':
            floor = np.where(env.alert[k], E, fixed_frac * E)
        env.step_dispatch(pc, pd_, floor)
    return env.results()
