import numpy as np
import pandas as pd
import yaml
from homefleet.paths import CONFIGS
from homefleet.sim.battery import make_fleet, clip_power, step_energy, envelope
from homefleet.sim.tariff import load_tariff, firm_margin
from homefleet.sim.outage import draw_outages
from homefleet.sim.profiles import forward_positive_energy, slot_quantile, slot_of_day, slot_groups
from homefleet.control.qp_layer import priority_disaggregate
from homefleet.control.baselines import SlotMedianForecaster, da_aware_price_forecast

TZ = "America/Chicago"


class FleetData:
    def __init__(self, npz_path):
        z = np.load(npz_path, allow_pickle=True)
        self.index = pd.DatetimeIndex(pd.to_datetime(z["index"], utc=True)).tz_convert(TZ)
        self.price_rt = z["price_rt"]
        self.price_da = z["price_da"]
        self.load = z["load_kw"].astype(float)
        self.pv = z["pv_kw"].astype(float)
        self.home_fips = z["home_fips"].astype(str)
        self.fips_list = list(z["fips_list"].astype(str))
        self.frac = pd.DataFrame(z["frac"], index=self.index, columns=self.fips_list)
        self.alert = pd.DataFrame(z["alert"], index=self.index, columns=self.fips_list)
        self.weather = z["weather"]
        self.weather_cols = list(z["weather_cols"].astype(str))
        self.hazard = z["hazard"] if "hazard" in z.files else None
        self.hazard_h = list(z["hazard_h"]) if "hazard_h" in z.files else None
        self.zone = str(z["zone"])
        self.year = int(z["year"])
        self.protocol = str(z['protocol']) if 'protocol' in z.files else 'submitted-v1'
        self.evaluation_start = int(z['evaluation_start']) if 'evaluation_start' in z.files else 28 * 96
        self.home_id = z['home_id'].astype(str) if 'home_id' in z.files else np.arange(self.load.shape[1]).astype(str)
        self.slot = slot_of_day(self.index)
        self.groups = slot_groups(self.index)
        self.T, self.N_pool = self.load.shape

    def county_idx(self, home_idx):
        return np.array([self.fips_list.index(f) for f in self.home_fips[home_idx]])


def load_cfg():
    with open(CONFIGS / "base.yaml") as f:
        return yaml.safe_load(f)


class HomeFleetEnv:
    # one fleet episode over a contiguous window of a FleetData year; homes are a subset of the pool
    def __init__(self, data, presets=None, cfg=None, tariff=None):
        self.d = data
        self.cfg = cfg or load_cfg()
        self.tariff = tariff or load_tariff()
        self.dt = self.cfg["step_hours"]
        self.presets = presets
        self.kappa = self.cfg.get("critical_load_fraction", 1.0)

    def reset(self, start, n_steps, home_idx, preset_names, seed=0, soc0=0.9, soh=None, dropout=0.0, tiers_h=None, need_hours=5.0):
        if start < 28 * 96 or n_steps <= 0 or start + n_steps > self.d.T:
            raise ValueError("episodes require 28 days of preceding history and a valid end")
        self.rng = np.random.default_rng(seed)
        self.t0, self.T, self.h = start, n_steps, np.asarray(home_idx)
        self.N = len(self.h)
        self.params = make_fleet(preset_names, self.presets)
        self.soh = np.ones(self.N) if soh is None else np.asarray(soh, dtype=float)
        self.e = soc0 * self.params.energy_kwh * self.soh
        self.cidx = self.d.county_idx(self.h)
        self.fips = self.d.home_fips[self.h]
        win = slice(start, start + n_steps)
        sub = self.d.frac.iloc[win][list(dict.fromkeys(self.fips))]
        self.out = draw_outages(sub, self.fips, self.rng, tau=self.cfg["event_window_threshold"])
        self.avail = (self.rng.random((n_steps, self.N)) >= dropout) if dropout > 0 else np.ones((n_steps, self.N), dtype=bool)
        self.load = self.d.load[win][:, self.h]
        self.pv = self.d.pv[win][:, self.h]
        self.net = self.load - self.pv
        self.price = self.d.price_rt[win]
        self.price_da = self.d.price_da[win]
        self.alert = self.d.alert.iloc[win].to_numpy()[:, self.cidx]
        self.hz = None if self.d.hazard is None else self.d.hazard[win][:, self.cidx, :]
        self._build_luts(need_hours, tiers_h)
        self.price_forecaster = SlotMedianForecaster(self.d.price_rt[:, None], self.d.slot, start=start)
        self.k = 0
        self.promise_mask = np.zeros_like(self.out)
        from homefleet.sim.outage import episodes
        for j in range(self.N):
            for s, end in episodes(self.out[:, j]):
                self.promise_mask[s:min(end, s + int(round(5.0 / self.dt))), j] = True
        self.initial_energy = self.e.copy()
        self.log = {key: [] for key in ("margin", "unserved", "pc", "pd", "e", "out", "shield", "floor_shortfall", "physical_viol", "floor_viol")}
        return self.obs()

    def _build_luts(self, need_hours, tiers_h):
        # reserve statistics are frozen from the 28 days preceding the episode
        a = self.t0 - 28 * 96
        net = self.d.load[a:self.t0, self.h] - self.d.pv[a:self.t0, self.h]
        def historical_lut(hours):
            st = int(round(hours / self.dt))
            valid = len(net) - st + 1
            fe = forward_positive_energy(net, st, self.dt)[:valid]
            return slot_quantile(fe, slot_groups(self.d.index[a:a + valid]), q=0.9) / self.params.eta_d
        self.need_lut = historical_lut(need_hours)
        E = self.params.energy_kwh * self.soh
        menu = [24, 12, 8, 6, 4, 2] if tiers_h is None else sorted(set(tiers_h), reverse=True)
        luts = {T: historical_lut(T) for T in menu}
        self.tier_lut = {}
        self.tiers_h = np.zeros(self.N)
        for j in range(self.N):
            cands = menu if tiers_h is None else [float(tiers_h[j])]
            for T in cands:
                lut = luts[T][:, j] + self.emin()[j]
                if lut.max() <= 0.8 * E[j] or T == cands[-1]:
                    self.tiers_h[j] = T
                    self.tier_lut[j] = np.minimum(lut, 0.8 * E[j])
                    break

    def need_now(self, k=None):
        k = self.k if k is None else k
        return self.need_lut[self.d.slot[self.t0 + k]]

    def tier_reserve_now(self, k=None):
        k = self.k if k is None else k
        s = self.d.slot[self.t0 + k]
        return np.array([self.tier_lut[j][s] for j in range(self.N)])

    def emin(self):
        return self.params.soc_min * self.params.energy_kwh

    def step_dispatch(self, pc, pd_, floor):
        # apply requested per-home charge/discharge under floor; islanded homes are overridden
        k = self.k
        out = self.out[k]
        av = self.avail[k].astype(float)
        floor_eff = np.where(out, self.emin(), floor)
        pc, pd_ = clip_power(self.e, np.asarray(pc, float), np.asarray(pd_, float), self.params, self.soh, av, floor_eff, self.dt)
        # islanded: battery must serve critical load net of pv; pv surplus charges
        crit = self.kappa * self.load[k]
        island_need = np.maximum(crit - self.pv[k], 0.0)
        island_surplus = np.maximum(self.pv[k] - crit, 0.0)
        pc_i, pd_i = clip_power(self.e, island_surplus, island_need, self.params, self.soh, np.ones(self.N), self.emin(), self.dt)
        pc = np.where(out, pc_i, pc)
        pd_ = np.where(out, pd_i, pd_)
        unserved = np.where(out, self.dt * np.maximum(island_need - pd_, 0.0), 0.0)
        self.unserved_promise = np.where(self.promise_mask[k], unserved, 0.0)
        p = pd_ - pc
        served_load = self.load[k] - unserved / self.dt
        margin = firm_margin(served_load, self.pv[k], p, self.price[k], self.tariff, self.dt, islanded=out)
        reachable_floor = np.minimum(floor_eff, self.e + self.dt * self.params.eta_c * self.params.p_charge_kw * np.where(out, 1.0, av))
        self.e = step_energy(self.e, pc, pd_, self.params, self.dt)
        self.log["floor_shortfall"].append(np.maximum(floor_eff - self.e, 0))
        self.log["floor_viol"].append(self.e < reachable_floor - 1e-7)
        self.log["physical_viol"].append((self.e < self.emin() - 1e-7) | (self.e > self.params.energy_kwh * self.soh + 1e-7))
        self.log["margin"].append(margin)
        self.log["unserved"].append(unserved)
        self.log["pc"].append(pc)
        self.log["pd"].append(pd_)
        self.log["e"].append(self.e.copy())
        self.log["out"].append(out.copy())
        self.k += 1
        done = self.k >= self.T
        return margin, unserved, done

    def floor_from_u(self, u_r):
        # learned reserve: fraction of the 90th-pct need, never below the hard min, never above 90% of capacity
        E = self.params.energy_kwh * self.soh
        rmin = self.cfg.get("floor_hard_min", 0.10) * E
        return np.clip(u_r * self.need_now(), rmin, 0.9 * E)

    def act(self, u_p, floor):
        # aggregate action in [-1,1] of envelope -> priority disaggregation -> step
        av = (self.avail[self.k] & ~self.out[self.k]).astype(float)
        pc_max, pd_max = envelope(self.e, self.params, self.soh, av, floor, self.dt)
        p_tgt = float(u_p) * (pd_max.sum() if u_p >= 0 else pc_max.sum())
        pc, pd_, shield = priority_disaggregate(p_tgt, self.e, self.params, self.soh, av, floor, self.dt)
        self.log["shield"].append(shield)
        margin, unserved, done = self.step_dispatch(pc, pd_, floor)
        thr = float(self.log["pc"][-1].sum() + self.log["pd"][-1].sum()) * self.dt
        return {"margin": float(margin.sum()), "unserved": float(unserved.sum()), "throughput_kwh": thr, "shield": float(abs(shield))}, done

    def rl_step(self, action, lam_r, deg_cost_per_kwh=0.0, shield_pen=0.01, reserve_mode="learned", promise_cost=False, floor_override=None):
        # action = [u_p in [-1,1], u_r in [0,1]]; reward per home; reserve_mode: learned (b6) | fixed (b5) | none (b4)
        u_p = float(np.clip(action[0], -1, 1))
        u_r = float(np.clip(action[1], 0, 1))
        E = self.params.energy_kwh * self.soh
        if reserve_mode == "learned":
            floor = self.floor_from_u(u_r)
        elif reserve_mode == "fixed":
            floor = self.cfg.get("floor_fixed", 0.20) * E
        else:
            floor = self.emin()
        if floor_override is not None:
            floor = np.maximum(floor, floor_override)
        terms, done = self.act(u_p, floor)
        cost = float(self.unserved_promise.sum()) if promise_cost else terms["unserved"]
        r = (terms["margin"] - lam_r * cost - deg_cost_per_kwh * terms["throughput_kwh"] - shield_pen * terms["shield"]) / self.N
        if done:
            r += float(np.sum(self.e - self.initial_energy) * self.params.eta_d * .03 / self.N)
        return self.obs(), r, done, terms

    def obs(self):
        k = self.k
        if k >= self.T:
            return None
        t = self.t0 + k
        E = self.params.energy_kwh * self.soh
        soc = self.e / E
        slot = self.d.slot[t]
        dow = self.d.index[t].dayofweek
        last = [self.d.price_rt[max(t - i, 0)] / 1000.0 for i in (1, 2, 3, 4)]
        self.price_forecaster.update_to(t)
        pm = self.price_forecaster.forecast(t, 96)[:, 0]
        prices = da_aware_price_forecast(t, 96, self.d.price_da, pm, lambda j: self.d.index[j].hour, index=self.d.index)
        da = [prices[4 * i] / 1000.0 for i in range(0, 24, 2)]
        need = self.need_now()
        av = self.avail[k].astype(float)
        pc_max, pd_max = envelope(self.e, self.params, self.soh, av, self.emin(), self.dt)
        w = self.d.weather[t, self.cidx, :]
        o = [np.sin(2 * np.pi * slot / 96), np.cos(2 * np.pi * slot / 96), np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
             *last, *da,
             np.quantile(soc, 0.1), np.quantile(soc, 0.5), np.quantile(soc, 0.9), soc.mean(), self.soh.mean(), av.mean(),
             float(np.mean(need / E)), float(pd_max.sum() / (self.params.p_discharge_kw.sum() + 1e-9)), float(pc_max.sum() / (self.params.p_charge_kw.sum() + 1e-9)),
             float(self.alert[k].mean()), float(self.out[k].mean()),
             float(np.mean(w[:, 0]) / 40.0), float(np.mean(w[:, 2]) / 100.0), float(np.mean(w[:, 3]) / 10.0)]
        if self.hz is not None:
            o += list(self.hz[k].mean(axis=0))
        else:
            o += [0.0] * 5
        return np.array(o, dtype=np.float32)

    def results(self):
        L = {key: np.array(v) for key, v in self.log.items() if len(v)}
        return L
