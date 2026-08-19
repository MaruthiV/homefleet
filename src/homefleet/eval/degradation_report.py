import numpy as np
from homefleet.sim.degradation import cycle_based, energy_throughput, linear_calendar


def episode_degradation(res, params, soh, dt, replacement_usd_per_kwh=300.0):
    # per-model capacity fade over the episode, annualized, fleet means; soc trace from energy log
    e = res["e"]
    T, N = e.shape
    years = T * dt / 8760.0
    E = params.energy_kwh * soh
    thr = (res["pc"] + res["pd"]).sum(axis=0) * dt
    fade_et = np.array([energy_throughput(thr[j], params.energy_kwh[j]) for j in range(N)])
    fade_cb = np.array([cycle_based(e[:, j] / E[j]) for j in range(N)])
    fade_lc = np.full(N, linear_calendar(years))
    out = {}
    for name, fade in [("lc", fade_lc), ("et", fade_et), ("cb", fade_cb)]:
        ann = fade / years
        cost = fade * params.energy_kwh * replacement_usd_per_kwh / (1 - 0.8)
        out[f"fade_{name}_pct_yr"] = float(100 * ann.mean())
        out[f"deg_cost_{name}_usd_home_week"] = float(cost.sum() / (N * (T * dt / 168.0)))
    return out
