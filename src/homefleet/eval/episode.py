import numpy as np
from homefleet.sim.outage import episodes
from homefleet.eval.metrics import cvar
from homefleet.eval.degradation_report import episode_degradation


def summarize(res, dt, subscription_usd_week=0.0, promise_hours=5.0, params=None, soh=None):
    # res = env.results(); returns dict of fleet-level metrics for one episode
    margin = res["margin"]
    unserved = res["unserved"]
    out = res["out"]
    T, N = margin.shape
    weeks = T * dt / 168.0
    daily = margin.sum(axis=1).reshape(-1, 96).sum(axis=1) if T % 96 == 0 else margin.sum(axis=1)
    n_ep, n_fail, n_fail_promise = 0, 0, 0
    ps = int(round(promise_hours / dt))
    for j in range(N):
        eps = episodes(out[:, j])
        n_ep += len(eps)
        n_fail += sum(1 for s, e in eps if unserved[s:e, j].sum() > 1e-9)
        # base-style promise: first `promise_hours` of every outage fully served
        n_fail_promise += sum(1 for s, e in eps if unserved[s:min(e, s + ps), j].sum() > 1e-9)
    return {
        "margin_per_home_week": float(margin.sum() / (N * weeks)) + subscription_usd_week,
        "cvar5_daily_margin_per_home": float(cvar(daily / N, 0.05)) if len(daily) >= 20 else float(daily.min() / N),
        "worst_day_margin_per_home": float(daily.min() / N),
        "unserved_kwh_per_home": float(unserved.sum() / N),
        "outage_episodes": int(n_ep),
        "ride_through_failures": int(n_fail),
        "ride_through_rate": float(1 - n_fail / n_ep) if n_ep else float("nan"),
        "promise_failures": int(n_fail_promise),
        "promise_rate": float(1 - n_fail_promise / n_ep) if n_ep else float("nan"),
        "out_home_hours": float(out.sum() * dt),
        "throughput_kwh_per_home": float((res["pc"].sum() + res["pd"].sum()) * dt / N),
        "mean_soc_frac": float(np.mean(res["e"]) / np.mean(res["e"].max(axis=0))) if "e" in res else float("nan"),
        "shield_abs_mean_kw": float(np.mean(np.abs(res["shield"]))) if "shield" in res and len(res["shield"]) else 0.0,
    } | (episode_degradation(res, params, soh, dt) if params is not None else {})
