import numpy as np
import json
from homefleet.sim.outage import episodes
from homefleet.eval.metrics import cvar
from homefleet.eval.degradation_report import episode_degradation


def summarize(res, dt, subscription_usd_week=0.0, promise_hours=5.0, params=None, soh=None, initial_energy=None, terminal_value=0.03):
    # res = env.results(); returns dict of fleet-level metrics for one episode
    margin = res["margin"]
    unserved = res["unserved"]
    out = res["out"]
    T, N = margin.shape
    weeks = T * dt / 168.0
    totals = margin.sum(axis=1)
    daily = np.array([totals[s:s + 96].sum() for s in range(0, T, 96)])
    n_ep, n_fail, n_fail_promise = 0, 0, 0
    ps = int(round(promise_hours / dt))
    home_promise = []
    home_episodes, home_failures = [], []
    promise_unserved = 0.0
    for j in range(N):
        eps = episodes(out[:, j])
        n_ep += len(eps)
        n_fail += sum(1 for s, e in eps if unserved[s:e, j].sum() > 1e-9)
        # base-style promise: first `promise_hours` of every outage fully served
        nf = sum(1 for s, e in eps if unserved[s:min(e, s + ps), j].sum() > 1e-9)
        n_fail_promise += nf
        promise_unserved += sum(float(unserved[s:min(e, s + ps), j].sum()) for s, e in eps)
        home_episodes.append(len(eps)); home_failures.append(nf)
        if eps:
            home_promise.append(1 - nf / len(eps))
    adjustment = 0.0
    if params is not None and initial_energy is not None:
        adjustment = float(np.sum(res["e"][-1] - initial_energy) * params.eta_d * terminal_value)
    return {
        "margin_per_home_week": float((margin.sum() + adjustment) / (N * weeks)) + subscription_usd_week,
        "cash_margin_per_home_week": float(margin.sum() / (N * weeks)) + subscription_usd_week,
        "terminal_adjustment_per_home": adjustment / N,
        "hours": T * dt,
        "cvar5_daily_margin_per_home": float(cvar(daily / N, 0.05)),
        "worst_day_margin_per_home": float(daily.min() / N),
        "unserved_kwh_per_home": float(unserved.sum() / N),
        "promise_unserved_kwh_per_home": promise_unserved / N,
        "outage_episodes": int(n_ep),
        "ride_through_failures": int(n_fail),
        "ride_through_rate": float(1 - n_fail / n_ep) if n_ep else float("nan"),
        "promise_failures": int(n_fail_promise),
        "promise_rate": float(1 - n_fail_promise / n_ep) if n_ep else float("nan"),
        "out_home_hours": float(out.sum() * dt),
        "throughput_kwh_per_home": float((res["pc"].sum() + res["pd"].sum()) * dt / N),
        "mean_soc_frac": float(np.mean(res["e"] / (params.energy_kwh * soh))) if params is not None else float("nan"),
        "home_promise_p05": float(np.quantile(home_promise, 0.05)) if home_promise else float("nan"),
        "home_unserved_p95_kwh": float(np.quantile(unserved.sum(axis=0), 0.95)),
        "physical_violations": int(res.get("physical_viol", np.zeros(1)).sum()),
        "reachable_floor_violations": int(res.get("floor_viol", np.zeros(1)).sum()),
        "floor_shortfall_kwh_mean": float(res.get("floor_shortfall", np.zeros(1)).mean()),
        "daily_cash_margin_per_home": json.dumps([float(totals[s:s + 96].sum() / N) for s in range(0, T - 95, 96)]),
        "home_unserved_kwh": json.dumps(unserved.sum(axis=0).tolist()),
        "home_episode_counts": json.dumps(home_episodes),
        "home_promise_failure_counts": json.dumps(home_failures),
        "shield_abs_mean_kw": float(np.mean(np.abs(res["shield"]))) if "shield" in res and len(res["shield"]) else 0.0,
    } | (episode_degradation(res, params, soh, dt) if params is not None else {})
