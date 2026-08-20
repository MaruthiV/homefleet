import numpy as np
import pandas as pd
import yaml
import opendssdirect as dss
from homefleet.paths import DATA_PROC, EXPERIMENTS, CONFIGS
from homefleet.env import FleetData, HomeFleetEnv
from homefleet.control.baselines import run_controller
from homefleet.rl.ppo import load_policy, run_policy

TZ = "America/Chicago"
FEEDER = "data/raw/smartds/feeder/p1uhs0_1247--p1udt2142/Master_static.dss"
DAYS = {"beryl": "2024-07-08", "heat": "2024-08-20", "normal": "2024-04-15"}
CTRLS = {"B0_fixed": ("dp", dict(mode="fixed")), "B3a_risk_lam5": ("dp", dict(mode="risk_priced", lam_r=5.0)),
         "B6_pc_lam20_guard": ("rlg", "experiments/ppo_pc_lam20_s1/policy.pt")}
KVAR_RATIO = 0.33


def run_day(d, day, kind, spec, homes, presets, n=100):
    start = int(d.index.get_indexer([pd.Timestamp(day, tz=TZ)])[0]) - 96
    env = HomeFleetEnv(d)
    env.reset(start, 96 * 2, homes, presets, seed=0, tiers_h=None)
    if kind == "dp":
        res = run_controller(env, S=32, **spec)
    else:
        agent, ostat = load_policy(spec)
        res = run_policy(env, agent, ostat, lam_r=20.0, guardrail=(kind == "rlg"))
    p = res["pd"] - res["pc"]
    g = env.load - env.pv - p
    return g[96:], env.load[96:]


def solve_series(g_batt, load_only, penetration):
    dss.Text.Command("Clear")
    dss.Text.Command(f"Redirect {FEEDER}")
    names = dss.Loads.AllNames()
    M = len(names)
    n_b = int(penetration * M)
    N = g_batt.shape[1]
    rows = []
    for t in range(g_batt.shape[0]):
        for j, nm in enumerate(names):
            kw = float(g_batt[t, j % N]) if j < n_b else float(load_only[t, j % N])
            dss.Loads.Name(nm)
            dss.Loads.kW(kw)
            dss.Loads.kvar(KVAR_RATIO * max(kw, 0.0))
        dss.Solution.Solve()
        v = np.array(dss.Circuit.AllBusMagPu())
        v = v[(v > 0.5) & (v < 1.5)]
        rows.append(dict(conv=dss.Solution.Converged(), vmin=v.min(), vmax=v.max(),
                         under=(v < 0.95).mean(), over=(v > 1.05).mean()))
    return pd.DataFrame(rows)


def main():
    d = FleetData(DATA_PROC / "houston_2024.npz")
    rng = np.random.default_rng(0)
    homes = rng.integers(0, d.N_pool, 100)
    presets = list(rng.choice(["gen2_single", "gen2_double", "core"], 100, p=[0.6, 0.2, 0.2]))
    out = []
    for cname, (kind, spec) in CTRLS.items():
        for dayname, day in DAYS.items():
            g, load = run_day(d, day, kind, spec, homes, presets)
            for pen in (0.3, 1.0):
                r = solve_series(g, load, pen)
                out.append(dict(controller=cname, day=dayname, penetration=pen, converged=float(r.conv.mean()),
                                vmin=float(r.vmin.min()), vmax=float(r.vmax.max()),
                                steps_with_undervolt=float((r.under > 0).mean()), mean_undervolt_bus_frac=float(r.under.mean()),
                                steps_with_overvolt=float((r.over > 0).mean())))
                print(out[-1], flush=True)
        # load-only reference once per day
    for dayname, day in DAYS.items():
        g, load = run_day(d, day, "dp", dict(mode="fixed"), homes, presets)
        r = solve_series(load, load, 0.0)
        out.append(dict(controller="load_only", day=dayname, penetration=0.0, converged=float(r.conv.mean()),
                        vmin=float(r.vmin.min()), vmax=float(r.vmax.max()),
                        steps_with_undervolt=float((r.under > 0).mean()), mean_undervolt_bus_frac=float(r.under.mean()),
                        steps_with_overvolt=float((r.over > 0).mean())))
        print(out[-1], flush=True)
    t = pd.DataFrame(out)
    t.to_csv(EXPERIMENTS / "ac_check.csv", index=False)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
