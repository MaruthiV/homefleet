import argparse
import time
import numpy as np
import pandas as pd
import yaml
from homefleet.paths import DATA_PROC, EXPERIMENTS, CONFIGS
from homefleet.env import FleetData, HomeFleetEnv
from homefleet.rl.ppo import load_policy, run_policy
from homefleet.eval.episode import summarize
from run_baselines import windows_for_year


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="experiments/<run>/policy.pt")
    ap.add_argument("--metro", default="houston")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows", default="all")
    ap.add_argument("--lam-r", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sp = yaml.safe_load(open(CONFIGS / "splits.yaml"))
    d = FleetData(DATA_PROC / f"{a.metro}_{a.year}.npz")
    agent, ostat = load_policy(a.policy)
    import torch
    lam_r = a.lam_r if a.lam_r is not None else float(torch.load(a.policy, weights_only=False)["lam_r"])
    wins = windows_for_year(d, a.year, events=sp["events"])
    if a.windows == "events":
        wins = [w for w in wins if not w[0].startswith("w")]
    elif a.windows != "all":
        keep = set(a.windows.split(","))
        wins = [w for w in wins if w[0] in keep]
    rng = np.random.default_rng(a.seed)
    homes = rng.integers(0, d.N_pool, a.n)
    presets = list(rng.choice(["gen2_single", "gen2_double", "core"], a.n, p=[0.6, 0.2, 0.2]))
    rows = []
    name = a.policy.split("/")[-2]
    out = EXPERIMENTS / (a.out or f"policy_{name}_{a.metro}_{a.year}_n{a.n}_s{a.seed}.csv")
    for wname, start, n_steps in wins:
        env = HomeFleetEnv(d)
        env.reset(start, n_steps, homes, presets, seed=a.seed, tiers_h=np.full(a.n, 5.0))
        t = time.time()
        res = run_policy(env, agent, ostat, lam_r=lam_r)
        s = summarize(res, env.dt)
        s.update({"window": wname, "start": str(d.index[start]), "controller": f"B6_{name}", "metro": a.metro, "year": a.year, "n": a.n, "seed": a.seed, "sec": round(time.time() - t, 1)})
        rows.append(s)
        print(wname, round(s["margin_per_home_week"], 2), round(s["unserved_kwh_per_home"], 2), s["promise_rate"], s["sec"], flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
