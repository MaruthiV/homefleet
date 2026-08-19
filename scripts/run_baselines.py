import argparse
import json
import time
import numpy as np
import pandas as pd
import yaml
from homefleet.paths import DATA_PROC, EXPERIMENTS, CONFIGS
from homefleet.env import FleetData, HomeFleetEnv
from homefleet.control.baselines import run_controller
from homefleet.eval.episode import summarize

TZ = "America/Chicago"
MODES = {
    "B0_fixed": dict(mode="fixed"),
    "B0h_fixed_hold90": dict(mode="fixed", idle_target_frac=0.9),
    "B1_storm_watch": dict(mode="storm_watch"),
    "B2_az_tier": dict(mode="az_tier"),
    "B3a_risk_priced": dict(mode="risk_priced"),
    "B3b_chance": dict(mode="chance"),
    "B7_oracle": dict(mode="oracle"),
}


def windows_for_year(d, year, weeks=2, events=None):
    # contiguous 2-week windows covering the year plus named event windows
    out = []
    n = 96 * 7 * weeks
    for s in range(96 * 28, d.T - n, n):
        out.append((f"w{s // n:02d}", s, n))
    if events:
        for name, ev in events.items():
            st = pd.Timestamp(ev["start"], tz=TZ)
            if st.year != year:
                continue
            s = d.index.get_indexer([st])[0]
            if s > 0:
                out.append((name, int(s), min(n, d.T - s)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metro", default="houston")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--lam-grid", nargs="+", type=float, default=[1, 5, 20])
    ap.add_argument("--eps-grid", nargs="+", type=float, default=[0.02, 0.1])
    ap.add_argument("--S", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows", default="all", help="all | events | wNN,...")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sp = yaml.safe_load(open(CONFIGS / "splits.yaml"))
    d = FleetData(DATA_PROC / f"{a.metro}_{a.year}.npz")
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
    out = EXPERIMENTS / (a.out or f"baselines_{a.metro}_{a.year}_n{a.n}_s{a.seed}.csv")
    for wname, start, n_steps in wins:
        for mname in a.modes:
            spec = MODES[mname]
            variants = [(mname, {})]
            if spec["mode"] == "risk_priced":
                variants = [(f"{mname}_lam{l:g}", {"lam_r": l}) for l in a.lam_grid]
            if spec["mode"] == "chance":
                variants = [(f"{mname}_eps{e:g}", {"eps_cc": e}) for e in a.eps_grid]
            for vname, kw in variants:
                env = HomeFleetEnv(d)
                env.reset(start, n_steps, homes, presets, seed=a.seed, tiers_h=np.full(a.n, 5.0))
                t = time.time()
                res = run_controller(env, S=a.S, **{**spec, **kw})
                s = summarize(res, env.dt)
                s.update({"window": wname, "start": str(d.index[start]), "controller": vname, "metro": a.metro, "year": a.year, "n": a.n, "seed": a.seed, "sec": round(time.time() - t, 1)})
                rows.append(s)
                print(wname, vname, round(s["margin_per_home_week"], 2), round(s["unserved_kwh_per_home"], 2), s["promise_rate"], s["sec"], flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
