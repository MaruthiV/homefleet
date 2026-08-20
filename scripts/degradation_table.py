import numpy as np
import pandas as pd
import yaml
from homefleet.paths import DATA_PROC, EXPERIMENTS, CONFIGS
from homefleet.env import FleetData, HomeFleetEnv
from homefleet.control.baselines import run_controller
from homefleet.rl.ppo import load_policy, run_policy
from homefleet.eval.blast_adapter import blast_fade
from homefleet.eval.degradation_report import episode_degradation

TZ = "America/Chicago"
WINDOWS = ["beryl_2024", "heat_aug_2024", "derecho_2024", "w10"]
CTRLS = {"B0_fixed": ("dp", dict(mode="fixed")), "B0h_hold90": ("dp", dict(mode="fixed", idle_target_frac=0.9)),
         "B1_storm_watch": ("dp", dict(mode="storm_watch")), "B3a_risk_lam5": ("dp", dict(mode="risk_priced", lam_r=5.0)),
         "B6_pc_lam20": ("rl", "experiments/ppo_pc_lam20_s1/policy.pt"), "B6_pc_lam20_guard": ("rlg", "experiments/ppo_pc_lam20_s1/policy.pt")}


def main():
    sp = yaml.safe_load(open(CONFIGS / "splits.yaml"))
    d = FleetData(DATA_PROC / "houston_2024.npz")
    import sys
    sys.path.insert(0, "scripts")
    from run_baselines import windows_for_year
    wins = {w[0]: w for w in windows_for_year(d, 2024, events=sp["events"])}
    rng = np.random.default_rng(0)
    homes = rng.integers(0, d.N_pool, 20)
    presets = list(rng.choice(["gen2_single", "gen2_double", "core"], 20, p=[0.6, 0.2, 0.2]))
    rows = []
    for cname, (kind, spec) in CTRLS.items():
        for wname in WINDOWS:
            _, start, n_steps = wins[wname]
            env = HomeFleetEnv(d)
            env.reset(start, n_steps, homes, presets, seed=0, tiers_h=None)
            if kind == "dp":
                res = run_controller(env, S=32, **spec)
            else:
                agent, ostat = load_policy(spec)
                res = run_policy(env, agent, ostat, lam_r=20.0, guardrail=(kind == "rlg"))
            temp = d.weather[start:start + n_steps, 0, 0]
            fade = blast_fade(res, env.params, env.soh, temp, env.dt)
            deg = episode_degradation(res, env.params, env.soh, env.dt)
            weeks = n_steps * env.dt / 168.0
            rows.append(dict(controller=cname, window=wname, mean_soc=float(np.mean(res["e"]) / np.mean(env.params.energy_kwh * env.soh)),
                             blast_pct_yr=float(100 * fade.mean()),
                             et_pct_yr=deg["fade_et_pct_yr"], cb_pct_yr=deg["fade_cb_pct_yr"],
                             margin_wk=float(res["margin"].sum() / (20 * weeks))))
            print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(EXPERIMENTS / "degradation_table.csv", index=False)
    g = t.groupby("controller")[["mean_soc", "blast_pct_yr", "et_pct_yr", "cb_pct_yr", "margin_wk"]].mean().round(3)
    print(g.to_string())


if __name__ == "__main__":
    main()
