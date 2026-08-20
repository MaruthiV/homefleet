import argparse
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from homefleet.paths import DATA_PROC, EXPERIMENTS, CONFIGS
from homefleet.env import FleetData, HomeFleetEnv
from homefleet.control.baselines import run_controller

TZ = "America/Chicago"
PANELS = [("uri_2021", "houston", 2021, "Winter Storm Uri (spike ∩ outage)"),
          ("beryl_2024", "houston", 2024, "Hurricane Beryl (outage only)"),
          ("heat_aug_2023", "houston", 2023, "Aug 2023 heat (spike only)")]
CTRLS = [("fixed 20%", dict(mode="fixed"), "0.4"),
         ("storm watch", dict(mode="storm_watch"), "tab:orange"),
         ("risk-priced λ=5", dict(mode="risk_priced", lam_r=5.0), "tab:blue")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--S", type=int, default=32)
    a = ap.parse_args()
    sp = yaml.safe_load(open(CONFIGS / "splits.yaml"))
    fig, axes = plt.subplots(3, len(PANELS), figsize=(4.2 * len(PANELS), 7.2), sharex="col", height_ratios=[1, 1, 1.3])
    for col, (ev, metro, year, title) in enumerate(PANELS):
        d = FleetData(DATA_PROC / f"{metro}_{year}.npz")
        st = pd.Timestamp(sp["events"][ev]["start"], tz=TZ) - pd.Timedelta(days=2)
        start = int(d.index.get_indexer([st])[0])
        n_steps = 96 * a.days
        idx = d.index[start:start + n_steps]
        rng = np.random.default_rng(0)
        homes = rng.integers(0, d.N_pool, a.n)
        presets = list(rng.choice(["gen2_single", "gen2_double", "core"], a.n, p=[0.6, 0.2, 0.2]))
        ax_p, ax_h, ax_s = axes[0, col], axes[1, col], axes[2, col]
        ax_p.plot(idx, d.price_rt[start:start + n_steps], color="k", lw=0.7)
        ax_p.set_yscale("log")
        ax_p.set_title(title, fontsize=10)
        hz = d.hazard[start:start + n_steps, :, -1].mean(axis=1)
        frac = d.frac.iloc[start:start + n_steps].mean(axis=1).to_numpy()
        ax_h.plot(idx, hz, color="tab:red", lw=1.0, label="hazard $\\hat h$ (24h)")
        ax_h.plot(idx, frac, color="tab:gray", lw=1.0, label="county frac out")
        al = d.alert.iloc[start:start + n_steps].mean(axis=1).to_numpy()
        ax_h.fill_between(idx, 0, al * 1.0, color="tab:orange", alpha=0.15, label="NWS alert")
        for name, spec, color in CTRLS:
            env = HomeFleetEnv(d)
            env.reset(start, n_steps, homes, presets, seed=0, tiers_h=None)
            res = run_controller(env, S=a.S, **spec)
            E = env.params.energy_kwh * env.soh
            soc = (res["e"] / E[None, :]).mean(axis=1)
            ax_s.plot(idx, soc, lw=1.2, color=color, label=name)
        ax_s.set_ylim(0, 1.02)
        if col == 0:
            ax_p.set_ylabel("RT price $/MWh")
            ax_h.set_ylabel("outage risk")
            ax_s.set_ylabel("fleet mean SOC")
            ax_h.legend(fontsize=7, loc="upper left")
            ax_s.legend(fontsize=7, loc="lower left")
        for ax in (ax_p, ax_h, ax_s):
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=7)
        ax_s.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    (EXPERIMENTS / "figs").mkdir(exist_ok=True)
    fig.savefig(EXPERIMENTS / "figs" / "fig4_event_panels.png", dpi=160)
    fig.savefig(EXPERIMENTS / "figs" / "fig4_event_panels.pdf")
    print(EXPERIMENTS / "figs" / "fig4_event_panels.png")


if __name__ == "__main__":
    main()
