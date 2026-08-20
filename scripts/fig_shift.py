import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from homefleet.paths import EXPERIMENTS

KEY = {"B0_fixed": ("floor 20%", "0.45"), "B1_storm_watch": ("storm watch", "tab:orange"),
       "B3b_chance_eps0.1": ("chance ε=0.1", "tab:cyan"), "B3a_risk_priced_lam5": ("risk-priced λ=5", "tab:blue"),
       "B6_ppo_pc_lam20_s1": ("learned (pc λ=20)", "tab:green"), "B6_ppo_pc_lam20_s1_guard": ("learned + guardrail", "darkgreen")}


def dm(df, ref="B0_fixed"):
    base = df[df.controller == ref].set_index("window").margin_per_home_week
    out = {}
    for c, g in df.groupby("controller"):
        d = (g.set_index("window").margin_per_home_week - base.reindex(g.window.to_numpy())).dropna()
        out[c] = d.mean() * 52
    return out


def promise(df):
    return {c: 1 - g.promise_failures.sum() / max(g.outage_episodes.sum(), 1) for c, g in df.groupby("controller")}


def main():
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    # (a) year and metro transfer of the delta-margin
    panels = {"Houston'24\n(val)": ("experiments/baselines_houston_2024_n100_s0.csv", "experiments/policy_pc_lam20_s1*houston_2024.csv"),
              "Houston'25\n(test)": ("experiments/baselines_houston_2025_n100_s0.csv", "experiments/policy_*pc_lam20_s1_houston_2025.csv"),
              "DFW'24": ("experiments/baselines_dfw_2024_n100_s0.csv", "experiments/policy_pc_lam20_s1_dfw_2024.csv"),
              "SanAnt'24": ("experiments/baselines_san_antonio_2024_n100_s0.csv", "experiments/policy_pc_lam20_s1_san_antonio_2024.csv")}
    ax = axes[0]
    xs = np.arange(len(panels))
    ctrls = ["B3b_chance_eps0.1", "B3a_risk_priced_lam5", "B6_ppo_pc_lam20_s1"]
    width = 0.25
    for k, c in enumerate(ctrls):
        vals = []
        for _, (bf, pf) in panels.items():
            df = pd.read_csv(bf)
            if pf:
                pol = [pd.read_csv(f) for f in glob.glob(pf)]
                if pol:
                    df = pd.concat([df] + pol, ignore_index=True)
            vals.append(dm(df).get(c, np.nan))
        ax.bar(xs + (k - 1) * width, vals, width, label=KEY[c][0], color=KEY[c][1])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xs, list(panels), fontsize=8)
    ax.set_ylabel("Δ margin vs floor-20 ($/home-yr)")
    ax.set_title("(a) year and metro transfer", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, axis="y")
    # (b) fleet-size / dropout / composition (learned policy, 2024 events)
    ax = axes[1]
    labels, margins, proms = [], [], []
    for lab, f in [("n=100", "experiments/policy_learned_lam5_s1_houston_2024.csv"),
                   ("n=500", "experiments/policy_learned_lam5_s1_houston_2024ev_n500.csv"),
                   ("n=1000", "experiments/policy_learned_lam5_s1_houston_2024ev_n1000.csv"),
                   ("20% drop", "experiments/policy_learned_lam5_s1_houston_2024ev_drop20.csv"),
                   ("70% Core", "experiments/policy_learned_lam5_s1_houston_2024ev_coremix.csv")]:
        g = pd.read_csv(f)
        g = g[~g.window.str.startswith("w")]
        labels.append(lab)
        margins.append(g.margin_per_home_week.mean())
        proms.append(1 - g.promise_failures.sum() / g.outage_episodes.sum())
    x = np.arange(len(labels))
    ax.bar(x, margins, 0.55, color="tab:green")
    ax2 = ax.twinx()
    ax2.plot(x, proms, "ko-", ms=4, lw=1)
    ax2.set_ylim(0.8, 1.02)
    ax2.set_ylabel("promise rate", fontsize=8)
    ax.set_xticks(x, labels, fontsize=8, rotation=20)
    ax.set_ylabel("margin ($/home-wk, events)")
    ax.set_title("(b) fleet shift (learned, 2024 events)", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    # (c) Uri: margin vs promise for key controllers
    ax = axes[2]
    u = pd.read_csv("experiments/baselines_houston_2021_events_n100_s0.csv")
    upol = pd.concat([pd.read_csv(f) for f in glob.glob("experiments/policy_*houston_2021ev.csv")], ignore_index=True)
    u = pd.concat([u, upol], ignore_index=True)
    show = {"B0_fixed": KEY["B0_fixed"], "B1_storm_watch": KEY["B1_storm_watch"], "B3a_risk_priced_lam5": KEY["B3a_risk_priced_lam5"],
            "B6_ppo_pc_lam20_s1": KEY["B6_ppo_pc_lam20_s1"], "B6_ppo_pc_lam20_s1_guard": KEY["B6_ppo_pc_lam20_s1_guard"], "B7_oracle": ("oracle", "goldenrod")}
    for c, (lab, col) in show.items():
        g = u[u.controller == c]
        if len(g) == 0:
            continue
        ax.scatter(g.promise_rate.mean(), g.margin_per_home_week.mean(), color=col, s=40, zorder=5)
        ax.annotate(lab, (g.promise_rate.mean(), g.margin_per_home_week.mean()), fontsize=7, xytext=(5, 3), textcoords="offset points")
    ax.set_xlabel("promise rate in Uri")
    ax.set_ylabel("margin in Uri ($/home-wk)")
    ax.set_title("(c) the joint extreme (Winter Storm Uri)", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(EXPERIMENTS / "figs" / "fig5_shift.png", dpi=170)
    fig.savefig(EXPERIMENTS / "figs" / "fig5_shift.pdf")
    print(EXPERIMENTS / "figs" / "fig5_shift.png")


if __name__ == "__main__":
    main()
