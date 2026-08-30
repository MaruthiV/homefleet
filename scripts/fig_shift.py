import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from homefleet.paths import EXPERIMENTS

KEY = {"B0_fixed": ("floor 20%", "0.45"), "B1_storm_watch": ("storm watch", "tab:orange"),
       "B3b_chance_eps0.1": ("chance", "tab:cyan"), "B3a_risk_priced_lam5": ("risk-priced", "tab:blue"),
       "B6_ppo_pc_lam20_s1": ("learned", "tab:green"), "B6_ppo_pc_lam20_s1_guard": ("learned + guardrail", "darkgreen")}


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
    plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6})
    fig, axes = plt.subplots(1, 4, figsize=(6.1, 1.50), gridspec_kw={"width_ratios": [1.0, 1.0, 0.95, 0.52]})
    axes[3].axis("off")
    # (a) year and metro transfer of the delta-margin
    panels = {"H'24": ("experiments/baselines_houston_2024_n100_s0.csv", "experiments/policy_pc_lam20_s1*houston_2024.csv"),
              "H'25": ("experiments/baselines_houston_2025_n100_s0.csv", "experiments/policy_*pc_lam20_s1_houston_2025.csv"),
              "DFW": ("experiments/baselines_dfw_2024_n100_s0.csv", "experiments/policy_pc_lam20_s1_dfw_2024.csv"),
              "SA": ("experiments/baselines_san_antonio_2024_n100_s0.csv", "experiments/policy_pc_lam20_s1_san_antonio_2024.csv")}
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
    ax.set_xticks(xs, list(panels), fontsize=5.0)
    ax.set_ylabel("Δ margin ($/home-yr)")
    ax.set_title("(a) year and metro transfer", fontsize=7)
    ax.legend(fontsize=4.6, loc="lower right", handlelength=1.0, handletextpad=0.4, labelspacing=0.25, borderpad=0.3, framealpha=0.9)
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
    ax2.set_ylabel("promise rate", fontsize=6.5)
    ax.set_xticks(x, labels, fontsize=5.6, rotation=25)
    ax.set_ylabel("margin ($/home-wk)")
    ax.set_title("(b) fleet shift (learned, 2024 events)", fontsize=7)
    ax.grid(alpha=0.25, axis="y")
    # (c) Uri: margin vs promise for key controllers
    ax = axes[2]
    CLAB = {"B0_fixed": ("floor 20%", "0.45"), "B1_storm_watch": ("storm watch", "tab:orange"),
            "B3a_risk_priced_lam5": ("risk-priced", "tab:blue"), "B7_oracle": ("oracle", "goldenrod"),
            "B6_ppo_pc_lam20_s1": ("best learned", "tab:green"), "B6_ppo_pc_lam20_s1_guard": ("+ guardrail", "darkgreen")}
    u = pd.read_csv("experiments/baselines_houston_2021_events_n100_s0.csv")
    upol = pd.concat([pd.read_csv(f) for f in glob.glob("experiments/policy_*houston_2021ev.csv")], ignore_index=True)
    u = pd.concat([u, upol], ignore_index=True)
    for c, (lab, col) in CLAB.items():
        g = u[u.controller == c]
        if len(g) == 0:
            continue
        ax.scatter(g.promise_rate.mean(), g.margin_per_home_week.mean(), color=col, s=20, zorder=5, label=lab)
    other = pd.concat([pd.read_csv(f) for f in glob.glob("experiments/policy_pc_lam*_houston_2021_events.csv")], ignore_index=True)
    other = other[other.window == "uri_2021"]
    op = other.groupby("controller").agg(pr=("promise_rate", "mean"), m=("margin_per_home_week", "mean"))
    ax.scatter(op.pr, op.m, s=6, color="tab:green", alpha=0.5, zorder=4, label="other seeds")
    h, l = ax.get_legend_handles_labels()
    axes[3].legend(h, l, loc="center left", bbox_to_anchor=(-0.55, 0.5), fontsize=4.6, handletextpad=0.25, borderpad=0.2, labelspacing=0.45, frameon=False)
    ax.set_ylim(top=-95)
    ax.set_xlabel("promise rate in Uri")
    ax.set_ylabel("margin in Uri ($/home-wk)")
    ax.set_title("(c) joint extreme (Uri)", fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(EXPERIMENTS / "figs" / "fig5_shift.png", dpi=170)
    fig.savefig(EXPERIMENTS / "figs" / "fig5_shift.pdf")
    print(EXPERIMENTS / "figs" / "fig5_shift.png")


if __name__ == "__main__":
    main()
