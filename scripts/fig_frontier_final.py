import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from homefleet.paths import EXPERIMENTS

LABEL = {"B0_fixed10": "floor 10%", "B0_fixed": "floor 20%", "B0_fixed30": "floor 30%", "B0_fixed50": "floor 50%",
         "B0h_fixed_hold90": "hold-90", "B1_storm_watch": "storm watch", "B2_az_tier": "quantile tier",
         "B7_oracle": "oracle"}


def fam(c):
    if c.startswith("B0"): return "fixed"
    if c.startswith("B1"): return "storm"
    if c.startswith("B2"): return "tier"
    if c.startswith("B3a"): return "risk-priced"
    if c.startswith("B3b"): return "chance"
    if c.startswith("B6"): return "learned"
    return "oracle"

KEY_LEARNED = {"B6_ppo_pc_lam20_s1_guard"}
COLOR = {"fixed": "0.45", "storm": "tab:orange", "tier": "tab:purple", "risk-priced": "tab:blue", "chance": "tab:cyan", "learned": "tab:green", "oracle": "goldenrod"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="experiments/baselines_houston_2024_n100_s0.csv")
    ap.add_argument("--policy-glob", default="experiments/policy_*_houston_2024*.csv")
    ap.add_argument("--ref", default="B0_fixed")
    ap.add_argument("--out", default="fig3_frontier_paired_2024.png")
    ap.add_argument("--title", default="Houston 2024 validation year (28 windows, n=100, paired vs 20% floor)")
    ap.add_argument("--drop", nargs="+", default=[])
    a = ap.parse_args()
    files = glob.glob(a.glob) + glob.glob(a.policy_glob)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    base_m = df[df.controller == a.ref].set_index("window").margin_per_home_week
    rng = np.random.default_rng(0)
    rows = []
    for c, g in df.groupby("controller"):
        if c in a.drop:
            continue
        d = (g.set_index("window").margin_per_home_week - base_m.reindex(g.window.to_numpy())).dropna().to_numpy()
        bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        promise = 1 - g.promise_failures.sum() / max(g.outage_episodes.sum(), 1)
        rows.append(dict(controller=c, family=fam(c), dm=d.mean() * 52, lo=np.quantile(bs, 0.025) * 52, hi=np.quantile(bs, 0.975) * 52, promise=promise))
    t = pd.DataFrame(rows)
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    fig, ax = plt.subplots(figsize=(5.5, 2.0))
    for famname, g in t.groupby("family"):
        g = g.sort_values("promise")
        if famname == "learned":
            key = g[g.controller.isin(KEY_LEARNED)]
            rest = g[~g.controller.isin(KEY_LEARNED)]
            ax.errorbar(key.promise, key.dm, yerr=[key.dm - key.lo, key.hi - key.dm], fmt="o", color=COLOR[famname], capsize=2, ms=4.5, lw=1.0, label="learned (best seed)", alpha=0.95)
            ax.scatter(rest.promise, rest.dm, s=9, color=COLOR[famname], alpha=0.55, label="learned (other seeds)", zorder=3)
            continue
        ax.errorbar(g.promise, g.dm, yerr=[g.dm - g.lo, g.hi - g.dm], fmt="o-" if len(g) > 1 and famname in ("fixed", "risk-priced", "chance") else "o",
                    color=COLOR[famname], capsize=2, ms=4, lw=1.0, label=famname, alpha=0.95)
    SHORT = {"B6_ppo_pc_lam20_s1_guard": "learned λ=20 + guard"}
    NUDGE = {"B3a_risk_priced_lam20": (5, -9), "B6_ppo_pc_lam20_s1_guard": (-76, 12),
             "B0h_fixed_hold90": (-34, -10), "B0_fixed50": (-36, -9),
             "B1_storm_watch": (-47, -3), "B2_az_tier": (-46, -4), "B3b_chance_eps0.1": (2, 7)}
    NOLABEL = {"B3b_chance_eps0.02", "B3a_risk_priced_lam1", "B3a_risk_priced_lam5", "B0_fixed30"}
    for _, r in t.iterrows():
        if r.controller in NOLABEL:
            continue
        if r.family == "learned" and r.controller not in SHORT:
            continue
        name = SHORT.get(r.controller) or LABEL.get(r.controller, r.controller.replace("B3a_risk_priced_lam", "λ=").replace("B3b_chance_eps", "ε=").replace("B6_ppo_", ""))
        dx, dy = NUDGE.get(r.controller, (4, 3))
        ax.annotate(name, (r.promise, r.dm), fontsize=6, xytext=(dx, dy), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("promise rate (first 5 h of every outage served)")
    ax.set_ylabel("Δ margin vs 20% floor ($/home-yr)")
    ax.grid(alpha=0.25)
    ax.set_xlim(right=1.0)
    ax.legend(fontsize=6, loc="lower left", ncol=2, framealpha=0.9)
    fig.tight_layout()
    (EXPERIMENTS / "figs").mkdir(exist_ok=True)
    fig.savefig(EXPERIMENTS / "figs" / a.out, dpi=170)
    fig.savefig(EXPERIMENTS / "figs" / a.out.replace(".png", ".pdf"))
    print(EXPERIMENTS / "figs" / a.out)
    print(t.sort_values("dm", ascending=False).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
