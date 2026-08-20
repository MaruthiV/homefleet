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
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for famname, g in t.groupby("family"):
        g = g.sort_values("promise")
        ax.errorbar(g.promise, g.dm, yerr=[g.dm - g.lo, g.hi - g.dm], fmt="o-" if len(g) > 1 and famname in ("fixed", "risk-priced", "chance") else "o",
                    color=COLOR[famname], capsize=2.5, ms=5, lw=1.1, label=famname, alpha=0.95)
    for _, r in t.iterrows():
        name = LABEL.get(r.controller, r.controller.replace("B3a_risk_priced_lam", "λ=").replace("B3b_chance_eps", "ε=").replace("B6_ppo_", ""))
        ax.annotate(name, (r.promise, r.dm), fontsize=6.8, xytext=(4, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("promise rate (first 5 h of every outage served)")
    ax.set_ylabel("Δ margin vs 20% floor ($/home-year, paired, 95% CI)")
    ax.set_title(a.title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower left", ncol=2)
    fig.tight_layout()
    (EXPERIMENTS / "figs").mkdir(exist_ok=True)
    fig.savefig(EXPERIMENTS / "figs" / a.out, dpi=170)
    fig.savefig(EXPERIMENTS / "figs" / a.out.replace(".png", ".pdf"))
    print(EXPERIMENTS / "figs" / a.out)
    print(t.sort_values("dm", ascending=False).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
