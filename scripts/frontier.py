import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from homefleet.paths import EXPERIMENTS


def boot_ci(x, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (np.nan, np.nan)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.quantile(m, 0.025), np.quantile(m, 0.975)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--events-only", action="store_true")
    ap.add_argument("--x", default="unserved_kwh_per_home", choices=["unserved_kwh_per_home", "promise_rate"])
    ap.add_argument("--out", default="frontier.png")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(f) for f in glob.glob(a.glob)], ignore_index=True)
    if a.events_only:
        df = df[~df.window.str.startswith("w")]
    rows = []
    for c, g in df.groupby("controller"):
        m = g.margin_per_home_week.to_numpy()
        if a.x == "promise_rate":
            x = 1 - g.promise_failures.sum() / max(g.outage_episodes.sum(), 1)
        else:
            x = g.unserved_kwh_per_home.mean()
        lo, hi = boot_ci(m)
        rows.append(dict(controller=c, margin=m.mean(), lo=lo, hi=hi, x=x, windows=len(g)))
    t = pd.DataFrame(rows).sort_values("margin", ascending=False)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(7, 5))
    fam = lambda c: c.split("_")[0]
    colors = {"B0": "gray", "B0h": "black", "B1": "tab:orange", "B2": "tab:purple", "B3a": "tab:blue", "B3b": "tab:cyan", "B4": "tab:red", "B5": "tab:pink", "B6": "tab:green", "B7": "gold"}
    for _, r in t.iterrows():
        col = colors.get(fam(r.controller), "k")
        ax.errorbar(r.x, r.margin, yerr=[[r.margin - r.lo], [r.hi - r.margin]], fmt="o", color=col, capsize=3)
        ax.annotate(r.controller.replace("B3a_risk_priced_", "λ=").replace("B3b_chance_", "ε=").replace("B0_", "").replace("B6_", ""), (r.x, r.margin), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("unserved backup kWh per home (lower = more reliable)" if a.x == "unserved_kwh_per_home" else "promise rate (first 5 h served)")
    ax.set_ylabel("fleet margin, $/home-week (mean over windows, 95% CI)")
    ax.set_title(a.title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    (EXPERIMENTS / "figs").mkdir(exist_ok=True)
    fig.savefig(EXPERIMENTS / "figs" / a.out, dpi=150)
    print(EXPERIMENTS / "figs" / a.out)


if __name__ == "__main__":
    main()
