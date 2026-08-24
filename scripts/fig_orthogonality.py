import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from homefleet.paths import DATA_RAW, EXPERIMENTS

EVENTS = {
    "Uri (Feb 2021)": ("2021-02-14", "2021-02-19"),
    "Elliott (Dec 2022)": ("2022-12-23", "2022-12-24"),
    "Austin ice (Feb 2023)": ("2023-02-01", "2023-02-02"),
    "Heat (Aug 2023)": ("2023-08-17", "2023-08-25"),
    "EEA2 (Sep 6 2023)": ("2023-09-06", "2023-09-08"),
    "Derecho (May 16 2024)": ("2024-05-16", "2024-05-17"),
    "Beryl (Jul 8 2024)": ("2024-07-08", "2024-07-09"),
    "Wind (Mar 4 2025)": ("2025-03-04", "2025-03-05"),
}


def main():
    d = pd.read_csv(DATA_RAW / "misc" / "daily_price_vs_outage_2021_2025.csv", index_col=0, parse_dates=True)
    d = d[d.index >= "2021-01-01"]
    d["out_frac"] = d.out_peak
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    fig, ax = plt.subplots(figsize=(2.5, 2.05))
    ax.scatter(d.out_peak / 1e6, d.pmax, s=8, alpha=0.35, color="tab:gray", edgecolors="none", label="days 2021–2025")
    ax.axhline(1000, color="tab:red", lw=0.8, ls="--")
    ax.axvline(0.25, color="tab:blue", lw=0.8, ls="--")
    for name, (a, b) in EVENTS.items():
        sub = d.loc[a:b]
        if len(sub) == 0:
            continue
        i = (sub.pmax.rank(pct=True) + sub.out_peak.rank(pct=True)).idxmax()
        r = d.loc[i]
        ax.scatter(r.out_peak / 1e6, r.pmax, s=30, color="tab:red" if "Uri" in name else "tab:blue", zorder=5)
        short = name.split(" (")[0]
        dx, dy = (3, 3)
        if "Heat" in name: dx, dy = (3, -9)
        if "Uri" in name: dx, dy = (-14, -3)
        if "Beryl" in name: dx, dy = (-8, 4)
        ax.annotate(short, (r.out_peak / 1e6, r.pmax), fontsize=5.8, xytext=(dx, dy), textcoords="offset points")
    n_spike = int((d.pmax >= 1000).sum())
    n_out = int((d.out_peak >= 250e3).sum())
    n_both = int(((d.pmax >= 1000) & (d.out_peak >= 250e3)).sum())
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_yscale("log")
    ax.set_xlabel("daily peak TX customers out (millions)")
    ax.set_ylabel("daily max LZ RT price ($/MWh)")
    ax.set_ylim(top=d.pmax.max() * 4)
    ax.set_xlim(left=-0.003)
    ax.set_xticks([1e-2, 1e-1, 1e0])
    ax.text(0.98, 0.03, f"{n_spike} spike days · {n_out} outage days · {n_both} joint (all Uri)", transform=ax.transAxes, fontsize=5.6, color="0.3", ha="right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    (EXPERIMENTS / "figs").mkdir(exist_ok=True)
    out = EXPERIMENTS / "figs" / "fig2_orthogonality.png"
    fig.savefig(out, dpi=170)
    fig.savefig(EXPERIMENTS / "figs" / "fig2_orthogonality.pdf")
    print(out, {"spike": n_spike, "outage": n_out, "both": n_both, "spearman": round(d[["pmax", "out_peak"]].corr(method="spearman").iloc[0, 1], 3)})


if __name__ == "__main__":
    main()
