import argparse
import glob
import pandas as pd


def paired_vs(df, ref="B0_fixed"):
    # per-window margin difference vs a reference controller, bootstrap ci over windows
    import numpy as np
    base = df[df.controller == ref].set_index(["metro", "year", "window"]).margin_per_home_week
    rows = []
    for c, g in df.groupby("controller"):
        key = list(zip(g.metro, g.year, g.window))
        diff = (g.margin_per_home_week.to_numpy() - base.reindex(key).to_numpy())
        diff = diff[~np.isnan(diff)]
        if len(diff) < 2:
            continue
        rng = np.random.default_rng(0)
        bs = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
        rows.append(dict(controller=c, dmargin=diff.mean(), lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975), n=len(diff)))
    return pd.DataFrame(rows).sort_values("dmargin", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="experiments/*houston_2024*.csv")
    ap.add_argument("--events-only", action="store_true")
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(f) for f in glob.glob(a.glob)], ignore_index=True)
    if a.events_only:
        df = df[~df.window.str.startswith("w")]
    # capture ratio vs oracle per window when the oracle ran in the same file set
    orc = df[df.controller == "B7_oracle"].set_index(["metro", "year", "window"]).margin_per_home_week
    if len(orc):
        key = list(zip(df.metro, df.year, df.window))
        df["capture"] = df.margin_per_home_week.to_numpy() / orc.reindex(key).to_numpy()
    agg = dict(windows=("window", "nunique"), margin=("margin_per_home_week", "mean"), cvar5=("cvar5_daily_margin_per_home", "mean"),
               unserved=("unserved_kwh_per_home", "mean"), episodes=("outage_episodes", "sum"), promise_fail=("promise_failures", "sum"),
               ride_fail=("ride_through_failures", "sum"), soc=("mean_soc_frac", "mean"), thr=("throughput_kwh_per_home", "mean"), sec=("sec", "mean"))
    if "capture" in df.columns:
        agg["capture"] = ("capture", "mean")
    if "fade_cb_pct_yr" in df.columns:
        agg["fade_cb"] = ("fade_cb_pct_yr", "mean")
    g = df.groupby("controller").agg(**agg)
    g["promise_rate"] = 1 - g.promise_fail / g.episodes
    g["ride_rate"] = 1 - g.ride_fail / g.episodes
    pd.set_option("display.width", 250)
    print(g.sort_values("margin", ascending=False).round(3))
    print()
    print("paired margin difference vs B0_fixed ($/home-week, 95% bootstrap CI over windows):")
    print(paired_vs(df).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
