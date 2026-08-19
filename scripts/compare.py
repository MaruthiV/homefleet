import argparse
import glob
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="experiments/*houston_2024*.csv")
    ap.add_argument("--events-only", action="store_true")
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(f) for f in glob.glob(a.glob)], ignore_index=True)
    if a.events_only:
        df = df[~df.window.str.startswith("w")]
    g = df.groupby("controller").agg(windows=("window", "nunique"), margin=("margin_per_home_week", "mean"), cvar5=("cvar5_daily_margin_per_home", "mean"),
                                     unserved=("unserved_kwh_per_home", "mean"), episodes=("outage_episodes", "sum"), promise_fail=("promise_failures", "sum"),
                                     ride_fail=("ride_through_failures", "sum"), soc=("mean_soc_frac", "mean"), thr=("throughput_kwh_per_home", "mean"), sec=("sec", "mean"))
    g["promise_rate"] = 1 - g.promise_fail / g.episodes
    g["ride_rate"] = 1 - g.ride_fail / g.episodes
    pd.set_option("display.width", 250)
    print(g.sort_values("margin", ascending=False).round(3))


if __name__ == "__main__":
    main()
