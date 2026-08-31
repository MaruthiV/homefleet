import argparse
import numpy as np
import pandas as pd
from homefleet.data.assemble import build_metro_year
from homefleet.paths import DATA_PROC


def truncate(path, end):
    d = dict(np.load(path, allow_pickle=True))
    idx = pd.DatetimeIndex(d["index"]).tz_localize("UTC").tz_convert("America/Chicago")
    cut = (idx <= pd.Timestamp(end, tz="America/Chicago") + pd.Timedelta("23:45:00")).sum()
    T = len(idx)
    for k, v in d.items():
        if hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] == T:
            d[k] = v[:cut]
    np.savez_compressed(path, **d)
    print(f"truncated {path} to {cut} steps (through {end})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metros", nargs="+", default=["houston"])
    ap.add_argument("--years", nargs="+", type=int, default=list(range(2019, 2026)))
    ap.add_argument("--end", default=None, help="truncate the last year, e.g. 2026-08-15 (partial-year data)")
    a = ap.parse_args()
    for m in a.metros:
        for y in a.years:
            out = DATA_PROC / f"{m}_{y}.npz"
            if out.exists():
                continue
            print(build_metro_year(m, y), flush=True)
            if a.end and y == max(a.years):
                truncate(out, a.end)


if __name__ == "__main__":
    main()
