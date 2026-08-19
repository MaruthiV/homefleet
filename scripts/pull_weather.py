import argparse
import time
import pandas as pd
from homefleet.data.weather import fetch_hourly
from homefleet.paths import DATA_RAW

DEFAULT = ["48201", "48113", "48439", "48085", "48121", "48029", "48453", "48157", "48339", "48491", "48167", "48039"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fips", nargs="+", default=DEFAULT)
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default="2026-08-15")
    a = ap.parse_args()
    cty = pd.read_csv(DATA_RAW / "misc" / "tx_counties.csv", dtype={"fips": str}).set_index("fips")
    out_dir = DATA_RAW / "weather"
    out_dir.mkdir(exist_ok=True)
    for f in a.fips:
        out = out_dir / f"openmeteo_{f}.parquet"
        if out.exists():
            continue
        parts = []
        for y in range(int(a.start[:4]), int(a.end[:4]) + 1):
            s = f"{y}-01-01" if y > int(a.start[:4]) else a.start
            e = f"{y}-12-31" if y < int(a.end[:4]) else a.end
            parts.append(fetch_hourly(cty.loc[f, "lat"], cty.loc[f, "lon"], s, e))
            time.sleep(3)
        df = pd.concat(parts)
        df = df[~df.index.isna()]
        df.to_parquet(out)
        print(out, df.shape, flush=True)


if __name__ == "__main__":
    main()
