import argparse
import time
import json
from pathlib import Path
import pandas as pd
from homefleet.data.weather import fetch_hourly, load_hourly
from homefleet.paths import DATA_RAW, ROOT

DEFAULT = ["48201", "48113", "48439", "48085", "48121", "48029", "48453", "48157", "48339", "48491", "48167", "48039"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fips", nargs="+", default=DEFAULT)
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default="2026-08-15")
    ap.add_argument('--force', action='store_true', help='replace a legacy cache with UTC/Unix-aligned weather')
    a = ap.parse_args()
    helper = DATA_RAW / 'misc/tx_counties.csv'
    if helper.exists():
        cty = pd.read_csv(helper, dtype={'fips': str}).set_index('fips')
    else:
        snapshot = Path(__file__).resolve().parents[1] / 'src/homefleet/assets/county_centroids.json'
        cty = pd.DataFrame.from_dict(json.loads(snapshot.read_text()), orient='index')
    out_dir = DATA_RAW / "weather"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in a.fips:
        out = out_dir / f"openmeteo_{f}.parquet"
        if out.exists() and not a.force:
            load_hourly(out)
            continue
        parts = []
        cache = ROOT / 'tmp/weather-parts' / f
        cache.mkdir(parents=True, exist_ok=True)
        for y in range(int(a.start[:4]), int(a.end[:4]) + 1):
            s = f"{y}-01-01" if y > int(a.start[:4]) else a.start
            e = f"{y}-12-31" if y < int(a.end[:4]) else a.end
            segment = cache / f'{s}_{e}.parquet'
            if segment.exists() and not a.force:
                frame = load_hourly(segment)
            else:
                frame = fetch_hourly(cty.loc[f, "lat"], cty.loc[f, "lon"], s, e)
                partial = segment.with_suffix('.parquet.part')
                frame.to_parquet(partial)
                load_hourly(partial)
                partial.replace(segment)
            parts.append(frame)
            print(f'{f}: fetched {s} through {e} ({len(parts[-1])} hours)', flush=True)
            time.sleep(3)
        df = pd.concat(parts)
        temporary = out.with_suffix('.parquet.part')
        df.to_parquet(temporary)
        load_hourly(temporary)
        temporary.replace(out)
        print(out, df.shape, flush=True)


if __name__ == "__main__":
    main()
