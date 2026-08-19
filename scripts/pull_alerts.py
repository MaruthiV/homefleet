import argparse
from homefleet.data.alerts import fetch_vtec
from homefleet.paths import DATA_RAW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=list(range(2014, 2027)))
    a = ap.parse_args()
    out_dir = DATA_RAW / "alerts"
    out_dir.mkdir(exist_ok=True)
    for y in a.years:
        out = out_dir / f"vtec_tx_{y}.parquet"
        if out.exists():
            continue
        d = fetch_vtec(f"{y}-01-01T00:00Z", f"{y + 1}-01-01T00:00Z")
        d.to_parquet(out)
        print(out, d.shape, flush=True)


if __name__ == "__main__":
    main()
