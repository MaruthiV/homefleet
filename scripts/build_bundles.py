import argparse
from homefleet.data.assemble import build_metro_year
from homefleet.paths import DATA_PROC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metros", nargs="+", default=["houston"])
    ap.add_argument("--years", nargs="+", type=int, default=list(range(2019, 2026)))
    a = ap.parse_args()
    for m in a.metros:
        for y in a.years:
            out = DATA_PROC / f"{m}_{y}.npz"
            if out.exists():
                continue
            print(build_metro_year(m, y), flush=True)


if __name__ == "__main__":
    main()
