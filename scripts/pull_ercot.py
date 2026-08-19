import argparse
from homefleet.data import ercot
from homefleet.paths import DATA_RAW

PULLERS = {"dam_as_mcpc": ercot.dam_as_mcpc, "dam_spp": ercot.dam_spp, "rt_as_mcpc": ercot.rt_as_mcpc_15min}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=sorted(PULLERS), required=True)
    ap.add_argument("--years", nargs="+", type=int, required=True)
    a = ap.parse_args()
    for y in a.years:
        df = PULLERS[a.what](y)
        out = DATA_RAW / "ercot" / f"{a.what}_{y}.parquet"
        df.to_parquet(out)
        print(out, df.shape, flush=True)


if __name__ == "__main__":
    main()
