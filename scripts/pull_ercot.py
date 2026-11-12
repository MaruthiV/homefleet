import argparse
import pandas as pd
from homefleet.data import ercot
from homefleet.paths import DATA_RAW

PULLERS = {"dam_as_mcpc": ercot.dam_as_mcpc, "dam_spp": ercot.dam_spp, "rt_as_mcpc": ercot.rt_as_mcpc_15min, 'rtm_spp': ercot.rtm_spp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=sorted(PULLERS), required=True)
    ap.add_argument("--years", nargs="+", type=int, required=True)
    a = ap.parse_args()
    (DATA_RAW / 'ercot').mkdir(parents=True, exist_ok=True)
    for y in a.years:
        df = PULLERS[a.what](y)
        out = DATA_RAW / "ercot" / f"{a.what}_{y}.parquet"
        df.to_parquet(out)
        print(out, df.shape, flush=True)
    if a.what == 'rtm_spp':
        files = sorted((DATA_RAW / 'ercot').glob('rtm_spp_[0-9][0-9][0-9][0-9].parquet'))
        pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).to_parquet(DATA_RAW / 'ercot/rtm_all.parquet')


if __name__ == "__main__":
    main()
