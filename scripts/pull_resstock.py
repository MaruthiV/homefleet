import argparse
import pandas as pd
from homefleet.data.resstock import load_tx_metadata, sample_building_ids, load_building_load_kw, META_COLS
from homefleet.paths import DATA_RAW

METROS = {"houston": ["48201"], "dfw": ["48113", "48439", "48085", "48121"], "san_antonio": ["48029"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--metros", nargs="+", default=list(METROS))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    meta = load_tx_metadata()
    out_dir = DATA_RAW / "resstock"
    for metro in a.metros:
        out = out_dir / f"loads_{metro}_n{a.n}.parquet"
        if out.exists():
            continue
        fips_list = METROS[metro]
        per = max(1, a.n // len(fips_list))
        ids, fips_of = [], {}
        for f in fips_list:
            s = sample_building_ids(f, per, seed=a.seed, meta=meta)
            ids += list(s)
            fips_of.update({str(i): f for i in s})
        cols = {}
        for k, b in enumerate(ids):
            cols[str(b)] = load_building_load_kw(b)
            if k % 25 == 0:
                print(metro, k, len(ids), flush=True)
        wide = pd.DataFrame(cols)
        wide.to_parquet(out)
        m = meta.loc[ids, META_COLS].copy()
        m["fips"] = [fips_of[str(i)] for i in ids]
        m.to_parquet(out_dir / f"meta_{metro}_n{a.n}.parquet")
        print(out, wide.shape, flush=True)


if __name__ == "__main__":
    main()
