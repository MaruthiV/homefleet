import numpy as np
import pandas as pd
from homefleet.paths import DATA_RAW

TZ = "America/Chicago"


def load_tx(year, root=None):
    # header differs by year: 2023 uses 'sum', 2024 adds total_customers
    root = root or DATA_RAW / "eaglei"
    df = pd.read_csv(root / f"eaglei_{year}_TX.csv", dtype={"fips_code": str}, on_bad_lines="skip")
    if "sum" in df.columns:
        df = df.rename(columns={"sum": "customers_out"})
    # some rows are date-only strings like 2018-10-3
    df["t"] = pd.to_datetime(df["run_start_time"], utc=True, format="mixed").dt.tz_convert(TZ)
    df = df.dropna(subset=["t"])
    df["customers_out"] = pd.to_numeric(df["customers_out"], errors="coerce").fillna(0)
    return df[["fips_code", "county", "customers_out", "t"]]


def load_mcc(root=None):
    root = root or DATA_RAW / "eaglei"
    m = pd.read_csv(root / "MCC.csv", dtype={"County_FIPS": str})
    cols = {c.lower(): c for c in m.columns}
    fips_col = cols.get("county_fips") or cols.get("fips_code") or m.columns[0]
    cust_col = [c for c in m.columns if "customer" in c.lower()][0]
    m[fips_col] = m[fips_col].astype(str).str.zfill(5)
    return m.set_index(fips_col)[cust_col].astype(float)


def county_fraction_panel(years, fips=None, root=None):
    # 15-min ct index x fips, customers_out / mcc, missing rows treated as 0 (caveat: missing != 0)
    mcc = load_mcc(root)
    parts = [load_tx(y, root) for y in years]
    df = pd.concat(parts)
    if fips is not None:
        df = df[df.fips_code.isin(fips)]
    wide = df.pivot_table(index="t", columns="fips_code", values="customers_out", aggfunc="sum")
    idx = pd.date_range(wide.index.min(), wide.index.max(), freq="15min", tz=TZ)
    wide = wide.reindex(idx).fillna(0.0)
    frac = wide.div(mcc.reindex(wide.columns).values, axis=1).clip(0, 1)
    return frac
