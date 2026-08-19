import numpy as np
import pandas as pd

TZ = "America/Chicago"


def year_index(year, tz=TZ):
    return pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="15min", tz=tz, inclusive="left")


def map_profile_to_year(s2018, year):
    # reuse a 2018 15-min profile in another year by (day-of-year, local clock); feb 29 -> feb 28; dst gaps ffilled
    src = s2018.copy()
    key = src.index.strftime("%m-%d %H:%M")
    lut = pd.Series(src.to_numpy(), index=key)
    lut = lut[~lut.index.duplicated(keep="first")]
    idx = year_index(year)
    k = idx.strftime("%m-%d %H:%M").str.replace("02-29", "02-28", regex=False)
    out = lut.reindex(k).to_numpy()
    out = pd.Series(out, index=idx).ffill().bfill()
    return out


def slot_of_day(index):
    return (index.hour * 4 + index.minute // 15).to_numpy()


def forward_positive_energy(net_kw, steps, dt=0.25):
    # energy needed to cover positive net load over the next `steps` intervals, per start step
    pos = np.maximum(np.asarray(net_kw, dtype=float), 0.0) * dt
    c = np.concatenate([[0.0], np.cumsum(pos)])
    n = len(pos)
    i = np.arange(n)
    j = np.minimum(i + steps, n)
    return c[j] - c[i]


def slot_quantile(values, index, q=0.9, window_slots=2):
    # quantile of `values` by 15-min slot of day pooled over days and +-window slots (a&z style), returns 96-vector
    slots = slot_of_day(index)
    v = np.asarray(values, dtype=float)
    out = np.zeros(96)
    for s in range(96):
        mask = np.zeros(len(v), dtype=bool)
        for w in range(-window_slots, window_slots + 1):
            mask |= slots == (s + w) % 96
        out[s] = np.quantile(v[mask], q) if mask.any() else 0.0
    return out
