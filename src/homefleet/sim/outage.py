import numpy as np
import pandas as pd


def event_windows(frac, tau=0.002):
    # maximal runs where county fraction >= tau; returns list of (start_idx, end_idx_exclusive)
    above = np.asarray(frac) >= tau
    if not above.any():
        return []
    d = np.diff(above.astype(int), prepend=0, append=0)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts, ends))


def draw_outages(frac_panel, home_fips, rng, tau=0.002):
    # quantile matching: home out at t iff u_home,window < county frac at t
    # frac_panel: DataFrame (time x fips), returns bool array (time x homes)
    T = len(frac_panel)
    out = np.zeros((T, len(home_fips)), dtype=bool)
    fips_arr = np.asarray(home_fips)
    for fips in np.unique(fips_arr):
        if fips not in frac_panel.columns:
            continue
        f = frac_panel[fips].to_numpy(dtype=float)
        idx = np.where(fips_arr == fips)[0]
        for s, e in event_windows(f, tau):
            u = rng.random(len(idx))
            out[s:e][:, idx] = f[s:e, None] > u[None, :]
    return out


def episodes(out_col, merge_gap=4):
    # contiguous out runs for one home -> list of (start, end); gaps <= merge_gap steps are merged (flicker)
    eps = event_windows(np.asarray(out_col).astype(float), tau=0.5)
    if not eps:
        return eps
    merged = [list(eps[0])]
    for s, e in eps[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]
