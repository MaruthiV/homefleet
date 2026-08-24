import numpy as np
import pandas as pd
from homefleet.paths import DATA_PROC as PROCESSED


def episodes(mask, index, merge_h=4):
    on = np.flatnonzero(mask)
    if len(on) == 0:
        return []
    gaps = np.diff(on) > merge_h * 4
    starts = np.r_[on[0], on[1:][gaps]]
    ends = np.r_[on[:-1][gaps], on[-1]]
    return list(zip(starts, ends))


def main(metro="houston", years=range(2019, 2026), thr=0.05):
    rows = []
    for y in years:
        d = np.load(PROCESSED / f"{metro}_{y}.npz", allow_pickle=True)
        idx = pd.DatetimeIndex(d["index"])
        alert = d["alert"][:, 0].astype(bool)
        frac = d["frac"][:, 0]
        price = d["price_rt"]
        eps = episodes(frac >= thr, idx)
        for s, e in eps:
            lb = max(0, s - 96)
            covered = alert[lb:s + 1].any()
            lead_h = np.nan
            if covered:
                a_on = np.flatnonzero(alert[lb:s + 1])
                first = lb + a_on[0]
                # walk back to the start of that alert run
                while first > 0 and alert[first - 1]:
                    first -= 1
                lead_h = (s - first) * 0.25
            rows.append(dict(year=y, onset=idx[s], dur_h=(e - s) * 0.25, peak_frac=frac[s:e + 1].max(),
                             covered=covered, lead_h=lead_h))
        d_duty = alert.mean()
        spike = price >= 1000
        hi = price >= np.quantile(price, 0.9)
        print(f"{y}: duty={d_duty:.4f}  alert&spike/alert={spike[alert].mean() if alert.any() else 0:.4f}  "
              f"alert&top-decile/alert={hi[alert].mean() if alert.any() else 0:.4f}  episodes>={thr:.0%}: {len(eps)}")
    t = pd.DataFrame(rows)
    if len(t):
        print(f"\nmass-outage episodes (frac>={thr:.0%}, {metro} {min(years)}-{max(years)}): n={len(t)}")
        print(f"recall r = {t.covered.mean():.3f}  ({t.covered.sum()}/{len(t)})")
        print(f"lead hours (covered): min={t.lead_h.min():.1f} median={t.lead_h.median():.1f}")
        print(t.to_string(index=False))
    return t


if __name__ == "__main__":
    import sys
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    main(thr=thr)
