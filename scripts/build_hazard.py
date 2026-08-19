import argparse
import json
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from homefleet.paths import DATA_RAW, DATA_PROC
from homefleet.data.eaglei import county_fraction_panel
from homefleet.data.alerts import zone_county_map
from homefleet.sim.hazard import hourly_frac, alert_group_panel, build_features, HazardModel, HORIZONS_H

DEFAULT_FIPS = ["48201", "48113", "48439", "48085", "48121", "48029", "48453", "48157", "48339", "48491", "48167", "48039"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fips", nargs="+", default=DEFAULT_FIPS)
    ap.add_argument("--train-years", nargs="+", type=int, default=[2014, 2015, 2016, 2017, 2018, 2019, 2020])
    ap.add_argument("--calib-year", type=int, default=2022)
    ap.add_argument("--no-forecast-proxy", action="store_true")
    a = ap.parse_args()
    fips = [f for f in a.fips if (DATA_RAW / "weather" / f"openmeteo_{f}.parquet").exists()]
    years_avail = [y for y in range(2014, 2026) if (DATA_RAW / "eaglei" / f"eaglei_{y}_TX.csv").exists()]
    frac = county_fraction_panel(years_avail, fips=fips)
    fh = hourly_frac(frac)
    zmap = zone_county_map()
    vt = pd.concat([pd.read_parquet(DATA_RAW / "alerts" / f"vtec_tx_{y}.parquet") for y in range(2014, 2027)])
    X, Y, yr, who = [], [], [], []
    for f in fips:
        w = pd.read_parquet(DATA_RAW / "weather" / f"openmeteo_{f}.parquet")
        w = w[~w.index.duplicated()].sort_index()
        al = alert_group_panel(vt, w.index, [f], zmap)
        al = {g: al[(g, f)] for g in ["tropical", "convective", "winter", "wind", "flood"]}
        Xi, Yi = build_features(w, fh[f].reindex(w.index).fillna(0.0), al, forecast_proxy=not a.no_forecast_proxy)
        X.append(Xi); Y.append(Yi); yr.append(pd.Series(w.index.year, index=w.index)); who.append(pd.Series(f, index=w.index))
    X = pd.concat(X); Y = pd.concat(Y); yr = pd.concat(yr); who = pd.concat(who)
    tr = yr.isin([y for y in a.train_years if y in years_avail]).to_numpy()
    ca = (yr == a.calib_year).to_numpy()
    hm = HazardModel().fit(X[tr], Y[tr])
    P = hm.predict(X)
    iso = {}
    for k in HORIZONS_H:
        iso[k] = IsotonicRegression(out_of_bounds="clip").fit(P[k][ca], Y[k][ca])
        P[k] = iso[k].predict(P[k])
    report = {}
    for y in sorted(set(yr)):
        m = (yr == y).to_numpy()
        if m.sum() == 0 or y not in years_avail:
            continue
        report[int(y)] = {int(k): {"auc": float(roc_auc_score(Y[k][m], P[k][m])) if Y[k][m].nunique() > 1 else None,
                                   "ap": float(average_precision_score(Y[k][m], P[k][m])) if Y[k][m].nunique() > 1 else None,
                                   "brier": float(brier_score_loss(Y[k][m], P[k][m])), "pos_rate": float(Y[k][m].mean())} for k in HORIZONS_H}
    DATA_PROC.mkdir(exist_ok=True)
    for f in fips:
        m = (who == f).to_numpy()
        P[m].to_parquet(DATA_PROC / f"hazard_{f}.parquet")
    json.dump({"train_years": a.train_years, "calib_year": a.calib_year, "fips": fips, "forecast_proxy": not a.no_forecast_proxy, "report": report},
              open(DATA_PROC / "hazard_report.json", "w"), indent=1)
    for y, r in report.items():
        print(y, {k: (round(v["auc"], 3) if v["auc"] else None) for k, v in r.items()})


if __name__ == "__main__":
    main()
