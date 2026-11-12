import argparse
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from homefleet.paths import DATA_RAW, DATA_PROC
from homefleet.data.eaglei import county_fraction_panel
from homefleet.data.alerts import zone_county_map
from homefleet.data.weather import load_hourly, CLOCK_PROTOCOL
from homefleet.sim.hazard import hourly_frac, alert_group_panel, build_features, HazardModel, HORIZONS_H

DEFAULT_FIPS = ["48201", "48113", "48439", "48085", "48121", "48029", "48453", "48157", "48339", "48491", "48167", "48039"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fips", nargs="+", default=DEFAULT_FIPS)
    ap.add_argument("--train-years", nargs="+", type=int, default=[2014, 2015, 2016, 2017, 2018, 2019])
    ap.add_argument("--calib-year", type=int, default=2020)
    ap.add_argument("--no-forecast-proxy", action="store_true")
    ap.add_argument("--forecast-proxy", action="store_true", help="optimistic sensitivity only")
    ap.add_argument("--output-dir", type=Path, default=DATA_PROC / "causal")
    a = ap.parse_args()
    fips = a.fips
    missing = [f for f in fips if not (DATA_RAW / 'weather' / f'openmeteo_{f}.parquet').exists()]
    if missing: raise FileNotFoundError('Missing requested county weather files: ' + ', '.join(missing))
    missing_years = [y for y in set(a.train_years + [a.calib_year]) if not (DATA_RAW / 'eaglei' / f'eaglei_{y}_TX.csv').exists()]
    if missing_years: raise FileNotFoundError(f'Missing requested outage training/calibration years: {sorted(missing_years)}')
    years_avail = [y for y in range(2014, 2026) if (DATA_RAW / "eaglei" / f"eaglei_{y}_TX.csv").exists()]
    frac = county_fraction_panel(years_avail, fips=fips)
    fh = hourly_frac(frac)
    zmap = zone_county_map()
    vt = pd.concat([pd.read_parquet(DATA_RAW / "alerts" / f"vtec_tx_{y}.parquet") for y in range(2014, 2027)])
    X, Y, yr, who = [], [], [], []
    for f in fips:
        print('Building risk-model features for county', f, flush=True)
        w = load_hourly(DATA_RAW / "weather" / f"openmeteo_{f}.parquet")
        al = alert_group_panel(vt, w.index, [f], zmap)
        al = {g: al[(g, f)] for g in ["tropical", "convective", "winter", "wind", "flood"]}
        Xi, Yi = build_features(w, fh[f].reindex(w.index).fillna(0.0), al, forecast_proxy=a.forecast_proxy)
        X.append(Xi); Y.append(Yi); yr.append(pd.Series(w.index.year, index=w.index)); who.append(pd.Series(f, index=w.index))
    X = pd.concat(X); Y = pd.concat(Y); yr = pd.concat(yr); who = pd.concat(who)
    tr = yr.isin([y for y in a.train_years if y in years_avail]).to_numpy()
    ca = (yr == a.calib_year).to_numpy()
    print('Fitting five risk classifiers:', int(tr.sum()), 'training rows,', X.shape[1], 'features', flush=True)
    hm = HazardModel().fit(X[tr], Y[tr])
    print('Predicting and calibrating on', int(ca.sum()), '2020 rows', flush=True)
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
    a.output_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump({'model': hm, 'calibrators': iso, 'feature_columns': hm.cols,
                 'train_years': a.train_years, 'calibration_year': a.calib_year,
                 'forecast_proxy': a.forecast_proxy, 'protocol': 'causal-v3', 'weather_clock_protocol': CLOCK_PROTOCOL},
                a.output_dir / 'hazard_model.joblib', compress=3)
    for f in fips:
        m = (who == f).to_numpy()
        P[m].to_parquet(a.output_dir / f"hazard_{f}.parquet")
    json.dump({"train_years": a.train_years, "calib_year": a.calib_year, "fips": fips, "forecast_proxy": a.forecast_proxy, "observation_lag_hours": 1, 'protocol': 'causal-v3', 'weather_clock_protocol': CLOCK_PROTOCOL, "report": report},
              open(a.output_dir / "hazard_report.json", "w"), indent=1)
    for y, r in report.items():
        print(y, {k: (round(v["auc"], 3) if v["auc"] else None) for k, v in r.items()})


if __name__ == "__main__":
    main()
