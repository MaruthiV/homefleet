import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

HORIZONS_H = (1, 3, 6, 12, 24)
ALERT_GROUPS = {"tropical": ["HU", "TR"], "convective": ["SV", "TO"], "winter": ["WS", "BZ", "IS", "WW", "EC", "WC"], "wind": ["HW"], "flood": ["FF", "FA", "FL"]}


def hourly_frac(frac_15min):
    return frac_15min.resample("1h").max()


def alert_group_panel(vtec, index_hourly, fips, zmap):
    # hourly bool frame per (group, fips): any alert of that group active
    from homefleet.data.alerts import county_alert_panel
    cols = {}
    for g, phs in ALERT_GROUPS.items():
        trig = {p: ("A", "W", "Y") for p in phs}
        p = county_alert_panel(vtec, index_hourly, fips, triggers=trig, zmap=zmap)
        for f in fips:
            cols[(g, f)] = p[f].to_numpy()
    return cols


def build_features(weather_h, frac_h, alerts_h, horizons=HORIZONS_H, tau=0.01, forecast_proxy=True):
    # one county: hourly weather frame, hourly frac series, dict group->bool array; returns X (frame) and Y (frame per horizon)
    w = weather_h.copy()
    w["gust_max6"] = w["wind_gusts_10m"].rolling(6, min_periods=1).max()
    w["precip6"] = w["precipitation"].rolling(6, min_periods=1).sum()
    w["precip24"] = w["precipitation"].rolling(24, min_periods=1).sum()
    w["tmin24"] = w["temperature_2m"].rolling(24, min_periods=1).min()
    w["dp3"] = w["surface_pressure"].diff(3)
    f = frac_h.reindex(w.index).fillna(0.0)
    w["frac_now"] = f.to_numpy()
    w["frac_max6"] = f.rolling(6, min_periods=1).max().to_numpy()
    w["frac_max24"] = f.rolling(24, min_periods=1).max().to_numpy()
    for g, arr in alerts_h.items():
        w[f"alert_{g}"] = np.asarray(arr, dtype=float)
    hrs = w.index.hour
    doy = w.index.dayofyear
    w["hour_sin"], w["hour_cos"] = np.sin(2 * np.pi * hrs / 24), np.cos(2 * np.pi * hrs / 24)
    w["doy_sin"], w["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    if forecast_proxy:
        # future-window weather as a stand-in for an nws forecast (optimistic; reported as such)
        for k in (6, 24):
            w[f"fc_gust_max{k}"] = w["wind_gusts_10m"][::-1].rolling(k, min_periods=1).max()[::-1]
            w[f"fc_tmin{k}"] = w["temperature_2m"][::-1].rolling(k, min_periods=1).min()[::-1]
            w[f"fc_precip{k}"] = w["precipitation"][::-1].rolling(k, min_periods=1).sum()[::-1]
    Y = {}
    fa = f.to_numpy()
    for k in horizons:
        fut = pd.Series(fa[::-1]).rolling(k, min_periods=1).max()[::-1].to_numpy()
        fut = np.r_[fut[1:], 0.0]
        Y[k] = (fut >= tau).astype(int)
    return w, pd.DataFrame(Y, index=w.index)


class HazardModel:
    def __init__(self, horizons=HORIZONS_H):
        self.horizons = horizons
        self.models = {}
        self.cols = None

    def fit(self, X, Y):
        self.cols = list(X.columns)
        for k in self.horizons:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, class_weight="balanced", random_state=0)
            m.fit(X[self.cols].to_numpy(), Y[k].to_numpy())
            self.models[k] = m
        return self

    def predict(self, X):
        out = {k: self.models[k].predict_proba(X[self.cols].to_numpy())[:, 1] for k in self.horizons}
        return pd.DataFrame(out, index=X.index)


def oracle_hazard(frac_h, horizons=HORIZONS_H, tau=0.01):
    fa = np.asarray(frac_h, dtype=float)
    out = {}
    for k in horizons:
        fut = pd.Series(fa[::-1]).rolling(k, min_periods=1).max()[::-1].to_numpy()
        out[k] = (np.r_[fut[1:], 0.0] >= tau).astype(float)
    return pd.DataFrame(out, index=frac_h.index)
