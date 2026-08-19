from dataclasses import dataclass
import numpy as np
import yaml
from homefleet.paths import CONFIGS


@dataclass
class Tariff:
    p_ret: float
    c_tdsp: float
    beta: float


def load_tariff():
    with open(CONFIGS / "tariff.yaml") as f:
        t = yaml.safe_load(f)
    return Tariff(t["p_ret"], t["c_tdsp"], t["beta_solar_credit"])


def firm_margin(load_kw, pv_kw, p_kw, price_usd_mwh, tariff, dt, islanded=None):
    # a&z per-step margin, usd; p_kw positive = discharge
    lam = np.asarray(price_usd_mwh, dtype=float) / 1000.0
    g = load_kw - pv_kw - p_kw
    m = np.maximum(g, 0)
    x = np.maximum(-g, 0)
    if islanded is not None:
        m = np.where(islanded, 0.0, m)
        x = np.where(islanded, 0.0, x)
    return dt * (tariff.p_ret * load_kw - (lam + tariff.c_tdsp) * m + lam * x - tariff.beta * pv_kw)
