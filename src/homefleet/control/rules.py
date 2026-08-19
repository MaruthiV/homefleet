import numpy as np


def fixed_floor(energy_kwh, soh, frac=0.20):
    return frac * energy_kwh * soh


def storm_watch_floor(energy_kwh, soh, alert_active, frac=0.20):
    # nws alert -> hold everything, else fixed floor
    base = frac * energy_kwh * soh
    return np.where(alert_active, energy_kwh * soh, base)


def need_kwh(net_load_kw_future, duration_steps, dt, eta_d, q=0.9, hist=None):
    # energy to ride out duration_steps of positive net load; hist gives quantile across days
    pos = np.maximum(net_load_kw_future, 0)
    if hist is None:
        return dt * pos[:duration_steps].sum() / eta_d
    return np.quantile(hist, q) / eta_d
