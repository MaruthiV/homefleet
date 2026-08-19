import numpy as np


def pv_kw_from_ghi(ghi_wm2, temp_c, kw_dc, losses=0.14, inv_eff=0.96, gamma=-0.004):
    # simple pvwatts-ish: dc = kw_dc * ghi/1000 * temp derate, then losses and inverter
    ghi = np.asarray(ghi_wm2, dtype=float)
    t_cell = np.asarray(temp_c, dtype=float) + 0.03 * ghi
    derate = 1.0 + gamma * (t_cell - 25.0)
    return np.maximum(kw_dc * ghi / 1000.0 * derate * (1 - losses) * inv_eff, 0.0)
