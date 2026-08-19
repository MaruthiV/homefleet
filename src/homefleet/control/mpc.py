def pooled_mpc(state, forecasts, floors, params, tariff, dt, horizon=96):
    # a&z style rolling horizon over the fleet, floors given per home (b0/b1/b2)
    raise NotImplementedError


def risk_priced_mpc(state, forecasts, hazard, need, lam_r, params, tariff, dt, horizon=96):
    # b3a: adds lam_r * sum_h hazard_h * shortfall_h, shortfall >= need - e
    raise NotImplementedError


def chance_constrained_mpc(state, forecasts, hazard, need, eps, params, tariff, dt, horizon=96):
    # b3b: e_{t+h} >= need when hazard_h >= eps
    raise NotImplementedError
