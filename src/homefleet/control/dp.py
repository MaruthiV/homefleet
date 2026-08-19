import numpy as np


def energy_grid(emin, emax, S):
    # (N,S) energy levels per home
    return emin[:, None] + (emax - emin)[:, None] * np.linspace(0.0, 1.0, S)[None, :]


def dp_plan(e0, load, pv, price, floors, params, soh, avail, tariff, dt, S=40, lam_r=0.0, need=None, hazard=None,
            deg_cost_per_kwh=0.0, terminal_price=None, hard_floor_penalty=1e6, idle_target=None, idle_penalty_per_kwh=0.0):
    # per-home finite-horizon dp over a soc grid, vectorized over homes
    # load/pv: (H,N) kw, price: (H,) or (H,N) usd/mwh, floors/need: (H,N) kwh, hazard: (H,) or (H,N) prob
    # returns planned next-energy (H,N) and planned signed power (H,N)
    H, N = load.shape
    emin = params.soc_min * params.energy_kwh
    emax = params.energy_kwh * soh
    lev = energy_grid(emin, emax, S)
    price = np.broadcast_to(np.asarray(price, dtype=float).reshape(H, -1), (H, N)) / 1000.0
    if hazard is not None:
        hazard = np.broadcast_to(np.asarray(hazard, dtype=float).reshape(H, -1), (H, N))
    tp = (np.nanmean(price) if terminal_price is None else terminal_price / 1000.0)
    V = lev * tp * params.eta_d
    de = lev[:, None, :] - lev[:, :, None]
    pc = np.where(de > 0, de / (dt * params.eta_c), 0.0)
    pd_ = np.where(de < 0, -de * params.eta_d / dt, 0.0)
    p = pd_ - pc
    pc_ok = pc <= (params.p_charge_kw * avail)[:, None, None] + 1e-9
    pd_ok = pd_ <= (params.p_discharge_kw * avail)[:, None, None] + 1e-9
    feas = pc_ok & pd_ok
    wear = deg_cost_per_kwh * dt * (pc + pd_)
    policy = np.zeros((H, N, S), dtype=np.int16)
    for t in range(H - 1, -1, -1):
        g = (load[t] - pv[t])[:, None, None] - p
        m = np.maximum(g, 0.0)
        x = np.maximum(-g, 0.0)
        lam = price[t][:, None, None]
        r = dt * (tariff.p_ret * load[t][:, None, None] - (lam + tariff.c_tdsp) * m + lam * x - tariff.beta * pv[t][:, None, None]) - wear
        below = lev[:, None, :] < floors[t][:, None, None] - 1e-9
        r = r - hard_floor_penalty * below
        if idle_target is not None and idle_penalty_per_kwh > 0:
            # base-style preference to sit near a high soc when idle
            r = r - idle_penalty_per_kwh * dt * np.maximum(idle_target[:, None, None] - lev[:, None, :], 0.0)
        if lam_r > 0 and need is not None and hazard is not None:
            short = np.maximum(need[t][:, None, None] - lev[:, None, :], 0.0)
            r = r - lam_r * hazard[t][:, None, None] * short
        Q = r + V[:, None, :]
        Q = np.where(feas, Q, -np.inf)
        best = Q.argmax(axis=2)
        V = np.take_along_axis(Q, best[:, :, None], axis=2)[:, :, 0]
        policy[t] = best
    # forward pass from e0 snapped to nearest level
    s = np.abs(lev - e0[:, None]).argmin(axis=1)
    plan_e = np.zeros((H, N))
    plan_p = np.zeros((H, N))
    rows = np.arange(N)
    for t in range(H):
        s_next = policy[t][rows, s]
        plan_e[t] = lev[rows, s_next]
        plan_p[t] = p[rows, s, s_next]
        s = s_next
    return plan_e, plan_p
