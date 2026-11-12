import numpy as np
from numba import njit


@njit(cache=True)
def backward(load, pv, price, floors, lev, p, pc, pd, feasible, values, dt,
             retail, delivery, solar, wear_cost, floor_penalty, idle, idle_penalty,
             risk, need, hazard, eta_d):
    H, N = load.shape
    S = lev.shape[1]
    policy = np.zeros((H, N, S), dtype=np.int16)
    for t in range(H - 1, -1, -1):
        next_values = np.empty_like(values)
        for i in range(N):
            for s in range(S):
                best = -np.inf
                choice = 0
                for j in range(S):
                    if not feasible[i, s, j]:
                        continue
                    g = load[t, i] - pv[t, i] - p[i, s, j]
                    m = max(g, 0.)
                    x = max(-g, 0.)
                    reward = dt * (retail * load[t, i] - (price[t, i] + delivery) * m + price[t, i] * x - solar * pv[t, i])
                    reward -= wear_cost * dt * (pc[i, s, j] + pd[i, s, j])
                    if lev[i, j] < floors[t, i] - 1e-9:
                        reward -= floor_penalty
                    reward -= idle_penalty * dt * max(idle[i] - lev[i, j], 0.)
                    reward -= risk * eta_d * hazard[t, i] * max(need[t, i] - lev[i, j], 0.)
                    q = reward + values[i, j]
                    if q > best:
                        best, choice = q, j
                next_values[i, s] = best
                policy[t, i, s] = choice
        values = next_values
    return policy
