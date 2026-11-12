import numpy as np
from homefleet.sim.battery import envelope, clip_power


def priority_disaggregate(p_target, e, params, soh, avail, floor, dt):
    # cheap feasible split of aggregate target: sort homes by normalized state of energy
    pc_max, pd_max = envelope(e, params, soh, avail, floor, dt)
    n = len(e)
    pc = np.zeros(n)
    pd_ = np.zeros(n)
    soe = (e - floor) / np.maximum(params.energy_kwh * soh - floor, 1e-9)
    if p_target >= 0:
        order = np.argsort(-soe)
        left = min(p_target, pd_max.sum())
        for i in order:
            take = min(pd_max[i], left)
            pd_[i] = take
            left -= take
            if left <= 0:
                break
    else:
        order = np.argsort(soe)
        left = min(-p_target, pc_max.sum())
        for i in order:
            take = min(pc_max[i], left)
            pc[i] = take
            left -= take
            if left <= 0:
                break
    pc, pd_ = clip_power(e, pc, pd_, params, soh, avail, floor, dt)
    shield = p_target - (pd_.sum() - pc.sum())
    return pc, pd_, shield
