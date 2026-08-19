import numpy as np

F_ET = 2.71e-5
F_CB = 5.24e-4
ALPHA_CB = 2.03


def linear_calendar(years, life_years=10.0, eol=0.8):
    return (1 - eol) * years / life_years


def energy_throughput(throughput_kwh, energy_kwh):
    # humiston et model: fade proportional to normalized throughput
    return F_ET * throughput_kwh / energy_kwh


def rainflow_cycles(soc):
    # simple astm rainflow on a soc trace -> list of (range, count)
    s = np.asarray(soc, dtype=float)
    pts = [s[0]]
    for v in s[1:]:
        if len(pts) >= 2 and (v - pts[-1]) * (pts[-1] - pts[-2]) > 0:
            pts[-1] = v
        else:
            pts.append(v)
    cycles = []
    stack = []
    for p in pts:
        stack.append(p)
        while len(stack) >= 3:
            x = abs(stack[-1] - stack[-2])
            y = abs(stack[-2] - stack[-3])
            if x < y:
                break
            if len(stack) == 3:
                cycles.append((y, 0.5))
                stack.pop(0)
            else:
                cycles.append((y, 1.0))
                del stack[-3:-1]
    for i in range(len(stack) - 1):
        cycles.append((abs(stack[i + 1] - stack[i]), 0.5))
    return cycles


def cycle_based(soc):
    return sum(F_CB * (r ** ALPHA_CB) * c for r, c in rainflow_cycles(soc))


def blast_lfp(time_s, soc, temp_c):
    # run in the blast env (requirements-blast.txt); returns relative capacity trace
    raise NotImplementedError
