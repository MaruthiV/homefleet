import numpy as np
import pandas as pd
import yaml
import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import opendssdirect as dss
from homefleet.paths import DATA_PROC, EXPERIMENTS, CONFIGS, ROOT
from homefleet.env import FleetData, HomeFleetEnv
from homefleet.control.baselines import run_controller
from homefleet.rl.ppo import load_policy, run_policy
from run_baselines import REPLAY_CODE_SHA, POLICY_CODE_SHA

TZ = "America/Chicago"
FEEDER = ROOT / "data/raw/smartds/feeder/p1uhs0_1247--p1udt2142/Master_static.dss"
DAYS = {"beryl": "2024-07-08", "heat": "2024-08-20", "normal": "2024-04-15"}
CTRLS = {"B0_fixed": ("dp", dict(mode="fixed")), "B3a_risk_lam5": ("dp", dict(mode="risk_priced", lam_r=5.0)),
         }
KVAR_RATIO = 0.33


def source_connected_buses():
    graph = {}
    for collection in [dss.Lines, dss.Transformers]:
        for name in collection.AllNames():
            collection.Name(name)
            if not dss.CktElement.Enabled(): continue
            buses = [b.split('.')[0].lower() for b in dss.CktElement.BusNames()]
            for bus in buses:
                graph.setdefault(bus, set()).update(buses)
    connected = set()
    for name in dss.Vsources.AllNames():
        dss.Vsources.Name(name)
        connected.add(dss.CktElement.BusNames()[0].split('.')[0].lower())
    pending = list(connected)
    while pending:
        for bus in graph.get(pending.pop(), []):
            if bus not in connected:
                connected.add(bus); pending.append(bus)
    return connected


def run_day(d, day, kind, spec, homes, presets, n=100):
    start = int(d.index.get_indexer([pd.Timestamp(day, tz=TZ)])[0]) - 96
    env = HomeFleetEnv(d)
    env.reset(start, 96 * 2, homes, presets, seed=0, tiers_h=None)
    if kind == "dp":
        res = run_controller(env, S=32, **spec)
    else:
        agent, ostat = load_policy(spec)
        res = run_policy(env, agent, ostat, lam_r=20.0, guardrail=(kind == "rlg"))
    p = res["pd"] - res["pc"]
    g = np.where(env.out, 0., env.load - env.pv - p)
    reference = np.where(env.out, 0., env.load - env.pv)
    return g[96:], reference[96:]


def solve_series(g_batt, load_only, penetration):
    temporary = tempfile.TemporaryDirectory()
    folder = Path(temporary.name) / 'feeder'
    shutil.copytree(FEEDER.resolve().parent, folder)
    master = folder / 'Master_audit.dss'
    lines = FEEDER.read_text().splitlines()
    lines = [l for l in lines if not l.strip().lower().startswith(('plot ', 'export ', 'new monitor.', 'new energymeter.'))]
    master.write_text('\n'.join(lines) + '\n')
    dss.Text.Command('Clear')
    dss.Text.Command(f'Redirect "{master}"')
    names = dss.Loads.AllNames()
    M = len(names)
    n_b = int(penetration * M)
    attached = set(np.random.default_rng(0).permutation(M)[:n_b])
    N = g_batt.shape[1]
    connected = source_connected_buses()
    nodes = dss.Circuit.AllNodeNames()
    phase = np.array([name.rsplit('.', 1)[-1] in {'1', '2', '3'} for name in nodes])
    energized = phase & np.array([name.rsplit('.', 1)[0].lower() in connected for name in nodes])
    rows = []
    for t in range(g_batt.shape[0]):
        for j, nm in enumerate(names):
            kw = float(g_batt[t, j % N]) if j in attached else float(load_only[t, j % N])
            dss.Loads.Name(nm)
            dss.Loads.kW(kw)
            dss.Loads.kvar(KVAR_RATIO * max(kw, 0.0))
        dss.Solution.Solve()
        all_v = np.array(dss.Circuit.AllBusMagPu())
        v = all_v[energized]
        if not np.isfinite(v).all():
            raise ValueError('non-finite AC voltages')
        line_ratios = []
        for name in dss.Lines.AllNames():
            dss.Lines.Name(name)
            rating = dss.Lines.NormAmps()
            if rating > 0:
                line_ratios.append(max(dss.CktElement.CurrentsMagAng()[::2]) / rating)
        rows.append(dict(conv=dss.Solution.Converged(), vmin=v.min(), vmax=v.max(),
                         under=(v < 0.95).mean(), over=(v > 1.05).mean(),
                         overloaded=any(r > 1. for r in line_ratios), max_line_loading=max(line_ratios, default=0.),
                         disconnected_phase_nodes=int((phase & ~energized).sum()), all_phase_vmin=float(all_v[phase].min())))
    temporary.cleanup()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', type=Path)
    a = ap.parse_args()
    if a.policy:
        import torch
        torch.set_num_threads(1)
    controls = dict(CTRLS)
    if a.policy: controls['PPO_guard'] = ('rlg', str(a.policy))
    d = FleetData(DATA_PROC / "revised/houston_2024.npz")
    rng = np.random.default_rng(0)
    homes = rng.integers(0, d.N_pool, 100)
    presets = list(rng.choice(["gen2_single", "gen2_double", "core"], 100, p=[0.6, 0.2, 0.2]))
    out = []
    for cname, (kind, spec) in controls.items():
        for dayname, day in DAYS.items():
            g, load = run_day(d, day, kind, spec, homes, presets)
            for pen in (0.3, 1.0):
                r = solve_series(g, load, pen)
                out.append(dict(controller=cname, day=dayname, penetration=pen, converged=float(r.conv.mean()),
                                vmin=float(r.vmin.min()), vmax=float(r.vmax.max()),
                                steps_with_undervolt=float((r.under > 0).mean()), mean_undervolt_bus_frac=float(r.under.mean()),
                                steps_with_overvolt=float((r.over > 0).mean()),
                                steps_with_line_overload=float(r.overloaded.mean()), max_line_loading=float(r.max_line_loading.max()),
                                disconnected_phase_nodes=int(r.disconnected_phase_nodes.max()), all_phase_vmin=float(r.all_phase_vmin.min())))
                print(out[-1], flush=True)
        # load-only reference once per day
    for dayname, day in DAYS.items():
        g, load = run_day(d, day, "dp", dict(mode="fixed"), homes, presets)
        r = solve_series(load, load, 0.0)
        out.append(dict(controller="load_PV_reference", day=dayname, penetration=0.0, converged=float(r.conv.mean()),
                        vmin=float(r.vmin.min()), vmax=float(r.vmax.max()),
                        steps_with_undervolt=float((r.under > 0).mean()), mean_undervolt_bus_frac=float(r.under.mean()),
                        steps_with_overvolt=float((r.over > 0).mean()),
                        steps_with_line_overload=float(r.overloaded.mean()), max_line_loading=float(r.max_line_loading.max()),
                        disconnected_phase_nodes=int(r.disconnected_phase_nodes.max()), all_phase_vmin=float(r.all_phase_vmin.min())))
        print(out[-1], flush=True)
    t = pd.DataFrame(out)
    learned = t.controller.eq('PPO_guard')
    t['protocol'] = d.protocol
    t['source_sha256'] = REPLAY_CODE_SHA
    t['policy_sha256'] = np.where(learned, hashlib.sha256(a.policy.read_bytes()).hexdigest() if a.policy else '', '')
    t['policy_source_sha256'] = np.where(learned, POLICY_CODE_SHA, '')
    t.to_csv(EXPERIMENTS / "revised/ac_check.csv", index=False)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
