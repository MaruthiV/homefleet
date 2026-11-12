import argparse
import copy
import hashlib
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from homefleet.env import FleetData, HomeFleetEnv
from homefleet.eval.episode import summarize
from homefleet.paths import ROOT, EXPERIMENTS
from run_baselines import REPLAY_CODE_SHA, POLICY_CODE_SHA


def scenarios():
    cases = []
    for price, duration, lead, initial in product([50., 1000., 5000.], [2., 8., 24.], [0., 1., 3., 6.], [.1, .9]):
        cases.append(dict(price=price, duration=duration, lead=lead, soc0=initial, alert='correct', dropout=0., load_multiplier=1., soh=1.))
    for alert, price in product(['missed', 'false'], [50., 1000., 5000.]):
        cases.append(dict(price=price, duration=8., lead=6., soc0=.1, alert=alert, dropout=0., load_multiplier=1., soh=1.))
    for dropout, multiplier, soh in product([0., .2], [1., 2.], [.85, 1.]):
        cases.append(dict(price=5000., duration=8., lead=3., soc0=.1, alert='correct', dropout=dropout, load_multiplier=multiplier, soh=soh))
    return cases


def run_case(base, case, guarded, policy=None):
    d = copy.copy(base)
    start = d.evaluation_start
    onset = start + 48
    end = min(d.T, onset + int(case['duration'] * 4))
    d.load = base.load.copy(); d.pv = base.pv.copy()
    d.load[start:] = 2. * case['load_multiplier']; d.pv[start:] = 0.
    d.price_rt = base.price_rt.copy(); d.price_da = base.price_da.copy()
    d.price_rt[start:] = 50.; d.price_rt[onset - 24:end] = case['price']
    d.price_da[start:] = d.price_rt[start:]
    d.frac = base.frac.copy(); d.frac.iloc[start:] = 0.
    if case['alert'] != 'false': d.frac.iloc[onset:end] = .5
    d.alert = base.alert.copy(); d.alert.iloc[start:] = False
    if case['alert'] != 'missed': d.alert.iloc[onset - int(case['lead'] * 4):end] = True
    d.hazard = np.zeros_like(base.hazard)
    env = HomeFleetEnv(d)
    env.reset(start, d.T - start, np.arange(32) % d.N_pool, ['gen2_single'] * 32, seed=0,
              soc0=case['soc0'], soh=np.full(32, case['soh']), dropout=case['dropout'])
    onset_energy = None
    while env.k < env.T:
        if env.k == 48: onset_energy = env.e.copy()
        override = np.where(env.alert[env.k], .9 * env.params.energy_kwh * env.soh, env.emin()) if guarded else None
        action = [1., 0.]
        if policy is not None:
            import torch
            from homefleet.rl.ppo import to_env_action
            agent, stat = policy
            with torch.no_grad(): action = to_env_action(agent.actor_mean(torch.as_tensor(stat.norm(env.obs()).astype(np.float32))).numpy())
        env.rl_step(action, lam_r=20, floor_override=override)
    trace = env.results()
    result = summarize(trace, env.dt, params=env.params, soh=env.soh, initial_energy=env.initial_energy)
    alert_steps = env.alert.any(axis=1)
    active = np.flatnonzero(alert_steps)
    result['cash_margin_during_alert_usd_per_home'] = float(trace['margin'][alert_steps].sum() / env.N)
    result['cash_margin_after_alert_usd_per_home'] = float(trace['margin'][active[-1] + 1:].sum() / env.N) if len(active) else np.nan
    return result | dict(onset_soc_mean=float(np.mean(onset_energy / (env.params.energy_kwh * env.soh))),
                         guarded=guarded, policy='always_discharge' if policy is None else 'ppo')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', type=Path)
    a = ap.parse_args()
    base = FleetData(ROOT / 'src/homefleet/assets/quickstart.npz')
    policies = [None]
    digest = hashlib.sha256(a.policy.read_bytes()).hexdigest() if a.policy else ''
    if a.policy:
        import torch
        torch.set_num_threads(1)
        from homefleet.rl.ppo import load_policy
        policies.append(load_policy(a.policy))
    rows = []
    for k, case in enumerate(scenarios()):
        for policy, guarded in product(policies, [False, True]):
            rows.append(dict(case=k, **case, **run_case(base, case, guarded, policy),
                             protocol=base.protocol, source_sha256=REPLAY_CODE_SHA,
                             policy_sha256=digest if policy is not None else '',
                             policy_source_sha256=POLICY_CODE_SHA if policy is not None else ''))
        if k % 10 == 0: print('case', k, flush=True)
    out = EXPERIMENTS / 'revised' / ('stress_guardrail_ppo.csv' if a.policy else 'stress_guardrail.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == '__main__':
    main()
