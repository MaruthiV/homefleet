"""Paired, pre-specified fleet sensitivities; not additional annual test years."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import yaml
from homefleet.paths import ROOT, DATA_PROC, EXPERIMENTS, CONFIGS
from homefleet.env import HomeFleetEnv
from homefleet.control.baselines import run_controller
from homefleet.eval.episode import summarize
from run_baselines import cached_data, windows_for_year, REPLAY_CODE_SHA, POLICY_CODE_SHA

SCENARIOS = {
    'base_100': (100, [.6, .2, .2], 0.),
    'size_500': (500, [.6, .2, .2], 0.),
    'size_1000': (1000, [.6, .2, .2], 0.),
    'core_70pct': (100, [.2, .1, .7], 0.),
    'dropout_20pct': (100, [.6, .2, .2], .2),
}
WINDOWS = [('houston', 2024, w) for w in ['w10', 'derecho_2024', 'beryl_2024', 'heat_aug_2024']]
WINDOWS += [('houston', 2021, 'uri_2021'), ('houston', 2022, 'elliott_2022')]
REFERENCES = {'fixed20': dict(mode='fixed'), 'storm': dict(mode='storm_watch'),
              'risk5': dict(mode='risk_priced', lam_r=5.)}


def job(spec):
    scenario, metro, year, window, controller, policy = spec
    n, mix, dropout = SCENARIOS[scenario]
    d = cached_data(DATA_PROC / 'revised' / f'{metro}_{year}.npz')
    events = yaml.safe_load((CONFIGS / 'splits.yaml').read_text())['events']
    wins = {w: (s, count) for w, s, count in windows_for_year(d, year, events=events)}
    start, steps = wins[window]
    rng = np.random.default_rng(0)
    homes = rng.integers(0, d.N_pool, n)
    presets = list(rng.choice(['gen2_single', 'gen2_double', 'core'], n, p=mix))
    env = HomeFleetEnv(d)
    env.reset(start, steps, homes, presets, seed=0, dropout=dropout)
    began = time.perf_counter()
    if controller in REFERENCES:
        trace = run_controller(env, S=32, **REFERENCES[controller])
    else:
        import torch
        from homefleet.rl.ppo import load_policy, run_policy
        torch.set_num_threads(1)
        agent, stat = load_policy(policy)
        trace = run_policy(env, agent, stat, guardrail=controller == 'ppo_guard')
    row = summarize(trace, env.dt, params=env.params, soh=env.soh, initial_energy=env.initial_energy)
    return row | dict(scenario=scenario, metro=metro, year=year, window=window, controller=controller,
                      n=n, seed=0, dropout=dropout, source_sha256=REPLAY_CODE_SHA,
                      policy_sha256=hashlib.sha256(Path(policy).read_bytes()).hexdigest() if policy else '',
                      policy_source_sha256=POLICY_CODE_SHA if policy else '',
                      protocol=d.protocol, sec=time.perf_counter() - began)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', type=Path)
    ap.add_argument('--references-only', action='store_true')
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    selected = a.policy
    if not a.references_only and selected is None:
        selected = ROOT / json.loads((EXPERIMENTS / 'revised/selected_policy.json').read_text())['path']
    controls = list(REFERENCES) + ([] if a.references_only else ['ppo', 'ppo_guard'])
    target = EXPERIMENTS / 'revised/structure.csv'
    rows = pd.read_csv(target).fillna('').to_dict('records') if target.exists() else []
    digest = hashlib.sha256(selected.read_bytes()).hexdigest() if selected else ''
    rows = [r for r in rows if r['source_sha256'] == REPLAY_CODE_SHA and
            (r['controller'] in REFERENCES or
             (r['policy_sha256'] == digest and r.get('policy_source_sha256') == POLICY_CODE_SHA))]
    done = {(r['scenario'], r['metro'], r['year'], r['window'], r['controller']) for r in rows}
    jobs = [(s, m, y, w, c, str(selected) if c not in REFERENCES else None)
            for s in SCENARIOS for m, y, w in WINDOWS for c in controls if (s, m, y, w, c) not in done]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(job, jobs):
            rows.append(row)
            pd.DataFrame(rows).to_csv(target, index=False)
            print(row['scenario'], row['window'], row['controller'], round(row['sec'], 2), flush=True)
    print(target)


if __name__ == '__main__':
    main()
