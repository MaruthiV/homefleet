"""Aging-model sensitivity on four repeated 2024 windows, not lifetime validation."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from homefleet.paths import ROOT, DATA_PROC, EXPERIMENTS, CONFIGS
from homefleet.env import FleetData, HomeFleetEnv
from homefleet.control.baselines import run_controller
from homefleet.rl.ppo import load_policy, run_policy
from homefleet.eval.blast_adapter import blast_fade
from homefleet.eval.degradation_report import episode_degradation
from homefleet.eval.episode import summarize
from run_baselines import windows_for_year, REPLAY_CODE_SHA, POLICY_CODE_SHA

WINDOWS = ['beryl_2024', 'heat_aug_2024', 'derecho_2024', 'w10']
REFERENCES = {'fixed20': dict(mode='fixed'), 'hold90': dict(mode='fixed', idle_target_frac=.9),
              'storm': dict(mode='storm_watch'), 'risk5': dict(mode='risk_priced', lam_r=5.)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', type=Path)
    ap.add_argument('--references-only', action='store_true')
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    policy = a.policy
    if not a.references_only and policy is None:
        policy = ROOT / json.loads((EXPERIMENTS / 'revised/selected_policy.json').read_text())['path']
    digest = hashlib.sha256(policy.read_bytes()).hexdigest() if policy else ''
    controls = list(REFERENCES) + ([] if a.references_only else ['ppo', 'ppo_guard'])
    d = FleetData(DATA_PROC / 'revised/houston_2024.npz')
    events = yaml.safe_load((CONFIGS / 'splits.yaml').read_text())['events']
    wins = {w: (s, n) for w, s, n in windows_for_year(d, 2024, events=events)}
    rng = np.random.default_rng(0)
    homes = rng.integers(0, d.N_pool, 20)
    presets = list(rng.choice(['gen2_single', 'gen2_double', 'core'], 20, p=[.6, .2, .2]))
    target = EXPERIMENTS / 'revised/degradation_table.csv'
    rows = pd.read_csv(target).to_dict('records') if target.exists() else []
    rows = [r for r in rows if r.get('source_sha256') == REPLAY_CODE_SHA and
            (r['controller'] in REFERENCES or
             (r.get('policy_sha256') == digest and r.get('policy_source_sha256') == POLICY_CODE_SHA))]
    done = {(r['controller'], r['window']) for r in rows}
    for controller in controls:
        for window in WINDOWS:
            if (controller, window) in done: continue
            start, steps = wins[window]
            env = HomeFleetEnv(d)
            env.reset(start, steps, homes, presets, seed=0)
            began = time.perf_counter()
            if controller in REFERENCES:
                trace = run_controller(env, S=32, **REFERENCES[controller])
            else:
                agent, stat = load_policy(policy)
                trace = run_policy(env, agent, stat, guardrail=controller == 'ppo_guard')
            # Ambient temperature is a proxy, not a cell thermal model.
            temp = d.weather[start:start + steps, 0, 0]
            fade = blast_fade(trace, env.params, env.soh, temp, env.dt)
            deg = episode_degradation(trace, env.params, env.soh, env.dt)
            score = summarize(trace, env.dt, params=env.params, soh=env.soh, initial_energy=env.initial_energy)
            row = dict(controller=controller, window=window, n=20, protocol=d.protocol,
                       source_sha256=REPLAY_CODE_SHA,
                       policy_sha256=digest if controller not in REFERENCES else '',
                       policy_source_sha256=POLICY_CODE_SHA if controller not in REFERENCES else '',
                       hours=steps * env.dt, mean_soc=score['mean_soc_frac'],
                       blast_pct_yr=float(100 * fade.mean()), et_pct_yr=deg['fade_et_pct_yr'],
                       cb_pct_yr=deg['fade_cb_pct_yr'], lc_pct_yr=deg['fade_lc_pct_yr'],
                       margin_per_home_week=score['margin_per_home_week'], sec=time.perf_counter() - began)
            rows.append(row)
            pd.DataFrame(rows).to_csv(target, index=False)
            print(row, flush=True)
    print(target)


if __name__ == "__main__":
    main()
