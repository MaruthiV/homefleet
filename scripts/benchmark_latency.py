"""CPU wall time for complete decision/simulation loops, excluding episode reset."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from homefleet.paths import ROOT, DATA_PROC, EXPERIMENTS
from homefleet.env import FleetData, HomeFleetEnv
from homefleet.control.baselines import run_controller
from homefleet.control.dp import backward
from homefleet.rl.ppo import load_policy, run_policy
from run_baselines import REPLAY_CODE_SHA, POLICY_CODE_SHA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', type=Path)
    a = ap.parse_args()
    torch.set_num_threads(1)
    policy = a.policy or ROOT / json.loads((EXPERIMENTS / 'revised/selected_policy.json').read_text())['path']
    agent, stat = load_policy(policy)
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    d = FleetData(DATA_PROC / 'revised/houston_2024.npz')
    rows = []
    for n in [100, 500, 1000]:
        rng = np.random.default_rng(0)
        homes = rng.integers(0, d.N_pool, n)
        presets = list(rng.choice(['gen2_single', 'gen2_double', 'core'], n, p=[.6, .2, .2]))
        for controller in ['fixed20', 'risk5', 'ppo', 'ppo_guard']:
            for repeat in range(4):
                env = HomeFleetEnv(d)
                env.reset(d.evaluation_start + 96 * 100, 96, homes, presets, seed=0)
                start = time.perf_counter()
                if controller == 'fixed20': run_controller(env, S=32, mode='fixed')
                elif controller == 'risk5': run_controller(env, S=32, mode='risk_priced', lam_r=5.)
                else: run_policy(env, agent, stat, guardrail=controller == 'ppo_guard')
                elapsed = time.perf_counter() - start
                if repeat == 0: continue  # warm-up, including JIT initialization
                row = dict(controller=controller, n=n, repeat=repeat, seconds=elapsed,
                           ms_per_interval=1000 * elapsed / 96, intervals=96,
                           dp_backend='numba' if backward is not None else 'numpy',
                           machine=platform.machine(), platform=platform.platform(), python=platform.python_version(),
                           torch_threads=torch.get_num_threads(), protocol=d.protocol,
                           source_sha256=REPLAY_CODE_SHA,
                           policy_sha256=digest if controller in ['ppo', 'ppo_guard'] else '',
                           policy_source_sha256=POLICY_CODE_SHA if controller in ['ppo', 'ppo_guard'] else '')
                rows.append(row)
                print(n, controller, round(row['ms_per_interval'], 3), flush=True)
    pd.DataFrame(rows).to_csv(EXPERIMENTS / 'revised/latency.csv', index=False)


if __name__ == '__main__':
    main()
