import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import yaml
import hashlib
from homefleet.paths import ROOT, EXPERIMENTS, CONFIGS

OUT = EXPERIMENTS / 'revised'


def training_runs():
    runs = []
    for penalty in (5, 20):
        for seed in range(1, 6):
            runs.append((f'ppo_pc_lam{penalty}_s{seed}', penalty, seed, 'learned', True))
    for penalty, seeds in [(1, [0, 1]), (5, [0, 1, 2]), (20, [0, 1])]:
        for seed in seeds:
            runs.append((f'ppo_learned_lam{penalty}_s{seed}', penalty, seed, 'learned', False))
    for mode in ['fixed', 'none']:
        for seed in (0, 1):
            runs.append((f'ppo_{mode}_lam5_s{seed}', 5, seed, mode, False))
    return runs


def execute(jobs, workers, phase):
    (OUT / 'logs').mkdir(parents=True, exist_ok=True)
    status_path = OUT / f'{phase}_status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    lock = threading.Lock()
    def run(job):
        name, args, output = job
        if output is not None and output.exists():
            return name, 'complete'
        with lock:
            status[name] = {'state': 'running', 'command': args, 'started': time.time()}
            status_path.write_text(json.dumps(status, indent=2))
        with open(OUT / 'logs' / f'{name}.log', 'a') as log:
            process = subprocess.Popen([sys.executable, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
            with lock:
                status[name]['pid'] = process.pid
                status_path.write_text(json.dumps(status, indent=2))
            code = process.wait()
        with lock:
            status[name].update(state='complete' if code == 0 else 'failed', exit_code=code, finished=time.time())
            status_path.write_text(json.dumps(status, indent=2))
        return name, status[name]['state']
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(run, j) for j in jobs]):
            name, state = future.result()
            print(phase, name, state, flush=True)
            if state == 'failed': failed.append(name)
    if failed:
        raise RuntimeError(f'failed jobs: {failed}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['train', 'baselines', 'policies'])
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--available-only', action='store_true', help='evaluate completed checkpoints while training continues')
    ap.add_argument('--runs', nargs='+', help='restrict training/policy evaluation to these exact run names')
    ap.add_argument('--status-label', help='separate status filename for a concurrent queue')
    a = ap.parse_args()
    if a.status_label and not a.status_label.replace('_', '').isalnum():
        ap.error('--status-label must contain only letters, digits, and underscores')
    runs = training_runs()
    if a.runs:
        if a.phase == 'baselines': ap.error('--runs applies to training or policy evaluation')
        unknown = set(a.runs) - {r[0] for r in runs}
        if unknown: ap.error(f'unknown run names: {sorted(unknown)}')
        runs = [r for r in runs if r[0] in a.runs]
    cfg = yaml.safe_load((CONFIGS / 'revision.yaml').read_text())
    jobs = []
    if a.phase == 'train':
        for name, penalty, seed, mode, promise in runs:
            args = ['scripts/train_ppo.py', '--steps', str(cfg['training_steps']), '--lam-r', str(penalty), '--seed', str(seed),
                    '--reserve-mode', mode, '--name', f'revised/{name}', '--threads', '1']
            if promise: args += ['--promise-cost', '--ent', '0.005']
            jobs.append((name, args, OUT / name / 'policy.pt'))
    elif a.phase == 'baselines':
        for metro, year in cfg['evaluation'] + cfg['events']:
            name = f'baselines_{metro}_{year}'
            args = ['scripts/run_baselines.py', '--metro', metro, '--year', str(year), '--procs', '1',
                    '--out', f'revised/{name}.csv']
            if [metro, year] in cfg['events']: args += ['--windows', 'events']
            jobs.append((name, args, None))
    else:
        from run_baselines import POLICY_CODE_SHA
        for name, penalty, seed, mode, promise in runs:
            for metro, year in cfg['evaluation'] + cfg['events']:
                if not (OUT / name / 'policy.pt').exists() or not (OUT / name / 'train_log.json').exists():
                    if a.available_only: continue
                    raise FileNotFoundError(OUT / name / 'policy.pt')
                for guard in [False, True]:
                    output_name = f'policy_{name}{"_guard" if guard else ""}_{metro}_{year}'
                    marker = OUT / f'{output_name}.complete.json'
                    if marker.exists():
                        completed = json.loads(marker.read_text())
                        if (completed.get('source_sha256') == POLICY_CODE_SHA and
                            completed.get('policy_sha256') == hashlib.sha256((OUT / name / 'policy.pt').read_bytes()).hexdigest() and
                            completed.get('n') == 100 and completed.get('seed') == 0 and completed.get('dropout') == 0 and
                            completed.get('mix') == '0.6,0.2,0.2' and completed.get('lam_override') is None and
                            (OUT / f'{output_name}.csv').exists()):
                            continue
                    args = ['scripts/eval_policy.py', '--policy', str(OUT / name / 'policy.pt'), '--metro', metro,
                            '--year', str(year), '--out', f'revised/{output_name}.csv']
                    if guard: args += ['--guardrail']
                    if [metro, year] in cfg['events']: args += ['--windows', 'events']
                    jobs.append((output_name, args, None))
    execute(jobs, a.workers, a.status_label or a.phase)


if __name__ == '__main__':
    main()
