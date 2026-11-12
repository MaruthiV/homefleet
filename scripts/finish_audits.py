"""Finish selected-policy audits while separate training/evaluation queues run.

This waits for their completion records. It never publishes or submits files.
The ordinary individual commands are documented in REPRODUCIBILITY.md.
"""
import json
import subprocess
import sys
import time
from homefleet.paths import ROOT, EXPERIMENTS
from reproduce import training_runs
from run_baselines import POLICY_CODE_SHA

OUT = EXPERIMENTS / 'revised'


def run(*args):
    print('Running', *args, flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def wait_for(predicate, label):
    print('Waiting for', label, flush=True)
    while not predicate(): time.sleep(30)


def baselines_complete():
    path = OUT / 'baselines_status.json'
    if not path.exists(): return False
    try: status = json.loads(path.read_text())
    except json.JSONDecodeError: return False
    if any(v['state'] == 'failed' for v in status.values()): raise RuntimeError('baseline worker failed')
    return len(status) == 11 and all(v['state'] == 'complete' for v in status.values())


def main():
    wait_for(baselines_complete, 'all baseline writers to finish')
    candidates = [r[0] for r in training_runs() if r[0].startswith('ppo_pc_')]
    def validation_complete():
        for name in candidates:
            marker = OUT / f'policy_{name}_houston_2024.complete.json'
            if not marker.exists(): return False
            try: info = json.loads(marker.read_text())
            except json.JSONDecodeError: return False
            if info.get('source_sha256') != POLICY_CODE_SHA: return False
        return True
    wait_for(validation_complete, 'all ten complete validation evaluations with current source hashes')
    run('scripts/select_policy.py')
    selected = ROOT / json.loads((OUT / 'selected_policy.json').read_text())['path']
    run('scripts/eval_structure.py', '--workers', '2')
    run('scripts/degradation_table.py')
    run('scripts/ac_check.py', '--policy', str(selected))
    run('scripts/stress_guardrail.py', '--policy', str(selected))
    def complete_summary():
        path = OUT / 'report.json'
        if not path.exists(): return False
        try: return json.loads(path.read_text()).get('complete', False)
        except json.JSONDecodeError: return False
    wait_for(complete_summary, 'complete main-grid verification and aggregation')
    run('scripts/benchmark_latency.py')
    run('scripts/verify_results.py', '--full')
    run('scripts/summarize_revision.py')
    run('scripts/build_paper_results.py')
    print('Numerical artifacts ready for manuscript layout review.', flush=True)


if __name__ == '__main__':
    main()
