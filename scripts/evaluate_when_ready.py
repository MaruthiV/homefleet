"""Drain completed checkpoints during a separate, already-running training queue."""
import subprocess
import sys
import time
from finish_audits import baselines_complete
from homefleet.paths import EXPERIMENTS, ROOT
from reproduce import training_runs


def main():
    seen = set()
    expected = {r[0] for r in training_runs()}
    candidates = {name for name in expected if name.startswith('ppo_pc_')}
    selected = False
    while True:
        ready = {name for name in expected if (EXPERIMENTS / 'revised' / name / 'train_log.json').exists()}
        if ready != seen:
            print(f'{len(ready)}/{len(expected)} completed checkpoints; evaluating available runs', flush=True)
            subprocess.run([sys.executable, 'scripts/reproduce.py', 'policies', '--available-only', '--workers', '2'], cwd=ROOT, check=True)
            seen = ready
        if candidates <= ready and not selected and baselines_complete():
            subprocess.run([sys.executable, 'scripts/select_policy.py'], cwd=ROOT, check=True)
            selected = True
        if ready == expected and baselines_complete():
            subprocess.run([sys.executable, 'scripts/verify_results.py'], cwd=ROOT, check=True)
            subprocess.run([sys.executable, 'scripts/summarize_revision.py'], cwd=ROOT, check=True)
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
