"""Select on completed Houston 2024 validation only, independently of test readiness."""
import hashlib
import json
import pandas as pd
from homefleet.paths import ROOT, EXPERIMENTS
from run_baselines import REPLAY_CODE_SHA, POLICY_CODE_SHA, is_regular_window
from reproduce import training_runs
from summarize_revision import summarize_group, save_selection


def main():
    out = EXPERIMENTS / 'revised'
    manifest = json.loads((ROOT / 'artifacts/experiment_manifest.json').read_text())
    specs = {s['path']: s for s in manifest['files']}
    baseline = pd.read_csv(out / 'baselines_houston_2024.csv')
    base_spec = specs['experiments/revised/baselines_houston_2024.csv']
    expected_base = {(w['name'], c) for w in base_spec['windows'] for c in base_spec['controllers']}
    if set(zip(baseline.window, baseline.controller)) != expected_base or (baseline.source_sha256 != REPLAY_CODE_SHA).any():
        raise ValueError('validation baseline is incomplete or has an outdated metric revision')
    reference = baseline[(baseline.controller == 'B0_fixed') & baseline.window.map(is_regular_window)]
    rows = []
    for run, *_ in training_runs():
        if not run.startswith('ppo_pc_'): continue
        relative = f'experiments/revised/policy_{run}_houston_2024.csv'
        spec = specs[relative]
        d = pd.read_csv(ROOT / relative)
        expected = {(w['name'], c) for w in spec['windows'] for c in spec['controllers']}
        checkpoint = out / run / 'policy.pt'
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if len(d) != len(expected) or set(zip(d.window, d.controller)) != expected:
            raise ValueError(f'{run}: incomplete validation windows')
        if (d.source_sha256 != POLICY_CODE_SHA).any() or (d.policy_sha256 != digest).any() or (d.protocol != 'causal-v3').any():
            raise ValueError(f'{run}: mismatched validation provenance')
        if (d.n != 100).any() or (d.seed != 0).any() or d.physical_violations.sum() or d.reachable_floor_violations.sum():
            raise ValueError(f'{run}: incorrect validation configuration or constraint violation')
        regular = d[d.window.map(is_regular_window)]
        rows.append(dict(run=run, controller=f'B6_{run}', **summarize_group(regular, reference)))
    candidates = pd.DataFrame(rows)
    best = candidates.sort_values(['validation_score', 'controller'], ascending=[False, True]).iloc[0]
    selected = save_selection(best.controller, best.validation_score)
    candidates.to_csv(out / 'validation_candidates.csv', index=False)
    print(json.dumps(selected, indent=2))


if __name__ == '__main__':
    main()
