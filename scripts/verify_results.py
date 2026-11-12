import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from homefleet.paths import ROOT
from run_baselines import REPLAY_CODE_SHA, POLICY_CODE_SHA


def exact_coverage_errors(frame, columns, expected, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        return [f'{label}: missing coverage fields {sorted(missing)}']
    actual = list(frame[columns].itertuples(index=False, name=None))
    if len(actual) != len(set(actual)) or set(actual) != expected:
        return [f'{label}: missing, unplanned, or repeated comparisons']
    return []


def policy_provenance_errors(frame, learned, expected_digest, label):
    if not learned.any():
        return []
    required = {'policy_sha256', 'policy_source_sha256'}
    if (expected_digest is None or not required.issubset(frame.columns) or
            not frame.loc[learned, 'policy_sha256'].eq(expected_digest).all() or
            not frame.loc[learned, 'policy_source_sha256'].eq(POLICY_CODE_SHA).all()):
        return [f'{label}: learned rows do not identify the selected checkpoint and policy source']
    return []


def verify(root=ROOT):
    manifest = json.loads((root / 'artifacts/experiment_manifest.json').read_text())
    errors = []
    bundles = json.loads((root / 'data/processed/revised/manifest.json').read_text())
    bundle_hashes = {s['file']: s['sha256'] for s in bundles['bundles']}
    if manifest['protocol'] != 'causal-v3' or bundles['protocol'] != 'causal-v3':
        errors.append('experiment and input manifests must use causal-v3')
    paired_episodes = {}
    required = ['daily_cash_margin_per_home', 'home_unserved_kwh', 'home_episode_counts',
                'home_promise_failure_counts', 'promise_unserved_kwh_per_home', 'hours']
    for spec in manifest['files']:
        path = root / spec['path']
        if not path.exists():
            errors.append(f'missing {spec["path"]}'); continue
        d = pd.read_csv(path)
        if 'protocol' not in d or (d.protocol != 'causal-v3').any():
            errors.append(f'{path.name}: missing or wrong protocol'); continue
        if path.name.startswith('baselines_') and ('controller_revision' not in d or (d.controller_revision != 'reserve-minimum-v3').any()):
            errors.append(f'{path.name}: outdated reserve-minimum controller revision'); continue
        if path.name.startswith('baselines_') and ('source_sha256' not in d or (d.source_sha256 != REPLAY_CODE_SHA).any()):
            errors.append(f'{path.name}: mismatched core source hash'); continue
        if path.name.startswith('policy_'):
            if 'source_sha256' not in d or (d.source_sha256 != POLICY_CODE_SHA).any():
                errors.append(f'{path.name}: mismatched policy evaluation code'); continue
            run = spec['controllers'][0].removeprefix('B6_').removesuffix('_guard')
            checkpoint = root / 'experiments/revised' / run / 'policy.pt'
            if checkpoint.exists() and ('policy_sha256' not in d or (d.policy_sha256 != hashlib.sha256(checkpoint.read_bytes()).hexdigest()).any()):
                errors.append(f'{path.name}: mismatched checkpoint'); continue
        missing = [c for c in required if c not in d]
        if missing:
            errors.append(f'{path.name}: missing metrics {missing}'); continue
        expected = {(w['name'], c) for w in spec['windows'] for c in spec['controllers']}
        got = list(zip(d.window, d.controller))
        if len(got) != len(set(got)) or set(got) != expected:
            errors.append(f'{path.name}: duplicate or incomplete controller/window coverage')
        if (d.n != 100).any() or (d.seed != 0).any() or (d.year != spec['year']).any():
            errors.append(f'{path.name}: wrong replay configuration')
        duration = {w['name']: w['steps'] * .25 for w in spec['windows']}
        if not np.allclose(d.hours, d.window.map(duration)):
            errors.append(f'{path.name}: inconsistent duration')
        if (d.physical_violations != 0).any() or (d.reachable_floor_violations != 0).any():
            errors.append(f'{path.name}: constraint violations')
        if not np.isfinite(d[['margin_per_home_week', 'cash_margin_per_home_week', 'unserved_kwh_per_home',
                              'promise_unserved_kwh_per_home', 'hours']].to_numpy()).all():
            errors.append(f'{path.name}: nonfinite economic or energy metrics')
        if not (d.promise_rate.isna().to_numpy() == (d.outage_episodes == 0).to_numpy()).all():
            errors.append(f'{path.name}: undefined reliability handled incorrectly')
        for key in required[:-2]:
            if d[key].isna().any(): errors.append(f'{path.name}: incomplete {key}')
        for _, row in d.iterrows():
            e = np.array(json.loads(row.home_episode_counts))
            f = np.array(json.loads(row.home_promise_failure_counts))
            if len(e) != row.n or e.sum() != row.outage_episodes or f.sum() != row.promise_failures or (f > e).any():
                errors.append(f'{path.name}: inconsistent per-home episode counts'); break
            pairing = (spec['metro'], spec['year'], row.window)
            if pairing in paired_episodes and not np.array_equal(e, paired_episodes[pairing]):
                errors.append(f'{path.name}: outage draws differ from paired controllers in {row.window}')
                break
            paired_episodes[pairing] = e
    for run in manifest['runs']:
        folder = root / 'experiments/revised' / run
        for file in ['policy.pt', 'config.json', 'train_log.json']:
            if not (folder / file).exists(): errors.append(f'missing {run}/{file}')
        if (folder / 'train_log.json').exists():
            log = json.loads((folder / 'train_log.json').read_text())
            if log[-1]['steps'] != 2998272: errors.append(f'{run}: wrong training budget')
        if (folder / 'config.json').exists():
            config = json.loads((folder / 'config.json').read_text())
            if config.get('protocol') != 'causal-v3': errors.append(f'{run}: wrong training protocol')
            saved = config.get('bundle_sha256', {})
            expected = {Path(p).name: bundle_hashes.get(Path(p).name) for p in config.get('bundles', [])}
            if not saved or saved != expected:
                errors.append(f'{run}: training inputs differ from the frozen bundle manifest')
    return errors


def verify_ancillary(root=ROOT):
    errors = []
    selected_digest = None
    selection = root / 'experiments/revised/selected_policy.json'
    if selection.exists():
        info = json.loads(selection.read_text())
        checkpoint = root / info['path']
        if checkpoint.exists():
            selected_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if (info.get('policy_sha256') != selected_digest or info.get('protocol') != 'causal-v3' or
                    info.get('source_sha256') != POLICY_CODE_SHA):
                errors.append('selected_policy.json: selected checkpoint provenance differs')
        else:
            errors.append('selected_policy.json: selected checkpoint is missing')
    expected = {'structure.csv': 150, 'degradation_table.csv': 24, 'ac_check.csv': 21,
                'stress_guardrail_ppo.csv': 344, 'latency.csv': 36}
    for filename, count in expected.items():
        path = root / 'experiments/revised' / filename
        if not path.exists():
            errors.append(f'missing {filename}'); continue
        d = pd.read_csv(path)
        if len(d) != count: errors.append(f'{filename}: expected {count} rows, got {len(d)}')
        if filename == 'structure.csv':
            from eval_structure import SCENARIOS, WINDOWS, REFERENCES
            planned = {(s, m, y, w, c) for s in SCENARIOS for m, y, w in WINDOWS
                       for c in [*REFERENCES, 'ppo', 'ppo_guard']}
            errors += exact_coverage_errors(d, ['scenario', 'metro', 'year', 'window', 'controller'], planned, filename)
            if (not d.n.eq(d.scenario.map({s: v[0] for s, v in SCENARIOS.items()})).all() or
                    not np.allclose(d.dropout, d.scenario.map({s: v[2] for s, v in SCENARIOS.items()}))):
                errors.append('structure.csv: wrong fleet size or dropout')
        if filename == 'degradation_table.csv':
            planned = {(c, w) for c in ['fixed20', 'hold90', 'storm', 'risk5', 'ppo', 'ppo_guard']
                       for w in ['beryl_2024', 'heat_aug_2024', 'derecho_2024', 'w10']}
            errors += exact_coverage_errors(d, ['controller', 'window'], planned, filename)
            if not d.n.eq(20).all(): errors.append('degradation_table.csv: wrong household count')
        if filename == 'ac_check.csv':
            planned = {(day, c, pen) for day in ['normal', 'beryl', 'heat']
                       for c in ['B0_fixed', 'B3a_risk_lam5', 'PPO_guard'] for pen in [.3, 1.]}
            planned |= {(day, 'load_PV_reference', 0.) for day in ['normal', 'beryl', 'heat']}
            errors += exact_coverage_errors(d, ['day', 'controller', 'penetration'], planned, filename)
        if filename == 'latency.csv':
            planned = {(n, c, r) for n in [100, 500, 1000] for c in ['fixed20', 'risk5', 'ppo', 'ppo_guard']
                       for r in [1, 2, 3]}
            errors += exact_coverage_errors(d, ['n', 'controller', 'repeat'], planned, filename)
            if (not d.intervals.eq(96).all() or not d.torch_threads.eq(1).all() or
                    not np.allclose(d.ms_per_interval, 1000*d.seconds/d.intervals)):
                errors.append('latency.csv: inconsistent timing units, interval count, or threads')
        if filename == 'stress_guardrail_ppo.csv':
            from stress_guardrail import scenarios
            cases = scenarios()
            planned = {(k, p, g) for k in range(len(cases)) for p in ['always_discharge', 'ppo'] for g in [False, True]}
            errors += exact_coverage_errors(d, ['case', 'policy', 'guarded'], planned, filename)
            for k, group in d.groupby('case'):
                if k not in range(len(cases)): continue
                for key, value in cases[int(k)].items():
                    if key not in group or (not group[key].eq(value).all() if isinstance(value, str)
                                            else not np.allclose(group[key], value)):
                        errors.append(f'{filename}: case {k} has wrong {key}')
        if (not {'protocol', 'source_sha256'}.issubset(d.columns) or
                (d.protocol != 'causal-v3').any() or (d.source_sha256 != REPLAY_CODE_SHA).any()):
            errors.append(f'{filename}: mismatched replay source or input protocol')
        learned = (d.policy.eq('ppo') if filename == 'stress_guardrail_ppo.csv' else
                   d.controller.eq('PPO_guard') if filename == 'ac_check.csv' else
                   d.controller.isin(['ppo', 'ppo_guard']))
        errors += policy_provenance_errors(d, learned, selected_digest, filename)
        if filename in ['structure.csv', 'stress_guardrail_ppo.csv']:
            if (d.physical_violations != 0).any() or (d.reachable_floor_violations != 0).any():
                errors.append(f'{filename}: battery constraint violations')
        if filename == 'structure.csv' and d.duplicated(['scenario', 'window', 'controller']).any():
            errors.append('structure.csv: repeated comparisons')
        if filename == 'degradation_table.csv':
            if d.duplicated(['controller', 'window']).any() or not np.isfinite(d[['blast_pct_yr', 'et_pct_yr', 'cb_pct_yr']]).all().all():
                errors.append('degradation_table.csv: invalid aging outputs')
        if filename == 'ac_check.csv':
            if not np.isfinite(d[['vmin', 'vmax', 'max_line_loading']]).all().all():
                errors.append('ac_check.csv: nonfinite solver output')
        if filename == 'stress_guardrail_ppo.csv':
            sufficient = d[(d.alert == 'correct') & (d.lead >= 3) & (d.dropout == 0) &
                           (d.soh == 1) & (d.load_multiplier == 1) & d.guarded]
            if (not len(sufficient) or sufficient.promise_rate.isna().any() or
                    (sufficient.outage_episodes <= 0).any() or (sufficient.promise_rate < 1 - 1e-8).any()):
                errors.append('synthetic guardrail violates its sufficient-condition cases')
        if filename == 'latency.csv' and (d.ms_per_interval <= 0).any():
            errors.append('latency.csv: invalid wall time')
    for filename in ['guardrail_constants.csv', 'alert_duty.csv', 'orthogonality_daily.csv', 'orthogonality.json', 'selected_policy.json']:
        if not (root / 'experiments/revised' / filename).exists(): errors.append(f'missing {filename}')
    return errors


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='also require all selected-policy and system audits')
    args = parser.parse_args()
    failures = verify()
    if args.full: failures += verify_ancillary()
    if failures:
        print('\n'.join(failures))
        raise SystemExit(f'{len(failures)} verification errors')
    print('Every planned run and result row is present and internally consistent.')
