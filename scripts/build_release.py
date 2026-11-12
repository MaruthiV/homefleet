import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from homefleet.paths import ROOT, DATA_PROC, DATA_RAW, EXPERIMENTS
from reproduce import training_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default='v0.2.0')
    ap.add_argument('--groups', nargs='+', choices=['bundles', 'raw', 'checkpoints', 'results'], default=['bundles', 'raw', 'checkpoints', 'results'])
    a = ap.parse_args()
    output = ROOT / 'release'
    output.mkdir(exist_ok=True)
    (ROOT / 'artifacts').mkdir(exist_ok=True)
    manifest_path = ROOT / 'artifacts/release_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(version=a.version, protocol='causal-v3', assets=[])
    if manifest['version'] != a.version:
        manifest = dict(version=a.version, protocol='causal-v3', assets=[])
    groups = {
        'bundles': list((DATA_PROC / 'revised').glob('*.npz')) + [DATA_PROC / 'revised/manifest.json'],
        'raw': [p for p in DATA_RAW.rglob('*') if p.is_file() and (p.suffix in ['.csv', '.parquet', '.dbx', '.dss'] or p.name in ['Buscoords.dss', 'SMART-DS_version.txt'])]
                + list((DATA_PROC / 'causal').glob('*')),
        'results': [],
        'checkpoints': []}
    if 'results' in a.groups:
        from verify_results import verify, verify_ancillary
        failures = verify() + verify_ancillary()
        if failures:
            raise ValueError(f'Result release requires complete verified outputs ({len(failures)} errors):\n' + '\n'.join(failures[:8]))
        planned = json.loads((ROOT / 'artifacts/experiment_manifest.json').read_text())['files']
        groups['results'] = [ROOT / spec['path'] for spec in planned]
        groups['results'] += [(ROOT / spec['path']).with_suffix('.complete.json') for spec in planned
                              if Path(spec['path']).name.startswith('policy_')]
        # Keep the scientific outputs; runtime status and literature caches are local.
        reports = ['summary.csv', 'seasonal.csv', 'events.csv', 'validation_candidates.csv', 'event_traces.csv',
                   'structure.csv', 'degradation_table.csv', 'ac_check.csv', 'stress_guardrail.csv',
                   'stress_guardrail_ppo.csv', 'latency.csv', 'guardrail_constants.csv', 'alert_duty.csv',
                   'orthogonality_daily.csv', 'orthogonality.json', 'guardrail_hardware.json',
                   'selected_policy.json', 'report.json']
        groups['results'] += [EXPERIMENTS / 'revised' / name for name in reports]
        for path in groups['results']:
            if not path.is_file(): raise FileNotFoundError(path)
    if 'checkpoints' in a.groups:
        for name, *_ in training_runs():
            run = EXPERIMENTS / 'revised' / name
            for filename in ['policy.pt', 'config.json', 'train_log.json']:
                p = run / filename
                if not p.exists(): raise FileNotFoundError(p)
                groups['checkpoints'].append(p)
            log = json.loads((run / 'train_log.json').read_text())
            if not log or log[-1]['steps'] != 2998272:
                raise ValueError(f'{name}: checkpoint has not completed the frozen training budget')
    for group in a.groups:
        name = f'homefleet-{group}-{a.version}.tar.gz'
        target = output / name
        files = sorted(set(groups[group]))
        if not files: raise ValueError(f'empty group {group}')
        with tarfile.open(target, 'w:gz', compresslevel=1) as archive:
            for path in files:
                archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
        with target.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        asset = dict(group=group, name=name, sha256=digest, bytes=target.stat().st_size,
                     url=f'https://github.com/MaruthiV/homefleet/releases/download/{a.version}/{name}', files=len(files))
        manifest['assets'] = [x for x in manifest['assets'] if x['group'] != group] + [asset]
        print(asset, flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
