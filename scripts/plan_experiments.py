import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import yaml
from homefleet.paths import ROOT, CONFIGS, DATA_PROC
from run_baselines import MODES, windows_for_year, is_regular_window
from reproduce import training_runs


def main():
    cfg = yaml.safe_load((CONFIGS / 'revision.yaml').read_text())
    splits = yaml.safe_load((CONFIGS / 'splits.yaml').read_text())
    controllers = []
    for name, spec in MODES.items():
        if spec['mode'] == 'risk_priced': controllers += [f'{name}_lam{x}' for x in [1, 5, 20]]
        elif spec['mode'] == 'chance': controllers += [f'{name}_eps{x}' for x in [.02, .1]]
        else: controllers.append(name)
    files = []
    for metro, year in cfg['evaluation'] + cfg['events']:
        path = DATA_PROC / 'revised' / f'{metro}_{year}.npz'
        with np.load(path, allow_pickle=True) as d:
            idx = pd.to_datetime(d['index'], utc=True).tz_convert('America/Chicago')
            data = SimpleNamespace(index=idx, T=len(idx), evaluation_start=int(d['evaluation_start']))
        wins = windows_for_year(data, year, events=splits['events'])
        if [metro, year] in cfg['events']: wins = [w for w in wins if not is_regular_window(w[0])]
        windows = [dict(name=w, start=str(idx[s]), steps=n) for w, s, n in wins]
        files.append(dict(path=f'experiments/revised/baselines_{metro}_{year}.csv', metro=metro, year=year,
                          windows=windows, controllers=controllers))
        for name, *_ in training_runs():
            for guard in [False, True]:
                policy = f'{name}{"_guard" if guard else ""}'
                files.append(dict(path=f'experiments/revised/policy_{policy}_{metro}_{year}.csv', metro=metro,
                                  year=year, windows=windows, controllers=[f'B6_{policy}']))
    planned = dict(protocol='causal-v3', config_sha256=hashlib.sha256((CONFIGS / 'revision.yaml').read_bytes()).hexdigest(),
                   runs=[r[0] for r in training_runs()], files=files)
    (ROOT / 'artifacts').mkdir(exist_ok=True)
    (ROOT / 'artifacts/experiment_manifest.json').write_text(json.dumps(planned, indent=2) + '\n')
    print(len(files), 'result files planned')


if __name__ == '__main__':
    main()
