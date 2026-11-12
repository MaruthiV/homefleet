import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from homefleet.paths import DATA_PROC, DATA_RAW
from homefleet.data.weather import load_hourly, CLOCK_PROTOCOL


def prepare(output, physical_dir):
    output.mkdir(parents=True, exist_ok=True)
    hazard_dir = DATA_PROC / 'causal'
    report = json.loads((hazard_dir / 'hazard_report.json').read_text())
    if report['forecast_proxy'] or report['calib_year'] >= 2021:
        raise ValueError('revised bundles require past-only features and pre-2021 calibration')
    if report.get('weather_clock_protocol') != CLOCK_PROTOCOL:
        raise ValueError('risk model must be rebuilt from UTC-aligned weather')
    manifest = {'protocol': 'causal-v3', 'weather_clock_protocol': CLOCK_PROTOCOL, 'hazard': report, 'bundles': []}
    for source in sorted(physical_dir.glob('*_20*.npz')):
        if not any(source.name.startswith(m + '_') for m in ['houston', 'dfw', 'san_antonio']):
            continue
        d = dict(np.load(source, allow_pickle=True))
        if str(d.get('weather_clock_protocol', '')) != CLOCK_PROTOCOL:
            raise ValueError(f'{source}: physical inputs must be rebuilt from UTC-aligned weather')
        year = int(d['year'])
        metro = str(d['metro'])
        T = len(d['index'])
        previous = physical_dir / f'{metro}_{year - 1}.npz'
        prefix = 0
        if previous.exists():
            prev = dict(np.load(previous, allow_pickle=True))
            prefix = 28 * 96
            for key, value in list(d.items()):
                if value.ndim and value.shape[0] == T:
                    d[key] = np.concatenate([prev[key][-prefix:], value], axis=0)
        idx = pd.to_datetime(d['index'], utc=True).tz_convert('America/Chicago')
        weather = []
        for f in d['fips_list']:
            w = load_hourly(DATA_RAW / 'weather' / f'openmeteo_{f}.parquet').shift(1)
            weather.append(w.reindex(idx, method='ffill').fillna(0.).to_numpy(dtype=np.float32))
        d['weather'] = np.stack(weather, axis=1)
        hz = [pd.read_parquet(hazard_dir / f'hazard_{f}.parquet').reindex(idx, method='ffill').fillna(0.) for f in d['fips_list']]
        d['hazard'] = np.stack([h.to_numpy(dtype=np.float32) for h in hz], axis=1)
        d['hazard_h'] = np.asarray([int(c) for c in hz[0].columns])
        meta = pd.read_parquet(DATA_RAW / 'resstock' / f'meta_{metro}_n300.parquet')
        d['home_id'] = meta.index.astype(str).to_numpy(dtype=str)
        d['evaluation_start'] = np.array(max(prefix, 28 * 96))
        d['protocol'] = np.array('causal-v3')
        d['weather_clock_protocol'] = np.array(CLOCK_PROTOCOL)
        target = output / source.name
        np.savez_compressed(target, **d)
        manifest['bundles'].append({'file': target.name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                                    'homes': len(d['home_fips']), 'home_ids': d['home_id'].tolist(),
                                    'first_timestamp': str(idx[0]), 'last_timestamp': str(idx[-1]),
                                    'evaluation_start': int(d['evaluation_start']), 'outage_truth': year <= 2025})
        print(target, flush=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=DATA_PROC / 'revised')
    ap.add_argument('--physical-dir', type=Path, default=DATA_PROC / 'physical_utc')
    args = ap.parse_args()
    prepare(args.output, args.physical_dir)
