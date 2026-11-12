"""Check frozen individual-home bundle hashes, schema, coverage and causal fields."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from homefleet.paths import ROOT, DATA_PROC, DATA_RAW
from homefleet.data.weather import load_hourly, CLOCK_PROTOCOL


def verify_saved_risk_model():
    """Reconstruct released forecasts using the serialized model and raw inputs."""
    import joblib
    from homefleet.data.eaglei import county_fraction_panel
    from homefleet.data.alerts import zone_county_map
    from homefleet.sim.hazard import hourly_frac, alert_group_panel, build_features
    folder = DATA_PROC / 'causal'
    report = json.loads((folder / 'hazard_report.json').read_text())
    saved = joblib.load(folder / 'hazard_model.joblib')
    assert saved['protocol'] == 'causal-v3' and saved['weather_clock_protocol'] == CLOCK_PROTOCOL
    assert saved['train_years'] == report['train_years'] == list(range(2014, 2020))
    assert saved['calibration_year'] == report['calib_year'] == 2020
    assert not saved['forecast_proxy'] and not report['forecast_proxy']
    fraction = hourly_frac(county_fraction_panel(range(2014, 2026), fips=report['fips']))
    warnings = pd.concat([pd.read_parquet(DATA_RAW / 'alerts' / f'vtec_tx_{y}.parquet') for y in range(2014, 2027)])
    mapping = zone_county_map()
    checked = []
    for fips in report['fips']:
        weather = load_hourly(DATA_RAW / 'weather' / f'openmeteo_{fips}.parquet')
        groups = alert_group_panel(warnings, weather.index, [fips], mapping)
        groups = {g: groups[(g, fips)] for g in ['tropical', 'convective', 'winter', 'wind', 'flood']}
        features, _ = build_features(weather, fraction[fips].reindex(weather.index).fillna(0), groups, forecast_proxy=False)
        predicted = saved['model'].predict(features)
        for horizon, calibrator in saved['calibrators'].items():
            predicted[horizon] = calibrator.predict(predicted[horizon])
        frozen = pd.read_parquet(folder / f'hazard_{fips}.parquet')
        pd.testing.assert_index_equal(predicted.index, frozen.index)
        np.testing.assert_allclose(predicted.to_numpy(), frozen.to_numpy(), rtol=0, atol=1e-12)
        checked.append(dict(fips=fips, hours=len(frozen), max_abs_error=float(np.abs(predicted.to_numpy()-frozen.to_numpy()).max())))
        print('Verified serialized risk-model predictions', fips, flush=True)
    result = dict(protocol='causal-v3', weather_clock_protocol=CLOCK_PROTOCOL, counties=checked,
                  model_sha256=hashlib.sha256((folder / 'hazard_model.joblib').read_bytes()).hexdigest())
    (ROOT / 'artifacts/hazard_model_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', action='store_true', help='also compare weather/risk arrays and building order against cached source files')
    a = ap.parse_args()
    folder = DATA_PROC / 'revised'
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['protocol'] == 'causal-v3'
    assert manifest['weather_clock_protocol'] == CLOCK_PROTOCOL
    checked, home_orders = [], {}
    for spec in manifest['bundles']:
        path = folder / spec['file']
        with path.open('rb') as handle: digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != spec['sha256']: raise ValueError(f'{path.name}: file hash mismatch')
        with np.load(path, allow_pickle=True) as d:
            index = pd.to_datetime(d['index'], utc=True).tz_convert('America/Chicago')
            T, N = d['load_kw'].shape
            C = len(d['fips_list'])
            assert N == 300 and len(index) == T and d['pv_kw'].shape == (T, N)
            assert d['frac'].shape == d['alert'].shape == (T, C)
            assert d['weather'].shape == (T, C, 7) and d['hazard'].shape == (T, C, 5)
            assert np.array_equal(d['hazard_h'], [1, 3, 6, 12, 24])
            assert np.all(np.diff(index.asi8) == pd.Timedelta('15min').value)
            assert int(d['evaluation_start']) >= 28 * 96 and str(d['protocol']) == 'causal-v3'
            assert str(d['weather_clock_protocol']) == CLOCK_PROTOCOL
            assert d['home_id'].astype(str).tolist() == spec['home_ids']
            assert len(d['home_fips']) == len(d['home_id']) == N
            assert set(d['home_fips'].astype(str)) <= set(d['fips_list'].astype(str))
            for field in ['price_rt', 'price_da', 'load_kw', 'pv_kw', 'weather', 'hazard', 'frac']:
                if not np.isfinite(d[field]).all(): raise ValueError(f'{path.name}: nonfinite {field}')
            assert (d['load_kw'] >= 0).all() and (d['pv_kw'] >= 0).all()
            assert ((d['hazard'] >= 0) & (d['hazard'] <= 1)).all()
            assert ((d['frac'] >= 0) & (d['frac'] <= 1)).all()
            if not spec['outage_truth']:
                assert not d['frac'][int(d['evaluation_start']):].any() and int(d['year']) == 2026
            metro = str(d['metro'])
            order = tuple(d['home_id'].astype(str))
            if metro in home_orders: assert home_orders[metro] == order
            home_orders[metro] = order
            if a.raw:
                meta = pd.read_parquet(DATA_RAW / 'resstock' / f'meta_{metro}_n300.parquet')
                loads = pd.read_parquet(DATA_RAW / 'resstock' / f'loads_{metro}_n300.parquet')
                assert tuple(meta.index.astype(str)) == tuple(loads.columns.astype(str)) == order
                for c, fips in enumerate(d['fips_list'].astype(str)):
                    w = load_hourly(DATA_RAW / 'weather' / f'openmeteo_{fips}.parquet').shift(1)
                    expected_weather = w.reindex(index, method='ffill').fillna(0).to_numpy(dtype=np.float32)
                    hazard = pd.read_parquet(DATA_PROC / 'causal' / f'hazard_{fips}.parquet')
                    expected_hazard = hazard.reindex(index, method='ffill').fillna(0).to_numpy(dtype=np.float32)
                    assert np.array_equal(d['weather'][:, c], expected_weather)
                    assert np.array_equal(d['hazard'][:, c], expected_hazard)
            checked.append(dict(file=path.name, sha256=digest, homes=N, steps=T, counties=C,
                                start=str(index[0]), end=str(index[-1]), outage_truth=spec['outage_truth']))
        print('Verified', path.name, flush=True)
    report = dict(protocol='causal-v3', bundles=len(checked), raw_compared=a.raw,
                  independent_profile_pool_per_metro=300, files=checked)
    if a.raw:
        report['serialized_risk_model_verified'] = bool(verify_saved_risk_model())
    destination = ROOT / 'artifacts/data_validation.json'
    destination.parent.mkdir(exist_ok=True, parents=True)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(destination)


if __name__ == '__main__':
    main()
