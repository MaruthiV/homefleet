import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from homefleet.paths import ROOT, EXPERIMENTS
from homefleet.eval.metrics import cvar
from verify_results import verify
from run_baselines import is_regular_window, POLICY_CODE_SHA

YEAR_HOURS = 8760.
SEASONS = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
           6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
SEASON_FIELDS = ['windows', 'observed_hours', 'margin_annual', 'cash_margin_annual',
                 'delta_margin_annual', 'delta_lo', 'delta_hi', 'promise', 'promise_lo', 'promise_hi',
                 'episodes', 'promise_failures', 'unserved_annual', 'promise_unserved_annual',
                 'mean_soc', 'throughput_annual']


def save_selection(controller, validation_score, root=ROOT):
    name = controller.removeprefix('B6_')
    relative = f'experiments/revised/{name}/policy.pt'
    selected = dict(run=name, controller=controller, path=relative, protocol='causal-v3',
                    source_sha256=POLICY_CODE_SHA,
                    policy_sha256=hashlib.sha256((root / relative).read_bytes()).hexdigest(),
                    criterion='Houston 2024 annualized margin minus $20/kWh of first-five-hour unserved energy',
                    validation_score=float(validation_score))
    path = root / 'experiments/revised/selected_policy.json'
    if path.exists() and json.loads(path.read_text())['run'] != name:
        raise ValueError('a prior selection differs; investigate inputs rather than silently reselecting')
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix='selected-policy-', suffix='.tmp', delete=False) as handle:
        handle.write(json.dumps(selected, indent=2) + '\n')
    Path(handle.name).replace(path)
    return selected


def summarize_group(g, baseline, draws=2000):
    if not g.window.map(is_regular_window).all():
        raise ValueError('annual summaries require regular wNN windows, without named event duplication')
    g = g.sort_values('window').reset_index(drop=True)
    reference = baseline.set_index('window').margin_per_home_week.reindex(g.window).to_numpy()
    if np.isnan(reference).any(): raise ValueError('unpaired evaluation windows')
    h = g.hours.to_numpy()
    margin = g.margin_per_home_week.to_numpy()
    delta = margin - reference
    samples = np.random.default_rng(0).integers(0, len(g), (draws, len(g)))
    denom = h[samples].sum(axis=1)
    boot_margin = (delta[samples] * h[samples]).sum(axis=1) / denom * YEAR_HOURS / 168.
    episodes = g.outage_episodes.to_numpy()
    failures = g.promise_failures.to_numpy()
    boot_e = episodes[samples].sum(axis=1)
    valid = boot_e > 0
    boot_p = 1 - failures[samples].sum(axis=1)[valid] / boot_e[valid]
    ref_fail = baseline.set_index('window').promise_failures.reindex(g.window).to_numpy()
    boot_delta_p = (ref_fail[samples].sum(axis=1)[valid] - failures[samples].sum(axis=1)[valid]) / boot_e[valid]
    annual = float(np.dot(margin, h) / h.sum() * YEAR_HOURS / 168.)
    promise_cost = float(g.promise_unserved_kwh_per_home.sum() / h.sum() * YEAR_HOURS)
    daily = np.array([v for values in g.daily_cash_margin_per_home for v in json.loads(values)])
    home_e = np.sum([json.loads(v) for v in g.home_episode_counts], axis=0)
    home_f = np.sum([json.loads(v) for v in g.home_promise_failure_counts], axis=0)
    home_u = np.sum([json.loads(v) for v in g.home_unserved_kwh], axis=0) / h.sum() * YEAR_HOURS
    eligible = home_e > 0
    p05 = np.quantile(1 - home_f[eligible] / home_e[eligible], .05) if eligible.any() else np.nan
    return dict(windows=len(g), observed_hours=float(h.sum()), margin_annual=annual,
                cash_margin_annual=float(np.dot(g.cash_margin_per_home_week, h) / h.sum() * YEAR_HOURS / 168.),
                delta_margin_annual=float(np.dot(delta, h) / h.sum() * YEAR_HOURS / 168.),
                delta_lo=float(np.quantile(boot_margin, .025)), delta_hi=float(np.quantile(boot_margin, .975)),
                promise=float(1 - failures.sum() / episodes.sum()) if episodes.sum() else np.nan,
                promise_lo=float(np.quantile(boot_p, .025)) if len(boot_p) else np.nan,
                promise_hi=float(np.quantile(boot_p, .975)) if len(boot_p) else np.nan,
                delta_promise=float((ref_fail.sum() - failures.sum()) / episodes.sum()) if episodes.sum() else np.nan,
                delta_promise_lo=float(np.quantile(boot_delta_p, .025)) if len(boot_delta_p) else np.nan,
                delta_promise_hi=float(np.quantile(boot_delta_p, .975)) if len(boot_delta_p) else np.nan,
                episodes=int(episodes.sum()), promise_failures=int(failures.sum()),
                unserved_annual=float(g.unserved_kwh_per_home.sum() / h.sum() * YEAR_HOURS),
                promise_unserved_annual=promise_cost, validation_score=annual - 20 * promise_cost,
                cvar5_daily=float(cvar(daily, .05)) if len(daily) else np.nan,
                worst_week=float(pd.Series(daily).rolling(7).sum().min()) if len(daily) >= 7 else np.nan,
                home_promise_p05=float(p05), home_unserved_p95=float(np.quantile(home_u, .95)), eligible_homes=int(eligible.sum()),
                mean_soc=float(np.dot(g.mean_soc_frac, h) / h.sum()),
                throughput_annual=float(g.throughput_kwh_per_home.sum() / h.sum() * YEAR_HOURS),
                seconds=float(g.sec.sum()), physical_violations=int(g.physical_violations.sum()),
                reachable_floor_violations=int(g.reachable_floor_violations.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--allow-incomplete', action='store_true', help='development summaries only, never a release gate')
    a = ap.parse_args()
    errors = verify()
    if errors and not a.allow_incomplete:
        raise SystemExit(f'{len(errors)} missing/inconsistent artifacts; run scripts/verify_results.py')
    planned = json.loads((ROOT / 'artifacts/experiment_manifest.json').read_text())['files']
    rows, events, sources, seasonal = [], [], [], []
    output = EXPERIMENTS / 'revised'
    for metro, year in [('houston', 2024), ('houston', 2025), ('houston', 2026), ('dfw', 2024), ('san_antonio', 2024)]:
        baseline_path = output / f'baselines_{metro}_{year}.csv'
        paths = [ROOT / spec['path'] for spec in planned if spec['metro'] == metro and spec['year'] == year
                 and (ROOT / spec['path']).is_file()]
        if not baseline_path.exists(): continue
        data = []
        for path in paths:
            g = pd.read_csv(path)
            if 'daily_cash_margin_per_home' not in g or 'promise_unserved_kwh_per_home' not in g: continue
            data.append(g)
            with path.open('rb') as handle: sources.append(dict(file=str(path.relative_to(ROOT)), sha256=hashlib.file_digest(handle, 'sha256').hexdigest()))
        if not data: continue
        data = pd.concat(data, ignore_index=True)
        regular = data[data.window.map(is_regular_window)].copy()
        regular['season'] = pd.to_datetime(regular.start, utc=True).dt.tz_convert('America/Chicago').dt.month.map(SEASONS)
        reference = regular[regular.controller == 'B0_fixed']
        for controller, g in regular.groupby('controller'):
            rows.append(dict(metro=metro, year=year, controller=controller, **summarize_group(g, reference)))
            for season, group in g.groupby('season'):
                scores = summarize_group(group, reference[reference.season == season])
                seasonal.append(dict(metro=metro, year=year, controller=controller, season=season,
                                     **{key: scores[key] for key in SEASON_FIELDS}))
        events.append(data[~data.window.map(is_regular_window)])
    summary = pd.DataFrame(rows)
    summary.to_csv(output / 'summary.csv', index=False)
    pd.DataFrame(seasonal).to_csv(output / 'seasonal.csv', index=False)
    recorded = {source['file'] for source in sources}
    for spec in planned:
        path = ROOT / spec['path']
        if not path.is_file(): continue
        d = pd.read_csv(path)
        event = d[~d.window.map(is_regular_window)]
        if len(event): events.append(event)
        relative = str(path.relative_to(ROOT))
        if relative not in recorded:
            with path.open('rb') as handle: sources.append(dict(file=relative, sha256=hashlib.file_digest(handle, 'sha256').hexdigest()))
            recorded.add(relative)
    if events:
        pd.concat(events, ignore_index=True).drop_duplicates(['metro', 'year', 'controller', 'window']).to_csv(output / 'events.csv', index=False)
    candidates = summary[(summary.metro == 'houston') & (summary.year == 2024) & summary.controller.str.match(r'B6_ppo_pc_lam(5|20)_s[1-5]$')]
    selected = None
    if len(candidates) == 10:
        best = candidates.sort_values(['validation_score', 'controller'], ascending=[False, True]).iloc[0]
        selected = save_selection(best.controller, best.validation_score)
    report = dict(protocol='causal-v3', complete=not errors, verification_errors=len(errors), source_files=sources, selected_policy=selected,
                  seasonal_definition='Whole regular windows grouped by the season of their first America/Chicago timestamp: DJF, MAM, JJA, SON. Boundary-crossing windows remain whole; annualized values are standardized cohort rates, not exact calendar-season totals.')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(summary[['metro', 'year', 'controller', 'delta_margin_annual', 'promise']].to_string(index=False) if len(summary) else 'No complete summary inputs yet.')


if __name__ == '__main__':
    main()
