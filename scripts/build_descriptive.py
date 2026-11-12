import json
import numpy as np
import pandas as pd
from homefleet.data.ercot import load_rtm_spp
from homefleet.data.eaglei import load_tx, county_fraction_panel
from homefleet.paths import EXPERIMENTS


def main():
    output = EXPERIMENTS / 'revised'
    output.mkdir(exist_ok=True, parents=True)
    prices = load_rtm_spp(zones=('LZ_HOUSTON', 'LZ_NORTH', 'LZ_SOUTH', 'LZ_AEN', 'LZ_CPS'))
    prices = prices.loc['2021-01-01':'2025-12-31']
    statewide = []
    for year in range(2021, 2026):
        outage = load_tx(year)
        statewide.append(outage.groupby('t').customers_out.sum())
        print('outage year', year, flush=True)
    totals = pd.concat(statewide).groupby(level=0).sum().sort_index()
    outage_daily = totals.resample('1D').max().loc['2021-01-01':'2025-12-31']
    daily = pd.concat([prices.max(axis=1).resample('1D').max().rename('pmax'), outage_daily.rename('out_peak')], axis=1).dropna()
    daily.to_csv(output / 'orthogonality_daily.csv')
    spike = daily.pmax >= 1000
    mass = daily.out_peak >= 250000
    report = dict(days=len(daily), spike_days=int(spike.sum()), outage_days=int(mass.sum()), joint_days=int((spike & mass).sum()),
                  joint_dates=[str(x.date()) for x in daily.index[spike & mass]],
                  spearman=float(daily.pmax.corr(daily.out_peak, method='spearman')))
    (output / 'orthogonality.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report)


if __name__ == '__main__':
    main()
