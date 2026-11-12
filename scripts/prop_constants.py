import json
import numpy as np
import pandas as pd
from homefleet.paths import DATA_PROC, EXPERIMENTS
from homefleet.env import FleetData
from homefleet.sim.outage import episodes
from homefleet.sim.battery import make_fleet


def main():
    output = EXPERIMENTS / 'revised'
    output.mkdir(exist_ok=True, parents=True)
    params = make_fleet(['gen2_single', 'gen2_double', 'core'])
    worst_charge_h = .25 * np.ceil((.9 - params.soc_min) * params.energy_kwh / (params.eta_c * params.p_charge_kw * .25))
    rows, duty = [], []
    for metro in ['houston', 'dfw', 'san_antonio']:
        for year in range(2019, 2026):
            data = FleetData(DATA_PROC / 'revised' / f'{metro}_{year}.npz')
            for j, fips in enumerate(data.fips_list):
                alert = data.alert.iloc[:, j].to_numpy(dtype=bool)
                fraction = data.frac.iloc[:, j].to_numpy()
                calendar = data.index.year == year
                duty.append(dict(metro=metro, year=year, fips=fips, duty=float(alert[calendar].mean()),
                                 alert_spike_hours=float(((data.price_rt >= 1000) & alert & calendar).sum() * .25)))
                for threshold in [.01, .05]:
                    for start, end in episodes(fraction >= threshold, merge_gap=16):
                        if data.index[start].year != year:
                            continue
                        first = start
                        if alert[start]:
                            while first > 0 and alert[first - 1]: first -= 1
                        lead = (start - first) * .25
                        rows.append(dict(metro=metro, year=year, fips=fips, threshold=threshold,
                                         onset=str(data.index[start]), duration_h=(end - start) * .25,
                                         peak_fraction=float(fraction[start:end].max()), active_at_onset=bool(alert[start]),
                                         recent_alert_24h=bool(alert[max(0, start - 96):start + 1].any()),
                                         continuous_lead_h=lead, lead_covers_worst_charge=bool(alert[start] and lead >= worst_charge_h.max())))
    pd.DataFrame(rows).to_csv(output / 'guardrail_constants.csv', index=False)
    pd.DataFrame(duty).to_csv(output / 'alert_duty.csv', index=False)
    (output / 'guardrail_hardware.json').write_text(json.dumps(dict(presets=['gen2_single', 'gen2_double', 'core'],
        worst_charge_hours=worst_charge_h.tolist(), assumption='continuous connection and full inverter availability; initial physical minimum; SOH 1'), indent=2) + '\n')
    print(pd.DataFrame(rows).groupby('threshold')[['active_at_onset', 'lead_covers_worst_charge']].mean())


if __name__ == '__main__':
    main()
