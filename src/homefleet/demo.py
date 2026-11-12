import argparse
import json
from pathlib import Path
import numpy as np
from homefleet.env import FleetData, HomeFleetEnv
from homefleet.control.baselines import run_controller
from homefleet.eval.episode import summarize


def main():
    ap = argparse.ArgumentParser(description='Run two days of the public Houston replay for eight simulated homes.')
    ap.add_argument('--check', action='store_true', help='compare numerical results with the released reference')
    args = ap.parse_args()
    assets = Path(__file__).parent / 'assets'
    data = FleetData(assets / 'quickstart.npz')
    output = {}
    for name, settings in [('fixed_20', {'mode': 'fixed'}), ('hazard_reserve', {'mode': 'chance', 'eps_cc': .1})]:
        env = HomeFleetEnv(data)
        env.reset(data.evaluation_start, data.T - data.evaluation_start, np.arange(data.N_pool), ['gen2_single'] * data.N_pool, seed=0)
        res = run_controller(env, S=32, **settings)
        stats = summarize(res, env.dt, params=env.params, soh=env.soh, initial_energy=env.initial_energy)
        output[name] = {key: stats[key] for key in ['margin_per_home_week', 'promise_rate', 'unserved_kwh_per_home', 'physical_violations', 'reachable_floor_violations']}
    print(json.dumps(output, indent=2))
    if args.check:
        expected = json.loads((assets / 'quickstart_expected.json').read_text())
        for name, row in output.items():
            for key, value in row.items():
                if not np.isclose(value, expected[name][key], rtol=1e-6, atol=1e-7, equal_nan=True):
                    raise AssertionError(f'{name}/{key}: {value} differs from reference {expected[name][key]}')
        print('Reference results match.')


if __name__ == '__main__':
    main()
