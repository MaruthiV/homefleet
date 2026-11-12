import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from homefleet.paths import DATA_PROC, ROOT


def main():
    source = DATA_PROC / 'revised/houston_2024.npz'
    original = dict(np.load(source, allow_pickle=True))
    idx = pd.to_datetime(original['index'], utc=True).tz_convert('America/Chicago')
    start = idx.get_loc(pd.Timestamp('2024-07-08', tz=idx.tz)) - 28 * 96
    end = start + 30 * 96
    result = {}
    for key, value in original.items():
        if value.ndim and value.shape[0] == len(idx): value = value[start:end]
        if key in ['load_kw', 'pv_kw']: value = value[:, :8]
        if key in ['home_fips', 'home_kw_dc', 'home_id']: value = value[:8]
        result[key] = value
    assets = ROOT / 'src/homefleet/assets'
    assets.mkdir(exist_ok=True)
    np.savez_compressed(assets / 'quickstart.npz', **result)
    output = subprocess.check_output([sys.executable, '-m', 'homefleet.demo'], text=True)
    json.loads(output)
    (assets / 'quickstart_expected.json').write_text(output)


if __name__ == '__main__':
    main()
