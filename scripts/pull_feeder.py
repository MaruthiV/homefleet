import argparse
from pathlib import Path
import re
import requests
from homefleet.paths import DATA_RAW

PREFIX = 'https://oedi-data-lake.s3.amazonaws.com/SMART-DS/v1.0/2018/AUS/P1U/scenarios/base_timeseries/opendss/p1uhs0_1247/p1uhs0_1247--p1udt2142'
FILES = ['Buscoords.dss', 'Capacitors.dss', 'LineCodes.dss', 'Lines.dss', 'Loads.dss', 'Master.dss', 'Transformers.dss', 'SMART-DS_version.txt']


def prepare(folder):
    loads = (folder / 'Loads.dss').read_text()
    loads = re.sub(r'\s+(?:yearly|daily|duty)=(?:"[^"]*"|\S+)', '', loads, flags=re.IGNORECASE)
    (folder / 'Loads_static.dss').write_text(loads)
    lines = []
    for line in (folder / 'Master.dss').read_text().splitlines():
        lower = line.strip().lower()
        if lower.startswith(('redirect loadshapes', 'new monitor.', 'new energymeter.', 'plot ', 'export ')): continue
        if lower.startswith('solve '):
            lines.append('set mode=snapshot')
        else:
            lines.append(line.replace('Loads.dss', 'Loads_static.dss'))
    (folder / 'Master_static.dss').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=DATA_RAW / 'smartds/feeder/p1uhs0_1247--p1udt2142')
    ap.add_argument('--prepare-only', action='store_true')
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if not a.prepare_only:
        for name in FILES:
            target = a.output / name
            if not target.exists():
                response = requests.get(f'{PREFIX}/{name}', timeout=120)
                response.raise_for_status()
                target.write_bytes(response.content)
                print(target, flush=True)
    prepare(a.output)
