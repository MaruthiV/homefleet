import argparse
import requests
from homefleet.paths import DATA_RAW

# figshare article 24237376 v4, ids checked 2026-08-19
FILE_IDS = {2014: 42547717, 2015: 42547822, 2016: 42547825, 2017: 42547828, 2018: 42547879, 2019: 42547885, 2020: 42547894, 2021: 42547891, 2022: 42547897, 2023: 44574907, 2024: 53581661, 2025: 62164877}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True)
    a = ap.parse_args()
    folder = DATA_RAW / 'eaglei'
    folder.mkdir(parents=True, exist_ok=True)
    for name, file_id in [('MCC.csv', 42547708), ('coverage_history.csv', 42547714), ('DQI.csv', 42547705)]:
        target = folder / name
        if not target.exists():
            response = requests.get(f'https://ndownloader.figshare.com/files/{file_id}', timeout=120)
            response.raise_for_status()
            target.write_bytes(response.content)
    for y in a.years:
        out = DATA_RAW / "eaglei" / f"eaglei_{y}_TX.csv"
        url = f"https://ndownloader.figshare.com/files/{FILE_IDS[y]}"
        if out.exists():
            continue
        partial = out.with_suffix('.part')
        count = 0
        with requests.get(url, stream=True, timeout=300) as response:
            response.raise_for_status()
            with partial.open('wb') as handle:
                for line in response.iter_lines():
                    if line.startswith(b'fips_code') or (len(line) > 5 and line[:2] == b'48' and line[2:5].isdigit() and line[5:6] == b','):
                        handle.write(line + b'\n')
                        count += 1
        if count < 2:
            raise RuntimeError(f'no Texas observations downloaded for {y}')
        partial.replace(out)
        print(out)


if __name__ == "__main__":
    main()
