import argparse
import subprocess
from homefleet.paths import DATA_RAW

# figshare article 24237376 v4, ids checked 2026-08-19
FILE_IDS = {2014: 42547717, 2015: 42547822, 2016: 42547825, 2017: 42547828, 2018: 42547879, 2019: 42547885, 2020: 42547894, 2021: 42547891, 2022: 42547897, 2023: 44574907, 2024: 53581661, 2025: 62164877}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True)
    a = ap.parse_args()
    for y in a.years:
        out = DATA_RAW / "eaglei" / f"eaglei_{y}_TX.csv"
        url = f"https://ndownloader.figshare.com/files/{FILE_IDS[y]}"
        # stream the 1gb file and keep texas rows only
        cmd = f"curl -sL {url} | grep -E '^48[0-9]{{3}},|^fips_code' > {out}"
        subprocess.run(cmd, shell=True, check=True)
        print(out)


if __name__ == "__main__":
    main()
