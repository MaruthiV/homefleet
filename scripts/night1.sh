#!/bin/bash
set -e
cd "$(dirname "$0")/.."
python3 scripts/run_baselines.py --metro houston --year 2024 --n 100 --S 32 --lam-grid 1 5 20 --eps-grid 0.02 0.1 --procs 6 > experiments/logs/baselines_houston_2024.log 2>&1
python3 scripts/run_baselines.py --metro houston --year 2021 --n 100 --S 32 --lam-grid 1 5 20 --eps-grid 0.02 0.1 --windows events --procs 6 --out baselines_houston_2021_events_n100_s0.csv > experiments/logs/baselines_houston_2021_events.log 2>&1
python3 scripts/run_baselines.py --metro houston --year 2023 --n 100 --S 32 --lam-grid 1 5 20 --eps-grid 0.02 0.1 --windows events --procs 6 --out baselines_houston_2023_events_n100_s0.csv > experiments/logs/baselines_houston_2023_events.log 2>&1
bash scripts/overnight.sh > experiments/logs/overnight.log 2>&1
