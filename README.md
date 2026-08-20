# homefleet-ercot / adaptive reliability reserves

Research code for learning risk-adjusted backup reserves for fleets of residential home batteries in ERCOT (RL + optimization layer, evaluated under market, outage, and fleet shift). Built on public data: ERCOT prices, ORNL EAGLE-I outages, NREL ResStock homes, NREL SMART-DS feeders.

## repro (code-only repo; all data is public and pulled by scripts)

```
pip install -e . && pip install -r requirements.txt
python -m venv envs/blast && envs/blast/bin/pip install -r requirements-blast.txt

# data (key-free): ercot prices/as, eagle-i outages, weather, nws alerts, resstock homes
python scripts/pull_eaglei.py --years 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
python scripts/pull_ercot.py --what dam_as_mcpc --years 2019 2020 2021 2022 2023 2024 2025 2026
python scripts/pull_ercot.py --what dam_spp --years 2019 2020 2021 2022 2023 2024 2025 2026
python scripts/pull_ercot.py --what rt_as_mcpc --years 2025 2026
python scripts/pull_weather.py && python scripts/pull_alerts.py && python scripts/pull_resstock.py --n 300

# hazard model + metro-year bundles
python scripts/build_hazard.py
python scripts/build_bundles.py --metros houston dfw san_antonio --years 2019 2020 2021 2022 2023 2024 2025

# baselines, training, evals, figures, tables
python scripts/run_baselines.py --metro houston --year 2024 --n 100 --procs 6
python scripts/train_ppo.py --metros houston --years 2019 2020 2021 2022 2023 --lam-r 20 --seed 1 --promise-cost
python scripts/eval_policy.py --policy experiments/ppo_pc_lam20_s1/policy.pt --metro houston --year 2024 --n 100
python scripts/fig_orthogonality.py && python scripts/fig_frontier_final.py && python scripts/fig_shift.py
python scripts/make_tables.py && python scripts/ac_check.py && python scripts/degradation_table.py
```

15-min RT SPP history: `homefleet.data.ercot.load_rtm_spp` expects yearly NP6-785-ER zips parsed to parquet (see the pull pattern in scripts/). Tests: `pytest`.
