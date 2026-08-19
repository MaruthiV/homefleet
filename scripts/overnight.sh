#!/bin/bash
# sequential-ish overnight plan: ppo runs (3 at a time, 2 threads each), then evals, then other metros
set -e
cd "$(dirname "$0")/.."
mkdir -p experiments/logs
run_ppo () { # lam seed mode
  python3 scripts/train_ppo.py --metros houston --years 2019 2020 2021 2022 2023 --steps 3000000 --lam-r $1 --seed $2 --reserve-mode $3 --n-envs 16 > experiments/logs/ppo_${3}_lam$1_s$2.log 2>&1
}
# wave 1: learned reserve, lam 5, three seeds
run_ppo 5 0 learned & run_ppo 5 1 learned & run_ppo 5 2 learned & wait
# wave 2: lam grid for frontier + ablations at lam 5
run_ppo 1 0 learned & run_ppo 20 0 learned & run_ppo 5 0 fixed & wait
run_ppo 5 0 none & run_ppo 1 1 learned & run_ppo 20 1 learned & wait
# evals on houston 2024 (val) + 2025 (test) + events; crn seed 0, n=100
for p in experiments/ppo_*/policy.pt; do
  python3 scripts/eval_policy.py --policy $p --metro houston --year 2024 --n 100 --seed 0 >> experiments/logs/eval_policies.log 2>&1
  python3 scripts/eval_policy.py --policy $p --metro houston --year 2025 --n 100 --seed 0 >> experiments/logs/eval_policies.log 2>&1
  python3 scripts/eval_policy.py --policy $p --metro houston --year 2021 --n 100 --seed 0 --windows events >> experiments/logs/eval_policies.log 2>&1
  python3 scripts/eval_policy.py --policy $p --metro houston --year 2023 --n 100 --seed 0 --windows events >> experiments/logs/eval_policies.log 2>&1
done
# baselines on houston 2025 (test) and other metros 2024
python3 scripts/run_baselines.py --metro houston --year 2025 --n 100 --S 32 --lam-grid 1 5 20 --eps-grid 0.02 0.1 --procs 8 > experiments/logs/baselines_houston_2025.log 2>&1
python3 scripts/run_baselines.py --metro dfw --year 2024 --n 100 --S 32 --lam-grid 1 5 20 --eps-grid 0.02 0.1 --procs 8 > experiments/logs/baselines_dfw_2024.log 2>&1
python3 scripts/run_baselines.py --metro san_antonio --year 2024 --n 100 --S 32 --lam-grid 1 5 20 --eps-grid 0.02 0.1 --procs 8 > experiments/logs/baselines_sa_2024.log 2>&1
echo overnight done
