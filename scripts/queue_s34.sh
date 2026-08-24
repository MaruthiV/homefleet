#!/bin/bash
# s4 oracle-hazard sweeps after 2026 baselines free up, then 2026 policy evals after trainers done
cd "$(dirname "$0")/.."
while pgrep -f "run_baselines" > /dev/null; do sleep 300; done
python3 scripts/run_baselines.py --metro houston --year 2024 --modes B3aOH_risk_priced_oh --lam-grid 1 5 20 --n 100 --seed 0 --procs 4 --out baselines_oh_houston_2024_n100_s0.csv > experiments/logs/baselines_oh_2024.log 2>&1
python3 scripts/run_baselines.py --metro houston --year 2021 --windows events --modes B3aOH_risk_priced_oh --lam-grid 1 5 20 --n 100 --seed 0 --procs 4 --out baselines_oh_houston_2021_events_n100_s0.csv > experiments/logs/baselines_oh_2021.log 2>&1
python3 scripts/run_baselines.py --metro houston --year 2023 --windows events --modes B3aOH_risk_priced_oh --lam-grid 1 5 20 --n 100 --seed 0 --procs 4 --out baselines_oh_houston_2023_events_n100_s0.csv > experiments/logs/baselines_oh_2023.log 2>&1
while pgrep -f "train_ppo" > /dev/null; do sleep 300; done
for pol in pc_lam20_s1 pc_lam5_s1; do
  python3 scripts/eval_policy.py --policy experiments/ppo_${pol}/policy.pt --metro houston --year 2026 --n 100 --seed 0 --out "policy_${pol}_houston_2026.csv" >> experiments/logs/eval_2026.log 2>&1
done
python3 scripts/eval_policy.py --policy experiments/ppo_pc_lam20_s1/policy.pt --metro houston --year 2026 --n 100 --seed 0 --guardrail --out "policy_pc_lam20_s1_guard_houston_2026.csv" >> experiments/logs/eval_2026.log 2>&1
echo QUEUE_S34_DONE
