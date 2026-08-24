#!/bin/bash
# spotlight upgrade: seeds 2-5 for both promise-cost configs, then 2024 + uri evals
cd "$(dirname "$0")/.."
for s in 2 3; do
  for lam in 5 20; do
    python3 scripts/train_ppo.py --metros houston --years 2019 2020 2021 2022 2023 --steps 3000000 --lam-r $lam --seed $s --promise-cost --ent 0.005 --name ppo_pc_lam${lam}_s${s} > experiments/logs/ppo_pc_lam${lam}_s${s}.log 2>&1 &
  done
done
wait
for s in 4 5; do
  for lam in 5 20; do
    python3 scripts/train_ppo.py --metros houston --years 2019 2020 2021 2022 2023 --steps 3000000 --lam-r $lam --seed $s --promise-cost --ent 0.005 --name ppo_pc_lam${lam}_s${s} > experiments/logs/ppo_pc_lam${lam}_s${s}.log 2>&1 &
  done
done
wait
for s in 2 3 4 5; do
  for lam in 5 20; do
    n="pc_lam${lam}_s${s}"
    python3 scripts/eval_policy.py --policy experiments/ppo_${n}/policy.pt --metro houston --year 2024 --n 100 --seed 0 --out "policy_${n}_houston_2024.csv" >> experiments/logs/eval_seeds.log 2>&1
    python3 scripts/eval_policy.py --policy experiments/ppo_${n}/policy.pt --metro houston --year 2021 --windows events --n 100 --seed 0 --out "policy_${n}_houston_2021_events.csv" >> experiments/logs/eval_seeds.log 2>&1
  done
done
echo SEEDS_DONE
