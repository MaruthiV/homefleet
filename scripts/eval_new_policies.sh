#!/bin/bash
# eval any policy without a 2024 csv yet, then regenerate fig3
cd "$(dirname "$0")/.."
for p in experiments/ppo_*/policy.pt; do
  name=$(basename $(dirname $p) | sed 's/^ppo_//')
  out="experiments/policy_${name}_houston_2024.csv"
  if [ ! -f "$out" ]; then
    echo "evaluating $name"
    python3 scripts/eval_policy.py --policy "$p" --metro houston --year 2024 --n 100 --seed 0 --out "policy_${name}_houston_2024.csv" >> experiments/logs/eval_new.log 2>&1
  fi
done
python3 scripts/fig_frontier_final.py --drop B7_oracle B6_ppo_learned_lam5_s0 2>&1 | tail -20
