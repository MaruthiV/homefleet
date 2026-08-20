#!/bin/bash
# seed-1 reruns of the ablations after current trainers free up
cd "$(dirname "$0")/.."
while pgrep -f "train_ppo" | head -6 | wc -l | grep -qv "^ *[0-2]$"; do sleep 120; done
python3 scripts/train_ppo.py --metros houston --years 2019 2020 2021 2022 2023 --steps 3000000 --lam-r 5 --seed 1 --reserve-mode fixed --n-envs 16 > experiments/logs/ppo_fixed_lam5_s1.log 2>&1 &
python3 scripts/train_ppo.py --metros houston --years 2019 2020 2021 2022 2023 --steps 3000000 --lam-r 5 --seed 1 --reserve-mode none --n-envs 16 > experiments/logs/ppo_none_lam5_s1.log 2>&1 &
wait
