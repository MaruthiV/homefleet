import argparse
from pathlib import Path
from homefleet.paths import DATA_PROC, EXPERIMENTS
from homefleet.rl.ppo import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metros", nargs="+", default=["houston"])
    ap.add_argument("--years", nargs="+", type=int, default=[2019, 2020, 2021, 2022, 2023])
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--lam-r", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-min", type=int, default=50)
    ap.add_argument("--n-max", type=int, default=200)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    bundles = [DATA_PROC / f"{m}_{y}.npz" for m in a.metros for y in a.years if (DATA_PROC / f"{m}_{y}.npz").exists()]
    name = a.name or f"ppo_lam{a.lam_r:g}_s{a.seed}"
    train(bundles, EXPERIMENTS / name, total_steps=a.steps, lam_r=a.lam_r, seed=a.seed, n_envs=a.n_envs, n_range=(a.n_min, a.n_max))


if __name__ == "__main__":
    main()
