import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import yaml
from homefleet.paths import DATA_PROC, EXPERIMENTS, CONFIGS
from homefleet.rl.ppo import train


def completed_run_matches(folder, args, bundles, holdouts):
    if not all((folder / file).exists() for file in ['policy.pt', 'config.json', 'train_log.json']):
        return False
    try:
        config = json.loads((folder / 'config.json').read_text())
        history = json.loads((folder / 'train_log.json').read_text())
    except json.JSONDecodeError:
        return False
    expected = dict(total_steps=args.steps, lam_r=args.lam_r, seed=args.seed, n_envs=args.n_envs,
                    n_range=[args.n_min, args.n_max], reserve_mode=args.reserve_mode,
                    promise_cost=args.promise_cost, ent=args.ent, threads=args.threads,
                    bundles=[str(path) for path in bundles], holdouts=holdouts, protocol='causal-v3')
    expected['bundle_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in bundles}
    mismatched = [key for key, value in expected.items() if config.get(key) != value]
    if mismatched:
        raise FileExistsError(f'{folder}: completed checkpoint has different settings {mismatched}; use a different run name')
    steps = args.steps // (args.n_envs * 256) * (args.n_envs * 256)
    return bool(history and history[-1]['steps'] == steps)


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
    ap.add_argument("--reserve-mode", default="learned", choices=["learned", "fixed", "none"])
    ap.add_argument("--promise-cost", action="store_true")
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument('--data-dir', type=Path, default=DATA_PROC / 'revised')
    ap.add_argument('--threads', type=int, default=1)
    a = ap.parse_args()
    bundles = [a.data_dir / f"{m}_{y}.npz" for m in a.metros for y in a.years]
    missing = [str(path) for path in bundles if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing requested training bundles; download the frozen bundles first: ' + ', '.join(missing))
    name = a.name or f"revised/ppo_{'pc' if a.promise_cost else a.reserve_mode}_lam{a.lam_r:g}_s{a.seed}"
    # event windows in splits.yaml are holdouts: never train on them
    events = yaml.safe_load(open(CONFIGS / "splits.yaml"))["events"]
    folder = EXPERIMENTS / name
    folder.mkdir(parents=True, exist_ok=True)
    # Separate launchers may request the same future job. Only one writer trains it.
    with (folder / '.train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if completed_run_matches(folder, a, bundles, list(events.values())):
            print('Preserving matching completed checkpoint:', folder, flush=True)
            return
        train(bundles, folder, total_steps=a.steps, lam_r=a.lam_r, seed=a.seed, n_envs=a.n_envs, n_range=(a.n_min, a.n_max), holdouts=list(events.values()), reserve_mode=a.reserve_mode, promise_cost=a.promise_cost, ent=a.ent, threads=a.threads)


if __name__ == "__main__":
    main()
