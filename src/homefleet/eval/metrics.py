import numpy as np


def margin_per_home_week(margin_steps, n_homes, dt):
    weeks = len(margin_steps) * dt / (24 * 7)
    return float(np.sum(margin_steps) / (n_homes * weeks))


def cvar(x, alpha=0.05):
    x = np.sort(np.asarray(x, dtype=float))
    k = max(1, int(np.ceil(alpha * len(x))))
    return float(x[:k].mean())


def ride_through_rate(failures, n_episodes):
    if n_episodes == 0:
        return float("nan")
    return 1.0 - failures / n_episodes


def bootstrap_ci(values, stat, n_boot=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    stats = [stat(values[rng.integers(0, len(values), len(values))]) for _ in range(n_boot)]
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))
