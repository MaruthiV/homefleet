import json
import time
import numpy as np
import torch
import torch.nn as nn
from homefleet.env import FleetData, HomeFleetEnv
from homefleet.sim.outage import event_windows


class RunningStat:
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = eps

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1, *self.mean.shape)
        bm, bv, bc = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def norm(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.critic = nn.Sequential(layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(), layer_init(nn.Linear(hidden, hidden)), nn.Tanh(), layer_init(nn.Linear(hidden, 1), std=1.0))
        self.actor_mean = nn.Sequential(layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(), layer_init(nn.Linear(hidden, hidden)), nn.Tanh(), layer_init(nn.Linear(hidden, act_dim), std=0.01))
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def value(self, x):
        return self.critic(x)

    def act(self, x, action=None):
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.critic(x)


def to_env_action(a):
    # gaussian sample -> [u_p in [-1,1], u_r in [0,1]]
    return np.array([np.clip(a[0], -1, 1), np.clip((a[1] + 1) / 2, 0, 1)])


class EpisodeSampler:
    # random training episodes across bundles; half start near an outage event window
    def __init__(self, bundles, episode_steps, n_range, presets, rng, event_frac=0.5, dropout_max=0.1, cfg=None):
        self.data = [FleetData(b) for b in bundles]
        self.ep = episode_steps
        self.n_range = n_range
        self.presets = presets
        self.rng = rng
        self.event_frac = event_frac
        self.dropout_max = dropout_max
        self.cfg = cfg
        self.events = []
        for i, d in enumerate(self.data):
            f = d.frac.max(axis=1).to_numpy()
            self.events += [(i, s) for s, e in event_windows(f, 0.01) if s > 96 * 2]

    def new_env(self, seed):
        if self.events and self.rng.random() < self.event_frac:
            i, s = self.events[self.rng.integers(len(self.events))]
            start = max(0, min(s - self.rng.integers(0, 96 * 3), self.data[i].T - self.ep - 1))
        else:
            i = self.rng.integers(len(self.data))
            start = self.rng.integers(96 * 28, self.data[i].T - self.ep - 1)
        d = self.data[i]
        n = int(self.rng.integers(self.n_range[0], self.n_range[1] + 1))
        homes = self.rng.integers(0, d.N_pool, n)
        names = list(self.rng.choice(self.presets, n))
        soh = self.rng.uniform(0.85, 1.0, n)
        env = HomeFleetEnv(d, cfg=self.cfg)
        env.reset(int(start), self.ep, homes, names, seed=seed, soh=soh, dropout=float(self.rng.uniform(0, self.dropout_max)), tiers_h=None)
        return env


def train(bundles, out_dir, total_steps=2_000_000, n_envs=16, rollout=256, epochs=6, minibatch=1024, lr=3e-4, gamma=0.997, lam=0.95,
          clip=0.2, ent=0.0, vf=0.5, lam_r=5.0, episode_steps=96 * 14, n_range=(50, 200), presets=("gen2_single", "gen2_double", "core"),
          seed=0, log_every=1, cfg=None, device="cpu"):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    sampler = EpisodeSampler(bundles, episode_steps, n_range, list(presets), rng, cfg=cfg)
    envs = [sampler.new_env(seed * 1000 + i) for i in range(n_envs)]
    obs_dim = envs[0].obs().shape[0]
    agent = Agent(obs_dim, 2).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    ostat = RunningStat((obs_dim,))
    ret_stat = RunningStat(())
    disc_ret = np.zeros(n_envs)
    obs = np.stack([e.obs() for e in envs])
    ostat.update(obs)
    n_updates = total_steps // (n_envs * rollout)
    hist = []
    t0 = time.time()
    gstep = 0
    ep_returns = []
    ep_cur = np.zeros(n_envs)
    for upd in range(1, n_updates + 1):
        frac = 1.0 - (upd - 1) / n_updates
        for g in opt.param_groups:
            g["lr"] = lr * frac
        B_obs = np.zeros((rollout, n_envs, obs_dim), dtype=np.float32)
        B_act = np.zeros((rollout, n_envs, 2), dtype=np.float32)
        B_logp = np.zeros((rollout, n_envs), dtype=np.float32)
        B_rew = np.zeros((rollout, n_envs), dtype=np.float32)
        B_done = np.zeros((rollout, n_envs), dtype=np.float32)
        B_val = np.zeros((rollout, n_envs), dtype=np.float32)
        for s in range(rollout):
            nobs = ostat.norm(obs).astype(np.float32)
            with torch.no_grad():
                a, logp, _, v = agent.act(torch.as_tensor(nobs, device=device))
            a = a.cpu().numpy()
            B_obs[s] = nobs; B_act[s] = a; B_logp[s] = logp.cpu().numpy(); B_val[s] = v.squeeze(-1).cpu().numpy()
            rews = np.zeros(n_envs)
            dones = np.zeros(n_envs)
            for i, e in enumerate(envs):
                o, r, done, _ = e.rl_step(to_env_action(a[i]), lam_r=lam_r)
                rews[i] = r
                ep_cur[i] += r
                if done:
                    dones[i] = 1.0
                    ep_returns.append(ep_cur[i]); ep_cur[i] = 0.0
                    envs[i] = sampler.new_env(seed * 1000 + gstep + i)
                    o = envs[i].obs()
                obs[i] = o
            disc_ret = disc_ret * gamma + rews
            ret_stat.update(disc_ret)
            B_rew[s] = rews / np.sqrt(ret_stat.var + 1e-8)
            B_done[s] = dones
            disc_ret = disc_ret * (1 - dones)
            ostat.update(obs)
            gstep += n_envs
        with torch.no_grad():
            next_v = agent.value(torch.as_tensor(ostat.norm(obs).astype(np.float32), device=device)).squeeze(-1).cpu().numpy()
        adv = np.zeros_like(B_rew)
        last = np.zeros(n_envs)
        for s in reversed(range(rollout)):
            nv = next_v if s == rollout - 1 else B_val[s + 1]
            nonterm = 1.0 - B_done[s]
            delta = B_rew[s] + gamma * nv * nonterm - B_val[s]
            last = delta + gamma * lam * nonterm * last
            adv[s] = last
        ret = adv + B_val
        b_obs = torch.as_tensor(B_obs.reshape(-1, obs_dim), device=device)
        b_act = torch.as_tensor(B_act.reshape(-1, 2), device=device)
        b_logp = torch.as_tensor(B_logp.reshape(-1), device=device)
        b_adv = torch.as_tensor(adv.reshape(-1), device=device)
        b_ret = torch.as_tensor(ret.reshape(-1), device=device)
        nb = rollout * n_envs
        idx = np.arange(nb)
        for _ in range(epochs):
            rng.shuffle(idx)
            for st in range(0, nb, minibatch):
                mb = idx[st:st + minibatch]
                _, newlogp, entropy, newv = agent.act(b_obs[mb], b_act[mb])
                ratio = torch.exp(newlogp - b_logp[mb])
                madv = b_adv[mb]
                madv = (madv - madv.mean()) / (madv.std() + 1e-8)
                pg = torch.max(-madv * ratio, -madv * torch.clamp(ratio, 1 - clip, 1 + clip)).mean()
                vloss = 0.5 * ((newv.squeeze(-1) - b_ret[mb]) ** 2).mean()
                loss = pg - ent * entropy.mean() + vf * vloss
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), 0.5); opt.step()
        if upd % log_every == 0:
            rec = {"update": upd, "steps": gstep, "ep_return_mean": float(np.mean(ep_returns[-20:])) if ep_returns else None, "sec": round(time.time() - t0, 1)}
            hist.append(rec)
            print(json.dumps(rec), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": agent.state_dict(), "obs_mean": ostat.mean, "obs_var": ostat.var, "obs_dim": obs_dim, "lam_r": lam_r}, out_dir / "policy.pt")
    json.dump(hist, open(out_dir / "train_log.json", "w"))
    return agent, ostat


def load_policy(path, device="cpu"):
    ck = torch.load(path, weights_only=False)
    agent = Agent(ck["obs_dim"], 2).to(device)
    agent.load_state_dict(ck["state_dict"])
    agent.eval()
    st = RunningStat((ck["obs_dim"],))
    st.mean, st.var = ck["obs_mean"], ck["obs_var"]
    return agent, st


def run_policy(env, agent, ostat, deterministic=True, lam_r=5.0):
    o = env.obs()
    done = False
    while not done:
        x = torch.as_tensor(ostat.norm(o).astype(np.float32))
        with torch.no_grad():
            mean = agent.actor_mean(x)
            a = mean.numpy() if deterministic else agent.act(x)[0].numpy()
        o, r, done, _ = env.rl_step(to_env_action(a), lam_r=lam_r)
    return env.results()
