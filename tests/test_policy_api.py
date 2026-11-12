from pathlib import Path
import numpy as np
import pytest


def test_stochastic_policy_replay_accepts_a_single_observation():
    torch = pytest.importorskip('torch')
    from homefleet.env import FleetData, HomeFleetEnv
    from homefleet.rl.ppo import Agent, RunningStat, run_policy
    import homefleet
    torch.set_num_threads(1)
    torch.manual_seed(0)
    data = FleetData(Path(homefleet.__file__).parent / 'assets/quickstart.npz')
    env = HomeFleetEnv(data)
    obs = env.reset(data.evaluation_start, 4, [0, 1], ['gen2_single'] * 2)
    agent = Agent(len(obs), 2)
    agent.protocol = data.protocol
    trace = run_policy(env, agent, RunningStat((len(obs),)), deterministic=False)
    assert trace['margin'].shape == (4, 2)
    assert np.isfinite(trace['margin']).all()
    assert not trace['physical_viol'].any()
    assert not ((trace['pc'] > 0) & (trace['pd'] > 0)).any()


def test_training_rejects_archived_input_protocol(monkeypatch):
    pytest.importorskip('torch')
    from types import SimpleNamespace
    from homefleet.rl import ppo
    monkeypatch.setattr(ppo, 'FleetData', lambda path: SimpleNamespace(protocol='causal-v2'))
    with pytest.raises(ValueError, match='training requires causal-v3'):
        ppo.EpisodeSampler(['archived.npz'], 96, (1, 2), ['gen2_single'], np.random.default_rng(0))
