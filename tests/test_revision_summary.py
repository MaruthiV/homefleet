import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from summarize_revision import summarize_group, save_selection
from run_baselines import is_regular_window
from verify_results import exact_coverage_errors, policy_provenance_errors
from run_baselines import POLICY_CODE_SHA
import pytest


def sample_rows():
    rows = []
    for k, (hours, margin, episodes, failures) in enumerate([(336., 10., [2, 2], [1, 0]), (168., 20., [1, 1], [0, 0])]):
        rows.append(dict(window=f'w{k:02d}', hours=hours, margin_per_home_week=margin, cash_margin_per_home_week=margin,
                         outage_episodes=sum(episodes), promise_failures=sum(failures), promise_unserved_kwh_per_home=.5,
                         daily_cash_margin_per_home=json.dumps([margin / 7] * int(hours / 24)),
                         home_episode_counts=json.dumps(episodes), home_promise_failure_counts=json.dumps(failures),
                         home_unserved_kwh=json.dumps([1., 2.]), unserved_kwh_per_home=1.5,
                         mean_soc_frac=.5, throughput_kwh_per_home=10., sec=1., physical_violations=0,
                         reachable_floor_violations=0))
    return pd.DataFrame(rows)


def test_annualization_weights_duration_and_reliability_weights_episodes():
    rows = sample_rows()
    reference = rows.copy(); reference['margin_per_home_week'] = 0.
    result = summarize_group(rows, reference, draws=100)
    assert np.isclose(result['margin_annual'], (40 / 3) * 8760 / 168)
    assert np.isclose(result['promise'], 5 / 6)
    assert np.isclose(result['promise_unserved_annual'], 8760 / 504)
    assert result['observed_hours'] == 504


def test_no_outage_truth_is_not_perfect_reliability():
    rows = sample_rows()
    rows['outage_episodes'] = rows['promise_failures'] = 0
    rows['home_episode_counts'] = rows['home_promise_failure_counts'] = '[0, 0]'
    result = summarize_group(rows, rows, draws=100)
    assert np.isnan(result['promise'])
    assert np.isnan(result['promise_lo'])
    assert np.isnan(result['promise_hi'])


def test_named_wind_event_cannot_be_counted_as_a_regular_year_window():
    names = ['w00', 'w26', 'wind_mar_2025', 'uri_2021']
    assert [name for name in names if is_regular_window(name)] == ['w00', 'w26']
    rows = sample_rows()
    rows.loc[1, 'window'] = 'wind_mar_2025'
    with pytest.raises(ValueError, match='named event duplication'):
        summarize_group(rows, rows)


@pytest.mark.parametrize('last', [('B', False), ('C', True)])
def test_equal_row_count_cannot_hide_a_missing_comparison(last):
    expected = {(s, g) for s in ['A', 'B'] for g in [False, True]}
    rows = pd.DataFrame([('A', False), ('A', True), ('B', False), last], columns=['scenario', 'guarded'])
    assert len(rows) == len(expected)
    assert exact_coverage_errors(rows, ['scenario', 'guarded'], expected, 'audit')


def test_complete_comparison_coverage_is_independent_of_row_order():
    expected = {(s, g) for s in ['A', 'B'] for g in [False, True]}
    rows = pd.DataFrame([('B', True), ('A', False), ('B', False), ('A', True)], columns=['scenario', 'guarded'])
    assert exact_coverage_errors(rows, ['scenario', 'guarded'], expected, 'audit') == []


@pytest.mark.parametrize('corruption', ['other_checkpoint', 'other_policy_source', 'unidentified_checkpoint'])
def test_ancillary_rows_cannot_reuse_another_or_unidentified_learned_policy(corruption):
    rows = pd.DataFrame([dict(controller='ppo', policy_sha256='selected', policy_source_sha256=POLICY_CODE_SHA)])
    if corruption == 'other_checkpoint': rows.loc[0, 'policy_sha256'] = 'previous'
    elif corruption == 'other_policy_source': rows.loc[0, 'policy_source_sha256'] = 'previous source'
    else: rows = rows.drop(columns='policy_sha256')
    assert policy_provenance_errors(rows, rows.controller.eq('ppo'), 'selected', 'audit')


def test_reference_rows_do_not_require_a_learned_checkpoint():
    rows = pd.DataFrame([dict(controller='fixed20'), dict(controller='ppo', policy_sha256='selected',
                                                       policy_source_sha256=POLICY_CODE_SHA)])
    assert policy_provenance_errors(rows, rows.controller.eq('ppo'), 'selected', 'audit') == []
    assert policy_provenance_errors(rows.iloc[:1], rows.iloc[:1].controller.eq('ppo'), None, 'audit') == []


def test_selection_rewrite_retains_checkpoint_provenance_and_rejects_reselection(tmp_path):
    for name in ['ppo_pc_lam5_s1', 'ppo_pc_lam20_s1']:
        folder = tmp_path / 'experiments/revised' / name
        folder.mkdir(parents=True)
        (folder / 'policy.pt').write_bytes(name.encode())
    first = save_selection('B6_ppo_pc_lam5_s1', 123., tmp_path)
    save_selection('B6_ppo_pc_lam5_s1', 123., tmp_path)
    saved = json.loads((tmp_path / 'experiments/revised/selected_policy.json').read_text())
    assert saved == first
    assert saved['policy_sha256'] == hashlib.sha256(b'ppo_pc_lam5_s1').hexdigest()
    assert saved['protocol'] == 'causal-v3' and saved['source_sha256'] == POLICY_CODE_SHA
    with pytest.raises(ValueError, match='prior selection differs'):
        save_selection('B6_ppo_pc_lam20_s1', 124., tmp_path)
    assert json.loads((tmp_path / 'experiments/revised/selected_policy.json').read_text()) == first
