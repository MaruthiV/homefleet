"""Generate every numerical paper table, figure, and headline macro from checked runs."""
import hashlib
import json
from pathlib import Path
import shutil
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from homefleet.paths import ROOT, EXPERIMENTS, DATA_PROC, CONFIGS
from verify_results import verify, verify_ancillary

OUT = EXPERIMENTS / 'revised'
PAPER = ROOT / 'paper'
FIG = PAPER / 'figs'
TABLE = PAPER / 'tables'
LABELS = {'B0_fixed': 'Fixed 20%', 'B0_fixed10': 'Fixed 10%', 'B0_fixed30': 'Fixed 30%',
          'B0_fixed50': 'Fixed 50%', 'B0h_fixed_hold90': 'Hold near 90%', 'B1_storm_watch': 'Storm rule',
          'B2_az_tier': 'Historical tiers', 'B3b_chance_eps0.1': 'Threshold 0.1',
          'B3b_chance_eps0.02': 'Threshold 0.02', 'B7_oracle': 'Market lookahead'}
for penalty in [1, 5, 20]:
    LABELS[f'B3a_risk_priced_lam{penalty}'] = f'Risk DP {penalty}'
    LABELS[f'B3aOH_risk_priced_oh_lam{penalty}'] = f'Oracle-onset DP {penalty}'
COLORS = {'fixed': '#636363', 'risk': '#2166ac', 'threshold': '#008b8b', 'ppo': '#2f7d32',
          'guard': '#7b3294', 'storm': '#d66000', 'reference': '#171717'}


def latex(value):
    return str(value).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def number(value, decimals=1, signed=False):
    if not np.isfinite(value): return '--'
    return f'{value:+,.{decimals}f}' if signed else f'{value:,.{decimals}f}'


def update_readme_summary(validation):
    fixed = validation.loc['B0_fixed']
    threshold = validation.loc['B3b_chance_eps0.1']
    candidates = validation.loc[validation.index.str.match(r'B6_ppo_pc_lam(5|20)_s[1-5]$')]
    paragraph = (
        'In the corrected Houston 2024 replay, the threshold 0.1 reserve raises simulated backup coverage '
        f'from **{100*fixed.promise:.1f}% to {100*threshold.promise:.1f}%** and adds '
        f'**USD {threshold.delta_margin_annual:.1f} in modeled margin per home-year** '
        f'(paired 95% interval: USD {threshold.delta_lo:.1f}–{threshold.delta_hi:.1f}). '
        'Coverage means serving the first five hours or until earlier restoration. '
        'The paired interval for the coverage gain includes zero, and the 2025 replay yields higher margin '
        'with lower observed coverage. These are conditional simulation comparisons.\n\n'
        f'Across the {len(candidates)} promise-cost PPO runs in that replay, backup coverage ranges from '
        f'**{100*candidates.promise.min():.1f}% to {100*candidates.promise.max():.1f}%**; '
        f'**{int((candidates.delta_margin_annual < 0).sum())} of {len(candidates)}** have lower modeled margin '
        'than fixed 20%. '
        '[The complete results table](artifacts/summary.csv) retains the full values and intervals.'
    )
    path = ROOT / 'README.md'
    source = path.read_text()
    runs = json.loads((ROOT / 'artifacts/experiment_manifest.json').read_text())['runs']
    hours = sum(json.loads((OUT / run / 'train_log.json').read_text())[-1]['sec'] for run in runs) / 3600
    hardware = json.loads((ROOT / 'artifacts/experiment_environment.json').read_text())
    machine = hardware['reference_hardware']
    compute = (
        f"Reference training used an {machine['architecture'].upper()} CPU workstation with "
        f"{machine['logical_processors']} logical processors and {machine['physical_memory_gib']} GiB RAM, "
        f"up to {hardware['maximum_concurrent_training_processes']} concurrent training workers, "
        f"and {hardware['torch_threads_per_run']} PyTorch thread per run. "
        f"Recorded training-loop wall times sum to **{hours:.1f} hours across all {len(runs)} runs**, excluding startup. "
        'This sum differs from elapsed time when runs execute in parallel. '
        'The checkpoint archive retains each run\'s configuration and training log; inference/evaluation timings '
        'are reported separately.'
    )
    for name, content in [('HEADLINE', paragraph), ('COMPUTE', compute)]:
        pattern = rf'(?s)(<!-- GENERATED_{name}_START -->).*?(<!-- GENERATED_{name}_END -->)'
        if len(re.findall(pattern, source)) != 1:
            raise ValueError(f'README must contain exactly one generated-{name.lower()} block')
        source = re.sub(pattern, lambda match: match[1]+'\n\n'+content+'\n\n'+match[2], source)
    path.write_text(source)


def table(filename, columns, rows, align=None):
    # Captions/labels live in the manuscript, keeping generated fragments reusable.
    lines = [r'\begin{tabular}{' + (align or ('l' + 'r' * (len(columns) - 1))) + '}', r'\toprule',
             ' & '.join(columns) + r' \\', r'\midrule']
    lines += [' & '.join(str(x) for x in row) + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular}']
    (TABLE / filename).write_text('\n'.join(lines) + '\n')


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(FIG / f'{name}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)


def orthogonality():
    d = pd.read_csv(OUT / 'orthogonality_daily.csv')
    spike, mass = d.pmax >= 1000, d.out_peak >= 250000
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for mask, label, color, marker in [(~spike & ~mass, 'Neither', '#a8a8a8', '.'),
                                       (spike & ~mass, 'Price threshold only', '#2166ac', 'o'),
                                       (~spike & mass, 'Outage threshold only', '#d66000', '^'),
                                       (spike & mass, 'Both (Uri)', '#b2182b', 's')]:
        ax.scatter(d.pmax[mask], d.out_peak[mask] / 1000., s=18 if marker != '.' else 9,
                   color=color, marker=marker, alpha=.75, label=f'{label}: {mask.sum():,} days')
    ax.axvline(1000, color='#555', ls='--', lw=.8)
    ax.axhline(250, color='#555', ls='--', lw=.8)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(d.pmax.min() * .8, d.pmax.max() * 1.2)
    ax.set_ylim(d.out_peak.min() / 1000 * .8, d.out_peak.max() / 1000 * 1.2)
    ax.set_xlabel('Daily max price across five load zones ($/MWh, log scale)')
    ax.set_ylabel('Texas daily peak customers out\n(thousands, log scale)')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.26), ncol=2, fontsize=9.5, frameon=False)
    ax.grid(alpha=.18)
    save(fig, 'orthogonality')


def frontier(summary, selected):
    d = summary[(summary.metro == 'houston') & (summary.year == 2024)].set_index('controller')
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for controllers, label, color, marker in [
            (['B0_fixed10', 'B0_fixed', 'B0_fixed30', 'B0_fixed50', 'B0h_fixed_hold90'], 'Fixed / hold-high', COLORS['fixed'], 'o'),
            ([f'B3a_risk_priced_lam{x}' for x in [1, 5, 20]], 'Risk DP', COLORS['risk'], 's'),
            (['B3b_chance_eps0.1', 'B3b_chance_eps0.02'], 'Hazard threshold', COLORS['threshold'], '^')]:
        g = d.loc[controllers].sort_values('promise')
        ax.plot(100 * g.promise, g.delta_margin_annual, marker=marker, color=color, label=label, lw=1.2, ms=5)
    candidates = d.loc[d.index.str.match(r'B6_ppo_pc_lam(5|20)_s[1-5]$')]
    ax.scatter(100 * candidates.promise, candidates.delta_margin_annual, color=COLORS['ppo'], marker='x', s=35, label='10 promise-cost PPO runs')
    for controller, color, marker in [('B1_storm_watch', COLORS['storm'], 'D'), ('B2_az_tier', '#9970ab', 'P'),
                                      (selected, COLORS['ppo'], '*'), (selected + '_guard', COLORS['guard'], 'o')]:
        r = d.loc[controller]
        label = LABELS[controller]
        if controller == selected + '_guard':
            ax.scatter(100 * r.promise, r.delta_margin_annual, s=170, facecolors='none',
                       edgecolors=color, linewidths=1.4, marker=marker, label=label, zorder=5)
        else:
            ax.scatter(100 * r.promise, r.delta_margin_annual, s=100 if marker == '*' else 35,
                       color=color, marker=marker, label=label, zorder=4)
    for controller in ['B0_fixed', 'B3b_chance_eps0.1', selected + '_guard']:
        r = d.loc[controller]
        ax.errorbar(100 * r.promise, r.delta_margin_annual,
                    xerr=[[100 * max(0., r.promise - r.promise_lo)], [100 * max(0., r.promise_hi - r.promise)]],
                    yerr=[[max(0., r.delta_margin_annual - r.delta_lo)], [max(0., r.delta_hi - r.delta_margin_annual)]],
                    color='#555', fmt='none', capsize=2, lw=.7, zorder=1)
    ax.axhline(0, color='#555', ls='--', lw=.8)
    ax.set_xlabel('Five-hour promise rate (%; higher is better)')
    ax.set_ylabel('Margin difference from fixed 20%\n($/home-year; higher is better)')
    ax.grid(alpha=.18)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.22), fontsize=9.5, ncol=2, frameon=False)
    save(fig, 'frontier')


def transfer(summary, selected):
    splits = [('houston', 2024), ('houston', 2025), ('houston', 2026), ('dfw', 2024), ('san_antonio', 2024)]
    controllers = ['B3b_chance_eps0.1', 'B3a_risk_priced_lam5', selected, selected + '_guard']
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.3), sharex=True, gridspec_kw={'height_ratios': [1.2, 1]})
    x = np.arange(len(splits)); width = .18
    for k, (c, color) in enumerate(zip(controllers, [COLORS['threshold'], COLORS['risk'], COLORS['ppo'], COLORS['guard']])):
        g = pd.DataFrame([summary[(summary.metro == m) & (summary.year == y) & (summary.controller == c)].iloc[0] for m, y in splits])
        pos = x + (k - 1.5) * width
        axes[0].bar(pos, g.delta_margin_annual, width, color=color, label=LABELS[c])
        axes[1].scatter(pos, 100 * g.promise, color=color, marker=['^', 's', 'o', 'D'][k], s=25)
    axes[0].axhline(0, color='#555', lw=.8)
    axes[0].set_ylabel('Margin difference\n($/home-year; higher is better)')
    axes[1].set_ylabel('Promise rate\n(%; higher is better)')
    axes[1].set_ylim(0, 105)
    axes[1].text(2, 50, 'No outage truth', ha='center', va='center', fontsize=9.5, color='#555', rotation=90)
    axes[1].set_xticks(x, ['Houston\n2024\nvalidation', 'Houston\n2025', 'Houston\n2026\npartial', 'DFW\n2024', 'San Antonio\n2024'])
    axes[0].legend(fontsize=9.5, ncol=2)
    for ax in axes: ax.grid(axis='y', alpha=.18)
    fig.tight_layout()
    save(fig, 'transfer')


def stress_plot():
    d = pd.read_csv(OUT / 'stress_guardrail_ppo.csv')
    subset = d[(d.price == 5000) & (d.duration == 8) & (d.soc0 == .1) & (d.alert == 'correct') &
               (d.dropout == 0) & (d.load_multiplier == 1) & (d.soh == 1)].drop_duplicates(['lead', 'policy', 'guarded'])
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.5))
    for policy, guarded, label, color, marker in [
            ('always_discharge', False, 'Always discharge', '#777', 'o'),
            ('always_discharge', True, 'Discharge + override', '#d66000', 's'),
            ('ppo', False, 'Selected PPO', COLORS['ppo'], '^'),
            ('ppo', True, 'PPO + override', COLORS['guard'], 'D')]:
        g = subset[(subset.policy == policy) & (subset.guarded == guarded)].sort_values('lead')
        style = dict(marker=marker, color=color, ms=8 if policy == 'always_discharge' else 4.5,
                     markerfacecolor='none' if policy == 'always_discharge' else color,
                     markeredgewidth=1.3, linestyle='--' if guarded and policy == 'ppo' else '-')
        axes[0].plot(g.lead, 100 * g.promise_rate, label=label, **style)
        axes[1].plot(g.lead, 100 * g.onset_soc_mean, **style)
    axes[0].set_ylabel('Promise rate (%; higher is better)')
    axes[1].set_ylabel('Mean SOC at outage onset (%)')
    for ax in axes:
        ax.set_xlabel('Continuous warning lead (hours)')
        ax.set_xticks([0, 1, 3, 6]); ax.set_ylim(-3, 103); ax.grid(alpha=.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, ncol=4, loc='lower center',
               bbox_to_anchor=(.5, .005), frameon=False)
    fig.tight_layout(rect=[0, .08, 1, 1])
    save(fig, 'stress')


def event_traces(selected_path):
    import yaml
    import torch
    from homefleet.env import FleetData, HomeFleetEnv
    from homefleet.control.baselines import run_controller
    from homefleet.rl.ppo import load_policy, run_policy
    from run_baselines import windows_for_year
    torch.set_num_threads(1)
    agent, stat = load_policy(selected_path)
    events = yaml.safe_load((CONFIGS / 'splits.yaml').read_text())['events']
    fig, axes = plt.subplots(3, 2, figsize=(6.8, 6.3), sharex='col')
    names = {'fixed20': 'Fixed 20%', 'risk5': 'Risk DP 5', 'ppo': 'Selected PPO', 'ppo_guard': 'PPO + override'}
    traces = []
    for col, (year, window, title) in enumerate([(2021, 'uri_2021', 'Winter Storm Uri'), (2024, 'beryl_2024', 'Hurricane Beryl')]):
        d = FleetData(DATA_PROC / 'revised' / f'houston_{year}.npz')
        start, steps = {w: (s, n) for w, s, n in windows_for_year(d, year, events=events)}[window]
        index = d.index[start:start + steps]
        axes[0, col].plot(index, d.price_rt[start:start + steps] / 1000, color='#555', lw=.8)
        ax2 = axes[0, col].twinx()
        ax2.plot(index, 100 * d.frac.iloc[start:start + steps, 0], color='#d66000', lw=.8)
        ax2.set_ylim(0, 100)
        if col == 1: ax2.set_ylabel('County customers out (%)', color='#a64d00')
        axes[0, col].set_title(title)
        rng = np.random.default_rng(0)
        homes = rng.integers(0, d.N_pool, 100)
        presets = list(rng.choice(['gen2_single', 'gen2_double', 'core'], 100, p=[.6, .2, .2]))
        for controller, color in [('fixed20', COLORS['fixed']), ('risk5', COLORS['risk']), ('ppo', COLORS['ppo']), ('ppo_guard', COLORS['guard'])]:
            env = HomeFleetEnv(d); env.reset(start, steps, homes, presets, seed=0)
            if controller == 'fixed20': trace = run_controller(env, mode='fixed', S=32)
            elif controller == 'risk5': trace = run_controller(env, mode='risk_priced', lam_r=5., S=32)
            else: trace = run_policy(env, agent, stat, guardrail=controller == 'ppo_guard')
            soc = np.mean(trace['e'] / (env.params.energy_kwh * env.soh), axis=1) * 100
            unserved = trace['unserved'].mean(axis=1).cumsum()
            style = dict(color=color, lw=1.2 if controller == 'ppo_guard' else 1.,
                         linestyle='--' if controller == 'ppo_guard' else '-')
            axes[1, col].plot(index, soc, label=names[controller], **style)
            axes[2, col].plot(index, unserved, **style)
            traces.extend(dict(event=window, time=str(t), controller=controller, mean_soc_pct=float(s), cumulative_unserved_per_home=float(u))
                          for t, s, u in zip(index, soc, unserved))
        axes[1, col].set_ylim(0, 105)
        axes[2, col].xaxis.set_major_locator(mdates.DayLocator(interval=2, tz=index.tz))
        axes[2, col].xaxis.set_major_formatter(mdates.DateFormatter('%d', tz=index.tz))
        axes[2, col].set_xlabel(index[0].strftime('%b %Y'))
        for row in range(3): axes[row, col].grid(alpha=.15)
    axes[0, 0].set_ylabel('Price ($/kWh)')
    axes[1, 0].set_ylabel('Mean SOC (%)')
    axes[2, 0].set_ylabel('Cumulative unserved\nenergy (kWh/home)\n(lower is better)')
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, ncol=4, loc='lower center',
               bbox_to_anchor=(.5, .005), frameon=False)
    fig.tight_layout(rect=[0, .05, 1, 1])
    save(fig, 'event_traces')
    pd.DataFrame(traces).to_csv(OUT / 'event_traces.csv', index=False)


def main():
    errors = verify() + verify_ancillary()
    if errors: raise SystemExit('Paper generation requires complete checked runs:\n' + '\n'.join(errors))
    FIG.mkdir(exist_ok=True, parents=True); TABLE.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({'font.size': 10.5, 'axes.titlesize': 11, 'axes.labelsize': 10.5,
                         'legend.fontsize': 9.5, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    summary = pd.read_csv(OUT / 'summary.csv')
    events = pd.read_csv(OUT / 'events.csv')
    selection = json.loads((OUT / 'selected_policy.json').read_text())
    selected = selection['controller']
    LABELS[selected] = 'Selected PPO'; LABELS[selected + '_guard'] = 'PPO + override'
    validation = summary[(summary.metro == 'houston') & (summary.year == 2024)].set_index('controller')
    controls = ['B0_fixed', 'B0_fixed50', 'B0h_fixed_hold90', 'B1_storm_watch', 'B2_az_tier',
                'B3a_risk_priced_lam5', 'B3b_chance_eps0.1', selected, selected + '_guard']
    rows = []
    for controller in controls:
        r = validation.loc[controller]
        rows.append([latex(LABELS[controller]), number(r.delta_margin_annual, signed=True) + f' [{number(r.delta_lo)}, {number(r.delta_hi)}]',
                     f'{100*r.promise:.1f} [{100*r.promise_lo:.1f}, {100*r.promise_hi:.1f}]', number(r.unserved_annual)])
    table('frontier.tex', ['Controller', r'$\Delta$ margin [95\% CI]', r'Promise [95\% CI]', 'Unserved'], rows)
    rows, home_rows = [], []
    for controller in controls:
        r = validation.loc[controller]
        rows.append([latex(LABELS[controller]), number(r.cash_margin_annual), number(r.margin_annual),
                     number(r.cvar5_daily, 2), number(r.worst_week, 2)])
        home_rows.append([latex(LABELS[controller]), int(r.eligible_homes), number(100*r.home_promise_p05), number(r.home_unserved_p95)])
    table('economic_tails.tex', ['Controller', 'Cash/yr', 'Adjusted/yr', 'Daily CVaR$_5$', 'Worst 7 blocks'], rows)
    table('households.tex', ['Controller', 'Homes with outages', r'Promise $p_5$\%', 'Unserved $p_{95}$/yr'], home_rows)
    seasonal = pd.read_csv(OUT / 'seasonal.csv')
    if seasonal.duplicated(['metro', 'year', 'controller', 'season']).any():
        raise ValueError('Repeated seasonal cohorts')
    for _, annual in summary.iterrows():
        g = seasonal[(seasonal.metro == annual.metro) & (seasonal.year == annual.year) &
                     (seasonal.controller == annual.controller)]
        if (not len(g) or not np.isclose(g.observed_hours.sum(), annual.observed_hours) or
                g.episodes.sum() != annual.episodes or g.promise_failures.sum() != annual.promise_failures or
                not np.isclose(np.average(g.delta_margin_annual, weights=g.observed_hours), annual.delta_margin_annual,
                               rtol=1e-10, atol=1e-10)):
            raise ValueError('Seasonal cohorts do not reconstruct annual hours, outage counts, or economics')
    rows = []
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        g = seasonal[(seasonal.metro == 'houston') & (seasonal.year == 2024) & (seasonal.season == season)].set_index('controller')
        for c in ['B0_fixed', 'B3b_chance_eps0.1', 'B3a_risk_priced_lam5', selected, selected + '_guard']:
            r = g.loc[c]
            rows.append([season, latex(LABELS[c]), number(r.observed_hours, 0),
                         number(r.delta_margin_annual * 168 / 8760, signed=True), number(100 * r.promise), int(r.episodes)])
    table('seasonal.tex', ['Season', 'Dispatch', 'Hours', r'$\Delta$ margin/wk', r'Promise\%', 'Episodes'], rows, 'llrrrr')
    candidates = validation.loc[validation.index.str.match(r'B6_ppo_pc_lam(5|20)_s[1-5]$')]
    for filename, mask in [('promise_seeds.tex', validation.index.str.match(r'B6_ppo_pc_lam(5|20)_s[1-5]$')),
                           ('other_seeds.tex', validation.index.str.startswith('B6_') & ~validation.index.str.contains('_guard') & ~validation.index.str.contains('ppo_pc_'))]:
        rows = []
        for controller, r in validation.loc[mask].sort_index().iterrows():
            run = controller.removeprefix('B6_ppo_')
            rows.append([latex(run) + ('*' if controller == selected else ''), number(r.delta_margin_annual, signed=True),
                         number(100*r.promise), number(100*r.mean_soc), number(r.throughput_annual)])
        table(filename, ['Run', r'$\Delta$ margin', r'Promise (\%)', r'Mean SOC (\%)', 'Throughput'], rows)
    main_events = [('houston', 2021, 'uri_2021'), ('houston', 2022, 'elliott_2022'),
                   ('houston', 2024, 'derecho_2024'), ('houston', 2024, 'beryl_2024'), ('dfw', 2022, 'elliott_2022')]
    rows = []
    for metro, year, window in main_events:
        g = events[(events.metro == metro) & (events.year == year) & (events.window == window)].set_index('controller')
        for c in ['B0_fixed', 'B1_storm_watch', 'B3a_risk_priced_lam5', selected, selected + '_guard']:
            r = g.loc[c]
            event_label = {'uri_2021': 'Uri', 'elliott_2022': 'Elliott', 'derecho_2024': 'Derecho', 'beryl_2024': 'Beryl'}[window]
            rows.append([f'{"DFW" if metro == "dfw" else "HOU"}: {event_label}', latex(LABELS[c]), number(r.margin_per_home_week),
                         number(100*r.promise_rate), number(r.unserved_kwh_per_home), str(int(r.outage_episodes))])
    table('events.tex', ['Window', 'Controller', 'Margin/wk', r'Promise (\%)', 'Unserved', 'Episodes'], rows, 'llrrrr')
    ac = pd.read_csv(OUT / 'ac_check.csv')
    ac_names = {'load_PV_reference': 'Load/PV reference', 'B0_fixed': 'Fixed 20%', 'B3a_risk_lam5': 'Risk DP 5', 'PPO_guard': 'PPO + override'}
    rows = []
    for _, r in ac.sort_values(['day', 'penetration', 'controller']).iterrows():
        rows.append([r.day, latex(ac_names[r.controller]), number(100*r.penetration, 0),
                     f'[{r.vmin:.3f}, {r.vmax:.3f}]', number(100*r.steps_with_undervolt),
                     number(100*r.steps_with_overvolt), number(100*r.steps_with_line_overload)])
    table('ac.tex', ['Day', 'Dispatch', r'Pen.\%', 'Voltage range', r'UV\%', r'OV\%', r'Line\%'], rows, 'llrrrrr')
    deg = pd.read_csv(OUT / 'degradation_table.csv')
    rows = []
    for c, g in deg.groupby('controller', sort=False):
        rows.append([latex(c.replace('_', ' ')), number(100*g.mean_soc.mean()),
                     number(g.et_pct_yr.mean(), 2), number(g.cb_pct_yr.mean(), 2), number(g.blast_pct_yr.mean(), 2)])
    table('degradation.tex', ['Dispatch', r'Mean SOC (\%)', 'ET', 'Rainflow', 'BLAST'], rows)
    guard = pd.read_csv(OUT / 'guardrail_constants.csv')
    rows = []
    for threshold, g in guard.groupby('threshold'):
        rows.append([number(100*threshold, 0), str(len(g)), number(100*g.active_at_onset.mean()),
                     number(100*g.recent_alert_24h.mean()), number(100*g.lead_covers_worst_charge.mean())])
    table('alert_coverage.tex', [r'Threshold\%', 'Events', r'Active\%', r'Past 24 h\%', r'Lead $\ge4$ h\%'], rows)
    hazard = json.loads((DATA_PROC / 'causal/hazard_report.json').read_text())
    rows = []
    for year in [2021, 2024, 2025]:
        for horizon in [1, 6, 24]:
            r = hazard['report'][str(year)][str(horizon)]
            rows.append([year, horizon, number(r['auc'], 3), number(r['ap'], 3), number(r['brier'], 4), number(100*r['pos_rate'], 2)])
    table('hazard.tex', ['Year', 'Horizon (h)', 'ROC AUC', 'Avg. precision', 'Brier', r'Positive\%'], rows)
    structure = pd.read_csv(OUT / 'structure.csv')
    rows = []
    for (scenario, controller), g in structure.groupby(['scenario', 'controller'], sort=False):
        ref = structure[(structure.scenario == scenario) & (structure.controller == 'fixed20')].set_index('window')
        delta = g.margin_per_home_week.to_numpy() - ref.margin_per_home_week.reindex(g.window).to_numpy()
        episodes = g.outage_episodes.sum()
        promise = 1 - g.promise_failures.sum() / episodes if episodes else np.nan
        rows.append([latex(scenario.replace('_', ' ')), latex(controller.replace('_', ' ')),
                     number(np.average(delta, weights=g.hours), signed=True), number(100*promise), int(episodes)])
    table('structure.tex', ['Scenario', 'Dispatch', r'$\Delta$ margin/wk', r'Promise\%', 'Episodes'], rows, 'llrrr')
    latency = pd.read_csv(OUT / 'latency.csv')
    rows = [[int(n), latex(c.replace('_', ' ')), number(g.ms_per_interval.mean(), 2), number(g.ms_per_interval.std(), 2)]
            for (n, c), g in latency.groupby(['n', 'controller'], sort=False)]
    table('latency.tex', ['Homes', 'Dispatch', 'Mean ms/interval', 'SD'], rows, 'rlrr')
    macros = {}
    for prefix, controller in [('Fixed', 'B0_fixed'), ('Threshold', 'B3b_chance_eps0.1'),
                               ('Risk', 'B3a_risk_priced_lam5'), ('Learned', selected), ('Guarded', selected + '_guard')]:
        r = validation.loc[controller]
        macros[prefix + 'Promise'] = number(100*r.promise) + r'\%'
        macros[prefix + 'Delta'] = number(r.delta_margin_annual, signed=True)
        macros[prefix + 'Low'] = number(r.delta_lo, signed=True)
        macros[prefix + 'High'] = number(r.delta_hi, signed=True)
        macros[prefix + 'Unserved'] = number(r.unserved_annual)
    macros['PromiseSeedLow'] = number(100*candidates.promise.min()) + r'\%'
    macros['PromiseSeedHigh'] = number(100*candidates.promise.max()) + r'\%'
    macros['PromiseSeedMedianDelta'] = number(candidates.delta_margin_annual.median(), signed=True)
    macros['PromiseCandidateCount'] = str(len(candidates))
    macros['PromiseBelowFixedCount'] = str(int((candidates.promise < validation.loc['B0_fixed', 'promise']).sum()))
    macros['PromiseNegativeMarginCount'] = str(int((candidates.delta_margin_annual < 0).sum()))
    threshold = validation.loc['B3b_chance_eps0.1']
    if not (threshold.delta_margin_annual > 0 and threshold.promise > validation.loc['B0_fixed', 'promise']):
        raise ValueError('The manuscript headline requires revision for these changed inputs.')
    if not (threshold.delta_lo > 0 and threshold.delta_promise_lo <= 0 <= threshold.delta_promise_hi):
        raise ValueError('Revise the manuscript interpretation of the paired economic/service intervals for these results.')
    test = summary[(summary.metro == 'houston') & (summary.year == 2025)].set_index('controller')
    if not (test.loc['B3b_chance_eps0.1', 'delta_margin_annual'] > 0 and
            test.loc['B3b_chance_eps0.1', 'promise'] < test.loc['B0_fixed', 'promise']):
        raise ValueError('Revise the manuscript claim about the 2025 threshold tradeoff for these results.')
    macros['ThresholdGain'] = number(threshold.delta_margin_annual)
    macros['ThresholdPromiseDelta'] = number(100 * threshold.delta_promise)
    macros['ThresholdPromiseDeltaLow'] = number(100 * threshold.delta_promise_lo, signed=True)
    macros['ThresholdPromiseDeltaHigh'] = number(100 * threshold.delta_promise_hi, signed=True)
    macros['SelectedRun'] = r'\texttt{' + latex(selection['run']) + '}'
    for prefix, year in [('Test', 2025), ('Partial', 2026)]:
        g = summary[(summary.metro == 'houston') & (summary.year == year)].set_index('controller')
        macros[prefix + 'FixedPromise'] = number(100 * g.loc['B0_fixed', 'promise']) + r'\%'
        for name, c in [('Threshold', 'B3b_chance_eps0.1'), ('Learned', selected), ('Guarded', selected + '_guard')]:
            macros[prefix + name + 'Delta'] = number(g.loc[c, 'delta_margin_annual'], signed=True)
            macros[prefix + name + 'Promise'] = number(100*g.loc[c, 'promise']) + r'\%'
    for prefix, metro in [('DFW', 'dfw'), ('SanAntonio', 'san_antonio')]:
        g = summary[(summary.metro == metro) & (summary.year == 2024)].set_index('controller')
        macros[prefix + 'FixedPromise'] = number(100*g.loc['B0_fixed', 'promise']) + r'\%'
        macros[prefix + 'ThresholdPromise'] = number(100*g.loc['B3b_chance_eps0.1', 'promise']) + r'\%'
        macros[prefix + 'ThresholdDelta'] = number(g.loc['B3b_chance_eps0.1', 'delta_margin_annual'], signed=True)
        macros[prefix + 'Episodes'] = str(int(g.loc['B0_fixed', 'episodes']))
    uri = events[(events.metro == 'houston') & (events.window == 'uri_2021')].set_index('controller')
    for name, c in [('Fixed', 'B0_fixed'), ('Risk', 'B3a_risk_priced_lam5'), ('Learned', selected), ('Guarded', selected + '_guard')]:
        macros['Uri' + name + 'Promise'] = number(100*uri.loc[c, 'promise_rate']) + r'\%'
        macros['Uri' + name + 'Margin'] = number(uri.loc[c, 'margin_per_home_week'])
    mass = guard[guard.threshold == .05]
    macros['MassEvents'] = str(len(mass))
    macros['AlertCoverage'] = number(100*mass.active_at_onset.mean()) + r'\%'
    macros['AlertLeadCoverage'] = number(100*mass.lead_covers_worst_charge.mean()) + r'\%'
    synthetic = pd.read_csv(OUT / 'stress_guardrail_ppo.csv')
    conditions = synthetic[(synthetic.alert == 'correct') & (synthetic.lead >= 3) & (synthetic.dropout == 0) &
                           (synthetic.soh == 1) & (synthetic.load_multiplier == 1) & synthetic.guarded]
    macros['SufficientTests'] = str(len(conditions))
    false = synthetic[synthetic.alert == 'false'].set_index(['case', 'policy', 'guarded'])
    differences, after = [], []
    for case, policy in false.index.droplevel('guarded').unique():
        a, b = false.loc[(case, policy, False)], false.loc[(case, policy, True)]
        differences.append((b.margin_per_home_week - a.margin_per_home_week) * a.hours / 168)
        after.append(b.cash_margin_after_alert_usd_per_home - a.cash_margin_after_alert_usd_per_home)
    macros['FalseAlertWorstCost'] = number(max(0., -min(differences)))
    macros['FalseAlertAfterDifference'] = number(max(abs(x) for x in after))
    macros['ACConvergence'] = number(100 * ac.converged.mean()) + r'\%'
    for prefix, c in [('Fixed', 'fixed20'), ('Hold', 'hold90')]:
        g = deg[deg.controller == c]
        macros[prefix + 'ETFade'] = number(g.et_pct_yr.mean(), 2) + r'\%'
        macros[prefix + 'BLASTFade'] = number(g.blast_pct_yr.mean(), 2) + r'\%'
    (PAPER / 'numbers.tex').write_text('% Generated by scripts/build_paper_results.py; do not edit.\n' +
        '\n'.join(r'\newcommand{' + chr(92) + name + '}{' + value + '}' for name, value in macros.items()) + '\n')
    orthogonality(); frontier(summary, selected); transfer(summary, selected); stress_plot()
    event_traces(ROOT / selection['path'])
    public = ROOT / 'artifacts'
    for filename in ['summary.csv', 'seasonal.csv', 'events.csv', 'structure.csv', 'degradation_table.csv', 'ac_check.csv',
                     'latency.csv', 'guardrail_constants.csv', 'alert_duty.csv', 'orthogonality_daily.csv',
                     'orthogonality.json', 'selected_policy.json', 'report.json', 'guardrail_hardware.json']:
        shutil.copy2(OUT / filename, public / filename)
    generated = [p for folder in [TABLE, FIG] for p in folder.glob('*') if p.suffix in {'.tex', '.pdf', '.png'}]
    generated.append(PAPER / 'numbers.tex')
    update_readme_summary(validation)
    report = dict(protocol='causal-v3', selection=selection, macros=macros,
                  readme_sha256=hashlib.sha256((ROOT / 'README.md').read_bytes()).hexdigest(),
                  generated=[dict(file=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(generated)])
    (public / 'paper_results.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Generated paper results from verified main grid and ancillary audits.')


if __name__ == '__main__':
    main()
