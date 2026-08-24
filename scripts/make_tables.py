import glob
import numpy as np
import pandas as pd
from homefleet.paths import EXPERIMENTS

OUT = EXPERIMENTS / "tables"
OUT.mkdir(exist_ok=True)


def dm_and_promise(df, ref="B0_fixed"):
    base = df[df.controller == ref].set_index("window").margin_per_home_week
    rows = {}
    for c, g in df.groupby("controller"):
        d = (g.set_index("window").margin_per_home_week - base.reindex(g.window.to_numpy())).dropna()
        rows[c] = dict(dm_yr=d.mean() * 52, margin=g.margin_per_home_week.mean(),
                       promise=1 - g.promise_failures.sum() / max(g.outage_episodes.sum(), 1),
                       cvar5=g.cvar5_daily_margin_per_home.mean(), soc=g.mean_soc_frac.mean())
    return rows


def t1_frontier():
    b = pd.read_csv("experiments/baselines_houston_2024_n100_s0.csv")
    pol_files = ["experiments/policy_learned_lam1_s1_houston_2024.csv", "experiments/policy_learned_lam5_s1_houston_2024.csv",
                 "experiments/policy_learned_lam5_s2_houston_2024.csv", "experiments/policy_pc_lam5_s1_houston_2024.csv",
                 "experiments/policy_pc_lam20_s1_houston_2024.csv", "experiments/policy_pc_lam20_s1_guard_houston_2024.csv",
                 "experiments/policy_fixed_lam5_s1_houston_2024.csv", "experiments/policy_none_lam5_s1_houston_2024.csv"]
    df = pd.concat([b] + [pd.read_csv(f) for f in pol_files], ignore_index=True)
    r = dm_and_promise(df)
    order = ["B7_oracle", "B0_fixed10", "B0_fixed", "B0_fixed30", "B0_fixed50", "B0h_fixed_hold90", "B1_storm_watch", "B2_az_tier",
             "B3b_chance_eps0.1", "B3b_chance_eps0.02", "B3a_risk_priced_lam1", "B3a_risk_priced_lam5", "B3a_risk_priced_lam20",
             "B6_ppo_learned_lam1_s1", "B6_ppo_learned_lam5_s1", "B6_ppo_learned_lam5_s2", "B6_ppo_pc_lam5_s1",
             "B6_ppo_pc_lam20_s1", "B6_ppo_pc_lam20_s1_guard", "B6_ppo_fixed_lam5_s1", "B6_ppo_none_lam5_s1"]
    names = {"B7_oracle": "oracle (perfect prices/loads)", "B0_fixed10": "fixed floor 10\\%", "B0_fixed": "fixed floor 20\\% (industry)",
             "B0_fixed30": "fixed floor 30\\%", "B0_fixed50": "fixed floor 50\\%", "B0h_fixed_hold90": "floor 20\\% + hold-90 posture",
             "B1_storm_watch": "storm watch (NWS alert rule)", "B2_az_tier": "quantile tier reserve",
             "B3b_chance_eps0.1": "chance-constrained $\\varepsilon{=}0.1$", "B3b_chance_eps0.02": "chance-constrained $\\varepsilon{=}0.02$",
             "B3a_risk_priced_lam1": "risk-priced MPC $\\lambda{=}1$", "B3a_risk_priced_lam5": "risk-priced MPC $\\lambda{=}5$",
             "B3a_risk_priced_lam20": "risk-priced MPC $\\lambda{=}20$",
             "B6_ppo_learned_lam1_s1": "learned, raw cost $\\lambda{=}1$ (s1)", "B6_ppo_learned_lam5_s1": "learned, raw cost $\\lambda{=}5$ (s1)",
             "B6_ppo_learned_lam5_s2": "learned, raw cost $\\lambda{=}5$ (s2)", "B6_ppo_pc_lam5_s1": "learned, promise cost $\\lambda{=}5$",
             "B6_ppo_pc_lam20_s1": "learned, promise cost $\\lambda{=}20$", "B6_ppo_pc_lam20_s1_guard": "\\quad + alert guardrail",
             "B6_ppo_fixed_lam5_s1": "ablation: hard 20\\% floor + learned trading", "B6_ppo_none_lam5_s1": "ablation: no floor, penalty only"}
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"controller & $\Delta$margin (\$/home-yr) & margin (\$/hm-wk) & promise & mean SOC \\", r"\midrule"]
    for c in order:
        if c not in r:
            continue
        v = r[c]
        lines.append(f"{names.get(c, c)} & {v['dm_yr']:+.0f} & {v['margin']:.2f} & {v['promise']:.3f} & {v['soc']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "t1_frontier_2024.tex").write_text("\n".join(lines))
    print("t1 done", len(r))


def t2_configs():
    b = pd.read_csv("experiments/baselines_houston_2024_n100_s0.csv")
    base = b[b.controller == "B0_fixed"].set_index("window").margin_per_home_week
    runs = [("raw $\\lambda{=}1$", "learned", 0, "collapsed"), ("raw $\\lambda{=}1$", "learned", 1, "ok"),
            ("raw $\\lambda{=}5$", "learned", 0, "collapsed"), ("raw $\\lambda{=}5$", "learned", 1, "ok"), ("raw $\\lambda{=}5$", "learned", 2, "ok"),
            ("raw $\\lambda{=}20$", "learned", 0, "collapsed"), ("raw $\\lambda{=}20$", "learned", 1, "collapsed"),
            ("raw $\\lambda{=}5$", "fixed floor", 0, "collapsed"), ("raw $\\lambda{=}5$", "fixed floor", 1, "ok"),
            ("raw $\\lambda{=}5$", "no floor", 0, "collapsed"), ("raw $\\lambda{=}5$", "no floor", 1, "hold-high"),
            ("promise $\\lambda{=}5$", "learned", 1, "ok"), ("promise $\\lambda{=}5$", "learned", 2, "ok"),
            ("promise $\\lambda{=}5$", "learned", 3, "ok"), ("promise $\\lambda{=}5$", "learned", 4, "ok"), ("promise $\\lambda{=}5$", "learned", 5, "low-SOC"),
            ("promise $\\lambda{=}20$", "learned", 1, "ok"), ("promise $\\lambda{=}20$", "learned", 2, "ok"),
            ("promise $\\lambda{=}20$", "learned", 3, "ok"), ("promise $\\lambda{=}20$", "learned", 4, "ok"), ("promise $\\lambda{=}20$", "learned", 5, "low-SOC")]
    FILE = {("raw $\\lambda{=}1$", "learned"): "learned_lam1", ("raw $\\lambda{=}5$", "learned"): "learned_lam5",
            ("raw $\\lambda{=}20$", "learned"): "learned_lam20", ("raw $\\lambda{=}5$", "fixed floor"): "fixed_lam5",
            ("raw $\\lambda{=}5$", "no floor"): "none_lam5", ("promise $\\lambda{=}5$", "learned"): "pc_lam5",
            ("promise $\\lambda{=}20$", "learned"): "pc_lam20"}
    lines = [r"\begin{tabular}{llcllr}", r"\toprule",
             r"cost & reserve mode & seed & outcome & $\Delta$margin \$/home-yr & promise \\", r"\midrule"]
    for cost, mode, seed, outcome in runs:
        f = EXPERIMENTS / f"policy_{FILE[(cost, mode)]}_s{seed}_houston_2024.csv"
        dm = pr = "--"
        if f.exists():
            g = pd.read_csv(f)
            d = (g.set_index("window").margin_per_home_week - base.reindex(g.window.to_numpy())).dropna()
            dm = f"{d.mean() * 52:+.0f}"
            pr = f"{1 - g.promise_failures.sum() / max(g.outage_episodes.sum(), 1):.2f}"
        lines.append(f"{cost} & {mode} & {seed} & {outcome} & {dm} & {pr} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "t2_collapse.tex").write_text("\n".join(lines))
    print("t2 done")


def t3_ac():
    t = pd.read_csv("experiments/ac_check.csv")
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"controller & day & pen. & $v_{\min}$ & $v_{\max}$ & steps w/ UV & steps w/ OV \\", r"\midrule"]
    for _, r_ in t.iterrows():
        nm = {"B0_fixed": "fixed 20\\%", "B3a_risk_lam5": "risk-priced $\\lambda{=}5$", "B6_pc_lam20_guard": "learned+guard", "load_only": "load only"}[r_.controller]
        lines.append(f"{nm} & {r_.day} & {r_.penetration:.0%}".replace("%", "\\%") + f" & {r_.vmin:.3f} & {r_.vmax:.3f} & {r_.steps_with_undervolt:.2f} & {r_.steps_with_overvolt:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "t3_ac.tex").write_text("\n".join(lines))
    print("t3 done", len(t))


def t4_degradation():
    t = pd.read_csv("experiments/degradation_table.csv")
    g = t.groupby("controller")[["mean_soc", "blast_pct_yr", "et_pct_yr", "cb_pct_yr", "margin_wk"]].mean()
    nm = {"B0_fixed": "fixed floor 20\\%", "B0h_hold90": "hold-90 posture", "B1_storm_watch": "storm watch",
          "B3a_risk_lam5": "risk-priced $\\lambda{=}5$", "B6_pc_lam20": "learned (promise $\\lambda{=}20$)", "B6_pc_lam20_guard": "\\quad + guardrail"}
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"controller & mean SOC & LFP (BLAST) \%/yr & throughput \%/yr & rainflow \%/yr \\", r"\midrule"]
    for c in ["B0_fixed", "B1_storm_watch", "B3a_risk_lam5", "B0h_hold90", "B6_pc_lam20", "B6_pc_lam20_guard"]:
        v = g.loc[c]
        lines.append(f"{nm[c]} & {v.mean_soc:.2f} & {v.blast_pct_yr:.1f} & {v.et_pct_yr:.2f} & {v.cb_pct_yr:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "t4_degradation.tex").write_text("\n".join(lines))
    print("t4 done")


def t5_events():
    rows = []
    for f, yr in [("experiments/baselines_houston_2021_events_n100_s0.csv", 2021), ("experiments/baselines_houston_2023_events_n100_s0.csv", 2023)]:
        rows.append(pd.read_csv(f))
    for f in glob.glob("experiments/policy_*_houston_2021ev.csv") + glob.glob("experiments/policy_*_houston_2023ev.csv"):
        rows.append(pd.read_csv(f))
    df = pd.concat(rows, ignore_index=True)
    keep = ["B0_fixed", "B1_storm_watch", "B3a_risk_priced_lam5", "B6_ppo_pc_lam20_s1", "B6_ppo_pc_lam20_s1_guard", "B7_oracle"]
    df = df[df.controller.isin(keep)].copy()
    cn = {"B0_fixed": "fixed 20\\%", "B1_storm_watch": "storm watch", "B3a_risk_priced_lam5": "risk-priced $\\lambda{=}5$",
          "B6_ppo_pc_lam20_s1": "learned", "B6_ppo_pc_lam20_s1_guard": "learned+guard", "B7_oracle": "oracle"}
    wn = {"uri_2021": "Uri (Feb 21)", "austin_ice_2023": "Austin ice (Feb 23)", "houston_storms_2023": "Houston storms (Jun 23)",
          "heat_aug_2023": "heat (Aug 23)", "eea2_sep_2023": "EEA2 (Sep 23)"}
    df["controller"] = df.controller.map(cn)
    df["window"] = df.window.map(wn)
    p1 = df.pivot_table(index="window", columns="controller", values="margin_per_home_week").round(0)
    p2 = df.pivot_table(index="window", columns="controller", values="promise_rate").round(2)
    cols = [c for c in ["fixed 20\\%".replace("\\\\","\\"), "storm watch", "risk-priced $\\lambda{=}5$", "learned", "learned+guard", "oracle"] if c in p1.columns]
    with open(OUT / "t5_events.tex", "w") as fh:
        fh.write(p1[cols].to_latex(escape=False, float_format="%.0f", na_rep="--"))
        fh.write("\\vspace{4pt}\n\n")
        fh.write(p2[cols].to_latex(escape=False, float_format="%.2f", na_rep="--"))
    print("t5 done", p1.shape)


t1_frontier(); t2_configs(); t3_ac(); t4_degradation(); t5_events()
