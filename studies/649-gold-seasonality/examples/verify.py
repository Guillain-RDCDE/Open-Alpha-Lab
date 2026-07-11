"""Reproducible headline run for Study 649 — Gold Seasonality.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached GLD / ^IRX tapes under ``_cache/``
(fetching once on a cache miss), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from gold_seasonality import data, strategy as st  # noqa: E402

print("# Gold Seasonality — is September really gold's best month, and summer a lull?")

if not data.have_real():
    print("(cache miss — fetching GLD / ^IRX once)")
    data.fetch()

gld, irx = data.load_real()
print(data_stamp("GLD adj close", gld.to_frame("close"), asof=data.AS_OF))
print(data_stamp("^IRX 13-week T-bill yield", irx.to_frame("close"), asof=data.AS_OF))

ret = st.monthly_log_returns(gld)
cash = st.monthly_cash_return(irx)
print(f"\nmonthly observations on tape: {len(ret)}  "
      f"({ret.index.min().date()} -> {ret.index.max().date()})")

print("\n# THE HEADLINE — 12-cell month-of-year table, Bonferroni-corrected")
crit = st.bonferroni_crit_t(12, df=len(ret) - 2)
print(f"Bonferroni bar for 12 simultaneous tests (df={len(ret)-2}): |t| >= {crit:.2f}")
ms = st.month_stats(ret)
n_survive = 0
for m, row in ms.iterrows():
    tag = ""
    if abs(row["tstat_hac"]) >= crit:
        n_survive += 1
        tag = "  <-- SURVIVES BONFERRONI"
    star = "  <-- September (the claim)" if int(m) == 9 else tag
    print(f"  month {int(m):>2}: mean {row['mean']*100:+.2f}%  n={int(row['n'])}  "
          f"t_naive={row['tstat']:+.2f}  t_hac={row['tstat_hac']:+.2f}{star if star else tag}")
print(f"cells clearing the Bonferroni bar: {n_survive}/12")
best_m = ms["mean"].idxmax()
print(f"the actual best month on the tape is month {int(best_m)} "
      f"({ms.loc[best_m,'mean']*100:+.2f}%, t_hac={ms.loc[best_m,'tstat_hac']:+.2f}) -- "
      f"not necessarily September")

print("\n# September vs the rest — the headline claim")
sep = st.month_vs_rest(ret, data.STRONG_MONTHS)
print(f"  September mean {sep['mean']*100:+.2f}% (n={sep['n']})  vs  "
      f"other 11 months {sep['rest_mean']*100:+.2f}% (n={sep['n_rest']})   "
      f"spread {sep['spread']*100:+.2f}%   Welch t = {sep['t']:+.2f}")
sep_ci = st.spread_bootstrap_ci(ret, data.STRONG_MONTHS)
print(f"  circular block-bootstrap 95% CI on the spread ({sep_ci['n_boot']} draws, "
      f"12-month blocks): [{sep_ci['lo']*100:+.2f}%, {sep_ci['hi']*100:+.2f}%]")

print("\n# Summer (May-Aug) vs the rest — the 'lull' half of the claim")
summer = st.month_vs_rest(ret, data.SUMMER_MONTHS)
print(f"  summer mean {summer['mean']*100:+.2f}% (n={summer['n']})  vs  "
      f"other 8 months {summer['rest_mean']*100:+.2f}% (n={summer['n_rest']})   "
      f"spread {summer['spread']*100:+.2f}%   Welch t = {summer['t']:+.2f}")

print("\n# Era contrast — September, pre vs post the 2013 gold crash "
      f"(split {data.ERA_SPLIT}, justified: the end of the 2001-2012 bull 'supercycle')")
ec = st.era_contrast(ret, data.STRONG_MONTHS, data.ERA_SPLIT)
print(f"  2004->2013-04: September mean {ec['early_mean']*100:+.2f}% (n={ec['n_early']}, "
      f"within-era Welch t = {ec['welch_t_early']:+.2f})")
print(f"  2013-04->2026: September mean {ec['late_mean']*100:+.2f}% (n={ec['n_late']}, "
      f"within-era Welch t = {ec['welch_t_late']:+.2f})")
print(f"  Welch t of the difference (late - early): {ec['welch_t_diff']:+.2f}")

print("\n# THIRD AXIS — 'own gold only in September' timer vs buy-and-hold (excess of ^IRX cash)")
print("  (calendar-known rule, zero look-ahead; 2 one-way legs/yr x cost x NAV, active month only)")
bh = st.summary(ret, rf=cash)
print(f"  buy & hold        : Sharpe {bh['sharpe']:+.2f}  CAGR {bh['cagr']*100:+.2f}%  "
      f"vol {bh['vol_ann']*100:.1f}%  maxDD {bh['max_drawdown']*100:.1f}%  (n={bh['n']})")
timer_gross = st.strong_month_timer(ret, data.STRONG_MONTHS, cash=cash)
sg = st.summary(timer_gross, rf=cash)
print(f"  timer, gross      : Sharpe {sg['sharpe']:+.2f}  CAGR {sg['cagr']*100:+.2f}%  "
      f"vol {sg['vol_ann']*100:.1f}%  maxDD {sg['max_drawdown']*100:.1f}%")
for cb in (5.0, 10.0):
    net = st.apply_timer_costs(timer_gross, data.STRONG_MONTHS, cost_bps_one_way=cb)
    sn = st.summary(net, rf=cash)
    print(f"  timer, net {cb:>4.1f}bps : Sharpe {sn['sharpe']:+.2f}  CAGR {sn['cagr']*100:+.2f}%")
hr = st.hit_rate(ret, data.STRONG_MONTHS)
print(f"  September hit rate: {hr['k']}/{hr['n']} = {hr['rate']*100:.1f}%  "
      f"(Wilson 95% [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# Synthetic positive control — deterministic, no network")
print("  the Welch detector must NOT fire on a null world (seasonal=0) and must recover a")
print("  planted September/summer calendar. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s_ in range(20):
    df = data.synthetic_world(seasonal=0.0, seed=649 + s_)
    null_ts.append(st.synthetic_detect(df, data.STRONG_MONTHS)["t"])
null_ts = np.asarray(null_ts)
print(f"  null (seasonal=0), 20 seeds: mean Welch t = {null_ts.mean():+.2f}  "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
df = data.synthetic_world(seasonal=0.03, seed=649)
planted = st.synthetic_detect(df, data.STRONG_MONTHS)
print(f"  planted seasonal=+3.0%/mo (seed 649): September {planted['mean']*100:+.2f}% vs "
      f"rest {planted['rest_mean']*100:+.2f}%  Welch t = {planted['t']:+.2f}")
