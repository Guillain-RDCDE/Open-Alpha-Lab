"""Reproducible headline run for Study 386 — NFCI-Conditions.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached weekly conditions proxy under
``_cache/nfci_proxy.csv`` if present (the real-tape numbers), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nfci_conditions import data, strategy as st

print("# NFCI-Conditions — financial-conditions PROXY (VIX+MOVE+credit+dollar) + S&P 500")
if data.have_real():
    f = data.load_real()
    span = (f.index.max() - f.index.min()).days / 365.25
    tm = st.tight_mask(f["fci"])
    print(f"weeks          : {len(f)}  ({f.index.min().date()} -> {f.index.max().date()}, "
          f"{span:.1f} years, weekly)")
    print(f"tight weeks    : {int(tm.sum())}  ({tm.mean()*100:.0f}% of weeks, fci >= +0.5 z)")
    print(f"contemp corr   : {st.contemporaneous_corr(f):+.3f}  (dFCI vs SPY return — "
          "MECHANICAL, the index embeds equity vol; predicts nothing)")

    print("\n# Forward S&P returns after a TIGHT week vs unconditional (1-week entry lag)")
    print(f"  {'H':>4} {'n':>4} {'cond_mean':>10} {'cond_win':>9} {'base_mean':>10} "
          f"{'base_win':>9} {'Welch_t':>8} {'p_placebo':>10}")
    for h in (1, 4, 13):
        s = st.summarize(f, h)
        print(f"  {str(h)+'w':>4} {s['n']:>4} {s['cond_mean']*100:>9.2f}% "
              f"{s['cond_win']*100:>8.0f}% {s['base_mean']*100:>9.2f}% "
              f"{s['base_win']*100:>8.0f}% {s['t']:>8.2f} {s['p_placebo']:>10.3f}")

    print("\n# Step-out-when-tight timing rule vs buy-and-hold (1-week lag, one-way costs)")
    for cb in (0.0, 5.0, 10.0):
        b = st.backtest_timing(f, cost_bps=cb)
        print(f"  cost={cb:>4.0f}bps  switches={b['n_switches']:>3}  cash={b['pct_in_cash']*100:>2.0f}%  "
              f"strat Sharpe={b['strat_sharpe']:.2f} (vol {b['strat_ann_vol']*100:.1f}%)  "
              f"bh Sharpe={b['bh_sharpe']:.2f} (vol {b['bh_ann_vol']*100:.1f}%)  "
              f"gap={b['sharpe_gap']:+.2f}  CAGR {b['strat_cagr']*100:.1f}% vs {b['bh_cagr']*100:.1f}%")

    print("\n# Regime concentration — drop the 2008-2009 crisis")
    f2 = f[~((f.index >= "2008-01-01") & (f.index < "2010-01-01"))]
    g_all = st.backtest_timing(f, cost_bps=5.0)["sharpe_gap"]
    g_ex = st.backtest_timing(f2, cost_bps=5.0)["sharpe_gap"]
    print(f"  Sharpe gap full sample = {g_all:+.3f}   ex-2008/09 = {g_ex:+.3f}  "
          "(the bump roughly halves)")

    print("\n# Robustness — shift the tightness threshold (13-week forward)")
    for th in (0.0, 0.5, 1.0):
        s = st.summarize(f, 13, thresh=th)
        print(f"  thresh={th:.1f}z: n={s['n']:>4}  cond13={s['cond_mean']*100:>5.2f}%  "
              f"t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
else:
    print("(no _cache/nfci_proxy.csv — run data.fetch_proxy() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  edge=0 must NOT manufacture a forward signal (despite a strong CONTEMPORANEOUS")
print("  corr); a planted forward edge must light the forward t up.")
print(f"  {'edge':>6} {'contemp':>8} {'cond13':>8} {'base13':>8} {'t13':>7} {'p13':>7} {'sharpe_gap':>11}")
for edge in (0.0, 0.006, 0.012):
    syn = data.synthetic_fci(edge=edge, seed=386)
    cc = st.contemporaneous_corr(syn)
    s = st.summarize(syn, 13)
    b = st.backtest_timing(syn, cost_bps=5.0)
    print(f"  {edge:>6.3f} {cc:>+8.2f} {s['cond_mean']*100:>7.2f}% {s['base_mean']*100:>7.2f}% "
          f"{s['t']:>7.2f} {s['p_placebo']:>7.3f} {b['sharpe_gap']:>+11.2f}")
