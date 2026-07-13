"""Reproducible headline run for Study 753 — Reverse-Repo-Drain.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached month-end SPY under ``_cache/``
plus the hardcoded ON RRP proxy (the real-tape numbers), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from reverse_repo_drain import data, strategy as st

print("# Reverse-Repo-Drain — hardcoded ON RRP proxy (FRED RRPONTSYD) + month-end SPY")
if data.have_real():
    f = data.build_real()
    span_years = (f.index.max() - f.index.min()).days / 365.25
    print(f"monthly frame  : {len(f)} month-ends  ({f.index.min().date()} -> "
          f"{f.index.max().date()}, {span_years:.1f} years)")
    print(f"ON RRP proxy   : peak ${f['rrp'].max():,.0f}B at {f['rrp'].idxmax().date()}  "
          f"-> ${f['rrp'].iloc[-1]:,.0f}B latest  (PROXY for FRED RRPONTSYD)")

    print("\n# Next-month SPY return by drain regime (RRP falling over trailing k months),")
    print("# 1-month execution lag; 'fill' = RRP rising. Welch t = drain vs fill.")
    print(f"  {'k':>3} {'n_dr':>5} {'n_fl':>5} {'drain':>8} {'fill':>8} {'base':>8} "
          f"{'spread':>8} {'Welch_t':>8} {'p_block':>8} {'dr_win':>7} {'fl_win':>7}")
    for k in (1, 2, 3, 6, 9, 12):
        s = st.summarize(f, k=k)
        print(f"  {k:>3} {s['n_drain']:>5} {s['n_fill']:>5} {s['drain_mean']*100:>7.2f}% "
              f"{s['fill_mean']*100:>7.2f}% {s['base_mean']*100:>7.2f}% "
              f"{s['spread']*100:>+7.2f} {s['t']:>+8.2f} {s['p_placebo']:>8.3f} "
              f"{s['drain_win']*100:>6.0f}% {s['fill_win']*100:>6.0f}%")

    d3 = f["rrp"].diff(3)
    nxt = f["spy"].pct_change().shift(-1)
    corr = pd.DataFrame({"d": d3, "r": nxt}).dropna().corr().iloc[0, 1]
    print(f"\n  corr( trailing-3m RRP change , next-month SPY return ) = {corr:+.3f}")

    print("\n# Timing backtest — hold SPY when the RRP is draining (1-month lag; SPY total-return)")
    for short in (False, True):
        b = st.timing_backtest(f, cost_bps=10.0, allow_short=short)
        tag = "long/short" if short else "long/flat "
        print(f"  {tag}: exposure={b['exposure']:.2f}  turns={b['n_turns']:.0f}  "
              f"gross Sharpe={b['gross']['sharpe']:+.2f}  net Sharpe={b['net']['sharpe']:+.2f}  "
              f"net ann={b['net']['ann_ret']*100:+.1f}%  (buy&hold Sharpe={b['buy_hold']['sharpe']:.2f}, "
              f"ann={b['buy_hold']['ann_ret']*100:.1f}%)")
else:
    print("(no _cache — run data.fetch_spy() once to build the SPY cache)")

print("\n# Synthetic positive control — deterministic, no network")
print("  inference must recover a PLANTED drain edge and must NOT manufacture significance")
print("  when the true edge is 0 (the drain carries no forward information).")
for edge in (0.0, 0.02):
    syn = data.synthetic(n_months=120, edge=edge, seed=753)
    s = st.summarize(syn, k=3)
    print(f"  planted edge={edge:+.2f}: n_drain={s['n_drain']:>3}  drain={s['drain_mean']*100:>6.2f}%  "
          f"fill={s['fill_mean']*100:>5.2f}%  t={s['t']:>5.2f}  p_block={s['p_placebo']:.3f}")
