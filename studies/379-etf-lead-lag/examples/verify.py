"""Reproducible headline run for Study 379 — ETF Lead-Lag.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached basket prices under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from etf_lead_lag import data, strategy as st

print("# ETF Lead-Lag — leader SPY vs a 37-name smaller/less-liquid member basket (daily bars)")
if data.have_real():
    F = data.load_real()
    span = (F.index.max() - F.index.min()).days / 365.25
    print(f"days           : {len(F)}  ({F.index.min().date()} -> {F.index.max().date()}, "
          f"{span:.1f} years)")
    print("returns        : daily log returns; members = equal-weight mean of member returns")

    print("\n# Lead-lag cross-correlation profile  corr(leader_{t-k}, members_t)")
    prof = st.crosscorr_profile(F, max_lag=5)
    for k in sorted(prof):
        tag = "  <-- contemporaneous (untradable)" if k == 0 else (
            "  <-- the tradable claim" if k == 1 else "")
        print(f"  k={k:+d}  {prof[k]:+.4f}{tag}")

    b = st.lag1_beta(F)
    print(f"\n# One-day lead-lag slope (members_t on leader_(t-1)), Newey-West HAC")
    print(f"  slope = {b['slope']:+.5f}   HAC t = {b['t_hac']:+.2f}   n = {b['n']}")

    print("\n# Next-bar rule — long the laggard basket the day after a big-leader day")
    print(f"  {'q':>5} {'n':>5} {'cond%':>9} {'cwin':>6} {'base%':>9} {'bwin':>6} "
          f"{'Welch_t':>8} {'p_plac':>8}")
    for q in (0.90, 0.95):
        ev = st.big_leader_days(F, q=q)
        s = st.summarize(F, ev)
        print(f"  {q:>5.2f} {s['n']:>5} {s['cond_mean']*100:>8.4f}% {s['cond_win']*100:>5.0f}% "
              f"{s['base_mean']*100:>8.4f}% {s['base_win']*100:>5.0f}% {s['t']:>8.2f} "
              f"{s['p_placebo']:>8.3f}")

    print("\n# Net of one-way costs (5 bps each side = 10 bps round-turn on a 1-day hold)")
    for q in (0.90, 0.95):
        ev = st.big_leader_days(F, q=q)
        c = st.net_of_costs(F, ev, cost_bps=5.0)
        print(f"  q={q:.2f}  n_trades={c['n_trades']:>4}  gross={c['gross_mean']*100:>8.4f}%  "
              f"net@5bps={c['net_mean']*100:>8.4f}%")
else:
    print("(no _cache/lead_lag_prices.csv — run data.fetch_basket() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  edge=0 must NOT manufacture significance; a planted one-day lead MUST light up.")
for lead in (0.0, 0.05, 0.25):
    syn = data.synthetic_leadlag(n_days=6000, lead=lead, seed=379)
    p = st.crosscorr_profile(syn, max_lag=2)
    bb = st.lag1_beta(syn)
    ev = st.big_leader_days(syn, q=0.90)
    s = st.summarize(syn, ev)
    print(f"  planted lead={lead:.2f}: xcorr[+1]={p[1]:+.4f}  slope={bb['slope']:+.4f} "
          f"HAC_t={bb['t_hac']:>6.2f} | rule n={s['n']} cond={s['cond_mean']*100:+.4f}% "
          f"base={s['base_mean']*100:+.4f}% t={s['t']:>5.2f} p={s['p_placebo']:.3f}")
