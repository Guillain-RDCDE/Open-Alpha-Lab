"""Reproducible headline run for Study 383 — SOFR-Repo-Stress.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached risk-asset prices under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sofr_repo_stress import data, strategy as st

print("# SOFR-Repo-Stress — hardcoded repo-stress episodes vs SPY / HYG / LQD")
if data.have_real():
    F = data.load_real()
    span_years = (F.index.max() - F.index.min()).days / 365.25
    ev = data.episode_dates()
    print(f"risk-asset days : {len(F)}  ({F.index.min().date()} -> {F.index.max().date()}, "
          f"{span_years:.1f} years)")
    print(f"episodes        : {len(ev)}  (hardcoded, sourced repo/funding-stress dates)")

    print("\n# Per-episode SPY 20-day forward return (1-day entry lag)")
    tab = data.episode_table()
    for d, row in tab.iterrows():
        r = st.event_returns(F["SPY"], pd.DatetimeIndex([d]), 20)
        rv = r[0] * 100 if len(r) else float("nan")
        print(f"  {d.date()}  {row['tag']:<20} {rv:+6.2f}%")

    print("\n# Forward returns after a repo spike vs unconditional base rate (1-day lag)")
    for asset in data.RISK_ASSETS:
        print(f"  -- {asset} --")
        print(f"     {'H':>4} {'n':>3} {'cond_mean':>10} {'cond_dn':>8} {'base_mean':>10} "
              f"{'base_dn':>8} {'Welch_t':>8} {'p_placebo':>10}")
        for h in (5, 20, 60):
            s = st.summarize(F[asset], ev, h)
            print(f"     {str(h)+'d':>4} {s['n']:>3} {s['cond_mean']*100:>9.2f}% "
                  f"{s['cond_down']*100:>7.0f}% {s['base_mean']*100:>9.2f}% "
                  f"{s['base_down']*100:>7.0f}% {s['t']:>8.2f} {s['p_placebo']:>10.3f}")

    print("\n# 'De-risk on the spike' rule — avoided loss net of one round-trip cost (SPY)")
    for h in (5, 20, 60):
        c = st.net_of_costs(F["SPY"], ev, h, cost_bps=5.0)
        print(f"  {h:>2}d  n_trades={c['n_trades']:>2}  gross_avoided={c['gross_mean']*100:>6.2f}%  "
              f"net@5bps={c['net_mean']*100:>6.2f}%")

    print("\n# Robustness — systemic-only subset (drop the routine quarter/year-end turns)")
    sys_dates = pd.DatetimeIndex([d for d, r in tab.iterrows() if r["tag"] in data.SYSTEMIC_ONLY])
    for h in (5, 20, 60):
        s = st.summarize(F["SPY"], sys_dates, h)
        print(f"  systemic SPY {h:>2}d: n={s['n']}  cond={s['cond_mean']*100:>6.2f}%  "
              f"base={s['base_mean']*100:>5.2f}%  t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
else:
    print("(no _cache/risk_assets.csv — run data.fetch_risk_assets() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  inference must recover a PLANTED bearish edge and must NOT manufacture")
print("  significance from ~a dozen events when the true edge is 0.")
for edge in (0.0, -0.08):
    syn = data.synthetic_tape(n_events=13, edge_20d=edge, seed=383)
    s = st.summarize(syn["prices"]["SPY"], syn["events"], 20)
    print(f"  planted edge_20d={edge:+.2f}: n={s['n']:>2}  "
          f"cond20={s['cond_mean']*100:>6.2f}%  base20={s['base_mean']*100:>5.2f}%  "
          f"t={s['t']:>5.2f}  p_placebo={s['p_placebo']:.3f}")
