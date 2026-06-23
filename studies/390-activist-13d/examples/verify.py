"""Reproducible headline run for Study 390 — Activist-13D.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached event prices under ``_cache/``
if present (the real-tape numbers from the hardcoded 13D table), and always runs the
synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from activist_13d import data, strategy as st

print("# Activist-13D — hardcoded table of famous activist campaigns + SPY (yfinance)")
if data.have_real():
    prices, tbl = data.load_real()
    span_years = (prices.index.max() - prices.index.min()).days / 365.25
    print(f"price days     : {len(prices)}  ({prices.index.min().date()} -> "
          f"{prices.index.max().date()}, {span_years:.1f} years)")
    print(f"events (priced): {len(tbl)} activist 13D campaigns (SELECTED famous basket)")

    pops = st.announcement_pops(prices, tbl)
    print("\n# Announcement pop (day-0 return — the news being priced in; NOT tradable)")
    print(f"  n={len(pops)}  mean={pops.mean()*100:+.2f}%  median={np.median(pops)*100:+.2f}%"
          f"  win={(pops>0).mean()*100:.0f}%  t={st.welch_t_vs_zero(pops):+.2f}")

    print("\n# Post-announcement excess drift vs SPY (buy 1 day after, hold H months)")
    print(f"  {'H':>4} {'n':>3} {'mean_ex':>9} {'median_ex':>10} {'win':>5} {'Welch_t':>8} {'p_placebo':>10}")
    for m in (1, 3, 6):
        s = st.summarize(prices, tbl, m)
        print(f"  {str(m)+'m':>4} {s['n']:>3} {s['mean_excess']*100:>8.2f}% "
              f"{s['median_excess']*100:>9.2f}% {s['win_rate']*100:>4.0f}% "
              f"{s['t']:>8.2f} {s['p_placebo']:>10.3f}")

    print("\n# Net of one round-trip cost (buy +1d, hold H months, excess vs SPY)")
    for m in (1, 3, 6):
        c = st.net_of_costs(prices, tbl, m, cost_bps=10.0)
        print(f"  {m:>2}m  n_trades={c['n_trades']:>2}  gross={c['gross_mean']*100:>6.2f}%  "
              f"net@10bps={c['net_mean']*100:>6.2f}%")

    print("\n# Robustness — entry-lag sensitivity (6-month excess drift)")
    for lag in (0, 1, 2, 5):
        s = st.summarize(prices, tbl, 6, lag=lag)
        print(f"  lag={lag}: n={s['n']:>2}  mean={s['mean_excess']*100:>5.2f}%  "
              f"t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")

    print("\n# Robustness — drop the 3 mega-caps (AAPL, MSFT, DIS), 6-month drift")
    sub = tbl[~tbl["ticker"].isin(["AAPL", "MSFT", "DIS"])].reset_index(drop=True)
    s = st.summarize(prices, sub, 6)
    print(f"  n={s['n']:>2}  mean={s['mean_excess']*100:>5.2f}%  t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
else:
    print("(no _cache/event_prices.csv — run data.fetch_events() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  engine must recover a PLANTED drift edge and must NOT manufacture significance")
print("  from ~25 events when the true post-announcement drift is 0.")
for edge in (0.0, 0.20):
    p, t = data.synthetic_events(n_events=25, drift_6m=edge, seed=390)
    s6 = st.summarize(p, t, 6)
    print(f"  planted drift_6m={edge:+.2f}: detected={s6['n']:>2}  "
          f"mean={s6['mean_excess']*100:>6.2f}%  win={s6['win_rate']*100:>3.0f}%  "
          f"t={s6['t']:>5.2f}  p_placebo={s6['p_placebo']:.3f}")
