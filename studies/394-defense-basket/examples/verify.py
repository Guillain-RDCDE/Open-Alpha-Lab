"""Reproducible headline run for Study 394 — Defense-Basket.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached defense prices under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from defense_basket import data, strategy as st

print("# Defense-Basket — equal-weight LMT/RTX/NOC/GD vs SPY, around 20 geopolitical shocks")
if data.have_real():
    f = data.load_real()
    yrs = (f.index.max() - f.index.min()).days / 365.25
    ev = data.shock_dates()
    print(f"tape days      : {len(f)}  ({f.index.min().date()} -> {f.index.max().date()}, "
          f"{yrs:.1f} years)")
    print(f"shock dates    : {len(ev)} hardcoded war/invasion/rearmament headlines")

    print("\n# Event-window excess (basket - SPY) after each shock vs unconditional (1-day lag)")
    print(f"  {'W':>4} {'n':>3} {'cond_mean':>10} {'cond_win':>9} {'base_mean':>10} "
          f"{'base_win':>9} {'t':>7} {'t_hac':>7} {'p_placebo':>10}")
    for w in (5, 21, 63):
        s = st.summarize(f, ev, window=w)
        print(f"  {str(w)+'d':>4} {s['n']:>3} {s['cond_mean']*100:>9.2f}% "
              f"{s['cond_win']*100:>8.0f}% {s['base_mean']*100:>9.2f}% "
              f"{s['base_win']*100:>8.0f}% {s['t']:>7.2f} {s['t_hac']:>7.2f} "
              f"{s['p_placebo']:>10.3f}")

    print("\n# Per-event 21-day excess-over-SPY")
    ee = st.event_excess(f, ev, window=21)
    for d, x in zip(ev, ee):
        print(f"  {d.date()}  {x*100:>+7.2f}%")

    print("\n# Robustness — vary the holding window")
    for w in (5, 10, 21, 42, 63):
        s = st.summarize(f, ev, window=w)
        print(f"  W={w:>2}: n={s['n']:>2}  cond={s['cond_mean']*100:>6.2f}%  "
              f"t={s['t']:>5.2f}  t_hac={s['t_hac']:>5.2f}  p={s['p_placebo']:.3f}")

    print("\n# ITA sector-ETF cross-check (from 2006; 21-day excess)")
    f2 = f.copy()
    f2["etf_excess"] = f2["etf"] - f2["mkt"]
    ev_ita = pd.DatetimeIndex([d for d in ev if d >= pd.Timestamp("2006-05-05")])
    s = st.summarize(f2, ev_ita, window=21, col="etf_excess")
    print(f"  ITA: n={s['n']}  cond={s['cond_mean']*100:.2f}%  win={s['cond_win']*100:.0f}%  "
          f"base={s['base_mean']*100:.2f}%  t={s['t']:.2f}  p={s['p_placebo']:.3f}")

    print("\n# Net of one round-trip cost (buy-basket / short-SPY, hold W days)")
    for w in (5, 21, 63):
        c = st.net_of_costs(f, ev, window=w, cost_bps=10.0)
        print(f"  W={w:>2}d  n_trades={c['n_trades']:>2}  gross={c['gross_mean']*100:>6.2f}%  "
              f"net@10bps={c['net_mean']*100:>6.2f}%")
else:
    print("(no _cache/defense_prices.csv — run data.fetch_prices() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  engine must recover a PLANTED window edge and must NOT manufacture significance")
print("  from ~20 events when the true edge is 0.")
for edge in (0.0, 0.06):
    syn, ev_s = data.synthetic_defense(n_events=20, edge_window=edge, seed=394, window=21)
    s = st.summarize(syn, ev_s, window=21)
    print(f"  planted edge={edge:+.2f}/win: detected={s['n']:>2}  "
          f"cond={s['cond_mean']*100:>6.2f}%  base={s['base_mean']*100:>5.2f}%  "
          f"t={s['t']:>5.2f}  p_placebo={s['p_placebo']:.3f}")
