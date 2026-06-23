"""Reproducible headline run for Study 369 — Earnings-Revision-Momentum.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached basket prices + earnings table
under ``_cache/`` if present (the real-tape numbers), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from earnings_revision_momentum import data, strategy as st

print("# Earnings-Revision-Momentum — revision proxy (surprise + est. change) on a 40-name basket")
if data.have_real():
    prices, earn = data.load_real()
    print(f"price tape     : {prices.index.min().date()} -> {prices.index.max().date()} "
          f"({len(prices)} days)")
    print(f"earnings events: {len(earn)}  ({earn['ticker'].nunique()} names, "
          f"{earn['date'].min().date()} -> {earn['date'].max().date()})")

    print("\n# Long-short tercile spread (excess of SPY), 1-day entry lag, by horizon")
    print(f"  {'H':>5} {'n_ev':>5} {'n_q':>4} {'long_ex':>8} {'short_ex':>9} "
          f"{'spread':>8} {'t':>6} {'p_plac':>7} {'hit':>5}")
    for h in (21, 63, 126):
        s = st.summarize(prices, earn, horizon=h)
        print(f"  {str(h)+'d':>5} {s['n_events']:>5} {s['n_quarters']:>4} "
              f"{s['long_ex_mean']*100:>7.2f}% {s['short_ex_mean']*100:>8.2f}% "
              f"{s['spread_mean']*100:>7.2f}% {s['spread_t']:>6.2f} "
              f"{s['p_placebo']:>7.3f} {s['long_hit_rate']*100:>4.0f}%")

    print("\n# Net of one-way costs (10 bps/leg) + short borrow (50 bps/yr)")
    for h in (21, 63, 126):
        c = st.net_of_costs(prices, earn, horizon=h)
        print(f"  {h:>3}d  gross={c['gross_mean']*100:>6.2f}% (t={c['gross_t']:>5.2f})  "
              f"net={c['net_mean']*100:>6.2f}% (t={c['net_t']:>5.2f})")

    print("\n# Robustness — tercile cut at 63d")
    for q in (0.20, 1 / 3, 0.40):
        s = st.summarize(prices, earn, horizon=63, q=q)
        print(f"  cut={q:.2f}: n_q={s['n_quarters']:>2}  spread={s['spread_mean']*100:>6.2f}%  "
              f"t={s['spread_t']:>5.2f}  p={s['p_placebo']:.3f}")
else:
    print("(no _cache/ — run data.fetch_prices() and data.fetch_earnings() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  engine must NOT fabricate a spread at edge=0 and MUST light up for a planted edge.")
for edge in (0.0, 0.02, 0.05):
    pan = data.synthetic_panel(edge=edge, seed=369)
    s = st.synthetic_spread(pan)
    p = st.placebo_pvalue_synth(pan)
    print(f"  planted edge={edge:.2f}: n_q={s['n_quarters']:>2}  "
          f"spread={s['spread_mean']*100:>6.2f}%  t={s['spread_t']:>6.2f}  p_placebo={p:.4f}")
