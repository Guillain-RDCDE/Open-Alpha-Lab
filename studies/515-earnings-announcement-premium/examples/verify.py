"""Reproducible headline run for Study 515 — Earnings-Announcement Premium.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached basket prices + earnings dates
under ``_cache/`` if present (the real-tape numbers), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from earnings_premium import data, strategy as st

NDRAWS = 20_000

print("# Earnings-Announcement Premium — fixed 30-name large-cap basket (yfinance)")
if data.have_real():
    prices, dates = data.load_real()
    flags = data.build_announce_flags(prices, dates, window=data.WINDOW)
    span = (prices.index.max() - prices.index.min()).days / 365.25
    n_names = len(flags)
    n_dates = len(dates[dates["ticker"].isin(flags.keys())])
    frac = st.announce_day_fraction(flags)
    print(f"prices         : {prices.shape[0]} days x {prices.shape[1]} names, "
          f"{prices.index.min().date()} -> {prices.index.max().date()} ({span:.1f} years)")
    print(f"earnings dates : {n_dates}  across {n_names} names "
          f"(window = [D-{data.WINDOW}, D+{data.WINDOW}] trading days)")
    print(f"announce-day % : {frac*100:.2f}% of all stock-days are in an announcement window")

    print("\n# Per-day premium — announce-window days vs the rest")
    s = st.summarize_premium(flags, n_draws=NDRAWS)
    print(f"  announce mean : {s['ann_mean_bps']:+.2f} bps/day  (n={s['n_ann']:,})")
    print(f"  rest mean     : {s['rest_mean_bps']:+.2f} bps/day  (n={s['n_rest']:,})")
    print(f"  premium       : {s['premium_bps']:+.2f} bps/day   Welch t = {s['welch_t']:+.2f}")
    print(f"  per-EVENT prem: {s['event_premium_bps']:+.2f} bps/day  one-sample t = "
          f"{s['t_event']:+.2f}  (n_events={s['n_events']:,})")
    print(f"  placebo p     : {s['p_placebo']:.4f}  (random same-density calendar tags)")

    print("\n# Share-of-return decomposition")
    sr = st.share_of_return(flags)
    print(f"  {sr['share_days']*100:.2f}% of days are announce days; they carry "
          f"{sr['share_return']*100:.2f}% of cumulative log return")
    print(f"  concentration ratio = {sr['share_return']/sr['share_days']:.2f}x "
          f"(share of return / share of days)")

    print("\n# Tradable overlay — long the basket ONLY in announcement windows")
    for cb in (2.0, 5.0, 10.0):
        ov = st.overlay_stats(flags, cost_bps=cb, window=data.WINDOW)
        print(f"  cost={cb:>4.1f} bps/leg: gross {ov['gross_daily_bps']:+.2f} bps/day -> "
              f"net {ov['net_daily_bps']:+.2f} bps/day | buy&hold {ov['bh_daily_bps']:+.2f} bps/day"
              f"  (net ann {ov['net_ann_pct']:+.1f}% vs B&H {ov['bh_ann_pct']:+.1f}%)")

    print("\n# Robustness — announcement window half-width")
    for w in (0, 1, 2):
        fl = data.build_announce_flags(prices, dates, window=w)
        s2 = st.summarize_premium(fl, placebo=False)
        print(f"  window=[D-{w},D+{w}]: premium {s2['premium_bps']:+.2f} bps/day  "
              f"Welch t={s2['welch_t']:+.2f}  per-event t={s2['t_event']:+.2f}  "
              f"(ann-day share {st.announce_day_fraction(fl)*100:.1f}%)")
else:
    print("(no _cache/ep_prices.csv — run data.fetch_panel() once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the announce-vs-rest detector must recover a PLANTED premium and must NOT manufacture")
print("  significance when the true premium is 0.")
for edge in (0.0, 0.0015):
    px, dt = data.synthetic_panel(edge=edge, seed=515)
    fl = data.build_announce_flags(px, dt, window=data.WINDOW)
    s = st.summarize_premium(fl, n_draws=4_000)
    print(f"  planted edge={edge*1e4:+6.1f} bps/day: premium={s['premium_bps']:+.2f} bps/day  "
          f"Welch t={s['welch_t']:+6.2f}  per-event t={s['t_event']:+6.2f}  "
          f"placebo p={s['p_placebo']:.3f}")
