"""Real-tape verification — Study 236 (Fifty-Two-Week-High). Regenerates docs/results.md numbers.

Fetches (or reads from cache) daily bars for the S&P 500 representative basket,
computes the 52-week-high proximity signal (George & Hwang 2004), runs the quintile
sort and the long-only top-decile backtest vs equal-weight baseline, and reports
honest HAC inference.  Network is touched only with --fetch.

    python studies/236-fifty-two-week-high/examples/verify.py            # cache-only
    python studies/236-fifty-two-week-high/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fifty_two_week_high import data, strategy as st  # noqa: E402

CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))
TICKERS = data.SP500_BASKET


def main(fetch: bool) -> None:
    if fetch:
        print("=== Fetching real data ===")
        for t in TICKERS:
            try:
                bars = data.fetch_daily(t, period="15y", fetch=True, cache_dir=CACHE)
                print(f"{t:5s} {bars.index[0].date()}..{bars.index[-1].date()} "
                      f"n={len(bars)} fp={data.fingerprint(bars[['close']])}")
            except Exception as e:
                print(f"{t:5s} ERROR: {e}")
        print()

    # Load panel
    avail = [t for t in TICKERS if os.path.exists(data._cache_path(t, CACHE))]
    if len(avail) < 5:
        print(f"Only {len(avail)} tickers cached — run with --fetch first.")
        sys.exit(1)

    print(f"Tickers available: {len(avail)} / {len(TICKERS)}")
    panel = data.load_panel(avail, cache_dir=CACHE)
    print(f"Panel: {panel.shape[0]} days x {panel.shape[1]} tickers "
          f"({panel.index[0].date()} to {panel.index[-1].date()})")
    print(f"Fingerprint: {data.fingerprint(panel)}\n")

    prox = st.proximity_to_52w_high(panel, window=252)

    print("=== Quintile sort — 5-day hold, rebalanced weekly ===")
    for hold in (5, 20, 65):
        fwd = st.forward_returns(panel, hold_days=hold)
        q = st.quintile_backtest(prox, fwd, n_quintiles=5, rebalance_freq=5)
        spread = q["Q5_minus_Q1"].dropna()
        s = st.summarize(spread)
        q1s = st.summarize(q["Q1"].dropna())
        q5s = st.summarize(q["Q5"].dropna())
        print(f"\n  hold={hold:3d}d | Q5-Q1 spread: {s['mean_bps']:+.2f}bps  "
              f"t={s['tstat']:+.2f}  n={s['n']}")
        print(f"           Q1 (far-from-high): {q1s['mean_bps']:+.2f}bps  t={q1s['tstat']:+.2f}")
        print(f"           Q5 (near-high):     {q5s['mean_bps']:+.2f}bps  t={q5s['tstat']:+.2f}")

    print("\n=== Long-top-decile vs equal-weight baseline (5-day hold) ===")
    fwd5 = st.forward_returns(panel, hold_days=5)
    strat, base = st.long_top_decile(prox, fwd5, decile_frac=0.10, rebalance_freq=5)
    s_strat = st.summarize(strat)
    s_base = st.summarize(base)
    spread_lb = strat - base
    s_spread = st.summarize(spread_lb)
    print(f"  Top decile:  {s_strat['mean_bps']:+.2f}bps/period  "
          f"ann={s_strat['ann_pct']:+.1f}%  t={s_strat['tstat']:+.2f}")
    print(f"  EW baseline: {s_base['mean_bps']:+.2f}bps/period   "
          f"ann={s_base['ann_pct']:+.1f}%  t={s_base['tstat']:+.2f}")
    print(f"  Spread:      {s_spread['mean_bps']:+.2f}bps/period  t={s_spread['tstat']:+.2f}")

    print("\n=== Hold-period sweep (Q5-Q1 spread) ===")
    print(f"{'hold':>6} | {'spread (bps)':>13} {'HAC t':>7} {'n':>6}")
    print("-" * 38)
    for hold in (1, 5, 10, 20, 65):
        fwd = st.forward_returns(panel, hold_days=hold)
        q = st.quintile_backtest(prox, fwd, n_quintiles=5, rebalance_freq=5)
        spread = q["Q5_minus_Q1"].dropna()
        s = st.summarize(spread)
        print(f"{hold:>6}d | {s['mean_bps']:>+13.2f} {s['tstat']:>+7.2f} {s['n']:>6}")

    print("\n=== Per-quintile summary (5d hold) ===")
    fwd5 = st.forward_returns(panel, hold_days=5)
    q_all = st.quintile_backtest(prox, fwd5, n_quintiles=5, rebalance_freq=5)
    for i in range(1, 6):
        s = st.summarize(q_all[f"Q{i}"].dropna())
        print(f"  Q{i}: {s['mean_bps']:+.2f}bps  t={s['tstat']:+.2f}  ann={s['ann_pct']:+.1f}%")
    spread_all = q_all["Q5_minus_Q1"].dropna()
    s_all = st.summarize(spread_all)
    print(f"  Q5-Q1 spread: n={s_all['n']}  mean={s_all['mean_bps']:+.2f}bps  t={s_all['tstat']:+.2f}")

    print("\n=== VERDICT ===")
    t_main = s_all["tstat"]
    if abs(t_main) >= 2.0 and s_all["mean_bps"] > 0:
        sig = "REAL (positive, momentum direction)"
    elif abs(t_main) >= 2.0:
        sig = "REAL (wrong direction)"
    elif abs(t_main) >= 1.3:
        sig = "WEAK"
    else:
        sig = "NONE"
    print(f"Signal: {sig} (HAC t = {t_main:+.2f})")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
