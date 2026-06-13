"""Real-tape verification — Study 103 (Turtle-Trader). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the daily tapes for the trend-following basket, runs both
Turtle System 1 (20-day entry / 10-day exit) and System 2 (55-day / 20-day), in both
directions (long + short), vs a random-entry control and vs buy-and-hold. Also computes
the pre/post-2003 split (rules published by Curtis Faith, 2003). Network is touched only
with --fetch.

    python studies/103-turtle-trader/examples/verify.py            # cache-only
    python studies/103-turtle-trader/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from turtle_trader import data, strategy as st  # noqa: E402

TICKERS = data.DEFAULT_TICKERS
SPLIT   = "2003-01-01"   # publication date: Curtis Faith's full Turtle rules


def _pool(fetch: bool, entry_n: int, exit_n: int, cost: float,
          direction: str = "both", start_date: str | None = None,
          end_date: str | None = None, random_seed: int | None = None,
          max_hold: int = 252) -> pd.DataFrame:
    """Pool trades across the basket for one set of parameters."""
    frames = []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=fetch)
        if start_date:
            bars = bars.loc[bars.index >= start_date]
        if end_date:
            bars = bars.loc[bars.index < end_date]
        if len(bars) < entry_n + 10:
            continue
        sig  = st.donchian_signals(bars, entry_n=entry_n, exit_n=exit_n,
                                   direction=direction)
        led  = st.run_trades(bars, sig, cost_bps=cost, max_hold=max_hold)
        if random_seed is not None and len(led) > 0:
            pairs = st.random_entry_pairs(len(led), len(bars),
                                          entry_n=entry_n, seed=random_seed)
            led = st.run_trades(bars, sig, cost_bps=cost,
                                random_entry_dates=pairs, max_hold=max_hold)
        frames.append(led)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    if fetch:
        print("Fetching / refreshing daily tapes...\n")
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True)
            print(f"  {t:5s}  {b.index[0].date()}..{b.index[-1].date()}  "
                  f"n={len(b)}  fp={data.fingerprint(b)}")
        print()

    # --- System 1: 20/10, both directions -----------------------------------
    print("=== System 1 (20-day entry / 10-day exit, both directions) ===")
    s1_cross = st.summarize(_pool(False, 20, 10, 0.0), "ret_gross")
    s1_rand  = st.summarize(_pool(False, 20, 10, 0.0, random_seed=103), "ret_gross")
    print(f"  BREAKOUT  n={s1_cross['n_trades']}  win={s1_cross['win_rate']:.3f}  "
          f"mean={s1_cross['mean_bps']:+.1f}bps  t={s1_cross['tstat']:+.2f}")
    print(f"  RANDOM    n={s1_rand['n_trades']}  win={s1_rand['win_rate']:.3f}  "
          f"mean={s1_rand['mean_bps']:+.1f}bps  t={s1_rand['tstat']:+.2f}")

    # --- System 2: 55/20, both directions -----------------------------------
    print()
    print("=== System 2 (55-day entry / 20-day exit, both directions) ===")
    s2_cross = st.summarize(_pool(False, 55, 20, 0.0, max_hold=365), "ret_gross")
    s2_rand  = st.summarize(_pool(False, 55, 20, 0.0, max_hold=365, random_seed=103), "ret_gross")
    print(f"  BREAKOUT  n={s2_cross['n_trades']}  win={s2_cross['win_rate']:.3f}  "
          f"mean={s2_cross['mean_bps']:+.1f}bps  t={s2_cross['tstat']:+.2f}")
    print(f"  RANDOM    n={s2_rand['n_trades']}  win={s2_rand['win_rate']:.3f}  "
          f"mean={s2_rand['mean_bps']:+.1f}bps  t={s2_rand['tstat']:+.2f}")

    # --- Cost sweep (System 1) ----------------------------------------------
    print()
    print("=== Cost sweep — System 1, net ===")
    for c in (0.0, 5.0, 10.0, 20.0):
        s = st.summarize(_pool(False, 20, 10, c), "ret_net")
        print(f"  cost={c:4.1f}bps  net mean={s['mean_bps']:+.2f}bps  t={s['tstat']:+.2f}")

    # --- Pre/post-publication split -----------------------------------------
    print()
    print("=== Pre/post-2003 split (System 1) ===")
    pre_led  = _pool(False, 20, 10, 0.0, end_date=SPLIT)
    post_led = _pool(False, 20, 10, 0.0, start_date=SPLIT)
    pre  = st.summarize(pre_led,  "ret_gross")
    post = st.summarize(post_led, "ret_gross")
    print(f"  PRE-2003  n={pre['n_trades']}  mean={pre['mean_bps']:+.1f}bps  "
          f"win={pre['win_rate']:.3f}  t={pre['tstat']:+.2f}")
    print(f"  POST-2003 n={post['n_trades']}  mean={post['mean_bps']:+.1f}bps  "
          f"win={post['win_rate']:.3f}  t={post['tstat']:+.2f}")

    # --- Long-only vs short-only --------------------------------------------
    print()
    print("=== Direction split — System 1, gross ===")
    s_long  = st.summarize(_pool(False, 20, 10, 0.0, direction="long"),  "ret_gross")
    s_short = st.summarize(_pool(False, 20, 10, 0.0, direction="short"), "ret_gross")
    print(f"  LONG   n={s_long['n_trades']}  mean={s_long['mean_bps']:+.1f}bps  "
          f"win={s_long['win_rate']:.3f}  t={s_long['tstat']:+.2f}")
    print(f"  SHORT  n={s_short['n_trades']}  mean={s_short['mean_bps']:+.1f}bps  "
          f"win={s_short['win_rate']:.3f}  t={s_short['tstat']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
