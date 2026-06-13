"""Real-tape verification — Study 104 (Bollinger-Reversion).

Fetches (or reads from cache) the daily tapes, runs the lower-band entry vs a random-day
control and a breakout-entry control, sweeps costs, and reports the honest numbers that
flow into docs/results.md.  Network is touched only with --fetch.

    python studies/104-bollinger-reversion/examples/verify.py            # cache-only
    python studies/104-bollinger-reversion/examples/verify.py --fetch    # refresh tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bollinger_reversion import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "JPM", "XLE"]


def _pool(
    fetch: bool,
    side: str = "reversion",
    exit_mode: str = "mid",
    cost: float = 0.0,
    max_hold: int = 40,
    random_seed: int | None = None,
) -> pd.DataFrame:
    frames = []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=fetch)
        if random_seed is not None:
            ent = st.random_entries(bars, n=200, seed=random_seed)
        else:
            ent = st.band_entries(bars["close"], side=side)
        frames.append(st.run_trades(bars, ent, exit_mode=exit_mode, cost_bps=cost, max_hold=max_hold))
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} "
                  f"n={len(b)} fp={data.fingerprint(b)}")
        print()

    # --- main result: lower-band entry vs random-day control (mid exit) ---
    rev = st.summarize(_pool(False, "reversion", "mid", 0.0), "ret_gross")
    rand = st.summarize(_pool(False, random_seed=7), "ret_gross")
    print("=== lower-band entry vs random-day control (mid exit, gross) ===")
    print(f"REVERSION n={rev['n_trades']} win={rev['win_rate']:.3f} "
          f"mean={rev['mean_bps']:+.2f}bps t={rev['tstat']:+.2f}")
    print(f"RANDOM    n={rand['n_trades']} win={rand['win_rate']:.3f} "
          f"mean={rand['mean_bps']:+.2f}bps t={rand['tstat']:+.2f}")
    print(f"delta reversion-random = {rev['mean_bps'] - rand['mean_bps']:+.2f} bps")

    # --- breakout (opposite rule) ---
    brk = st.summarize(_pool(False, "breakout", "mid", 0.0), "ret_gross")
    print("\n=== upper-band breakout entry (mid exit, gross) ===")
    print(f"BREAKOUT  n={brk['n_trades']} win={brk['win_rate']:.3f} "
          f"mean={brk['mean_bps']:+.2f}bps t={brk['tstat']:+.2f}")

    # --- cost sweep ---
    print("\n=== cost sweep (lower-band entry, mid exit, net) ===")
    for c in (0.0, 1.0, 2.0, 5.0):
        s = st.summarize(_pool(False, "reversion", "mid", c), "ret_net")
        print(f"cost={c:4.1f}bps  net mean={s['mean_bps']:+.2f}bps  t={s['tstat']:+.2f}")

    # --- per-instrument breakdown ---
    print("\n=== per-instrument breakdown (lower-band entry, mid exit, gross) ===")
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=False)
        ent = st.band_entries(bars["close"], side="reversion")
        led = st.run_trades(bars, ent, exit_mode="mid", cost_bps=0.0)
        s = st.summarize(led, "ret_gross")
        print(f"{t:5s} n={s['n_trades']:4d} win={s['win_rate']:.3f} "
              f"mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    # --- time exit for comparison ---
    rev_time = st.summarize(_pool(False, "reversion", "time", 0.0, max_hold=20), "ret_gross")
    rand_time = st.summarize(_pool(False, random_seed=7), "ret_gross")
    print("\n=== lower-band entry, time exit (20-day hold), gross ===")
    print(f"REVERSION n={rev_time['n_trades']} mean={rev_time['mean_bps']:+.2f}bps "
          f"t={rev_time['tstat']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
