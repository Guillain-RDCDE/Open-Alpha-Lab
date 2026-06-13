"""Real-tape verification — Study 106 (Supertrend). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the four daily tapes, runs the Supertrend(10, 3) flip
signal with symmetric ±1 ATR barriers against a random-direction control, sweeps costs,
and shows the fixed-day forward return at 5/20 day horizons.  Network only with --fetch.

    python studies/106-supertrend/examples/verify.py            # cache-only
    python studies/106-supertrend/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from supertrend import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM", "AAPL"]


def _pool(fetch: bool, tp: float, sl: float, cost: float, seed: int | None = None):
    frames = []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=fetch)
        ent = st.flip_entries(bars, atr_n=10, mult=3.0)
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(bars, ent, tp_R=tp, sl_R=sl, cost_bps=cost, directions=dirs))
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    if fetch:  # warm the cache + stamp fingerprints
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")
        print()

    cross = st.summarize(_pool(False, 1, 1, 0), "ret_gross")
    rand = st.summarize(_pool(False, 1, 1, 0, seed=106), "ret_gross")
    print("=== honest symmetric ±1 ATR (gross) ===")
    print(f"FLIP   n={cross['n_trades']} win={cross['win_rate']:.3f} "
          f"mean={cross['mean_bps']:+.2f}bps t={cross['tstat']:+.2f}")
    print(f"RANDOM n={rand['n_trades']} win={rand['win_rate']:.3f} "
          f"mean={rand['mean_bps']:+.2f}bps t={rand['tstat']:+.2f}")
    print(f"Delta (flip - random): {cross['mean_bps'] - rand['mean_bps']:+.2f} bps")

    print("\n=== per-instrument breakdown (flip, symmetric, gross) ===")
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=False)
        ent = st.flip_entries(bars, atr_n=10, mult=3.0)
        s = st.summarize(st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0), "ret_gross")
        print(f"  {t:5s}  n={s['n_trades']}  win={s['win_rate']:.3f}  "
              f"mean={s['mean_bps']:+.2f}bps  t={s['tstat']:+.2f}")

    print("\n=== cost sweep (flip arm, symmetric ±1 ATR, net) ===")
    for c in (0.0, 0.5, 1.0, 2.0, 5.0):
        s = st.summarize(_pool(False, 1, 1, c), "ret_net")
        print(f"cost={c:4.1f}bps net mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== fixed-day forward returns (d5 / d20, flip gross) ===")
    for hd in (5, 20):
        frames = []
        for t in TICKERS:
            bars = data.fetch_daily(t, fetch=False)
            ent = st.flip_entries(bars, atr_n=10, mult=3.0)
            frames.append(st.run_forward_returns(bars, ent, hold_days=hd, cost_bps=0))
        led = pd.concat(frames, ignore_index=True)
        s = st.summarize(led, "ret_gross")
        print(f"  d{hd:02d}: n={s['n_trades']} win={s['win_rate']:.3f} "
              f"mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== data fingerprints ===")
    for t in TICKERS:
        b = data.fetch_daily(t, fetch=False)
        print(f"  {t:5s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
