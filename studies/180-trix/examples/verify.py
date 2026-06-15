"""Real-tape verification — Study 180 (TRIX). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the daily tapes for the basket, runs the TRIX zero-line and
signal-line cross strategies against a random-direction control, sweeps holding windows and
costs, and prints the headline numbers that are frozen in docs/results.md.

    python studies/180-trix/examples/verify.py            # cache-only
    python studies/180-trix/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from trix import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT"]
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))


def _pool_zero(fetch: bool, hold: int, cost: float, seed: int | None = None):
    frames = []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=fetch, cache_dir=CACHE_DIR)
        tx = st.trix(bars["close"], period=15)
        entries = st.zero_line_entries(tx)
        dirs = st.random_directions(len(entries), seed=seed) if seed is not None else None
        frames.append(st.run_trades(bars, entries, hold_days=hold, cost_bps=cost, directions=dirs))
    return pd.concat(frames, ignore_index=True)


def _pool_signal(fetch: bool, hold: int, cost: float, seed: int | None = None):
    frames = []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=fetch, cache_dir=CACHE_DIR)
        tx = st.trix(bars["close"], period=15)
        sig = st.trix_signal(tx, signal_period=9)
        entries = st.signal_line_entries(tx, sig)
        dirs = st.random_directions(len(entries), seed=seed) if seed is not None else None
        frames.append(st.run_trades(bars, entries, hold_days=hold, cost_bps=cost, directions=dirs))
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    if fetch:
        print("=== Fetching / refreshing tapes ===")
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True, cache_dir=CACHE_DIR)
            print(f"{t:6s} {str(b.index[0].date())}..{str(b.index[-1].date())} "
                  f"n={len(b)} fp={data.fingerprint(b)}")
        print()

    print("=== TRIX(15) zero-line cross, gross (hold=20d) ===")
    cross_z = st.summarize(_pool_zero(False, 20, 0), "ret_gross")
    rand_z = st.summarize(_pool_zero(False, 20, 0, seed=180), "ret_gross")
    print(f"ZERO-CROSS  n={cross_z['n_trades']} win={cross_z['win_rate']:.3f} "
          f"mean={cross_z['mean_bps']:+.2f}bps t={cross_z['tstat']:+.2f}")
    print(f"RANDOM      n={rand_z['n_trades']} win={rand_z['win_rate']:.3f} "
          f"mean={rand_z['mean_bps']:+.2f}bps t={rand_z['tstat']:+.2f}")
    print(f"Delta (cross - random): {cross_z['mean_bps'] - rand_z['mean_bps']:+.2f} bps")

    print("\n=== TRIX(15) signal-line cross, gross (hold=10d) ===")
    cross_s = st.summarize(_pool_signal(False, 10, 0), "ret_gross")
    rand_s = st.summarize(_pool_signal(False, 10, 0, seed=180), "ret_gross")
    print(f"SIG-CROSS   n={cross_s['n_trades']} win={cross_s['win_rate']:.3f} "
          f"mean={cross_s['mean_bps']:+.2f}bps t={cross_s['tstat']:+.2f}")
    print(f"RANDOM      n={rand_s['n_trades']} win={rand_s['win_rate']:.3f} "
          f"mean={rand_s['mean_bps']:+.2f}bps t={rand_s['tstat']:+.2f}")

    print("\n=== Hold-period sweep (zero-line cross, gross) ===")
    for hold in (5, 10, 20, 40, 60):
        s = st.summarize(_pool_zero(False, hold, 0), "ret_gross")
        print(f"hold={hold:2d}d  n={s['n_trades']} mean={s['mean_bps']:+.2f}bps "
              f"t={s['tstat']:+.2f} win={s['win_rate']:.3f}")

    print("\n=== Cost sweep (zero-line cross, hold=20d) ===")
    for c in (0.0, 1.0, 2.0, 5.0):
        s = st.summarize(_pool_zero(False, 20, c), "ret_net")
        print(f"cost={c:4.1f}bps  net mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== Per-instrument breakdown (zero-line cross, hold=20d, gross) ===")
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=False, cache_dir=CACHE_DIR)
        tx = st.trix(bars["close"], period=15)
        entries = st.zero_line_entries(tx)
        s = st.summarize(st.run_trades(bars, entries, hold_days=20, cost_bps=0), "ret_gross")
        print(f"{t:6s} n={s['n_trades']:3d} mean={s['mean_bps']:+.2f}bps "
              f"t={s['tstat']:+.2f} win={s['win_rate']:.3f}")

    print("\n=== Period sensitivity (hold=20d, gross) ===")
    for period in (9, 15, 21, 30):
        frames = []
        for t in TICKERS:
            bars = data.fetch_daily(t, fetch=False, cache_dir=CACHE_DIR)
            tx = st.trix(bars["close"], period=period)
            entries = st.zero_line_entries(tx)
            frames.append(st.run_trades(bars, entries, hold_days=20, cost_bps=0))
        pooled = pd.concat(frames, ignore_index=True)
        s = st.summarize(pooled, "ret_gross")
        print(f"period={period:2d}  n={s['n_trades']} mean={s['mean_bps']:+.2f}bps "
              f"t={s['tstat']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
