"""Real-tape verification -- Study 74 (Mind-the-Gap). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the five daily tapes, computes conditional fill rates by
gap-size bucket, runs the symmetric gap-fade barrier backtest against a random-direction
control, sweeps costs, and prints the decisive numbers.

    python studies/74-mind-the-gap/examples/verify.py            # cache-only
    python studies/74-mind-the-gap/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mind_the_gap import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "AAPL", "MSFT", "TSLA", "NVDA"]


def _pool(fetch: bool, tp_R: float, stop_R: float, cost: float, seed: int | None = None):
    frames = []
    for t in TICKERS:
        df = data.fetch_daily(t, fetch=fetch)
        if seed is not None:
            n = len(st.run_trades(df, min_gap_pct=0.0, tp_R=tp_R, stop_R=stop_R, cost_bps=0))
            dirs = st.random_directions(n, seed=seed)
        else:
            dirs = None
        frames.append(
            st.run_trades(df, min_gap_pct=0.0, tp_R=tp_R, stop_R=stop_R, cost_bps=cost,
                          directions=dirs)
        )
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} "
                  f"n={len(b)} fp={data.fingerprint(b)}")
        print()

    # --- Fill rates ---
    frames = [data.fetch_daily(t, fetch=False) for t in TICKERS]
    pooled_daily = pd.concat(frames)
    tbl = st.fill_rate_by_bucket(pooled_daily)
    print("=== Fill-rate by gap-size bucket (pooled, 5 tickers) ===")
    print(tbl.to_string(index=False))
    print(f"Overall fill rate: {pooled_daily['gap_fill'].mean():.4f}  "
          f"(N = {len(pooled_daily[pooled_daily['gap_pct'].abs() > 0])} sessions)")

    # --- Main test: symmetric 1:1 ATR, gross ---
    cross = st.summarize(_pool(False, 1.0, 1.0, 0), "ret_gross")
    rand  = st.summarize(_pool(False, 1.0, 1.0, 0, seed=74), "ret_gross")
    print("\n=== Honest symmetric 1:1 ATR (gross) ===")
    print(f"FADE  n={cross['n_trades']} win={cross['win_rate']:.3f} "
          f"mean={cross['mean_bps']:+.2f}bps t={cross['tstat']:+.2f}")
    print(f"RAND  n={rand['n_trades']} win={rand['win_rate']:.3f} "
          f"mean={rand['mean_bps']:+.2f}bps t={rand['tstat']:+.2f}")
    print(f"Fade - Rand = {cross['mean_bps'] - rand['mean_bps']:+.2f} bps")

    # --- Per instrument ---
    print("\n=== Per-instrument (all gaps, 1:1 ATR, gross) ===")
    for t in TICKERS:
        df = data.fetch_daily(t, fetch=False)
        led = st.run_trades(df, min_gap_pct=0.0, tp_R=1.0, stop_R=1.0, cost_bps=0)
        s = st.summarize(led, "ret_gross")
        print(f"{t:5s}: n={s['n_trades']:4d} win={s['win_rate']:.3f} "
              f"mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    # --- Cost sweep ---
    print("\n=== Cost sweep (all gaps, 1:1 ATR) ===")
    for c in (0.0, 0.5, 1.0, 2.0):
        s = st.summarize(_pool(False, 1.0, 1.0, c), "ret_net")
        print(f"cost={c:4.1f}bps  net mean={s['mean_bps']:+.2f}bps  t={s['tstat']:+.2f}")

    # --- Data fingerprints ---
    print("\n=== Data stamps ===")
    for t in TICKERS:
        df = data.fetch_daily(t, fetch=False)
        print(f"{t:5s}: {df.index[0].date()}..{df.index[-1].date()} "
              f"n={len(df)} fp={data.fingerprint(df)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
