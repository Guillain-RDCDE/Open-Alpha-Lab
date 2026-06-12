"""Real-tape verification — Study 77 (Golden-Mean). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the SPY and QQQ daily tapes, identifies swings, finds
Fibonacci level touches and placebo-level touches, and compares bounce rates and
forward returns. Also tests round-number levels. Network is touched only with --fetch.

    python studies/77-golden-mean/examples/verify.py            # cache-only
    python studies/77-golden-mean/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from golden_mean import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "NVDA"]


def _pool(fetch: bool, use_placebo: bool, fwd_bars: int = 5, cost: float = 1.0):
    frames = []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=fetch)
        swings = st.find_swings(bars, lookback=60, min_swing_pct=0.05)
        if swings.empty:
            continue
        frames.append(
            st.run_bounces(bars, swings, fwd_bars=fwd_bars, cost_bps=cost,
                           use_placebo=use_placebo)
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _pool_rn(fetch: bool, use_placebo: bool, fwd_bars: int = 5, cost: float = 1.0):
    frames = []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=fetch)
        frames.append(
            st.run_round_number_bounces(bars, fwd_bars=fwd_bars, cost_bps=cost,
                                        use_placebo=use_placebo)
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} "
                  f"n={len(b)} fp={data.fingerprint(b)}")
        print()

    print("=== Fibonacci retracement levels vs placebo (gross, 5-day forward) ===")
    fib_l = _pool(False, use_placebo=False, cost=0.0)
    plac_l = _pool(False, use_placebo=True, cost=0.0)
    fib_s = st.summarize(fib_l, "ret_gross")
    plac_s = st.summarize(plac_l, "ret_gross")
    print(f"FIB   n={fib_s['n_touches']} bounce={fib_s['bounce_rate']:.3f} "
          f"mean={fib_s['mean_bps']:+.2f}bps t={fib_s['tstat']:+.2f}")
    print(f"PLAC  n={plac_s['n_touches']} bounce={plac_s['bounce_rate']:.3f} "
          f"mean={plac_s['mean_bps']:+.2f}bps t={plac_s['tstat']:+.2f}")
    delta = fib_s['mean_bps'] - plac_s['mean_bps']
    print(f"Fib - placebo: {delta:+.2f} bps")

    print("\n=== Round numbers vs mid-points (gross, 5-day forward) ===")
    rn_l = _pool_rn(False, use_placebo=False, cost=0.0)
    rn_plac_l = _pool_rn(False, use_placebo=True, cost=0.0)
    rn_s = st.summarize(rn_l, "ret_gross")
    rn_plac_s = st.summarize(rn_plac_l, "ret_gross")
    print(f"RN    n={rn_s['n_touches']} bounce={rn_s['bounce_rate']:.3f} "
          f"mean={rn_s['mean_bps']:+.2f}bps t={rn_s['tstat']:+.2f}")
    print(f"MID   n={rn_plac_s['n_touches']} bounce={rn_plac_s['bounce_rate']:.3f} "
          f"mean={rn_plac_s['mean_bps']:+.2f}bps t={rn_plac_s['tstat']:+.2f}")

    print("\n=== Fibonacci cost sweep (net, 5-day forward) ===")
    for c in (0.0, 1.0, 5.0):
        l = _pool(False, use_placebo=False, cost=c)
        s = st.summarize(l, "ret_net")
        print(f"cost={c:4.1f}bps net mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== Per-instrument Fibonacci (gross, 5-day) ===")
    for t in TICKERS:
        try:
            bars = data.fetch_daily(t, fetch=False)
            swings = st.find_swings(bars, lookback=60, min_swing_pct=0.05)
            if swings.empty:
                print(f"{t:5s} no swings found")
                continue
            l_fib = st.run_bounces(bars, swings, fwd_bars=5, cost_bps=0.0)
            l_plac = st.run_bounces(bars, swings, fwd_bars=5, cost_bps=0.0,
                                    use_placebo=True)
            sf = st.summarize(l_fib, "ret_gross")
            sp = st.summarize(l_plac, "ret_gross")
            print(f"{t:5s} fib n={sf['n_touches']} mean={sf['mean_bps']:+.2f}bps "
                  f"t={sf['tstat']:+.2f}  |  plac n={sp['n_touches']} "
                  f"mean={sp['mean_bps']:+.2f}bps t={sp['tstat']:+.2f}")
        except FileNotFoundError:
            print(f"{t:5s} cache not found (run with --fetch)")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
