"""Real-tape verification — Study 87 (Center-Line). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the eight 5-minute tapes, runs the VWAP-fade with
symmetric ±1 ATR barriers against a random-direction control, sweeps costs and thresholds,
and shows the full honest verdict. Network is touched only with --fetch.

    python studies/87-center-line/examples/verify.py            # cache-only
    python studies/87-center-line/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from center_line import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA", "ES=F", "NQ=F"]


def _pool(fetch: bool, tp: float, sl: float, cost: float,
          seed: int | None = None, threshold: float = 1.0):
    frames = []
    for t in TICKERS:
        bars = data.fetch_5m(t, fetch=fetch)
        ent = st.vwap_fade_entries(bars, threshold=threshold)
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(
            st.run_trades(bars, ent, tp_R=tp, sl_R=sl, cost_bps=cost, directions=dirs)
        )
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    if fetch:  # warm the cache and stamp fingerprints
        for t in TICKERS:
            b = data.fetch_5m(t, fetch=True)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")
        print()

    print("=== data fingerprints (cache) ===")
    for t in TICKERS:
        b = data.fetch_5m(t, fetch=False)
        print(f"  {t:5s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")

    print("\n=== honest symmetric ±1 ATR (gross, threshold=1.0) ===")
    fade = st.summarize(_pool(False, 1, 1, 0), "ret_gross")
    rand = st.summarize(_pool(False, 1, 1, 0, seed=87), "ret_gross")
    print(f"FADE  n={fade['n_trades']} win={fade['win_rate']*100:.1f}% "
          f"mean={fade['mean_bps']:+.2f}bps t={fade['tstat']:+.2f}")
    print(f"RAND  n={rand['n_trades']} win={rand['win_rate']*100:.1f}% "
          f"mean={rand['mean_bps']:+.2f}bps t={rand['tstat']:+.2f}")
    print(f"DELTA fade−rand = {fade['mean_bps']-rand['mean_bps']:+.2f} bps")

    print("\n=== per-instrument breakdown ===")
    for t in TICKERS:
        b = data.fetch_5m(t, fetch=False)
        ent = st.vwap_fade_entries(b, threshold=1.0)
        s = st.summarize(st.run_trades(b, ent, tp_R=1, sl_R=1, cost_bps=0), "ret_gross")
        print(f"  {t:5s} n={s['n_trades']:5d} win={s['win_rate']*100:.1f}% "
              f"mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== threshold sweep (symmetric 1:1 ATR, gross) ===")
    for thr in (0.5, 1.0, 2.0):
        s = st.summarize(_pool(False, 1, 1, 0, threshold=thr), "ret_gross")
        print(f"  thr={thr:.1f}ATR n={s['n_trades']:6d} win={s['win_rate']*100:.1f}% "
              f"mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== cost sweep (fade, symmetric 1:1 ATR, threshold=1.0) ===")
    for c in (0.0, 0.5, 1.0, 2.0):
        s = st.summarize(_pool(False, 1, 1, c), "ret_net")
        print(f"  cost={c:.1f}bps net mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
