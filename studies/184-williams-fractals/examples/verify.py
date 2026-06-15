"""Real-tape verification — Study 184 (Williams-Fractals). Regenerates docs/results.md numbers.

Fetches (or reads from cache) daily OHLCV for a basket of six liquid instruments, detects
Williams fractals, runs both the reversal and breakout framings against a random-direction
control, sweeps hold periods, and sweeps costs. Network is touched only with --fetch.

    python studies/184-williams-fractals/examples/verify.py            # cache-only
    python studies/184-williams-fractals/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from williams_fractals import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GLD"]


def _pool(fetch: bool, framing: str, hold_days: int, cost: float,
          seed: int | None = None) -> pd.DataFrame:
    frames = []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=fetch)
        f = st.detect_fractals(bars)
        if framing == "reversal":
            ent = st.reversal_entries(f)
        else:
            ent = st.breakout_entries(bars, f)
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(bars, ent, hold_days=hold_days, cost_bps=cost,
                                    directions=dirs))
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} "
                  f"fp={data.fingerprint(b)}  n={len(b)}")
        print()

    print("=== Reversal framing — hold 5 days, gross (cost=0) ===")
    rev = st.summarize(_pool(False, "reversal", 5, 0), "ret_gross")
    rand = st.summarize(_pool(False, "reversal", 5, 0, seed=184), "ret_gross")
    print(f"REVERSAL  n={rev['n_trades']} win={rev['win_rate']:.3f} "
          f"mean={rev['mean_bps']:+.2f}bps t={rev['tstat']:+.2f}")
    print(f"RANDOM    n={rand['n_trades']} win={rand['win_rate']:.3f} "
          f"mean={rand['mean_bps']:+.2f}bps t={rand['tstat']:+.2f}")

    print("\n=== Breakout framing — hold 5 days, gross (cost=0) ===")
    brk = st.summarize(_pool(False, "breakout", 5, 0), "ret_gross")
    rbrk = st.summarize(_pool(False, "breakout", 5, 0, seed=184), "ret_gross")
    print(f"BREAKOUT  n={brk['n_trades']} win={brk['win_rate']:.3f} "
          f"mean={brk['mean_bps']:+.2f}bps t={brk['tstat']:+.2f}")
    print(f"RANDOM    n={rbrk['n_trades']} win={rbrk['win_rate']:.3f} "
          f"mean={rbrk['mean_bps']:+.2f}bps t={rbrk['tstat']:+.2f}")

    print("\n=== Hold-period sweep (reversal, gross) ===")
    for h in (1, 3, 5, 10, 20):
        s = st.summarize(_pool(False, "reversal", h, 0), "ret_gross")
        print(f"hold={h:2d}d  n={s['n_trades']} mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== Cost sweep (reversal, hold=5) ===")
    for c in (0.0, 0.5, 1.0, 2.0, 5.0):
        s = st.summarize(_pool(False, "reversal", 5, c), "ret_net")
        print(f"cost={c:4.1f}bps  net mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== Per-instrument (reversal, hold=5, gross) ===")
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=False)
        f = st.detect_fractals(bars)
        ent = st.reversal_entries(f)
        s = st.summarize(st.run_trades(bars, ent, hold_days=5, cost_bps=0), "ret_gross")
        print(f"{t:5s}  n={s['n_trades']:4d}  win={s['win_rate']:.3f}  "
              f"mean={s['mean_bps']:+.2f}bps  t={s['tstat']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
