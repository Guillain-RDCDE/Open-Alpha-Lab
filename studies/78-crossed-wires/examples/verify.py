"""Real-tape verification — Study 78 (Crossed-Wires).  Regenerates docs/results.md numbers.

Fetches (or reads from cache) the daily tapes for the basket, runs the MACD(12,26,9)
signal-line crossover against a random-direction control with symmetric ±1 ATR barriers,
sweeps costs, and reports the fixed-tick skew trap.  Network is touched only with --fetch.

    python studies/78-crossed-wires/examples/verify.py             # cache-only
    python studies/78-crossed-wires/examples/verify.py --fetch     # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crossed_wires import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA"]


def _pool(fetch: bool, tp: float, sl: float, cost: float, seed: int | None = None):
    frames = []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=fetch)
        ent = st.crossover_entries(bars["close"])
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(bars, ent, tp_R=tp, sl_R=sl, cost_bps=cost, directions=dirs))
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")
        print()

    cross = st.summarize(_pool(False, 1, 1, 0), "ret_gross")
    rand = st.summarize(_pool(False, 1, 1, 0, seed=78), "ret_gross")
    print("=== honest symmetric ±1 ATR (gross) ===")
    print(f"CROSS  n={cross['n_trades']} win={cross['win_rate']:.3f} "
          f"mean={cross['mean_bps']:+.2f}bps t={cross['tstat']:+.2f}")
    print(f"RANDOM n={rand['n_trades']} win={rand['win_rate']:.3f} "
          f"mean={rand['mean_bps']:+.2f}bps t={rand['tstat']:+.2f}")
    print(f"cross - random = {cross['mean_bps'] - rand['mean_bps']:+.2f} bps")

    print("\n=== per-instrument breakdown ===")
    for t in TICKERS:
        b = data.fetch_daily(t, fetch=False)
        ent = st.crossover_entries(b["close"])
        s = st.summarize(st.run_trades(b, ent, tp_R=1, sl_R=1, cost_bps=0), "ret_gross")
        print(f"{t:5s} n={s['n_trades']:3d} win={s['win_rate']:.3f} "
              f"mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== cost sweep (cross, net) ===")
    for c in (0.0, 0.5, 1.0, 2.0, 5.0):
        s = st.summarize(_pool(False, 1, 1, c), "ret_net")
        print(f"cost={c:4.1f}bps net mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== fixed-tick trap (gross) ===")
    for tp, sl, lbl in [(1.0, 1.0, "symmetric 1:1"),
                        (0.5, 2.0, "TP0.5/SL2"),
                        (0.25, 8.0, "TP0.25/~no-stop")]:
        s = st.summarize(_pool(False, tp, sl, 0), "ret_gross")
        print(f"{lbl:18s} win={s['win_rate']:.3f} mean={s['mean_bps']:+.2f}bps "
              f"skew={s['skew']:+.2f} t={s['tstat']:+.2f}")

    print("\n=== forward return horizons (d5, d20, d60) ===")
    for hold in (5, 20, 60):
        frames = []
        rframes = []
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=False)
            ent = st.crossover_entries(b["close"])
            frames.append(st.run_forward_returns(b, ent, hold_days=hold, cost_bps=0))
            rframes.append(st.run_forward_returns(b, ent, hold_days=hold, cost_bps=0,
                           directions=st.random_directions(len(ent), seed=78)))
        c = st.summarize(pd.concat(frames, ignore_index=True), "ret_gross")
        r = st.summarize(pd.concat(rframes, ignore_index=True), "ret_gross")
        print(f"d{hold:2d} cross mean={c['mean_bps']:+.2f}bps t={c['tstat']:+.2f} | "
              f"random mean={r['mean_bps']:+.2f}bps t={r['tstat']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
