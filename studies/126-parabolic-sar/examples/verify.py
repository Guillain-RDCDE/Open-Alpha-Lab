"""Real-tape verification — Study 126 (Parabolic-SAR). Regenerates docs/results.md numbers.

Fetches (or reads from cache) daily tapes for the basket, runs the Parabolic SAR
stop-and-reverse signal against a random-direction control with symmetric ±1 ATR(20)
barriers, sweeps costs, and computes turnover.  Network is touched only with --fetch.

    python studies/126-parabolic-sar/examples/verify.py            # cache-only
    python studies/126-parabolic-sar/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from parabolic_sar import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "EEM"]


def _pool(fetch: bool, tp: float, sl: float, cost: float, seed: int | None = None):
    frames = []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=fetch)
        ent = st.flip_entries(bars)
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(bars, ent, tp_R=tp, sl_R=sl, cost_bps=cost, directions=dirs))
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} "
                  f"fp={data.fingerprint(b)}")
        print()

    cross = st.summarize(_pool(False, 1, 1, 0), "ret_gross")
    rand = st.summarize(_pool(False, 1, 1, 0, seed=126), "ret_gross")
    print("=== honest symmetric ±1 ATR (gross) ===")
    print(f"SAR   n={cross['n_trades']} win={cross['win_rate']:.3f} "
          f"mean={cross['mean_bps']:+.2f}bps t={cross['tstat']:+.2f}")
    print(f"RAND  n={rand['n_trades']} win={rand['win_rate']:.3f} "
          f"mean={rand['mean_bps']:+.2f}bps t={rand['tstat']:+.2f}")

    # Turnover: trades per ticker per year
    tpy_list = []
    for t in TICKERS:
        b = data.fetch_daily(t, fetch=False)
        ent = st.flip_entries(b)
        n_years = len(b) / 252.0
        tpy_list.append(len(ent) / n_years)
    tpy = sum(tpy_list) / len(tpy_list)
    print(f"\nTurnover: ~{tpy:.1f} flips/ticker/year")

    print("\n=== cost sweep (SAR, net) ===")
    for c in (0.0, 1.0, 5.0, 10.0):
        s = st.summarize(_pool(False, 1, 1, c), "ret_net")
        ann = s["mean_bps"] * tpy / 1e4 * 100
        print(f"cost={c:5.1f}bps net mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f} "
              f"ann~{ann:+.1f}%/yr")

    print("\n=== per-ticker breakdown (gross) ===")
    for t in TICKERS:
        b = data.fetch_daily(t, fetch=False)
        ent = st.flip_entries(b)
        s = st.summarize(st.run_trades(b, ent, tp_R=1, sl_R=1, cost_bps=0), "ret_gross")
        print(f"{t:5s} n={s['n_trades']:4d} win={s['win_rate']:.3f} "
              f"mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
