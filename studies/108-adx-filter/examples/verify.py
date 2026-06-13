"""Real-tape verification -- Study 108 (ADX-Filter).  Regenerates docs/results.md numbers.

Fetches (or reads from cache) the daily tapes, runs the 20/50 MA cross ungated and
ADX(14) > 25 gated, compares both against a random-direction control, and sweeps
costs.  Network is touched only with --fetch.

    python studies/108-adx-filter/examples/verify.py            # cache-only
    python studies/108-adx-filter/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adx_filter import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA"]


def _pool(fetch: bool, gated: bool, cost: float, seed: int | None = None):
    frames = []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=fetch)
        ent = st.ma_cross_entries(bars["close"])
        if gated:
            adx_s = st.adx(bars, n=14)
            ent = st.apply_adx_gate(ent, adx_s, threshold=25.0)
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(bars, ent, hold_days=20, cost_bps=cost, directions=dirs))
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True)
            print(f"{t:5s}  {b.index[0].date()}..{b.index[-1].date()}  fp={data.fingerprint(b)}")
        print()

    # Ungated arm vs random (the baseline test: does the MA cross itself have signal?)
    ungated = st.summarize(_pool(False, False, 0), "ret_gross")
    rand_ungated = st.summarize(_pool(False, False, 0, seed=108), "ret_gross")

    print("=== UNGATED MA(20/50) cross vs random (gross) ===")
    print(f"UNGATED  n={ungated['n_trades']} win={ungated['win_rate']:.3f} "
          f"mean={ungated['mean_bps']:+.2f}bps t={ungated['tstat']:+.2f}")
    print(f"RANDOM   n={rand_ungated['n_trades']} win={rand_ungated['win_rate']:.3f} "
          f"mean={rand_ungated['mean_bps']:+.2f}bps t={rand_ungated['tstat']:+.2f}")

    # Gated arm vs random (the direct test: does gating beat a coin?)
    gated = st.summarize(_pool(False, True, 0), "ret_gross")
    rand_gated = st.summarize(_pool(False, True, 0, seed=108), "ret_gross")

    print("\n=== GATED (ADX>25) MA(20/50) cross vs random (gross) ===")
    print(f"GATED    n={gated['n_trades']} win={gated['win_rate']:.3f} "
          f"mean={gated['mean_bps']:+.2f}bps t={gated['tstat']:+.2f}")
    print(f"RANDOM   n={rand_gated['n_trades']} win={rand_gated['win_rate']:.3f} "
          f"mean={rand_gated['mean_bps']:+.2f}bps t={rand_gated['tstat']:+.2f}")

    print("\n=== Key differential: GATED minus UNGATED (bps/trade) ===")
    print(f"Delta mean bps/trade: {gated['mean_bps'] - ungated['mean_bps']:+.2f}")

    print("\n=== Cost sweep (UNGATED, net) ===")
    for c in (0.0, 0.5, 1.0, 2.0):
        s = st.summarize(_pool(False, False, c), "ret_net")
        print(f"cost={c:4.1f}bps  net mean={s['mean_bps']:+.2f}bps  t={s['tstat']:+.2f}")

    print("\n=== Cost sweep (GATED, net) ===")
    for c in (0.0, 0.5, 1.0, 2.0):
        s = st.summarize(_pool(False, True, c), "ret_net")
        print(f"cost={c:4.1f}bps  net mean={s['mean_bps']:+.2f}bps  t={s['tstat']:+.2f}")

    print("\n=== Per-instrument breakdown (ungated, gross) ===")
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=False)
        ent = st.ma_cross_entries(bars["close"])
        s = st.summarize(st.run_trades(bars, ent, cost_bps=0), "ret_gross")
        adx_s = st.adx(bars, n=14)
        g_ent = st.apply_adx_gate(ent, adx_s, threshold=25.0)
        gs = st.summarize(st.run_trades(bars, g_ent, cost_bps=0), "ret_gross")
        print(f"{t:5s}  ug n={s['n_trades']:3d} mean={s['mean_bps']:+.1f}bps t={s['tstat']:+.2f} | "
              f"g  n={gs['n_trades']:3d} mean={gs['mean_bps']:+.1f}bps t={gs['tstat']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
