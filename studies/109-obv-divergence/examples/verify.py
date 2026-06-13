"""Real-tape verification — Study 109 (OBV-Divergence). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the six daily tapes, runs both OBV signals
(trend-cross and price-divergence) against random-direction controls, sweeps
costs for Signal A, and reports per-instrument and pooled headline numbers.
Network is touched only with --fetch.

    python studies/109-obv-divergence/examples/verify.py            # cache-only
    python studies/109-obv-divergence/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from obv_divergence import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA"]
HORIZON = 5
COST_BPS = 2.0


def _load(fetch: bool) -> dict[str, pd.DataFrame]:
    bars_map = {}
    for t in TICKERS:
        bars_map[t] = data.fetch_daily(t, fetch=fetch)
        if fetch:
            b = bars_map[t]
            print(f"  {t}: {b.index[0].date()} -> {b.index[-1].date()}  fp={data.fingerprint(b)}")
    return bars_map


def main(fetch: bool) -> None:
    if fetch:
        print("=== Refreshing cache ===")
    bars_map = _load(fetch)

    print("\n=== Signal A: OBV trend vs SMA(OBV, 20) — per instrument (gross, 5d) ===")
    for t, bars in bars_map.items():
        sig = st.signal_obv_trend(bars, obv_sma_n=20)
        led = st.run_signals(bars, sig, horizon=HORIZON, cost_bps=0.0)
        s = st.summarize(led, "ret_gross")
        print(f"  {t}: n={s['n_signals']} win={s['win_rate']*100:.1f}% "
              f"mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    led_A = st.run_basket(
        bars_map, lambda b: st.signal_obv_trend(b, obv_sma_n=20),
        horizon=HORIZON, cost_bps=0.0,
    )
    sA = st.summarize(led_A, "ret_gross")
    led_A_rand = st.run_basket(
        bars_map, lambda b: st.signal_obv_trend(b, obv_sma_n=20),
        horizon=HORIZON, cost_bps=0.0, random_seed=109,
    )
    sA_rand = st.summarize(led_A_rand, "ret_gross")
    print(f"Pooled A: n={sA['n_signals']} win={sA['win_rate']*100:.1f}% "
          f"mean={sA['mean_bps']:+.2f}bps t={sA['tstat']:+.2f}")
    print(f"Random A: n={sA_rand['n_signals']} win={sA_rand['win_rate']*100:.1f}% "
          f"mean={sA_rand['mean_bps']:+.2f}bps t={sA_rand['tstat']:+.2f}")
    print(f"A − random: {sA['mean_bps'] - sA_rand['mean_bps']:+.2f} bps/signal")

    print("\n=== Signal B: OBV-price divergence (lookback=10, sma=5) ===")
    for t, bars in bars_map.items():
        sig = st.signal_obv_divergence(bars, lookback=10, obv_sma_n=5)
        led = st.run_signals(bars, sig, horizon=HORIZON, cost_bps=0.0)
        s = st.summarize(led, "ret_gross")
        print(f"  {t}: n={s['n_signals']} win={s['win_rate']*100:.1f}% "
              f"mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    led_B = st.run_basket(
        bars_map, lambda b: st.signal_obv_divergence(b, lookback=10, obv_sma_n=5),
        horizon=HORIZON, cost_bps=0.0,
    )
    sB = st.summarize(led_B, "ret_gross")
    led_B_rand = st.run_basket(
        bars_map, lambda b: st.signal_obv_divergence(b, lookback=10, obv_sma_n=5),
        horizon=HORIZON, cost_bps=0.0, random_seed=109,
    )
    sB_rand = st.summarize(led_B_rand, "ret_gross")
    print(f"Pooled B: n={sB['n_signals']} win={sB['win_rate']*100:.1f}% "
          f"mean={sB['mean_bps']:+.2f}bps t={sB['tstat']:+.2f}")
    print(f"Random B: mean={sB_rand['mean_bps']:+.2f}bps t={sB_rand['tstat']:+.2f}")
    print(f"B − random: {sB['mean_bps'] - sB_rand['mean_bps']:+.2f} bps/signal")

    print("\n=== Cost sweep: Signal A ===")
    for c in [0.0, 1.0, 2.0, 5.0]:
        led_c = st.run_basket(
            bars_map, lambda b: st.signal_obv_trend(b, obv_sma_n=20),
            horizon=HORIZON, cost_bps=c,
        )
        s = st.summarize(led_c, "ret_net")
        print(f"  cost={c:.1f}bps: net mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
