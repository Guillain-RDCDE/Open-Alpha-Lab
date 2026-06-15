"""Real-tape verification — Study 189 (Double-Top / Double-Bottom).
Regenerates the headline numbers in docs/results.md.

Fetches (or reads from cache) the ten daily tapes, runs the double-top /
double-bottom detector against a random-day placebo, and prints the HAC t-stats
for each pattern and horizon.  Network is touched only with --fetch.

    python studies/189-double-top/examples/verify.py            # cache-only
    python studies/189-double-top/examples/verify.py --fetch    # refresh tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from double_top import data, strategy as st  # noqa: E402

TICKERS = data.DEFAULT_TICKERS
HORIZONS = (1, 5, 20)


def main(fetch: bool) -> None:
    if fetch:
        print("=== Refreshing caches ===")
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True)
            print(f"  {t:6s}  {b.index[0].date()} -> {b.index[-1].date()}  "
                  f"fp={data.fingerprint(b)}  n={len(b)}")
        print()

    print("=== Pooled pattern study (10 tickers, default parameters) ===\n")
    pooled = st.run_pooled_study(
        TICKERS, fetch=False, horizons=HORIZONS, cost_bps=0.0, seed=189,
        min_bars=10, tolerance=0.04, lookback=120, atr_prominence_mult=0.5,
    )

    print(f"{'Pattern':>15} {'h':>3} {'n':>5} {'win%':>6} "
          f"{'sig bps':>8} {'rnd bps':>8} {'excess':>8} {'t(exc)':>8} {'t(sig)':>7}")
    print("-" * 80)
    for pname in st.PATTERN_NAMES:
        if pname not in pooled:
            print(f"  {pname}: no signals detected")
            continue
        ledger = pooled[pname]
        for h in HORIZONS:
            res = st.summarize_pattern(ledger, horizon=h)
            if not res:
                continue
            t_exc = res["tstat_excess"]
            t_sig = res["tstat_signal"]
            print(
                f"{pname:>15} {h:>3}d  {res['n']:>4}  "
                f"{res['win_rate']*100:>5.1f}%  "
                f"{res['mean_bps']:>+7.2f}  {res['baseline_bps']:>+7.2f}  "
                f"{res['excess_bps']:>+7.2f}  "
                f"{t_exc:>+7.2f}  {t_sig:>+6.2f}"
            )
        print()

    print()
    print("=== Data stamp ===")
    for t in TICKERS:
        try:
            bars = data.fetch_daily(t, fetch=False)
            print(f"  {t:6s}  {bars.index[0].date()} -> {bars.index[-1].date()}  "
                  f"fp={data.fingerprint(bars)}")
        except FileNotFoundError:
            print(f"  {t:6s}  (cache missing)")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
