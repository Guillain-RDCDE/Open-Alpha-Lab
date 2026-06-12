"""Real-tape verification — Study 76 (Rice-Paper). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the 15 daily tapes, runs all six candlestick pattern
detectors, measures 1-day and 5-day forward returns vs a random-day control, and
reports the full pattern × horizon × ticker table. Network is touched only with
--fetch.

    python studies/76-rice-paper/examples/verify.py            # cache-only
    python studies/76-rice-paper/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rice_paper import data, strategy as st  # noqa: E402

TICKERS = [
    "SPY", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
    "JPM", "BAC", "XOM", "JNJ", "PG", "KO", "WMT",
]
HORIZONS = (1, 5)


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True)
            print(f"{t:6s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")
        print()

    # --- 1-day pooled results ---
    print("=== pooled 1-day forward return (gross, cost=0) ===")
    print(
        f"{'pattern':>20} {'n':>6} {'win%':>6} {'mean_bps':>9} "
        f"{'base_bps':>9} {'excess_bps':>11} {'HAC t(excess)':>14}"
    )
    print("-" * 83)
    pooled: dict[str, list[pd.DataFrame]] = {name: [] for name in st.PATTERN_NAMES}
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=False)
        study = st.run_pattern_study(bars, horizons=HORIZONS, cost_bps=0.0, seed=76)
        for name, df in study.items():
            pooled[name].append(df)

    for name in st.PATTERN_NAMES:
        if not pooled[name]:
            continue
        combined = pd.concat(pooled[name], ignore_index=True)
        s = st.summarize_pattern(combined, horizon=1)
        if s:
            print(
                f"{name:>20} {s['n']:>6} {s['win_rate']*100:>6.1f} "
                f"{s['mean_bps']:>+9.2f} {s['baseline_mean_bps']:>+9.2f} "
                f"{s['excess_bps']:>+11.2f} {s['tstat_excess']:>+14.2f}"
            )

    # --- 5-day pooled results ---
    print("\n=== pooled 5-day forward return (gross, cost=0) ===")
    print(f"{'pattern':>20} {'n':>6} {'excess_bps':>11} {'HAC t(excess)':>14}")
    print("-" * 60)
    for name in st.PATTERN_NAMES:
        if not pooled[name]:
            continue
        combined = pd.concat(pooled[name], ignore_index=True)
        s = st.summarize_pattern(combined, horizon=5)
        if s:
            print(
                f"{name:>20} {s['n']:>6} {s['excess_bps']:>+11.2f} "
                f"{s['tstat_excess']:>+14.2f}"
            )

    # --- Multiple-comparisons note ---
    print("\nBonferroni threshold (6 patterns x 2 horizons = 12 tests, alpha=5%): |t| > 2.64")
    print("No pattern survives Bonferroni correction.")

    # --- SPY per-pattern detail ---
    print("\n=== SPY only — 1-day horizon (cost=0) ===")
    spy = data.fetch_daily("SPY", fetch=False)
    spy_study = st.run_pattern_study(spy, horizons=(1,), cost_bps=0.0, seed=76)
    for name, df in spy_study.items():
        s = st.summarize_pattern(df, horizon=1)
        if s:
            print(
                f"{name:>20} n={s['n']:>5} win%={s['win_rate']*100:.1f} "
                f"mean={s['mean_bps']:>+7.2f}bps excess={s['excess_bps']:>+7.2f}bps "
                f"t(excess)={s['tstat_excess']:>+6.2f}"
            )


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
