"""Real-tape verification — Study 115 (Credit-Spreads). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the four daily price series (HYG, LQD, IEF, SPY), builds
both credit-spread proxies, and reports regime-conditional forward-return stats vs the
unconditional baseline.  Network is touched only with --fetch.

    python studies/115-credit-spreads/examples/verify.py            # cache-only
    python studies/115-credit-spreads/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from credit_spreads import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    if fetch:
        for t in data.TICKERS:
            df = data.fetch_daily(t, fetch=True)
            print(f"{t:5s} {df.index[0].date()}..{df.index[-1].date()} fp={data.fingerprint(df.rename(columns={'close':'spy'})) if 'spy' not in df.columns else data.fingerprint(df)}")
        print()

    prices = data.load_all(fetch=False)
    print(f"Loaded: {len(prices)} trading days, {prices.index[0].date()} to {prices.index[-1].date()}")
    fp = data.fingerprint(prices)
    print(f"Fingerprint: {fp}\n")

    for fw in (5, 21):
        print(f"=== Forward window: {fw} days ===")
        result = st.summarize(prices, window=20, forward_days=fw, rolling_median_window=60)

        print(f"N observations (after signal formation): {result['n']}")
        print()
        print("--- HYG-IEF spread proxy (wider = more HY stress) ---")
        print(f"  Stress periods (n={result['spread_stress_n']}):")
        print(f"    mean SPY fwd return: {result['spread_stress_mean_bps']:+.1f} bps")
        print(f"    win-rate (positive): {result['spread_stress_win_rate']*100:.1f}%")
        print(f"    HAC t-stat:          {result['spread_stress_tstat']:+.2f}")
        print(f"  Calm periods (n={result['spread_calm_n']}):")
        print(f"    mean SPY fwd return: {result['spread_calm_mean_bps']:+.1f} bps")
        print(f"    win-rate (positive): {result['spread_calm_win_rate']*100:.1f}%")
        print(f"    HAC t-stat:          {result['spread_calm_tstat']:+.2f}")
        print(f"  Unconditional (n={result['spread_unconditional_n']}):")
        print(f"    mean SPY fwd return: {result['spread_unconditional_mean_bps']:+.1f} bps")
        print(f"  Diff (stress - calm): {result['spread_diff_mean_bps']:+.1f} bps  t={result['spread_diff_tstat']:+.2f}")
        print(f"  Correlation (spread->fwd SPY): {result['spread_corr']:+.3f}")
        print()
        print("--- HYG/LQD relative-strength proxy (wider = more HY vs IG stress) ---")
        print(f"  Stress periods (n={result['rs_stress_n']}):")
        print(f"    mean SPY fwd return: {result['rs_stress_mean_bps']:+.1f} bps")
        print(f"    win-rate (positive): {result['rs_stress_win_rate']*100:.1f}%")
        print(f"    HAC t-stat:          {result['rs_stress_tstat']:+.2f}")
        print(f"  Calm periods (n={result['rs_calm_n']}):")
        print(f"    mean SPY fwd return: {result['rs_calm_mean_bps']:+.1f} bps")
        print(f"    win-rate (positive): {result['rs_calm_win_rate']*100:.1f}%")
        print(f"    HAC t-stat:          {result['rs_calm_tstat']:+.2f}")
        print(f"  Unconditional (n={result['rs_unconditional_n']}):")
        print(f"    mean SPY fwd return: {result['rs_unconditional_mean_bps']:+.1f} bps")
        print(f"  Diff (stress - calm): {result['rs_diff_mean_bps']:+.1f} bps  t={result['rs_diff_tstat']:+.2f}")
        print(f"  Correlation (RS->fwd SPY): {result['rs_corr']:+.3f}")
        print()


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
