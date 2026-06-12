"""Offline, deterministic demo — Study 82 (Witching-Hour).

No network. Builds synthetic daily tapes and shows the study's spine: the vol and
return detectors only find an effect when one is planted; on a null tape (no premium)
both read near zero. Run:

    python studies/82-witching-hour/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from witching_hour import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 82 -- Witching-Hour -- synthetic positive control\n")

    # ---- vol leg: sweep vol_premium ---
    print(f"{'vol_premium':>12} | {'range diff':>11} {'t':>6} | "
          f"{'logvol diff':>11} {'t':>6} | detects vol?")
    print("-" * 62)
    for vp in (0.0, 0.5, 1.0, 2.0):
        bars, _ = data.synthetic_daily(n_years=30, vol_premium=vp,
                                       ret_premium_bps=0.0, seed=82)
        idx = pd.DatetimeIndex(bars.index)
        mask = data.witching_day_mask(idx)
        rng_res = st.compare_witching_vs_baseline(st.daily_range_pct(bars), mask)
        vol_res = st.compare_witching_vs_baseline(st.log_volume_demeaned(bars), mask)
        detects = "yes" if abs(rng_res["tstat"]) > 2 or abs(vol_res["tstat"]) > 2 else "no"
        print(f"{vp:>12.1f} | {rng_res['diff']:>+11.5f} {rng_res['tstat']:>+6.2f} | "
              f"{vol_res['diff']:>+11.4f} {vol_res['tstat']:>+6.2f} | {detects}")

    print()

    # ---- return leg: sweep ret_premium ---
    print(f"{'ret_premium':>12} | {'return diff bps':>14} {'t':>6} | detects return?")
    print("-" * 45)
    for rp in (0.0, 5.0, 10.0, 20.0):
        bars, _ = data.synthetic_daily(n_years=30, vol_premium=0.0,
                                       ret_premium_bps=rp, seed=82)
        idx = pd.DatetimeIndex(bars.index)
        mask = data.witching_day_mask(idx)
        ret_res = st.compare_witching_vs_baseline(st.daily_return(bars).dropna(), mask)
        detects = "yes" if abs(ret_res["tstat"]) > 2 else "no"
        print(f"{rp:>12.1f} | {ret_res['diff']*1e4:>+14.2f} {ret_res['tstat']:>+6.2f} | {detects}")

    print()
    print("The detectors are faithful: they find the planted effect when it exists and are")
    print("near-flat on the null. The real tape shows volume REAL, range NONE, return MIXED.")
    print("See docs/results.md for the real-tape headline numbers.")


if __name__ == "__main__":
    main()
