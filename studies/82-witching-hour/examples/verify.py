"""Real-tape verification — Study 82 (Witching-Hour). Regenerates docs/results.md numbers.

Fetches (or reads from cache) SPY daily bars from 1993, identifies all witching days
and weeks, and tests the vol/volume and return effects.

    python studies/82-witching-hour/examples/verify.py            # cache-only
    python studies/82-witching-hour/examples/verify.py --fetch    # refresh the tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd

from witching_hour import data, strategy as st  # noqa: E402

TICKER = "SPY"


def main(fetch: bool) -> None:
    bars = data.fetch_daily(TICKER, start="1993-01-01", fetch=fetch)
    if fetch:
        print(f"{TICKER} {bars.index[0].date()}..{bars.index[-1].date()} "
              f"n={len(bars)} fp={data.fingerprint(bars)}")
        print()

    idx = pd.DatetimeIndex(bars.index)
    day_mask = data.witching_day_mask(idx)
    week_mask = data.witching_week_mask(idx)

    # ---- vol / volume leg -----------------------------------------------
    vol_results = st.witching_vol_test(bars)
    rng_day = vol_results["range_day"]
    rng_week = vol_results["range_week"]
    vol_day = vol_results["vol_day"]
    vol_week = vol_results["vol_week"]
    vol_raw = vol_results["vol_raw_day"]

    print("=== RANGE (high-low)/close ===")
    print(f"Witching DAY  : mean={rng_day['mean_witching']:.4f}  "
          f"baseline={rng_day['mean_baseline']:.4f}  "
          f"diff={rng_day['diff']:+.4f}  HAC t={rng_day['tstat']:+.2f}  "
          f"n_witch={rng_day['n_witching']}")
    print(f"Witching WEEK : mean={rng_week['mean_witching']:.4f}  "
          f"baseline={rng_week['mean_baseline']:.4f}  "
          f"diff={rng_week['diff']:+.4f}  HAC t={rng_week['tstat']:+.2f}  "
          f"n_witch={rng_week['n_witching']}")

    print("\n=== LOG VOLUME (year-demeaned) ===")
    print(f"Witching DAY  : diff={vol_day['diff']:+.4f}  HAC t={vol_day['tstat']:+.2f}  "
          f"volume uplift ~{(np.exp(vol_day['diff'])-1)*100:.1f}%")
    print(f"Witching WEEK : diff={vol_week['diff']:+.4f}  HAC t={vol_week['tstat']:+.2f}  "
          f"volume uplift ~{(np.exp(vol_week['diff'])-1)*100:.1f}%")
    print(f"  (raw log-vol for reference: HAC t={vol_raw['tstat']:+.2f})")

    # ---- return leg ------------------------------------------------------
    ret_results = st.witching_return_test(bars)
    ret_day = ret_results["day"]
    ret_week = ret_results["week"]

    print("\n=== DAILY RETURN (close-to-close) ===")
    print(f"Witching DAY  : mean={ret_day['mean_witching']*1e4:+.2f}bps  "
          f"baseline={ret_day['mean_baseline']*1e4:+.2f}bps  "
          f"diff={ret_day['diff']*1e4:+.2f}bps  HAC t={ret_day['tstat']:+.2f}  "
          f"n_witch={ret_day['n_witching']}")
    print(f"Witching WEEK : mean={ret_week['mean_witching']*1e4:+.2f}bps  "
          f"baseline={ret_week['mean_baseline']*1e4:+.2f}bps  "
          f"diff={ret_week['diff']*1e4:+.2f}bps  HAC t={ret_week['tstat']:+.2f}  "
          f"n_witch={ret_week['n_witching']}")

    # ---- tradability check -----------------------------------------------
    week_ret = st.witching_week_returns(bars, long_week=True)
    s = st.summarize(week_ret[week_ret != 0])
    print(f"\n=== TRADABILITY (long witching week only) ===")
    print(f"n={s['n']}  mean={s['mean_bps']:+.2f}bps  sharpe={s['sharpe_ann']:+.2f}  "
          f"CAGR={s['cagr']:+.2%}  HAC t={s['tstat']:+.2f}")

    # ---- data stamp ------------------------------------------------------
    print(f"\n=== DATA STAMP ===")
    print(f"{TICKER}: {bars.index[0].date()} to {bars.index[-1].date()}  "
          f"n={len(bars)}  fp={data.fingerprint(bars)}")
    print(f"witching days: {int(day_mask.sum())}  witching-week days: {int(week_mask.sum())}")
    wdates = data.witching_dates("1993-01-01", "2026-12-31")
    print(f"witching events in tape: {len(wdates)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
