"""Real-tape verification — Study 195 (Monthly-OpEx). Regenerates docs/results.md numbers.

Fetches (or reads from cache) SPY daily bars from 1993, identifies all monthly OpEx
days and weeks, tests the vol/volume and return effects, and shows the pre/post-2010
structural break and the incremental monthly-only vs quarterly split.

    python studies/195-monthly-opex/examples/verify.py            # cache-only
    python studies/195-monthly-opex/examples/verify.py --fetch    # refresh the tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd

from monthly_opex import data, strategy as st  # noqa: E402

TICKER = "SPY"


def main(fetch: bool) -> None:
    bars = data.fetch_daily(TICKER, start="1993-01-01", fetch=fetch)
    if fetch:
        print(f"{TICKER} {bars.index[0].date()}..{bars.index[-1].date()} "
              f"n={len(bars)} fp={data.fingerprint(bars)}")
        print()

    idx = pd.DatetimeIndex(bars.index)
    od = data.opex_dates("1993-01-01", "2026-12-31")
    day_mask = data.opex_day_mask(idx)
    week_mask = data.opex_week_mask(idx)
    monthly_mask = data.monthly_only_opex_week_mask(idx)
    qrt_mask = data.quarterly_opex_week_mask(idx)

    # ---- vol / range leg ------------------------------------------------
    vol_results = st.opex_vol_test(bars)
    rng_day = vol_results["range_day"]
    rng_week = vol_results["range_week"]
    vol_day = vol_results["vol_day"]
    vol_week = vol_results["vol_week"]

    print("=== RANGE (high-low)/close ===")
    print(f"OpEx DAY  : mean={rng_day['mean_opex']:.4f}  "
          f"base={rng_day['mean_baseline']:.4f}  "
          f"diff={rng_day['diff']:+.4f}  HAC t={rng_day['tstat']:+.2f}  "
          f"n={rng_day['n_opex']}")
    print(f"OpEx WEEK : mean={rng_week['mean_opex']:.4f}  "
          f"base={rng_week['mean_baseline']:.4f}  "
          f"diff={rng_week['diff']:+.4f}  HAC t={rng_week['tstat']:+.2f}  "
          f"n={rng_week['n_opex']}")

    print("\n=== LOG VOLUME (year-demeaned) ===")
    print(f"OpEx DAY  : diff={vol_day['diff']:+.4f}  HAC t={vol_day['tstat']:+.2f}  "
          f"uplift ~{(np.exp(vol_day['diff'])-1)*100:.1f}%  n={vol_day['n_opex']}")
    print(f"OpEx WEEK : diff={vol_week['diff']:+.4f}  HAC t={vol_week['tstat']:+.2f}  "
          f"uplift ~{(np.exp(vol_week['diff'])-1)*100:.1f}%  n={vol_week['n_opex']}")

    # ---- return leg -----------------------------------------------------
    ret_results = st.opex_return_test(bars)
    ret_day = ret_results["day"]
    ret_week = ret_results["week"]

    print("\n=== DAILY RETURN (close-to-close) ===")
    print(f"OpEx DAY  : mean={ret_day['mean_opex']*1e4:+.2f}bps  "
          f"base={ret_day['mean_baseline']*1e4:+.2f}bps  "
          f"diff={ret_day['diff']*1e4:+.2f}bps  HAC t={ret_day['tstat']:+.2f}  "
          f"n={ret_day['n_opex']}")
    print(f"OpEx WEEK : mean={ret_week['mean_opex']*1e4:+.2f}bps  "
          f"base={ret_week['mean_baseline']*1e4:+.2f}bps  "
          f"diff={ret_week['diff']*1e4:+.2f}bps  HAC t={ret_week['tstat']:+.2f}  "
          f"n={ret_week['n_opex']}")

    # ---- incremental monthly-only test ----------------------------------
    incr = st.opex_incremental_monthly_test(bars)
    print("\n=== INCREMENTAL MONTHLY-ONLY (non-quarterly OpEx weeks) ===")
    for k, v in incr.items():
        if "vol" in k:
            print(f"  {k}: diff={v['diff']:+.4f} ({(np.exp(v['diff'])-1)*100:+.1f}%) t={v['tstat']:+.2f} n={v['n_opex']}")
        elif "range" in k:
            print(f"  {k}: diff={v['diff']:+.4f} t={v['tstat']:+.2f} n={v['n_opex']}")
        else:
            print(f"  {k}: diff={v['diff']*1e4:+.2f}bps t={v['tstat']:+.2f} n={v['n_opex']}")

    # ---- pre/post 2010 split -------------------------------------------
    split = st.opex_pre_post_2010_split(bars)
    print("\n=== PRE/POST-2010 SPLIT ===")
    for period, res in split.items():
        print(f"  {period}:")
        v = res["vol"]
        print(f"    vol: diff={v['diff']:+.4f} ({(np.exp(v['diff'])-1)*100:+.1f}%) t={v['tstat']:+.2f}")
        v = res["range"]
        print(f"    range: diff={v['diff']:+.4f} t={v['tstat']:+.2f}")
        v = res["ret"]
        print(f"    ret: diff={v['diff']*1e4:+.2f}bps t={v['tstat']:+.2f} n={v['n_opex']}")

    # ---- tradability check ----------------------------------------------
    week_ret = st.opex_week_trade_returns(bars, long_week=True)
    s = st.summarize(week_ret[week_ret != 0])
    bh_ret = bars["close"].pct_change().dropna()
    bh = st.summarize(bh_ret)
    print(f"\n=== TRADABILITY (long OpEx week only) ===")
    print(f"OpEx strat: n={s['n']}  mean={s['mean_bps']:+.2f}bps  sharpe={s['sharpe_ann']:+.2f}  "
          f"CAGR={s['cagr']:+.2%}  HAC t={s['tstat']:+.2f}")
    print(f"Buy-and-hold: mean={bh['mean_bps']:+.2f}bps  sharpe={bh['sharpe_ann']:+.2f}  "
          f"CAGR={bh['cagr']:+.2%}")

    # ---- data stamp -----------------------------------------------------
    print(f"\n=== DATA STAMP ===")
    print(f"{TICKER}: {bars.index[0].date()} to {bars.index[-1].date()}  "
          f"n={len(bars)}  fp={data.fingerprint(bars)}")
    print(f"Total OpEx events defined: {len(od)}")
    print(f"OpEx days in tape: {int(day_mask.sum())}  "
          f"OpEx-week days: {int(week_mask.sum())}  "
          f"Monthly-only: {int(monthly_mask.sum())}  "
          f"Quarterly: {int(qrt_mask.sum())}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
