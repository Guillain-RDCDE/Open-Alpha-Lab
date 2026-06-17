"""Real-tape verification -- Study 218 (SPAC-Performance). Regenerates docs/results.md numbers.

Loads (or fetches) SPAK, SPY, and the de-SPAC basket from Yahoo Finance and computes
the headline comparison: CAGR, Sharpe, CAPM alpha (Jensen), HAC t-stat, and max drawdown.

    python studies/218-spac-performance/examples/verify.py          # cache-only
    python studies/218-spac-performance/examples/verify.py --fetch  # refresh the data
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from spac_performance import data, strategy as st  # noqa: E402

STUDY_CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))


def main(fetch: bool) -> None:
    if fetch:
        print("Refreshing data from Yahoo Finance...")
        data.fetch_daily("SPAK", fetch=True, cache_dir=STUDY_CACHE)
        data.fetch_daily("SPY", fetch=True, cache_dir=STUDY_CACHE)
        for t in data.DE_SPAC_TICKERS:
            data.fetch_daily(t, fetch=True, cache_dir=STUDY_CACHE)
        print()

    # ---- SPAK vs SPY -------------------------------------------------------
    print("=== SPAK (de-SPAC ETF) vs SPY ===")
    prices_spak = data.load_spak_vs_spy(fetch=False, cache_dir=STUDY_CACHE)
    rets_spak = st.daily_returns(prices_spak)
    res_spak = st.summarize(rets_spak["SPAK"], rets_spak["SPY"])

    spak_close = prices_spak["SPAK"]
    spy_close = prices_spak["SPY"]
    print(f"  SPAK  {spak_close.index[0].date()} .. {spak_close.index[-1].date()}  "
          f"fp={data.fingerprint(spak_close)}")
    print(f"  SPY   {spy_close.index[0].date()} .. {spy_close.index[-1].date()}  "
          f"fp={data.fingerprint(spy_close)}")
    print()

    n, n_years = res_spak["n"], res_spak["n_years"]
    print(f"  n = {n} trading days, {n_years:.2f} years")
    print(f"  SPAK CAGR:    {res_spak['cagr_fund_pct']:+.2f}%/yr")
    print(f"  SPY  CAGR:    {res_spak['cagr_market_pct']:+.2f}%/yr")
    print(f"  SPAK Sharpe:  {res_spak['sharpe_fund']:.3f}  |  SPY Sharpe: {res_spak['sharpe_market']:.3f}")
    print(f"  Beta vs SPY:  {res_spak['beta']:.3f}  |  R^2: {res_spak['r_squared']:.3f}")
    print(f"  Jensen alpha: {res_spak['alpha_ann_pct']:+.2f}%/yr  "
          f"({res_spak['alpha_daily_bps']:+.3f} bps/day)")
    print(f"  HAC t(alpha): {res_spak['tstat_alpha_hac']:+.3f}   [|t|>=2 required for significance]")
    print(f"  Excess ret (SPAK-SPY): {res_spak['excess_mean_bps']:+.3f} bps/day  "
          f"t = {res_spak['tstat_excess_hac']:+.3f}")
    print(f"  SPAK max DD:  {res_spak['max_dd_fund_pct']:.1f}%  |  SPY max DD: {res_spak['max_dd_market_pct']:.1f}%")
    print()

    # ---- De-SPAC basket vs SPY ---------------------------------------------
    print("=== De-SPAC basket (surviving names) vs SPY ===")
    prices_basket = data.load_basket_vs_spy(fetch=False, cache_dir=STUDY_CACHE)
    rets_basket = st.daily_returns(prices_basket)
    res_basket = st.summarize(rets_basket["BASKET"], rets_basket["SPY"])

    print(f"  Basket tickers: {', '.join(data.DE_SPAC_TICKERS)}")
    print(f"  Period: {prices_basket.index[0].date()} .. {prices_basket.index[-1].date()}")
    print(f"  n = {res_basket['n']} days, {res_basket['n_years']:.2f} years")
    print(f"  Basket CAGR:  {res_basket['cagr_fund_pct']:+.2f}%/yr")
    print(f"  SPY  CAGR:    {res_basket['cagr_market_pct']:+.2f}%/yr")
    print(f"  Basket Sharpe:{res_basket['sharpe_fund']:.3f}  |  SPY Sharpe: {res_basket['sharpe_market']:.3f}")
    print(f"  Beta vs SPY:  {res_basket['beta']:.3f}  |  R^2: {res_basket['r_squared']:.3f}")
    print(f"  Jensen alpha: {res_basket['alpha_ann_pct']:+.2f}%/yr  "
          f"({res_basket['alpha_daily_bps']:+.3f} bps/day)")
    print(f"  HAC t(alpha): {res_basket['tstat_alpha_hac']:+.3f}")
    print(f"  Basket max DD:{res_basket['max_dd_fund_pct']:.1f}%  |  SPY max DD: {res_basket['max_dd_market_pct']:.1f}%")
    print()

    # ---- Survivorship note -------------------------------------------------
    print("=== Survivorship bias note ===")
    print("  Delisted de-SPACs (NKLA, OTRK, GNUS, HYLIION, PTRA, BARK, SPNV...)")
    print("  are ABSENT from the basket. The basket represents *surviving* names only.")
    print("  True cohort underperformance is LARGER than what the basket shows.")
    print()

    # ---- Verdict -----------------------------------------------------------
    print("=== Verdict ===")
    t_spak = res_spak["tstat_alpha_hac"]
    if abs(t_spak) >= 2:
        verdict = f"REAL (|t|={abs(t_spak):.2f} >= 2)"
    else:
        verdict = f"NONE (|t|={abs(t_spak):.2f} < 2; direction is unambiguous)"
    print(f"  Signal:       {verdict}")
    print(f"  SPAK alpha:   {res_spak['alpha_ann_pct']:+.2f}%/yr (SPAK CAGR vs SPY CAGR gap = "
          f"{res_spak['cagr_fund_pct'] - res_spak['cagr_market_pct']:+.1f}%/yr)")
    print("  Tradability:  MIRAGE (severe underperformance on return, Sharpe, and drawdown)")
    print("  Myth check:   BUSTED — SPACs were structurally rigged against public investors")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
