"""Real-tape verification -- Study 139 (AI-Powered-ETF). Regenerates docs/results.md numbers.

Loads (or fetches) AIEQ, SPY, IVV daily data from Yahoo Finance and computes the headline
comparison: CAGR, Sharpe, CAPM alpha (Jensen), HAC t-stat, and maximum drawdown.

    python studies/139-ai-powered-etf/examples/verify.py            # cache-only
    python studies/139-ai-powered-etf/examples/verify.py --fetch    # refresh the data
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ai_powered_etf import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    if fetch:
        print("Refreshing data from Yahoo Finance...")
        for t in data.TICKERS:
            df = data.fetch_daily(t, fetch=True)
            close = df["close"] if "close" in df.columns else df["Close"]
            print(f"  {t:5s}  {close.index[0].date()} .. {close.index[-1].date()}  "
                  f"fp={data.fingerprint(close)}")
        print()

    prices = data.load_prices(fetch=False)
    r = st.daily_returns(prices)

    print("=== Data stamp ===")
    for t in data.TICKERS:
        close = prices[t]
        print(f"  {t:5s}  {close.index[0].date()} .. {close.index[-1].date()}  "
              f"fp={data.fingerprint(close)}")
    print()

    # Main comparison
    res = st.summarize(r["AIEQ"], r["SPY"])
    print("=== AIEQ vs SPY (full window) ===")
    print(f"  n = {res['n']} trading days, {res['n_years']:.2f} years")
    print(f"  AIEQ CAGR:    {res['cagr_fund_pct']:+.2f}%/yr")
    print(f"  SPY  CAGR:    {res['cagr_market_pct']:+.2f}%/yr")
    print(f"  AIEQ Sharpe:  {res['sharpe_fund']:.3f}  |  SPY Sharpe: {res['sharpe_market']:.3f}")
    print(f"  Beta vs SPY:  {res['beta']:.3f}  |  R^2: {res['r_squared']:.3f}")
    print(f"  Jensen alpha: {res['alpha_ann_pct']:+.2f}%/yr  "
          f"({res['alpha_daily_bps']:+.3f} bps/day)")
    print(f"  HAC t(alpha): {res['tstat_alpha_hac']:+.3f}   [|t|>=2 required for significance]")
    print(f"  Excess ret (AIEQ-SPY): {res['excess_mean_bps']:+.3f} bps/day  "
          f"t = {res['tstat_excess_hac']:+.3f}")
    print(f"  AIEQ max DD:  {res['max_dd_fund_pct']:.1f}%  |  SPY max DD: {res['max_dd_market_pct']:.1f}%")
    print()

    # Fraction of rolling 12m windows AIEQ beats SPY
    roll_out = st.rolling_outperformance(r["AIEQ"], r["SPY"], window=252)
    frac_beat = (roll_out > 0).mean()
    print(f"=== Rolling 12m outperformance ===")
    print(f"  AIEQ beats SPY in {frac_beat*100:.1f}% of rolling 12m windows")
    print()

    # Bootstrap Sharpe CI for AIEQ
    try:
        from quantlab.stats import sharpe_ci_bootstrap
        ci = sharpe_ci_bootstrap(r["AIEQ"], n_boot=2000, periods_per_year=252, seed=139)
        print(f"=== Bootstrap Sharpe CI (AIEQ) ===")
        print(f"  Point: {ci['sharpe']:.3f}  95% CI: [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  "
              f"frac_neg: {ci['frac_negative']*100:.1f}%")
        ci_spy = sharpe_ci_bootstrap(r["SPY"], n_boot=2000, periods_per_year=252, seed=139)
        print(f"  SPY:   {ci_spy['sharpe']:.3f}  95% CI: [{ci_spy['ci_low']:.3f}, {ci_spy['ci_high']:.3f}]  "
              f"frac_neg: {ci_spy['frac_negative']*100:.1f}%")
    except ImportError:
        print("(quantlab not found -- skipping bootstrap CI)")
    print()

    print("=== Verdict ===")
    t = res["tstat_alpha_hac"]
    if abs(t) >= 2:
        verdict = "REAL (|t|>=2)" if t > 0 else "REAL (negative alpha, |t|>=2)"
    else:
        verdict = f"NONE/WEAK (|t|={abs(t):.2f} < 2; power-limited on {res['n_years']:.1f} yrs)"
    print(f"  Signal: {verdict}")
    print(f"  Alpha: {res['alpha_ann_pct']:+.2f}%/yr (below zero = underperformance vs market-beta)")
    print(f"  Tradability: MIRAGE (AIEQ CAGR {res['cagr_fund_pct']:+.2f}% vs SPY {res['cagr_market_pct']:+.2f}%)")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
