"""Real-panel verification -- Study 238 (Betting-Against-Beta). Regenerates docs/results.md numbers.

Reads the study's own yfinance cache (studies/238-betting-against-beta/_cache/bab_prices.parquet),
runs the Frazzini-Pedersen BAB portfolio construction (rolling beta ranking, leveraged low-beta
long leg, deleveraged high-beta short leg), and reports vs SPY.

    python studies/238-betting-against-beta/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from betting_against_beta import data, strategy as st  # noqa: E402


def main() -> None:
    prices, spy = data.fetch_panel()

    if prices.empty:
        print("Cache not found. Run from repo root with network access to populate the cache.")
        return

    print("Study 238 -- Betting-Against-Beta -- real yfinance panel\n")
    fp_prices = data.fingerprint(prices)
    fp_spy = data.fingerprint(spy)
    print(f"Prices panel: {prices.shape}, fingerprint={fp_prices}")
    print(f"SPY series:   {spy.shape}, fingerprint={fp_spy}")
    print(f"Date range: {prices.index[0].date()} -- {prices.index[-1].date()}")
    print(f"As-of: 2026-06-16 | Source: yfinance daily adjusted-close prices\n")

    result = st.bab_portfolio(prices, spy, beta_window=252, min_stocks=20)

    if result.empty:
        print("Could not construct portfolio (too few months).")
        return

    s_bab = st.summary(result["bab_ret"])
    s_low = st.summary(result["low_beta_ret"])
    s_high = st.summary(result["high_beta_ret"])
    s_mkt = st.summary(result["mkt_ret"])

    print("=== BAB Portfolio (leveraged low-beta long, deleveraged high-beta short) ===")
    print(f"  N months:            {s_bab['n']}")
    print(f"  Mean return (ann):   {s_bab['mean']*100:+.2f}%/yr")
    print(f"  Vol (ann):           {s_bab['vol']*100:.2f}%/yr")
    print(f"  Sharpe (ann):        {s_bab['sharpe']:+.3f}")
    print(f"  HAC t-stat:          {s_bab['tstat']:+.3f}")
    print(f"  Hit rate:            {s_bab['hit_rate']*100:.1f}%")
    print(f"  Max drawdown:        {s_bab['max_drawdown']*100:.1f}%")

    print("\n=== Legs vs SPY ===")
    print(f"  Low-beta leg (ann):  {s_low['mean']*100:+.2f}%/yr")
    print(f"  High-beta leg (ann): {s_high['mean']*100:+.2f}%/yr")
    print(f"  SPY (market, ann):   {s_mkt['mean']*100:+.2f}%/yr")
    print(f"  Avg beta (low leg):  {result['avg_beta_low'].mean():.3f}")
    print(f"  Avg beta (high leg): {result['avg_beta_high'].mean():.3f}")
    print(f"  Avg leverage (long): {result['leverage'].mean():.2f}x (cap={st.MAX_LEVERAGE}x)")
    print(f"  Avg deleverage (sh): {result['deleverage'].mean():.2f}x")

    print("\n=== Year-by-year BAB returns (compounded over months) ===")
    result_copy = result.copy()
    result_copy["year"] = result_copy.index.year
    annual = (
        result_copy.groupby("year")["bab_ret"]
        .apply(lambda x: float((1 + x).prod() - 1))
    )
    for yr, val in annual.items():
        print(f"  {yr}: {val*100:+.1f}%")

    print("\nNOTE Survivorship bias: universe = current S&P 500 names projected backwards.")
    print("     Results are upper-bound estimates. Leverage cap = 2.0x (stated assumption).")
    print("     Risk-free rate approximated at 3%/yr (constant) -- no FRED dependency.")


if __name__ == "__main__":
    main()
