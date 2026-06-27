"""Real-panel verification -- Study 535 (Mispricing-Score). Prints every number.

Reads the study's own yfinance cache (studies/535-mispricing-score/_cache/),
builds the Stambaugh-Yu-Yuan composite mispricing score (average of six
price-only anomaly ranks), runs the quintile long-short, the long-only and
short-only legs, costs + borrow, the placebo label-shuffle null, and the
synthetic positive control.

    python studies/535-mispricing-score/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mispricing_score import data, strategy as st  # noqa: E402


def main() -> None:
    print("=" * 70)
    print("Study 535 -- Mispricing-Score -- synthetic positive control (engine check)")
    print("=" * 70)
    ctrl = st.synthetic_control()
    print(ctrl.round(3).to_string(index=False))
    print("(monotone in the planted premium, ~0 at the null = faithful engine)\n")

    prices, spy = data.fetch_panel()
    if prices.empty:
        print("Cache not found. Run from repo root with network access to populate the cache.")
        return

    print("=" * 70)
    print("Study 535 -- Mispricing-Score -- real yfinance panel")
    print("=" * 70)
    fp_prices = data.fingerprint(prices)
    fp_spy = data.fingerprint(spy)
    print(f"Prices panel: {prices.shape}, fingerprint={fp_prices}")
    print(f"SPY series:   {spy.shape}, fingerprint={fp_spy}")
    print(f"Date range:   {prices.index[0].date()} -- {prices.index[-1].date()}")
    print(f"As-of: 2026-06-27 | Source: yfinance daily adjusted-close\n")

    result = st.long_short(prices)
    if result.empty:
        print("Could not construct the long-short (too few months).")
        return

    s_ls = st.summary(result["ls_ret"])
    s_long = st.summary(result["long_ret"])
    s_short = st.summary(result["short_ret"])
    # short LEG as a standalone book = short the overpriced names -> -short_ret
    s_shortbook = st.summary(-result["short_ret"])

    net = st.apply_costs(result)
    s_net = st.summary(net)

    print("=== Composite long-short (long cheap quintile, short overpriced quintile) ===")
    print(f"  N months:            {s_ls['n']}")
    print(f"  Mean return (ann):   {s_ls['mean']*100:+.2f}%/yr  GROSS")
    print(f"  Vol (ann):           {s_ls['vol']*100:.2f}%/yr")
    print(f"  Sharpe (ann):        {s_ls['sharpe']:+.3f}")
    print(f"  HAC t-stat:          {s_ls['tstat']:+.3f}")
    print(f"  Hit rate:            {s_ls['hit_rate']*100:.1f}%")
    print(f"  Max drawdown:        {s_ls['max_drawdown']*100:.1f}%")
    print(f"  Avg turnover/mo:     {result['turnover'].mean()*100:.1f}%")

    print("\n=== Net of costs (10 bps/side x turnover x 2 legs + 75 bps/yr borrow) ===")
    print(f"  Mean return (ann):   {s_net['mean']*100:+.2f}%/yr  NET")
    print(f"  HAC t-stat:          {s_net['tstat']:+.3f}")
    print(f"  Sharpe (ann):        {s_net['sharpe']:+.3f}")

    print("\n=== Legs (where is the edge? SYY say the SHORT leg) ===")
    print(f"  Long (cheap) leg:      {s_long['mean']*100:+.2f}%/yr  (t={s_long['tstat']:+.2f})")
    print(f"  Short-names leg (raw): {s_short['mean']*100:+.2f}%/yr")
    print(f"  Short BOOK (short them): {s_shortbook['mean']*100:+.2f}%/yr (t={s_shortbook['tstat']:+.2f})")

    print("\n=== Placebo label-shuffle null (200 shuffles) ===")
    pl = st.placebo_null(prices, n_shuffles=200)
    print(f"  Observed L-S mean:   {pl['observed_mean']*100:+.2f}%/yr")
    print(f"  Null mean:           {pl['null_mean']*100:+.2f}%/yr")
    print(f"  Null std:            {pl['null_std']*100:.2f}%/yr")
    print(f"  Placebo p-value:     {pl['p_value']:.3f}")

    print("\n=== Year-by-year composite L-S (compounded) ===")
    rc = result.copy()
    rc["year"] = rc.index.year
    annual = rc.groupby("year")["ls_ret"].apply(lambda x: float((1 + x).prod() - 1))
    for yr, val in annual.items():
        print(f"  {yr}: {val*100:+.1f}%")

    print("\nNOTE Survivorship bias: universe = current S&P 500 names projected backwards.")
    print("     The SHORT (overpriced) leg -- where SYY locate the edge -- is exactly")
    print("     the leg most damaged by survivorship (the worst names delisted).")
    print("     Fundamental anomalies (accruals/issuance/profitability/distress) need")
    print("     clean per-name financials yfinance does not deliver; the composite here")
    print("     uses the SIX price-computable anomalies. Results are upper-bound.")


if __name__ == "__main__":
    main()
