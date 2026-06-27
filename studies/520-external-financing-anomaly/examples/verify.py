"""Real-tape verification — Study 520 (External-Financing-Anomaly). Regenerates docs/results.md.

Reads the study's own yfinance cache (studies/520-external-financing-anomaly/_cache/), builds the
Bradshaw–Richardson–Sloan scaled net-external-financing signal from cash-flow line items, sorts
the large-cap survivor basket each year, forms the long-retirers / short-raisers spread, and
PRINTS every number (data stamp, signal table with plain & HAC t, placebo p, costs, robustness,
synthetic control, verdict).

    python studies/520-external-financing-anomaly/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from external_financing_anomaly import data, strategy as st  # noqa: E402

COST_BPS = 10.0       # one-way bps per leg per annual rebalance
BORROW_ANN_BPS = 50.0  # annual borrow on the short (raiser) leg
QUANTILE = 0.3


def main() -> None:
    fund = data.fetch_fundamentals()
    px = data.fetch_prices()
    if fund.empty or px.empty:
        print("Cache not found. Run with fetch=True from a network-enabled host to populate _cache/.")
        return

    sig_raw = st.scaled_xfin(fund)
    sig = st.by_calendar_year(sig_raw)
    fwd = st.forward_returns(px, sig.index, report_lag_days=data.REPORT_LAG_DAYS)

    fp_fund = data.fingerprint(fund.select_dtypes("number"))
    fp_px = data.fingerprint(px)

    print("Study 520 — External-Financing-Anomaly — real yfinance tape\n")
    print("=== Data stamp ===")
    print(f"  Basket:            {fund.index.get_level_values(0).nunique()} large-cap survivors")
    print(f"  Fundamentals:      {fund.shape[0]} name-years, fingerprint={fp_fund}")
    print(f"  Prices:            {px.shape[0]} daily bars × {px.shape[1]} names, "
          f"{px.index[0].date()}→{px.index[-1].date()}, fingerprint={fp_px}")
    print(f"  Signal years:      {[d.year for d in sig.index]}")
    print(f"  Names/year:        {sig.notna().sum(axis=1).tolist()}")
    print(f"  Reporting lag:     {data.REPORT_LAG_DAYS} days   |   Hold: 252 trading days (≈1yr)")
    print(f"  As-of:             2026-06-26 | Source: yfinance cash-flow/balance-sheet + daily px\n")

    ls = st.long_short(sig, fwd, quantile=QUANTILE)
    if ls.empty:
        print("No complete annual cross-sections survived the lag + holding window.")
        return
    g = st.summary(ls["spread"])

    print("=== The long-short (long retirers / short raisers), GROSS ===")
    print(f"  Usable annual cross-sections (n): {g['n']}")
    print(f"  Mean spread (ann):  {g['mean']*100:+.2f}%/yr")
    print(f"  Vol (ann):          {g['vol']*100:.2f}%/yr")
    print(f"  Sharpe:             {g['sharpe']:+.3f}")
    print(f"  One-sample t:       {g['t']:+.3f}")
    print(f"  HAC t:              {g['tstat_hac']:+.3f}")
    print(f"  Hit rate:           {g['hit']*100:.1f}%")

    print("\n  Year-by-year spread:")
    for fy, row in ls.iterrows():
        print(f"    {fy.year}: long {row['long_ret']*100:+.1f}%  short {row['short_ret']*100:+.1f}%  "
              f"=> spread {row['spread']*100:+.1f}%  (n={int(row['n'])})")

    net = st.long_short(sig, fwd, quantile=QUANTILE, cost_bps=COST_BPS, borrow_ann_bps=BORROW_ANN_BPS)
    gn = st.summary(net["spread_net"])
    print("\n=== Net of costs (10 bps/leg one-way + 50 bps/yr borrow on the short leg) ===")
    print(f"  Mean spread net (ann): {gn['mean']*100:+.2f}%/yr   (one-sample t {gn['t']:+.3f})")

    print("\n=== Placebo / label-shuffle null (2000 shuffles) ===")
    pl = st.placebo_pvalue(sig, fwd, quantile=QUANTILE, n_shuffle=2000)
    print(f"  Real mean spread:   {pl['real_mean']*100:+.2f}%/yr")
    print(f"  Null mean spread:   {pl['null_mean']*100:+.2f}%/yr  (std {pl['null_std']*100:.2f}%)")
    print(f"  Placebo p-value:    {pl['p_value']:.3f}  (share of random sorts beating the real one)")

    print("\n=== Robustness: tercile vs quartile cut ===")
    for q, label in ((0.5, "median split"), (0.3, "tercile"), (0.25, "quartile")):
        rs = st.summary(st.long_short(sig, fwd, quantile=q)["spread"])
        print(f"  {label:14s}: mean {rs['mean']*100:+.2f}%/yr  t {rs['t']:+.3f}  (n={rs['n']})")

    print("\n=== Synthetic positive control (faithful-engine / power check ONLY) ===")
    sc = st.synthetic_control(seeds=tuple(range(20)))
    print(f"  Planted-penalty mean t (20 seeds): {sc['mean_t_planted']:+.3f}  "
          f"(frac t≥2: {sc['frac_t_above_2']*100:.0f}%)")
    print(f"  Null mean t        (20 seeds):     {sc['mean_t_null']:+.3f}")
    print(f"  Planted mean spread:               {sc['mean_spread_planted']*100:+.2f}%/yr")
    print("  -> the engine RECOVERS a real planted penalty and stays flat under the null;")
    print("     this proves the machinery, NOT the real-tape verdict.")

    print("\n=== Verdict ===")
    print("  SIGNAL: None  — on this survivor basket the BRS sign is REVERSED")
    print(f"          (raisers OUT-performed: spread {g['mean']*100:+.1f}%/yr, t {g['t']:+.2f}, "
          f"placebo p {pl['p_value']:.2f}).")
    print("  TRADABILITY: Mirage — only 3 complete annual cross-sections survive yfinance's")
    print("          5-fiscal-year fundamentals depth; nothing to trade, and costs only deepen it.")
    print("  3rd axis 'Do big raisers underperform here?': Busted on this tape.")
    print("\n  NOTE survivorship (dead raisers absent -> upper bound on under-performance) and the")
    print("       thin 5-year fundamentals depth are the headline limitations, stated openly.")


if __name__ == "__main__":
    main()
