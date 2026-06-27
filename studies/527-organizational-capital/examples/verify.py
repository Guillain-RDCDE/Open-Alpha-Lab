"""Real-tape verification -- Study 527 (Organizational-Capital). Regenerates docs/results.md.

Reads the study's own caches (EDGAR fundamentals + yfinance prices under
studies/527-organizational-capital/_cache/), builds the org-capital stock by perpetual
inventory of SG&A (Eisfeldt-Papanikolaou 2013), scales by assets, sorts the basket annually,
forms the high-minus-low long-short, and PRINTS every number: leg returns, the long-short
mean and Newey-West HAC t, the label-shuffle placebo p-value, net-of-cost results, robustness,
and a deterministic synthetic positive control.

    python studies/527-organizational-capital/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd  # noqa: E402

from organizational_capital import data, strategy as st  # noqa: E402


def main() -> None:
    fund = data.fetch_fundamentals()
    prices = data.fetch_prices()

    if fund.empty or prices.empty:
        print("Cache not found. Run from repo root with network access to populate _cache/.")
        return

    print("Study 527 -- Organizational-Capital -- real EDGAR + yfinance tape\n")
    fp_fund = data.fingerprint(fund.set_index(["ticker", "fyear"]))
    fp_px = data.fingerprint(prices)
    print(f"Fundamentals (SG&A + assets): {fund.shape}, "
          f"{fund['ticker'].nunique()} firms, fp={fp_fund}")
    print(f"Prices (daily adj-close):     {prices.shape}, fp={fp_px}")
    print(f"Fund years: {fund['fyear'].min()} -- {fund['fyear'].max()} | "
          f"Prices: {prices.index[0].date()} -- {prices.index[-1].date()}")
    print("As-of: 2026-06-26 | Source: EDGAR data.sec.gov (10-K) + yfinance\n")

    oc = st.org_capital_stock(fund)
    monthly = prices.resample("ME").last().pct_change().dropna(how="all")
    ls = st.long_short(oc, monthly)

    if ls.empty:
        print("Could not construct long-short (too few names).")
        return

    s_ls = st.summary(ls["ls"])
    s_hi = st.summary(ls["high"])
    s_lo = st.summary(ls["low"])

    print("=== Org-capital long-short (HIGH oc/assets minus LOW oc/assets) ===")
    print(f"  N months:           {s_ls['n']} ({ls.index[0].date()} -- {ls.index[-1].date()})")
    print(f"  Basket (priced):    {monthly.shape[1]} names")
    print(f"  Mean LS (ann):      {s_ls['mean']*100:+.2f}%/yr  [GROSS]")
    print(f"  Vol (ann):          {s_ls['vol']*100:.2f}%/yr")
    print(f"  Sharpe (ann):       {s_ls['sharpe']:+.3f}")
    print(f"  HAC t-stat:         {s_ls['tstat']:+.3f}   <-- the inference bar (|t|>=2)")
    print(f"  Hit rate:           {s_ls['hit_rate']*100:.1f}%")
    print(f"  Max drawdown:       {s_ls['max_drawdown']*100:.1f}%")

    print("\n=== Legs ===")
    print(f"  High-OC leg (ann):  {s_hi['mean']*100:+.2f}%/yr")
    print(f"  Low-OC leg (ann):   {s_lo['mean']*100:+.2f}%/yr")
    print(f"  E-P (2013) predict: high-OC > low-OC (a POSITIVE long-short)")

    print("\n=== Net of costs ===")
    net = st.apply_costs(ls["ls"], ls["rebalance"])
    s_net = st.summary(net)
    print(f"  Costs: {st.COST_BPS:.0f} bps/leg one-way + {st.BORROW_BPS_ANNUAL:.0f} bps/yr "
          f"borrow on short leg (annual rebalance, 100% turnover assumed)")
    print(f"  Net LS (ann):       {s_net['mean']*100:+.2f}%/yr")
    print(f"  Net HAC t-stat:     {s_net['tstat']:+.3f}")

    print("\n=== Placebo (shuffle oc/assets labels within each year) ===")
    pb = st.placebo_pvalue(oc, monthly, n_shuffles=500)
    print(f"  Real LS mean (mo):  {pb['real_mean']*100:+.4f}%/mo")
    print(f"  Null mean (mo):     {pb['null_mean']*100:+.4f}%  std {pb['null_std']*100:.4f}%")
    print(f"  Placebo p-value:    {pb['p_value']:.3f}  ({pb['n_shuffles']} shuffles)")

    print("\n=== Robustness: depreciation rate delta ===")
    for d in (0.10, 0.15, 0.20, 0.30):
        oc_d = st.org_capital_stock(fund, delta=d)
        ls_d = st.long_short(oc_d, monthly)
        s_d = st.summary(ls_d["ls"])
        print(f"  delta={d:.2f}: LS {s_d['mean']*100:+.2f}%/yr  HAC t {s_d['tstat']:+.3f}")

    print("\n=== Year-by-year long-short ===")
    ls2 = ls.copy()
    ls2["year"] = pd.DatetimeIndex(ls2.index).year
    annual = ls2.groupby("year")["ls"].apply(lambda x: float((1 + x).prod() - 1))
    for yr, val in annual.items():
        print(f"  {yr}: {val*100:+.1f}%")

    print("\n=== Synthetic positive control (engine power check ONLY, 12-seed avg) ===")
    ctrl = st.synthetic_control()
    for _, r in ctrl.iterrows():
        print(f"  planted premium {r['premium']:+.2f}: "
              f"LS {r['mean_ls_pct']:+.2f}%/yr  mean t {r['mean_t']:+.2f}")

    print("\nNOTE Survivorship bias: basket = current large-caps still trading in 2026.")
    print("     Delisted firms (the high-risk tail E-P put on the long leg) are absent.")
    print("     Results are upper-bound estimates. Fundamentals = EDGAR 10-K SG&A + assets.")


if __name__ == "__main__":
    main()
