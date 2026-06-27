"""Real-panel verification -- Study 539 (Cash-Flow-Volatility). Regenerates docs/results.md numbers.

Reads the study's own yfinance cache (studies/539-cash-flow-volatility/_cache/*.parquet),
forms trailing cash-flow volatility per name (std of quarterly operating cash flow / total
assets), and runs the Huang (2009) quality-like long-short: LONG low-CF-vol tercile, SHORT
high-CF-vol tercile, equal-weight, one execution-lag day, monthly returns. Prints every number.

    python studies/539-cash-flow-volatility/examples/verify.py

With no cache and no network, prints a friendly notice and exits 0.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from cash_flow_volatility import data, strategy as st  # noqa: E402


def main() -> None:
    prices, spy, cfvol = data.fetch_panel()

    print("Study 539 -- Cash-Flow-Volatility -- synthetic positive control\n")
    ctrl = st.synthetic_control(premiums=(0.0, 0.04, 0.08), n_seeds=20)
    print(ctrl.round(3).to_string(index=False))
    print("  (faithful-engine check ONLY -- never cited for a real-tape stamp)\n")

    if prices.empty or cfvol.empty:
        print("Cache not found. Run once with network access:")
        print("  python -c \"import sys;sys.path.insert(0,'studies/539-cash-flow-volatility');"
              "from cash_flow_volatility import data;data.fetch_panel(fetch=True)\"")
        return

    print("Study 539 -- Cash-Flow-Volatility -- real yfinance panel\n")
    fp_prices = data.fingerprint(prices)
    fp_cfvol = data.fingerprint(cfvol[["cfvol"]])
    print(f"Prices panel: {prices.shape}, fingerprint={fp_prices}")
    print(f"CF-vol panel: {cfvol.shape}, fingerprint={fp_cfvol}")
    print(f"Date range:   {prices.index[0].date()} -- {prices.index[-1].date()}")
    print(f"Quarters used per name: {int(cfvol['n_quarters'].min())}"
          f"--{int(cfvol['n_quarters'].max())} (yfinance fundamentals are thin)")
    print("As-of: 2026-06-26 | Source: yfinance daily adjusted-close + quarterly statements\n")

    ls = st.build_real(prices, cfvol, exec_lag=1)
    if ls.empty:
        print("Could not construct long-short (too few names).")
        return

    s_ls = st.summary(ls["ls_ret"])
    s_low = st.summary(ls["low_ret"])
    s_high = st.summary(ls["high_ret"])
    s_spy = st.summary(spy.pct_change().resample("ME").prod().sub(0) if False
                       else (1 + spy.pct_change()).resample("ME").prod().sub(1).iloc[1:])

    print("=== Long-short (LONG low-CF-vol tercile, SHORT high-CF-vol tercile) ===")
    print(f"  N months:            {s_ls['n']}  ({ls.index.min()} -- {ls.index.max()})")
    print(f"  Names per leg:       {int(ls['n_low'].iloc[0])} long / {int(ls['n_high'].iloc[0])} short")
    print(f"  Mean return (ann):   {s_ls['mean']*100:+.2f}%/yr")
    print(f"  Vol (ann):           {s_ls['vol']*100:.2f}%/yr")
    print(f"  Sharpe (ann):        {s_ls['sharpe']:+.3f}")
    print(f"  HAC t-stat:          {s_ls['tstat']:+.3f}")
    print(f"  Hit rate:            {s_ls['hit_rate']*100:.1f}%")
    print(f"  Max drawdown:        {s_ls['max_drawdown']*100:.1f}%")

    print("\n=== Legs ===")
    print(f"  Low-CF-vol leg (ann):  {s_low['mean']*100:+.2f}%/yr  (the LONG / Huang winners)")
    print(f"  High-CF-vol leg (ann): {s_high['mean']*100:+.2f}%/yr  (the SHORT)")
    print(f"  SPY (market, ann):     {s_spy['mean']*100:+.2f}%/yr")

    print("\n=== Placebo / label-shuffle null (1000 permutations) ===")
    monthly = st.monthly_returns_from_prices(prices, exec_lag=1)
    pl = st.placebo_pvalue(monthly, cfvol["cfvol"], n_perm=1000)
    print(f"  Real mean LS (monthly):  {pl['real_mean']*100:+.3f}%/mo")
    print(f"  Null mean / std:         {pl['null_mean']*100:+.3f}% / {pl['null_std']*100:.3f}%")
    print(f"  Permutation p-value:     {pl['p_value']:.3f}  (two-sided)")

    print("\n=== Costs (10 bps one-way x turnover + 100 bps/yr borrow on short) ===")
    c = st.apply_costs(ls)
    print(f"  Gross mean (ann):  {c['gross_mean']*100:+.2f}%/yr  | HAC t {c['gross_t']:+.3f}")
    print(f"  Net mean (ann):    {c['net_mean']*100:+.2f}%/yr  | HAC t {c['net_t']:+.3f}")
    print(f"  Cost drag (ann):   {c['cost_drag']*100:.2f}%/yr  (turnover {c['turnover']*100:.0f}%/mo)")

    print("\n=== Robustness -- split-fraction sensitivity ===")
    for frac, name in [(0.20, "quintile"), (1.0 / 3.0, "tercile "), (0.50, "half    ")]:
        ls_r = st.build_real(prices, cfvol, tercile=frac)
        s_r = st.summary(ls_r["ls_ret"])
        print(f"  {name} (k={int(ls_r['n_low'].iloc[0])}):  "
              f"mean {s_r['mean']*100:+.2f}%/yr  t {s_r['tstat']:+.2f}  Sharpe {s_r['sharpe']:+.2f}")

    print("\n=== Year-by-year long-short returns ===")
    yb = ls.copy()
    yb["year"] = [p.year for p in yb.index]
    annual = yb.groupby("year")["ls_ret"].apply(lambda x: float((1 + x).prod() - 1))
    for yr, val in annual.items():
        print(f"  {yr}: {val*100:+.1f}%")

    print("\nNOTE Survivorship bias: basket = large-cap names still trading in 2026.")
    print("     Firms whose cash flows blew up and that delisted -- the natural high-CF-vol")
    print("     short -- are absent. Results are upper-bound estimates.")
    print("LIMITATION yfinance serves only ~5 quarters of cash-flow statements, so the CF-vol")
    print("     characteristic is estimated on a thin, recent history.")


if __name__ == "__main__":
    main()
