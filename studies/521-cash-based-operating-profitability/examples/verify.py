"""Real-panel verification -- Study 521 (Cash-Based Operating Profitability).

Pulls yfinance fundamentals + prices for the survivor basket (cache-first to this
study's own _cache/), builds the cash-based OP signal and the accrual-laden
gross-profitability sibling, runs the top-quintile / bottom-quintile long-short with a
one-year reporting lag, charges costs, runs the label-shuffle placebo, and the synthetic
positive control. Prints EVERY number that docs/results.md quotes.

    python studies/521-cash-based-operating-profitability/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cash_op import data, strategy as st  # noqa: E402


def main() -> None:
    cop, gpa, fwd = data.fetch_panel(fetch=True)

    if cop.empty:
        print("No data: yfinance fetch failed and no cache present.")
        return

    print("Study 521 -- Cash-Based Operating Profitability -- real yfinance panel\n")
    print(f"Cash-OP signal:  {cop.shape} (year x ticker)  fingerprint={data.fingerprint(cop)}")
    print(f"Gross-prof (GPA):{gpa.shape}                  fingerprint={data.fingerprint(gpa)}")
    print(f"Forward returns: {fwd.shape}                  fingerprint={data.fingerprint(fwd)}")
    print(f"Years covered:   {int(cop.index.min())} - {int(cop.index.max())}")
    print(f"As-of: 2026-06-26 | Source: yfinance annual statements + daily adj close\n")

    # --- Cash-based OP hedge ---
    h_cop = st.quintile_hedge(cop, fwd, q=0.20)
    h_cop = st.apply_costs(h_cop)
    s_cop = st.summary(h_cop["hedge"])
    s_cop_net = st.summary(h_cop["net"])
    s_hi = st.summary(h_cop["high"])
    s_lo = st.summary(h_cop["low"])
    s_mkt = st.summary(h_cop["market"])

    print("=== Cash-based OP: top-quintile vs bottom-quintile long-short ===")
    print(f"  N years:         {s_cop['n']}")
    print(f"  Mean hedge:      {s_cop['mean']*100:+.2f}%/yr  (gross)")
    print(f"  Mean hedge:      {s_cop_net['mean']*100:+.2f}%/yr  (net of costs)")
    print(f"  Vol:             {s_cop['vol']*100:.2f}%/yr")
    print(f"  Sharpe (annual): {s_cop['sharpe']:+.3f}")
    print(f"  One-sample t:    {s_cop['tstat']:+.3f}")
    print(f"  Hit rate:        {s_cop['hit_rate']*100:.1f}%")
    print(f"  Max drawdown:    {s_cop['max_drawdown']*100:.1f}%")
    print(f"  Mean turnover:   {h_cop['turnover'].mean()*100:.1f}%/yr")
    print(f"  Mean cost:       {h_cop['cost'].mean()*100:.2f}%/yr")

    print("\n=== Legs vs equal-weight market (cash-OP sort) ===")
    print(f"  High cash-OP:    {s_hi['mean']*100:+.2f}%/yr")
    print(f"  Low cash-OP:     {s_lo['mean']*100:+.2f}%/yr")
    print(f"  EW market:       {s_mkt['mean']*100:+.2f}%/yr")
    print(f"  High vs market:  {(s_hi['mean']-s_mkt['mean'])*100:+.2f}%/yr")

    # --- Gross-profitability sibling (Study 122 contrast) ---
    h_gpa = st.quintile_hedge(gpa, fwd, q=0.20)
    s_gpa = st.summary(h_gpa["hedge"])
    print("\n=== Head-to-head: cash-OP vs accrual-laden gross-profitability (GP/A) ===")
    print(f"  cash-OP hedge:   {s_cop['mean']*100:+.2f}%/yr  t={s_cop['tstat']:+.2f}")
    print(f"  GP/A hedge:      {s_gpa['mean']*100:+.2f}%/yr  t={s_gpa['tstat']:+.2f}")
    print(f"  BGLN claim (cash beats accrual): "
          f"{'SUPPORTED' if s_cop['mean'] > s_gpa['mean'] else 'NOT SUPPORTED'} on this tape")

    # --- Placebo ---
    pl = st.placebo_pvalue(cop, fwd, n_shuffles=500)
    print("\n=== Label-shuffle placebo (cash-OP) ===")
    print(f"  Real hedge mean: {pl['real_mean']*100:+.2f}%/yr")
    print(f"  Null mean:       {pl['null_mean']*100:+.2f}%/yr  (std {pl['null_std']*100:.2f}%)")
    print(f"  Placebo p-value: {pl['p_value']:.3f}  ({pl['n_shuffles']} shuffles)")

    # --- Synthetic positive control ---
    ctrl = st.synthetic_control()
    print("\n=== Synthetic positive control (engine faithfulness) ===")
    print(ctrl.round(3).to_string(index=False))

    # --- Year-by-year ---
    print("\n=== Year-by-year (cash-OP hedge) ===")
    for yr in h_cop.index:
        r = h_cop.loc[yr]
        print(f"  {yr}: hedge={r['hedge']*100:+6.1f}%  high={r['high']*100:+6.1f}%  "
              f"low={r['low']*100:+6.1f}%  mkt={r['market']*100:+6.1f}%  n={int(r['n'])}")

    print("\nNOTE Survivorship: basket = large-caps still trading 2026; results are upper bounds.")
    print("NOTE yfinance annual statements expose only ~4-5 fiscal years -> the panel is SHORT.")


if __name__ == "__main__":
    main()
