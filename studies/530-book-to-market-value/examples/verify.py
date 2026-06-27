"""Real-panel verification -- Study 530 (Book-To-Market-Value). PRINTS every number.

Reads the study's own yfinance cache (studies/530-book-to-market-value/_cache/*.parquet),
runs the Fama-French book-to-market quintile sort (long high-B/M value Q5, short low-B/M
growth Q1), reports the HAC t-stat, a label-shuffle placebo p-value, costs x turnover
(+ borrow on the short leg), and the deterministic synthetic positive control.

    python studies/530-book-to-market-value/examples/verify.py

On a cold cache, populate it once (network required) with:
    python studies/530-book-to-market-value/examples/verify.py --fetch
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from book_to_market_value import data, strategy as st  # noqa: E402

AS_OF = "2026-06-27"


def main() -> None:
    fetch = "--fetch" in sys.argv
    bm, fwd = data.fetch_panel(fetch=fetch)

    if bm.empty:
        print("Cache not found. Run once with network access: "
              "python examples/verify.py --fetch")
        return

    print("Study 530 -- Book-To-Market-Value (Fama-French HML) -- real yfinance panel\n")
    print(f"B/M panel:   {bm.shape[0]} years x {bm.shape[1]} tickers | "
          f"fingerprint={data.fingerprint(bm)}")
    print(f"Fwd-ret:     {fwd.shape[0]} years x {fwd.shape[1]} tickers | "
          f"fingerprint={data.fingerprint(fwd)}")
    print(f"Signal years: {list(bm.index)}  (forward years = signal+1)")
    print(f"As-of: {AS_OF} | Source: yfinance balance_sheet (Stockholders Equity / "
          f"Ordinary Shares Number) + adjusted close\n")

    sort = st.quintile_sort(bm, fwd)
    if sort.empty:
        print("Could not build the sort (too few firms/years).")
        return

    s_hedge = st.summary(sort["hedge"])
    s_long = st.summary(sort["long"])
    s_short = st.summary(sort["short"])
    s_mkt = st.summary(sort["market"])

    print("=== B/M quintile sort: long value (Q5) / short growth (Q1) ===")
    print(f"  N years (annual hedge obs):  {s_hedge['n']}")
    print(f"  Hedge mean (Q5 - Q1):        {s_hedge['mean']*100:+.2f}%/yr")
    print(f"  Hedge vol:                   {s_hedge['vol']*100:.2f}%/yr")
    print(f"  Hedge Sharpe:                {s_hedge['sharpe']:+.3f}")
    print(f"  HAC t-stat:                  {s_hedge['tstat']:+.3f}   (bar = |t| >= 2)")
    print(f"  Hit rate (years > 0):        {s_hedge['hit_rate']*100:.1f}%")
    print(f"  Max drawdown:                {s_hedge['max_drawdown']*100:.1f}%")

    print("\n=== Legs ===")
    print(f"  Value leg Q5 (long):         {s_long['mean']*100:+.2f}%/yr")
    print(f"  Growth leg Q1 (short):       {s_short['mean']*100:+.2f}%/yr")
    print(f"  Equal-weight universe:       {s_mkt['mean']*100:+.2f}%/yr")

    print("\n=== Year-by-year hedge (Q5 value - Q1 growth) ===")
    for y, row in sort.iterrows():
        print(f"  {int(y)} signal -> {int(y)+1} return:  "
              f"value {row['long']*100:+6.1f}%  growth {row['short']*100:+6.1f}%  "
              f"hedge {row['hedge']*100:+6.1f}%")

    print("\n=== Placebo (label-shuffle null, 2000 shuffles) ===")
    pb = st.placebo_pvalue(bm, fwd, n_shuffles=2000)
    print(f"  Real mean hedge:             {pb['real_mean']*100:+.2f}%/yr")
    print(f"  Null mean:                   {pb['null_mean']*100:+.2f}%/yr "
          f"(std {pb['null_std']*100:.2f}%)")
    print(f"  Two-sided p-value:           {pb['p_value']:.3f}   "
          f"(< 0.05 would be significant)")

    print("\n=== Costs (one-way bps x NAV x turnover + borrow on short leg) ===")
    cs = st.apply_costs(sort, bm)
    print(f"  Avg annual turnover:         {cs['avg_turnover']*100:.1f}%")
    print(f"  Cost drag (10 bps one-way):  {cs['cost_drag']*100:.2f}%/yr")
    print(f"  Borrow drag (50 bps short):  {cs['borrow_drag']*100:.2f}%/yr")
    print(f"  GROSS hedge mean:            {cs['gross_mean']*100:+.2f}%/yr "
          f"(t={cs['gross_t']:+.3f})")
    print(f"  NET hedge mean:              {cs['net_mean']*100:+.2f}%/yr "
          f"(t={cs['net_t']:+.3f})")

    print("\n=== Synthetic positive control (faithful-engine check ONLY) ===")
    ctrl = st.synthetic_control()
    for _, r in ctrl.iterrows():
        print(f"  planted premium {r['premium']:+.2f} -> hedge "
              f"{r['hedge_mean_%']:+7.2f}%/yr  t={r['tstat']:+.2f}")
    print("  (Engine recovers a positive value premium ONLY when one is planted; "
          "monotone in the premium.)")

    print("\nNOTE Survivorship bias: basket = large-caps still trading in 2026, projected "
          "backwards.")
    print("     Delisted value traps (the natural deep-value long) are absent -> "
          "upper-bound magnitudes.")
    print("     DATA LIMITATION: yfinance serves only ~4-5 annual balance sheets per name, "
          "so the")
    print("     real B/M time series is only "
          f"{s_hedge['n']} cross-sectional years -- a low-powered HAC test by construction.")


if __name__ == "__main__":
    main()
