"""Real-panel verification -- Study 505 (Left-Tail-Momentum). Prints every number.

Reads the study's own yfinance cache (studies/505-left-tail-momentum/_cache/), runs the
Atilgan-Bali-Demirtas-Gunaydin (2020) left-tail-momentum book (sort on trailing left-tail
VaR / worst day; long the safe tail, short the crashed tail; one execution lag; monthly
hold), and reports the long-short spread with a one-sample t, a placebo label-shuffle null,
costs x turnover (+ borrow), seed-robust synthetic control, and a signal-flavour sweep.

    python studies/505-left-tail-momentum/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from left_tail_momentum import data, strategy as st  # noqa: E402

COST_BPS = 5.0          # one-way per leg, per rebalance
BORROW_BPS = 50.0       # annual borrow on the short (crashed) leg


def main() -> None:
    prices, spy = data.fetch_panel()
    if prices.empty:
        print("Cache not found. Run from repo root WITH network access to populate the cache.")
        return

    print("Study 505 -- Left-Tail-Momentum -- real yfinance panel\n")
    fp_prices = data.fingerprint(prices)
    fp_spy = data.fingerprint(spy)
    print(f"Prices panel: {prices.shape}, fingerprint={fp_prices}")
    print(f"SPY series:   {spy.shape}, fingerprint={fp_spy}")
    print(f"Date range:   {prices.index[0].date()} -- {prices.index[-1].date()}")
    print("As-of: 2026-06-26 | Source: yfinance daily adjusted-close (auto_adjust=True)\n")

    # --- Headline: var5 quintile long-short -------------------------------------
    book = st.long_short_book(prices, kind="var5", lookback=252, min_stocks=20,
                              cost_bps=COST_BPS, borrow_ann_bps=BORROW_BPS)
    if book.empty:
        print("Could not construct the book (too few months).")
        return

    s_gross = st.summary(book["ls_ret"])
    s_net = st.summary(book["ls_ret_net"])
    s_long = st.summary(book["long_ret"])
    s_short = st.summary(book["short_ret"])

    print("=== Left-tail-momentum long-short (var5 quintiles: long safe, short crashed) ===")
    print(f"  N months:            {s_gross['n']} "
          f"({book.index[0].date()} -- {book.index[-1].date()})")
    print(f"  Mean spread (ann):   {s_gross['mean']*100:+.2f}%/yr  (gross)")
    print(f"  Vol (ann):           {s_gross['vol']*100:.2f}%/yr")
    print(f"  Sharpe (ann):        {s_gross['sharpe']:+.3f}")
    print(f"  One-sample t:        {s_gross['tstat']:+.3f}   <-- inference bar |t|>=2")
    print(f"  HAC t:               {s_gross['hac_t']:+.3f}")
    print(f"  Hit rate:            {s_gross['hit_rate']*100:.1f}%")
    print(f"  Max drawdown:        {s_gross['max_drawdown']*100:.1f}%")
    print(f"  Avg turnover/mo:     {book['turnover'].mean()*100:.1f}%")

    print("\n=== Legs ===")
    print(f"  Long  (safe tail):   {s_long['mean']*100:+.2f}%/yr")
    print(f"  Short (crashed tail):{s_short['mean']*100:+.2f}%/yr")
    print(f"  (continuation expects short leg to keep underperforming -> positive spread)")

    print("\n=== Net of costs ===")
    print(f"  Cost: {COST_BPS:.0f} bps one-way/leg x turnover, borrow {BORROW_BPS:.0f} bps/yr on short")
    print(f"  Mean spread (net):   {s_net['mean']*100:+.2f}%/yr")
    print(f"  One-sample t (net):  {s_net['tstat']:+.3f}")

    # --- Placebo label-shuffle null ---------------------------------------------
    print("\n=== Placebo (label-shuffle null, 200 random sorts) ===")
    pl = st.placebo_pvalue(prices, kind="var5", lookback=252, min_stocks=20,
                           n_shuffles=200, seed=505)
    print(f"  Real mean spread/mo: {pl['real_mean']*100:+.3f}%")
    print(f"  Placebo p-value:     {pl['p_value']:.3f}  "
          f"(fraction of random sorts >= real)")

    # --- Signal-flavour sweep ---------------------------------------------------
    print("\n=== Signal-flavour sweep (each long-short, gross) ===")
    for kind in ("var5", "var1", "worst"):
        b = st.long_short_book(prices, kind=kind, lookback=252, min_stocks=20)
        s = st.summary(b["ls_ret"]) if not b.empty else None
        if s:
            print(f"  {kind:6s}: mean {s['mean']*100:+6.2f}%/yr | "
                  f"t {s['tstat']:+.3f} | Sharpe {s['sharpe']:+.3f} | n {s['n']}")

    # --- Lookback sensitivity ----------------------------------------------------
    print("\n=== Lookback sensitivity (var5, gross) ===")
    for lb in (126, 252, 504):
        if len(prices) <= lb:
            continue
        b = st.long_short_book(prices, kind="var5", lookback=lb, min_stocks=20)
        s = st.summary(b["ls_ret"]) if not b.empty else None
        if s:
            print(f"  lookback {lb:4d}d: mean {s['mean']*100:+6.2f}%/yr | "
                  f"t {s['tstat']:+.3f} | n {s['n']}")

    # --- Synthetic positive control (seed-robust) -------------------------------
    print("\n=== Synthetic positive control (20 seeds, Welch-style averaged) ===")
    ctrl = st.synthetic_control(data, premiums=(0.0, 0.04, 0.08), n_seeds=20,
                                kind="var5", lookback=252, base_seed=505)
    for _, row in ctrl.iterrows():
        print(f"  premium {row['premium']:+.2f}: mean LS {row['mean_ls_pct']:+6.2f}%/yr | "
              f"avg t {row['mean_t']:+.3f} | share(t>=2) {row['share_t2']*100:.0f}%")

    print("\n=== Year-by-year long-short (var5, gross, compounded) ===")
    bc = book.copy()
    bc["year"] = bc.index.year
    annual = bc.groupby("year")["ls_ret"].apply(lambda x: float((1 + x).prod() - 1))
    for yr, val in annual.items():
        print(f"  {yr}: {val*100:+.1f}%")

    print("\nNOTE Survivorship bias: basket = large-cap names still trading in 2026.")
    print("     The worst left-tail names of the past (firms that crashed into delisting)")
    print("     are ABSENT -- exactly the bias that flatters a short-the-crashed book.")
    print("     One execution lag (enter close 1 day after signal). Costs labelled gross vs net.")


if __name__ == "__main__":
    main()
