"""Real-panel verification -- Study 508 (Momentum-Crashes). Regenerates docs/results.md numbers.

Reads the study's own yfinance cache (studies/508-momentum-crashes/_cache/momcrash_prices.parquet
and momcrash_spy.parquet), builds the Jegadeesh-Titman 12-1 winners-minus-losers book, dissects
its crash dynamics (worst months, deepest drawdown), conditions it on Daniel-Moskowitz bear/panic
regimes, and tests whether vol-scaling repairs the left tail. Prints every number.

    python studies/508-momentum-crashes/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from momentum_crashes import data, strategy as st  # noqa: E402

FRAC = 0.2          # quintile sort
COST_BPS = 10.0     # one-way bps per leg per rebalance
BORROW_BPS = 50.0   # annual short-borrow on the loser leg


def main() -> None:
    prices = data.fetch_panel()
    market = data.fetch_market()

    if prices.empty:
        print("Cache not found. Run with fetch=True and network access to populate the cache.")
        return

    prices = data.drop_partial_last_month(prices)
    mp = st.to_monthly(prices)
    mkt_m = st.to_monthly_series(market) if not market.empty else market

    print("Study 508 -- Momentum-Crashes -- real yfinance panel\n")
    fp_prices = data.fingerprint(prices)
    fp_spy = data.fingerprint(market) if not market.empty else "n/a"
    print(f"Prices panel: {prices.shape}, fingerprint={fp_prices}")
    print(f"SPY proxy:    {market.shape}, fingerprint={fp_spy}")
    print(f"Date range:   {prices.index[0].date()} -- {prices.index[-1].date()}")
    print("As-of: 2026-06-26 | Source: yfinance daily adjusted-close prices\n")

    # ---- The WML book (gross & net) -------------------------------------
    ls = st.long_short(mp, frac=FRAC, cost_bps=COST_BPS, borrow_ann_bps=BORROW_BPS)
    if ls.empty:
        print("Could not construct the WML book (too few months).")
        return

    s_g = st.summary(ls["wml_gross"])
    s_n = st.summary(ls["wml_net"])
    print("=== 12-1 Winners-minus-Losers (quintile, dollar-neutral) ===")
    print(f"  N holding months:   {s_g['n']} ({ls.index[0].date()} -- {ls.index[-1].date()})")
    print(f"  GROSS mean (ann):   {s_g['mean']*100:+.2f}%/yr")
    print(f"  NET mean (ann):     {s_n['mean']*100:+.2f}%/yr  (10bps/leg + 50bps borrow)")
    print(f"  Vol (ann):          {s_g['vol']*100:.2f}%/yr")
    print(f"  Sharpe (gross):     {s_g['sharpe']:+.3f}")
    print(f"  HAC t-stat (gross): {s_g['tstat']:+.3f}")
    print(f"  Hit rate:           {s_g['hit_rate']*100:.1f}%")
    print(f"  Max drawdown:       {s_g['max_dd']*100:.1f}%")
    print(f"  Worst month:        {s_g['worst']*100:+.1f}%")
    print(f"  Skewness:           {s_g['skew']:+.2f}  (negative = crash-prone left tail)")
    print(f"  Avg turnover:       {ls['turnover'].mean()*100:.1f}%/mo")

    # ---- Placebo null ---------------------------------------------------
    pb = st.placebo_pvalue(mp, frac=FRAC, n_perm=2000, seed=508)
    print("\n=== Label-shuffle placebo null (2000 permutations) ===")
    print(f"  Real WML mean (ann):    {pb['real_mean_ann']*100:+.2f}%/yr")
    print(f"  Placebo mean (ann):     {pb['placebo_mean_ann']*100:+.2f}%/yr")
    print(f"  One-sided p-value:      {pb['p_value']:.3f}")

    # ---- Crash anatomy --------------------------------------------------
    ep = st.max_drawdown_episode(ls["wml_gross"])
    print("\n=== Crash anatomy ===")
    print(f"  Deepest drawdown:   {ep['max_dd']*100:.1f}%  "
          f"({ep['peak'].date()} -> {ep['trough'].date()})")
    print("  Worst 6 WML months (gross):")
    ct = st.crash_table(ls, k=6)
    for dt, row in ct.iterrows():
        print(f"    {dt.date()}: WML {row['wml']*100:+.1f}%  "
              f"(win {row['win_leg']*100:+.1f}%, los {row['los_leg']*100:+.1f}%)")

    # ---- Regime conditioning (Daniel-Moskowitz) -------------------------
    if not market.empty:
        rs = st.regime_split(ls, mkt_m, col="wml_gross")
        print("\n=== WML conditioned on market regime (annualised mean) ===")
        for reg, row in rs.iterrows():
            print(f"    {reg:<26s}: {row['wml_mean_ann']*100:+7.2f}%/yr  (n={int(row['n_months'])})")

    # ---- Vol-scaling repair --------------------------------------------
    vs = st.vol_scaled(ls, col="wml_gross")
    s_vs = st.summary(vs)
    print("\n=== Vol-scaled (dynamic) momentum -- the Daniel-Moskowitz repair ===")
    print(f"  Mean (ann):         {s_vs['mean']*100:+.2f}%/yr")
    print(f"  Vol (ann):          {s_vs['vol']*100:.2f}%/yr")
    print(f"  Sharpe:             {s_vs['sharpe']:+.3f}  (vs gross {s_g['sharpe']:+.3f})")
    print(f"  HAC t-stat:         {s_vs['tstat']:+.3f}")
    print(f"  Max drawdown:       {s_vs['max_dd']*100:.1f}%  (vs gross {s_g['max_dd']*100:.1f}%)")
    print(f"  Worst month:        {s_vs['worst']*100:+.1f}%  (vs gross {s_g['worst']*100:+.1f}%)")
    print(f"  Skewness:           {s_vs['skew']:+.2f}  (vs gross {s_g['skew']:+.2f})")

    # ---- Synthetic positive control -------------------------------------
    print("\n=== Synthetic positive control (engine faithfulness, NOT a real-tape claim) ===")
    ctrl = st.synthetic_control(strengths=(0.0, 0.15, 0.30, 0.50), frac=FRAC, seed=508)
    for _, row in ctrl.iterrows():
        print(f"    mom_strength={row['mom_strength']:.2f}: "
              f"WML {row['wml_mean_ann']*100:+6.2f}%/yr  t={row['tstat']:+.2f}  (n={int(row['n'])})")

    print("\nNOTE Survivorship: basket = current large-caps still trading in 2026, projected back.")
    print("     The loser leg's natural shorts (firms that trended to delisting) are absent, so")
    print("     the premium is an upper bound and the crash an under-statement. Named on SIGNAL.")
    print("     One execution lag (form on month-end close, hold next month). Costs: 10bps/leg")
    print("     + 50bps/yr borrow. Gross vs net labelled. Price-only (adjusted-close).")


if __name__ == "__main__":
    main()
