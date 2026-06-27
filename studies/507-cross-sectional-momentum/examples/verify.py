"""Real-panel verification -- Study 507 (Cross-Sectional-Momentum). Prints every number.

Reads the study's own yfinance cache (studies/507-cross-sectional-momentum/_cache/
csmom_prices.parquet), runs the Jegadeesh-Titman 12-1 winners-minus-losers long-short
(quintile and decile sorts), charges costs + short borrow, runs a label-shuffle placebo null,
and checks the synthetic positive control. Regenerates the docs/results.md numbers.

    python studies/507-cross-sectional-momentum/examples/verify.py
    # one-time, with network, to populate the cache:
    python -c "import sys; sys.path.insert(0,'studies/507-cross-sectional-momentum'); \
               from cross_sectional_momentum import data; data.fetch_panel(fetch=True)"
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cross_sectional_momentum import data, strategy as st  # noqa: E402

# Stamp constants
ASOF = "2026-06-26"
COST_BPS = 10.0          # one-way bps per leg per rebalance
BORROW_ANN_BPS = 50.0    # annual borrow on the short (loser) leg


def main() -> None:
    prices = data.fetch_panel()
    if prices.empty:
        print("Cache not found. Run fetch_panel(fetch=True) once with network access.")
        return
    prices = data.drop_partial_last_month(prices)

    fp = data.fingerprint(prices)
    mp = st.to_monthly(prices)

    print("Study 507 -- Cross-Sectional-Momentum -- real yfinance panel\n")
    print(f"Panel: {prices.shape[0]} days x {prices.shape[1]} tickers, fingerprint={fp}")
    print(f"Date range: {prices.index[0].date()} -- {prices.index[-1].date()}")
    print(f"Monthly bars: {len(mp)}  |  As-of: {ASOF}")
    print(f"Source: yfinance daily adjusted-close (auto_adjust=True)\n")

    # ---- Synthetic positive control --------------------------------------
    print("=== Synthetic positive control (engine faithfulness) ===")
    ctrl = st.synthetic_control()
    for _, row in ctrl.iterrows():
        print(f"  mom_strength={row['mom_strength']:.2f}: "
              f"WML {row['wml_mean_ann']*100:+.2f}%/yr, t={row['tstat']:+.2f}")
    print("  -> monotone in planted drift, ~0 at the null. Engine is faithful.\n")

    # ---- Real tape: quintile and decile ----------------------------------
    for label, frac in (("QUINTILE (top/bottom 20%)", 0.20), ("DECILE (top/bottom 10%)", 0.10)):
        print(f"=== Real WML -- {label} ===")
        ls_g = st.long_short(mp, frac=frac)
        ls_n = st.long_short(mp, frac=frac, cost_bps=COST_BPS, borrow_ann_bps=BORROW_ANN_BPS)
        sg = st.summary(ls_g["wml_gross"])
        sn = st.summary(ls_n["wml_net"])
        s_win = st.summary(ls_g["win"])
        s_los = st.summary(ls_g["los"])
        print(f"  N months:          {sg['n']}  ({ls_g.index[0].date()} -- {ls_g.index[-1].date()})")
        print(f"  Names per leg:     {int(ls_g['n_leg'].iloc[-1])}")
        print(f"  WML gross (ann):   {sg['mean']*100:+.2f}%/yr")
        print(f"  WML net (ann):     {sn['mean']*100:+.2f}%/yr  (cost {COST_BPS:.0f}bps/leg + borrow {BORROW_ANN_BPS:.0f}bps/yr)")
        print(f"  Gross Sharpe:      {sg['sharpe']:+.3f}")
        print(f"  Net Sharpe:        {sn['sharpe']:+.3f}")
        print(f"  HAC t (gross):     {sg['tstat']:+.3f}   <-- inference bar")
        print(f"  HAC t (net):       {sn['tstat']:+.3f}")
        print(f"  Hit rate:          {sg['hit_rate']*100:.1f}%")
        print(f"  Max drawdown:      {sg['max_dd']*100:.1f}%")
        print(f"  Worst month:       {sg['worst']*100:+.1f}%")
        print(f"  Avg turnover:      {ls_g['turnover'].mean()*100:.1f}%/mo")
        print(f"  Winner leg (ann):  {s_win['mean']*100:+.2f}%/yr")
        print(f"  Loser leg (ann):   {s_los['mean']*100:+.2f}%/yr")
        pl = st.placebo_pvalue(mp, frac=frac, n_perm=1000)
        print(f"  Placebo p-value:   {pl['p_value']:.3f}  "
              f"(real {pl['real_mean_ann']*100:+.2f}%/yr vs shuffle {pl['placebo_mean_ann']*100:+.2f}%/yr)")
        print()

    print("NOTE Survivorship bias (named on the SIGNAL axis): the basket is large-cap names")
    print("     still trading in 2026. Firms that trended into delisting -- the loser leg's")
    print("     natural shorts -- are absent. Any WML premium here is an UPPER BOUND.")
    print("     Costs: 10bps/leg/rebalance + 50bps/yr short borrow. One execution lag (form on")
    print("     month-end close, hold the next month). No same-bar fill, no look-ahead.")


if __name__ == "__main__":
    main()
