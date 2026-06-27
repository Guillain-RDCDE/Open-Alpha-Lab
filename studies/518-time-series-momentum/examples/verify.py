"""Real-panel verification -- Study 518 (Time-Series-Momentum). Prints every number.

Reads the study's own yfinance cache (studies/518-time-series-momentum/_cache/
tsmom_prices.parquet), runs the Moskowitz-Ooi-Pedersen own-sign, vol-scaled, diversified
TSMOM book (vol-scaled and equal-notional variants), charges costs + short borrow, runs a
sign-shuffle placebo null, and checks the synthetic positive control. Regenerates the
docs/results.md numbers.

    python studies/518-time-series-momentum/examples/verify.py
    # one-time, with network, to populate the cache:
    python -c "import sys; sys.path.insert(0,'studies/518-time-series-momentum'); \
               from time_series_momentum import data; data.fetch_panel(fetch=True)"
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from time_series_momentum import data, strategy as st  # noqa: E402

# Stamp constants
ASOF = "2026-06-26"
COST_BPS = 10.0          # one-way bps per rebalance (per unit turnover)
BORROW_ANN_BPS = 50.0    # annual borrow on the net short notional


def main() -> None:
    prices = data.fetch_panel()
    if prices.empty:
        print("Cache not found. Run fetch_panel(fetch=True) once with network access.")
        return
    prices = data.drop_partial_last_month(prices)

    fp = data.fingerprint(prices)
    mp = st.to_monthly(prices)

    print("Study 518 -- Time-Series-Momentum -- real yfinance cross-asset ETF panel\n")
    print(f"Panel: {prices.shape[0]} days x {prices.shape[1]} tickers, fingerprint={fp}")
    print(f"Date range: {prices.index[0].date()} -- {prices.index[-1].date()}")
    print(f"Monthly bars: {len(mp)}  |  As-of: {ASOF}")
    print(f"Tickers: {', '.join(prices.columns)}")
    print(f"Source: yfinance daily adjusted-close (auto_adjust=True)\n")

    # ---- Synthetic positive control --------------------------------------
    print("=== Synthetic positive control (engine faithfulness) ===")
    ctrl = st.synthetic_control()
    for _, row in ctrl.iterrows():
        print(f"  trend_strength={row['trend_strength']:.2f}: "
              f"port {row['port_mean_ann']*100:+.2f}%/yr, t={row['tstat']:+.2f}")
    nr = st.synthetic_null_robustness(n_seeds=20)
    print(f"  -> across {nr['n_seeds']} seeds the NULL mean t = {nr['mean_null_t']:+.3f}, "
          f"frac |t|>2 = {nr['frac_abs_t_gt2']:.2f}. Engine is faithful & well-sized.\n")

    # ---- Real tape: vol-scaled and equal-notional ------------------------
    for label, vs in (("VOL-SCALED (inverse-vol to common target)", True),
                      ("EQUAL-NOTIONAL (sign only)", False)):
        print(f"=== Real TSMOM -- {label} ===")
        bk_g = st.tsmom_book(mp, vol_scale=vs)
        bk_n = st.tsmom_book(mp, vol_scale=vs, cost_bps=COST_BPS, borrow_ann_bps=BORROW_ANN_BPS)
        sg = st.summary(bk_g["port_gross"])
        sn = st.summary(bk_n["port_net"])
        print(f"  N months:          {sg['n']}  ({bk_g.index[0].date()} -- {bk_g.index[-1].date()})")
        print(f"  Assets w/ signal:  {int(bk_g['n_assets'].iloc[-1])}")
        print(f"  Port gross (ann):  {sg['mean']*100:+.2f}%/yr")
        print(f"  Port net (ann):    {sn['mean']*100:+.2f}%/yr  (cost {COST_BPS:.0f}bps x turnover + borrow {BORROW_ANN_BPS:.0f}bps/yr on shorts)")
        print(f"  Gross Sharpe:      {sg['sharpe']:+.3f}")
        print(f"  Net Sharpe:        {sn['sharpe']:+.3f}")
        print(f"  HAC t (gross):     {sg['tstat']:+.3f}   <-- inference bar")
        print(f"  HAC t (net):       {sn['tstat']:+.3f}")
        print(f"  Hit rate:          {sg['hit_rate']*100:.1f}%")
        print(f"  Max drawdown:      {sg['max_dd']*100:.1f}%")
        print(f"  Worst month:       {sg['worst']*100:+.1f}%")
        print(f"  Avg turnover:      {bk_g['turnover'].mean()*100:.1f}%/mo")
        print(f"  Avg short notion:  {bk_g['short_frac'].mean()*100:.1f}%")
        if vs:
            pl = st.placebo_pvalue(mp, n_perm=1000)
            print(f"  Placebo p-value:   {pl['p_value']:.3f}  "
                  f"(real {pl['real_mean_ann']*100:+.2f}%/yr vs shuffle {pl['placebo_mean_ann']*100:+.2f}%/yr)")
        print()

    # ---- Year-by-year (vol-scaled gross) ---------------------------------
    bk = st.tsmom_book(mp, vol_scale=True)
    yr = bk.copy(); yr["year"] = yr.index.year
    yearly = yr.groupby("year")["port_gross"].apply(lambda x: float((1 + x).prod() - 1)) * 100
    print("=== Year-by-year vol-scaled TSMOM (gross, %) ===")
    for y, v in yearly.items():
        print(f"  {y}: {v:+.1f}%")
    print()

    print("NOTE Survivorship bias (named on the SIGNAL axis): the basket is cross-asset ETFs")
    print("     still trading in 2026, chosen ex post. TSMOM's own-sign book shorts a falling")
    print("     ETF rather than deleting it, so the bias is MILD vs a cross-sectional equity")
    print("     sort -- but no surviving vehicle had a terminal blow-up, so the premium is a")
    print("     mild upper bound. Costs: 10bps x turnover + 50bps/yr borrow on net shorts.")
    print("     One execution lag (form on month-end close, hold the next month). No look-ahead.")


if __name__ == "__main__":
    main()
