"""Real-panel verification -- Study 511 (Volume-Momentum, Lee-Swaminathan).

Regenerates the numbers in docs/results.md. Reads the study's own yfinance cache
(studies/511-volume-momentum/_cache/vm_prices.parquet + vm_dollarvol.parquet), builds the 12-1
momentum signal and a trailing turnover (dollar-volume) measure, double-sorts momentum x volume,
races the HIGH-volume winners-minus-losers book against the LOW-volume book, reports the
Lee-Swaminathan interaction with HAC t and a seed-robust label-shuffle placebo, charges costs +
short borrow, prints the volume-conditioned reversal term-structure, and runs the synthetic
positive control.

    python studies/511-volume-momentum/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from volume_momentum import data, strategy as st  # noqa: E402

FRAC = 0.30
COST_BPS = 5.0      # one-way bps per leg per rebalance
BORROW_BPS = 50.0   # annual borrow on the short (loser) leg


def main() -> None:
    prices, dvol = data.fetch_panel()
    if prices.empty:
        print("Cache not found. Run with fetch=True and network access to populate the cache.")
        return

    prices = data.drop_partial_last_month(prices)
    dvol = dvol.loc[dvol.index.isin(prices.index)]
    mp = st.to_monthly(prices)

    print("Study 511 -- Volume-Momentum (Lee-Swaminathan) -- real yfinance panel\n")
    fp = data.fingerprint(prices)
    fpv = data.fingerprint(dvol)
    print(f"Prices panel: {prices.shape}, fingerprint={fp}")
    print(f"Volume panel: {dvol.shape}, fingerprint={fpv}")
    print(f"Date range: {prices.index[0].date()} -- {prices.index[-1].date()}")
    print(f"Monthly month-ends: {mp.shape[0]}  |  basket: {prices.shape[1]} names")
    print("As-of: 2026-06-26 | Source: yfinance daily adjusted-close prices + dollar volume\n")

    # The plain momentum baseline (whole basket, no volume conditioning)
    base = st.long_short(mp, dollar_volume=None, frac=FRAC)
    s_base = st.summary(base["wml_gross"])

    # The two volume halves, gross
    hi = st.long_short(mp, dollar_volume=dvol, vol_side="high", frac=FRAC)
    lo = st.long_short(mp, dollar_volume=dvol, vol_side="low", frac=FRAC)
    s_hi = st.summary(hi["wml_gross"])
    s_lo = st.summary(lo["wml_gross"])
    vm = st.vm_spread(hi, lo)
    s_vm = st.summary(vm)

    # Net of costs + borrow (high-volume book -- the literature's strongest)
    hi_net = st.long_short(mp, dollar_volume=dvol, vol_side="high", frac=FRAC,
                           cost_bps=COST_BPS, borrow_ann_bps=BORROW_BPS)
    s_hi_net = st.summary(hi_net["wml_net"])

    print("=== Plain momentum WML (whole basket, no volume split) ===")
    _print_block(s_base)

    print("\n=== HIGH-volume WML (Lee-Swaminathan strongest slice) ===")
    _print_block(s_hi)
    print(f"  Avg turnover/mo:     {hi['turnover'].mean()*100:.1f}%")
    print(f"  Avg names/leg:       {hi['n_leg'].mean():.1f}")

    print("\n=== LOW-volume WML ===")
    _print_block(s_lo)

    print("\n=== Lee-Swaminathan interaction = HIGH-vol WML minus LOW-vol WML ===")
    print(f"  N months:            {s_vm['n']}")
    print(f"  Mean (ann):          {s_vm['mean']*100:+.2f}%/yr")
    print(f"  HAC t-stat:          {s_vm['tstat']:+.3f}")
    print(f"  Hit rate:            {s_vm['hit_rate']*100:.1f}%")

    print("\n=== HIGH-volume WML, NET of costs + borrow (5 bps/leg, 50 bps/yr borrow) ===")
    print(f"  Mean (ann):          {s_hi_net['mean']*100:+.2f}%/yr")
    print(f"  HAC t-stat:          {s_hi_net['tstat']:+.3f}")
    print(f"  Sharpe:              {s_hi_net['sharpe']:+.3f}")

    print("\n=== Placebo (label-shuffle null) on the HIGH-volume WML ===")
    pl = st.placebo_pvalue(mp, dvol, vol_side="high", frac=FRAC, n_perm=1000, seed=511)
    print(f"  Real mean (ann):     {pl['real_mean_ann']*100:+.2f}%/yr")
    print(f"  Placebo mean (ann):  {pl['placebo_mean_ann']*100:+.2f}%/yr")
    print(f"  Permutation p-value: {pl['p_value']:.3f}  (share of shuffles >= real)")

    print("\n=== Seed-robustness of the placebo p (5 seeds, high-vol WML) ===")
    for seed in (1, 7, 42, 511, 2024):
        pls = st.placebo_pvalue(mp, dvol, vol_side="high", frac=FRAC, n_perm=400, seed=seed)
        print(f"  seed {seed:>4}: p = {pls['p_value']:.3f}")

    print("\n=== Volume-conditioned reversal term-structure (cumulative WML by hold) ===")
    rts = st.reversal_term_structure(mp, dvol, holds=(1, 3, 6, 12), frac=FRAC)
    print(rts.round(4).to_string(index=False))

    print("\n=== Synthetic positive control (engine faithfulness, NOT a real-tape claim) ===")
    ctrl = st.synthetic_control(strengths=(0.0, 0.20, 0.40, 0.60), frac=FRAC, seed=511)
    print(ctrl.round(3).to_string(index=False))

    print("\n=== Year-by-year HIGH-volume WML returns (compounded over months) ===")
    hi_y = hi.copy()
    hi_y["year"] = hi_y.index.year
    annual = hi_y.groupby("year")["wml_gross"].apply(lambda x: float((1 + x).prod() - 1))
    for yr, val in annual.items():
        print(f"  {yr}: {val*100:+.1f}%")

    print("\nNOTE Survivorship bias: basket = current large-caps projected backwards. The")
    print("     low-volume loser leg (Lee-Swaminathan's strongest short) is the quiet names that")
    print("     fade into delisting -- absent here, so the volume slice is an upper-bound estimate.")


def _print_block(s: dict) -> None:
    print(f"  N months:            {s['n']}")
    print(f"  Mean return (ann):   {s['mean']*100:+.2f}%/yr")
    print(f"  Vol (ann):           {s['vol']*100:.2f}%/yr")
    print(f"  Sharpe (ann):        {s['sharpe']:+.3f}")
    print(f"  HAC t-stat:          {s['tstat']:+.3f}")
    print(f"  Hit rate:            {s['hit_rate']*100:.1f}%")
    print(f"  Max drawdown:        {s['max_dd']*100:.1f}%")


if __name__ == "__main__":
    main()
