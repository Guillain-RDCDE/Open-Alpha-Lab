"""Real-panel verification -- Study 510 (Frog-In-The-Pan). Regenerates docs/results.md numbers.

Reads the study's own yfinance cache (studies/510-frog-in-the-pan/_cache/fip_prices.parquet),
builds the 12-1 momentum signal and the Da-Gurun-Warachka information-discreteness (ID) measure,
double-sorts momentum x ID, races the LOW-ID (gradual) winners-minus-losers book against the
HIGH-ID (discrete) book, reports the FIP interaction with HAC t and a label-shuffle placebo, and
charges costs + short borrow. Also prints the synthetic positive control.

    python studies/510-frog-in-the-pan/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from frog_in_the_pan import data, strategy as st  # noqa: E402

FRAC = 0.30
COST_BPS = 5.0      # one-way bps per leg per rebalance
BORROW_BPS = 50.0   # annual borrow on the short (loser) leg


def main() -> None:
    prices = data.fetch_panel()
    if prices.empty:
        print("Cache not found. Run with fetch=True and network access to populate the cache.")
        return

    prices = data.drop_partial_last_month(prices)
    mp = st.to_monthly(prices)

    print("Study 510 -- Frog-In-The-Pan -- real yfinance panel\n")
    fp = data.fingerprint(prices)
    print(f"Prices panel: {prices.shape}, fingerprint={fp}")
    print(f"Date range: {prices.index[0].date()} -- {prices.index[-1].date()}")
    print(f"Monthly month-ends: {mp.shape[0]}  |  basket: {prices.shape[1]} names")
    print("As-of: 2026-06-26 | Source: yfinance daily adjusted-close prices\n")

    # The plain momentum baseline (whole basket, no ID conditioning)
    base = st.long_short(mp, prices=None, frac=FRAC)
    s_base = st.summary(base["wml_gross"])

    # The two ID halves, gross
    lo = st.long_short(mp, prices=prices, id_side="low", frac=FRAC)
    hi = st.long_short(mp, prices=prices, id_side="high", frac=FRAC)
    s_lo = st.summary(lo["wml_gross"])
    s_hi = st.summary(hi["wml_gross"])
    fip = st.fip_spread(lo, hi)
    s_fip = st.summary(fip)

    # Net of costs + borrow
    lo_net = st.long_short(mp, prices=prices, id_side="low", frac=FRAC,
                           cost_bps=COST_BPS, borrow_ann_bps=BORROW_BPS)
    s_lo_net = st.summary(lo_net["wml_net"])

    print("=== Plain momentum WML (whole basket, no ID split) ===")
    _print_block(s_base)

    print("\n=== LOW-ID (gradual / Frog-In-The-Pan) WML ===")
    _print_block(s_lo)
    print(f"  Avg turnover/mo:     {lo['turnover'].mean()*100:.1f}%")
    print(f"  Avg names/leg:       {lo['n_leg'].mean():.1f}")

    print("\n=== HIGH-ID (discrete / jumpy) WML ===")
    _print_block(s_hi)

    print("\n=== FIP interaction = LOW-ID WML minus HIGH-ID WML ===")
    print(f"  N months:            {s_fip['n']}")
    print(f"  Mean (ann):          {s_fip['mean']*100:+.2f}%/yr")
    print(f"  HAC t-stat:          {s_fip['tstat']:+.3f}")
    print(f"  Hit rate:            {s_fip['hit_rate']*100:.1f}%")

    print("\n=== LOW-ID WML, NET of costs + borrow (5 bps/leg, 50 bps/yr borrow) ===")
    print(f"  Mean (ann):          {s_lo_net['mean']*100:+.2f}%/yr")
    print(f"  HAC t-stat:          {s_lo_net['tstat']:+.3f}")
    print(f"  Sharpe:              {s_lo_net['sharpe']:+.3f}")

    print("\n=== Placebo (label-shuffle null) on the LOW-ID WML ===")
    pl = st.placebo_pvalue(mp, prices, id_side="low", frac=FRAC, n_perm=1000, seed=510)
    print(f"  Real mean (ann):     {pl['real_mean_ann']*100:+.2f}%/yr")
    print(f"  Placebo mean (ann):  {pl['placebo_mean_ann']*100:+.2f}%/yr")
    print(f"  Permutation p-value: {pl['p_value']:.3f}  (share of shuffles >= real)")

    print("\n=== Seed-robustness of the placebo p (5 seeds) ===")
    for seed in (1, 7, 42, 510, 2024):
        pls = st.placebo_pvalue(mp, prices, id_side="low", frac=FRAC, n_perm=400, seed=seed)
        print(f"  seed {seed:>4}: p = {pls['p_value']:.3f}")

    print("\n=== Synthetic positive control (engine faithfulness, NOT a real-tape claim) ===")
    ctrl = st.synthetic_control(strengths=(0.0, 0.20, 0.40, 0.60), frac=FRAC, seed=510)
    print(ctrl.round(3).to_string(index=False))

    print("\n=== Year-by-year LOW-ID WML returns (compounded over months) ===")
    lo_y = lo.copy()
    lo_y["year"] = lo_y.index.year
    annual = lo_y.groupby("year")["wml_gross"].apply(lambda x: float((1 + x).prod() - 1))
    for yr, val in annual.items():
        print(f"  {yr}: {val*100:+.1f}%")

    print("\nNOTE Survivorship bias: basket = current large-caps projected backwards. The LOW-ID")
    print("     (gradual) loser leg's natural shorts -- firms drifting quietly into delisting --")
    print("     are absent, so the FIP slice is the MORE inflated of the two. Upper-bound estimate.")


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
