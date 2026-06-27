"""Real-panel verification -- Study 509 (Intermediate-Momentum). Prints every number.

Reads the study's own yfinance cache (studies/509-intermediate-momentum/_cache/im_prices.parquet),
builds the intermediate (t-12..t-7) and recent (t-6..t-1) momentum long-shorts, and contrasts
which window carries the drift -- with one execution lag, a label-shuffle placebo, and
turnover-based costs + borrow.

    python studies/509-intermediate-momentum/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from intermediate_momentum import data, strategy as st  # noqa: E402


def _print_leg(name: str, s: dict) -> None:
    print(f"=== {name} ===")
    print(f"  N months:          {s['n']}")
    print(f"  Mean (ann):        {s['mean']*100:+.2f}%/yr")
    print(f"  Vol (ann):         {s['vol']*100:.2f}%/yr")
    print(f"  Sharpe (ann):      {s['sharpe']:+.3f}")
    print(f"  HAC t-stat:        {s['tstat']:+.3f}")
    print(f"  Hit rate:          {s['hit_rate']*100:.1f}%")
    print(f"  Max drawdown:      {s['max_drawdown']*100:.1f}%")


def main() -> None:
    prices, spy = data.fetch_panel()
    if prices.empty:
        print("Cache not found. Run from repo root with network access to populate the cache.")
        return

    print("Study 509 -- Intermediate-Momentum -- real yfinance panel\n")
    fp_prices = data.fingerprint(prices)
    fp_spy = data.fingerprint(spy)
    print(f"Prices panel: {prices.shape}, fingerprint={fp_prices}")
    print(f"SPY series:   {spy.shape}, fingerprint={fp_spy}")
    print(f"Date range: {prices.index[0].date()} -- {prices.index[-1].date()}")
    print("As-of: 2026-06-26 | Source: yfinance daily adjusted-close prices\n")

    inter = st.momentum_longshort(prices, window="intermediate", min_stocks=18)
    recent = st.momentum_longshort(prices, window="recent", min_stocks=18)

    s_int = st.summary(inter["ls_ret"])
    s_rec = st.summary(recent["ls_ret"])

    _print_leg("INTERMEDIATE long-short (t-12..t-7), gross", s_int)
    print()
    _print_leg("RECENT long-short (t-6..t-1), gross", s_rec)

    # Net of costs.
    net_int = st.apply_costs(inter)
    net_rec = st.apply_costs(recent)
    s_int_net = st.summary(net_int)
    s_rec_net = st.summary(net_rec)

    print("\n=== Net of costs (5 bps/leg-side + 50 bps annual borrow) ===")
    print(f"  Intermediate avg turnover: {inter['turnover'].mean()*100:.1f}% | "
          f"net mean {s_int_net['mean']*100:+.2f}%/yr (t={s_int_net['tstat']:+.2f})")
    print(f"  Recent       avg turnover: {recent['turnover'].mean()*100:.1f}% | "
          f"net mean {s_rec_net['mean']*100:+.2f}%/yr (t={s_rec_net['tstat']:+.2f})")

    # Placebo.
    print("\n=== Label-shuffle placebo (200 shuffles) ===")
    plac_int = st.placebo_pvalue(prices, window="intermediate", n_shuffles=200, min_stocks=18)
    plac_rec = st.placebo_pvalue(prices, window="recent", n_shuffles=200, min_stocks=18)
    print(f"  Intermediate: real mean/mo {plac_int['real_mean']*100:+.3f}% | "
          f"placebo p = {plac_int['p_value']:.3f}")
    print(f"  Recent:       real mean/mo {plac_rec['real_mean']*100:+.3f}% | "
          f"placebo p = {plac_rec['p_value']:.3f}")

    # Synthetic positive control.
    print("\n=== Synthetic positive control (engine faithfulness) ===")
    for prem in [0.0, 0.06, 0.12]:
        ps, sp, truth = data.synthetic_panel(mom_premium=prem, seed=509)
        ri = st.momentum_longshort(ps, window="intermediate", min_stocks=10)
        rr = st.momentum_longshort(ps, window="recent", min_stocks=10)
        si = st.summary(ri["ls_ret"]); srr = st.summary(rr["ls_ret"])
        print(f"  planted prem={prem:+.2f} -> INT mean {si['mean']*100:+.2f}%/yr "
              f"(t={si['tstat']:+.2f}) | REC mean {srr['mean']*100:+.2f}%/yr (t={srr['tstat']:+.2f})")

    print("\n=== Year-by-year intermediate long-short ===")
    ic = inter.copy()
    ic["year"] = ic.index.year
    annual = ic.groupby("year")["ls_ret"].apply(lambda x: float((1 + x).prod() - 1))
    for yr, val in annual.items():
        print(f"  {yr}: {val*100:+.1f}%")

    print("\nNOTE Survivorship bias: universe = current large-cap names projected backwards.")
    print("     Results are upper-bound estimates. One execution lag, costs charged.")


if __name__ == "__main__":
    main()
