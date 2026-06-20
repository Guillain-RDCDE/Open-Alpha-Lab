"""Real-tape verification -- Study 329 (One-Month-Reversal). Regenerates docs/results.md.

Reads (cache-first) the monthly S&P 500 price panel, forms loser/winner quintiles on
*last month's* return, and reports the loser-minus-winner spread, the microstructure-safe
``skip=1`` variant, the beta decomposition, the sub-period decay, and the turnover/cost
sweep. Network is touched only with --fetch.

    python studies/329-one-month-reversal/examples/verify.py          # cache-only
    python studies/329-one-month-reversal/examples/verify.py --fetch  # refresh the panel
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from one_month_reversal import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    prices = data.fetch_monthly(fetch=fetch) if fetch else data.load_real()
    print(f"Panel: {prices.shape[0]} months x {prices.shape[1]} tickers")
    print(f"Range: {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"Fingerprint (last row): {data.fingerprint(prices)}\n")

    # --- skip=0: the Jegadeesh same-month-rank specification ------------------
    sig0 = st.trailing_return(prices, skip=0)
    res0 = st.quintile_returns(sig0, prices, q=0.20)
    s_sp = st.summarize(res0["spread"])
    s_lo = st.summarize(res0["loser"])
    s_wi = st.summarize(res0["winner"])
    loser_x = (res0["loser"] - res0["market"]).dropna()
    s_lx = st.summarize(loser_x)

    print("=== skip=0 (last month's rank; Jegadeesh 1990) ===")
    print(f"Portfolio months: {len(res0)}  avg stocks/quintile: {res0['n_loser'].mean():.0f}")
    print(f"LOSER : {s_lo['mean']*12*100:+.2f}%/yr  t={s_lo['tstat']:+.2f}")
    print(f"WINNER: {s_wi['mean']*12*100:+.2f}%/yr  t={s_wi['tstat']:+.2f}")
    print(f"SPREAD: {s_sp['mean']*1e4:+.1f}bps/mo = {s_sp['mean']*12*100:+.2f}%/yr  HAC t={s_sp['tstat']:+.2f}  hit={s_sp['hit_rate']:.3f}")
    print(f"LOSER excess over market: {s_lx['mean']*12*100:+.2f}%/yr  t={s_lx['tstat']:+.2f}")

    beta_s, alpha_s = st.beta_alpha(res0["spread"], res0["market"])
    beta_l, _ = st.beta_alpha(res0["loser"], res0["market"])
    beta_w, _ = st.beta_alpha(res0["winner"], res0["market"])
    print(f"\nSpread (L-S) beta={beta_s:.3f}  alpha={alpha_s*100:+.2f}%/yr")
    print(f"Loser beta={beta_l:.3f}  Winner beta={beta_w:.3f}")

    # --- skip=1: the microstructure-safe variant -----------------------------
    sig1 = st.trailing_return(prices, skip=1)
    res1 = st.quintile_returns(sig1, prices, q=0.20)
    s_sp1 = st.summarize(res1["spread"])
    print(f"\n=== skip=1 (one-month gap; bid-ask-bounce-safe) ===")
    print(f"SPREAD: {s_sp1['mean']*1e4:+.1f}bps/mo = {s_sp1['mean']*12*100:+.2f}%/yr  HAC t={s_sp1['tstat']:+.2f}")

    # --- sub-periods ---------------------------------------------------------
    print("\n=== Sub-period decay (skip=0 spread) ===")
    for lab, a, b in [("1990-2002", "1990", "2002"), ("2003-2014", "2003", "2014"),
                      ("2015-2026", "2015", "2026")]:
        s = st.summarize(res0["spread"][a:b])
        print(f"  {lab}: {s['mean']*1e4:+.1f}bps/mo  t={s['tstat']:+.2f}  n={s['n']}")

    # --- turnover & cost sweep ----------------------------------------------
    print(f"\n=== Turnover & costs (skip=0) ===")
    print(f"Avg loser-leg one-way turnover: {res0['loser_turn'].mean():.1%}")
    print(f"Break-even one-way cost: {st.break_even_cost(res0):.1f} bps")
    for c in [0, 5, 10, 20]:
        s = st.summarize(st.net_spread(res0, c))
        print(f"  net @ {c:2d} bps one-way: {s['mean']*1e4:+.1f}bps/mo  t={s['tstat']:+.2f}")

    # --- random-portfolio null ----------------------------------------------
    rand = st.random_portfolio_returns(sig0, prices, n_draws=100, seed=329)
    print(f"\nRandom-portfolio null: mean={rand.mean()*1e4:+.1f}bps  (centred near 0: {abs(rand.mean()*1e4) < 5})")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
