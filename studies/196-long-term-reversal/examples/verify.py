"""Real-tape verification — Study 196 (Long-Term-Reversal). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the monthly S&P 500 price panel, forms loser/winner quintile
portfolios on trailing 36-month returns, compares against a random-portfolio control and the
equal-weight market, sweeps the result across sub-periods, and reports the beta decomposition.
Network is touched only with --fetch.

    python studies/196-long-term-reversal/examples/verify.py          # cache-only
    python studies/196-long-term-reversal/examples/verify.py --fetch  # refresh the panel
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from long_term_reversal import data, strategy as st  # noqa: E402

try:
    from quantlab.stats import sharpe_ci_bootstrap
    _HAS_QUANTLAB = True
except ImportError:
    _HAS_QUANTLAB = False


def main(fetch: bool) -> None:
    prices = data.fetch_monthly(fetch=fetch)
    print(f"Panel: {prices.shape[0]} months × {prices.shape[1]} tickers")
    print(f"Range: {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"Fingerprint (last row): {data.fingerprint(prices)}\n")

    # ---- 36-month formation, 1-month hold -----------------------------------
    sig36 = st.trailing_return(prices, formation=36, skip=1)
    res36 = st.quintile_returns(sig36, prices, q=0.20)
    spread = res36["spread"].dropna()
    loser_xret = (res36["loser"] - res36["market"]).dropna()
    winner_xret = (res36["winner"] - res36["market"]).dropna()

    s_spread = st.summarize(spread)
    s_loser  = st.summarize(res36["loser"].dropna())
    s_winner = st.summarize(res36["winner"].dropna())
    s_market = st.summarize(res36["market"].dropna())
    s_loser_x = st.summarize(loser_xret)

    print("=== 36-month formation, 1-month hold ===")
    print(f"Portfolio months: {len(res36)}")
    print(f"Avg stocks in universe / loser quintile: {res36['n_total'].mean():.0f} / {res36['n_loser'].mean():.0f}")
    print()
    print(f"LOSER  : {s_loser['mean']*12*100:+.2f}%/yr  vol={s_loser['vol']*np.sqrt(12)*100:.1f}%  t={s_loser['tstat']:+.2f}  SR(ann)={s_loser['sharpe']*np.sqrt(12):+.2f}")
    print(f"WINNER : {s_winner['mean']*12*100:+.2f}%/yr  vol={s_winner['vol']*np.sqrt(12)*100:.1f}%  t={s_winner['tstat']:+.2f}  SR(ann)={s_winner['sharpe']*np.sqrt(12):+.2f}")
    print(f"MARKET : {s_market['mean']*12*100:+.2f}%/yr")
    print(f"SPREAD : {s_spread['mean']*12*100:+.2f}%/yr = {s_spread['mean']*10000:+.1f}bps/mo  t={s_spread['tstat']:+.2f}  hit={s_spread['hit_rate']:.3f}")
    print(f"LOSER-excess-mkt: {s_loser_x['mean']*12*100:+.2f}%/yr  t={s_loser_x['tstat']:+.2f}")

    if _HAS_QUANTLAB:
        ci = sharpe_ci_bootstrap(spread, n_boot=2000, periods_per_year=12, seed=196)
        print(f"\nSpread Sharpe (ann) 95% CI: [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]  ({ci['frac_negative']*100:.0f}% of resamples negative)")

    # Beta decomposition
    common_idx = res36["loser"].dropna().index.intersection(res36["market"].dropna().index)
    l = res36["loser"].loc[common_idx].to_numpy()
    m = res36["market"].loc[common_idx].to_numpy()
    w = res36["winner"].loc[common_idx].to_numpy()
    beta_l = np.cov(l, m)[0, 1] / np.var(m) if np.var(m) > 0 else np.nan
    beta_w = np.cov(w, m)[0, 1] / np.var(m) if np.var(m) > 0 else np.nan
    alpha_l = l.mean() - beta_l * m.mean()
    alpha_w = w.mean() - beta_w * m.mean()
    print(f"\nBeta decomposition (vs equal-weight market):")
    print(f"  Loser  beta={beta_l:.3f}  alpha={alpha_l*12*100:+.2f}%/yr")
    print(f"  Winner beta={beta_w:.3f}  alpha={alpha_w*12*100:+.2f}%/yr")

    # Sub-period analysis
    print("\n=== Sub-period analysis (loser excess over market) ===")
    for label, start, end in [("1993-2005", "1993", "2005"),
                               ("2006-2015", "2006", "2015"),
                               ("2016-2026", "2016", "2026")]:
        sub = loser_xret[start:end]
        s = st.summarize(sub)
        print(f"  {label}: mean={s['mean']*12*100:+.2f}%/yr  t={s['tstat']:+.2f}  n={s['n']}")

    # 60-month formation
    sig60 = st.trailing_return(prices, formation=60, skip=1)
    res60 = st.quintile_returns(sig60, prices, q=0.20)
    s60 = st.summarize(res60["spread"].dropna())
    print(f"\n=== 60-month formation, 1-month hold ===")
    print(f"Portfolio months: {len(res60)}")
    print(f"SPREAD: {s60['mean']*12*100:+.2f}%/yr = {s60['mean']*10000:+.1f}bps/mo  t={s60['tstat']:+.2f}")

    # Turnover and cost drag
    drag = st.turnover_cost_drag(sig36, prices, q=0.20, one_way_bps=10.0)
    avg_turnover = drag.mean() / 0.002  # drag = turnover * 2 * 10bps
    avg_drag = drag.mean() * 10000
    net_spread = spread - drag.reindex(spread.index).fillna(drag.mean())
    s_net = st.summarize(net_spread)
    print(f"\n=== Turnover and cost drag ===")
    print(f"Avg monthly portfolio turnover: {avg_turnover:.2%}")
    print(f"Avg monthly cost drag @10bps one-way: {avg_drag:.1f} bps")
    print(f"Net spread: {s_net['mean']*12*100:+.2f}%/yr = {s_net['mean']*10000:+.1f}bps/mo  t={s_net['tstat']:+.2f}")

    # Random portfolio null
    print("\nComputing random-portfolio null (n_draws=200)...")
    rand = st.random_portfolio_returns(sig36, prices, q=0.20, n_draws=200, seed=196)
    print(f"Random excess: mean={rand.mean()*10000:+.1f}bps  std={rand.std()*10000:.1f}bps  (null centred at zero: {abs(rand.mean()*10000) < 10})")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
