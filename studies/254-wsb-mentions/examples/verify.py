"""Real-tape verification -- Study 254 (WSB-Mentions). Regenerates docs/results.md numbers.

Reads (or fetches) the meme-basket monthly price panel, joins the hardcoded WSB
mention-count table, forms loud-half / quiet-half portfolios on this month's
mention rank (predicting next month's return), compares against a random-portfolio
control and the equal-weight basket, sweeps calendar-year sub-periods, reports a
pooled rank-IC, and a turnover cost sweep. Network is touched only with --fetch.

    python studies/254-wsb-mentions/examples/verify.py          # cache-only
    python studies/254-wsb-mentions/examples/verify.py --fetch  # refresh the panel
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from wsb_mentions import data, strategy as st  # noqa: E402

try:
    from quantlab.stats import sharpe_ci_bootstrap
    _HAS_QUANTLAB = True
except ImportError:
    _HAS_QUANTLAB = False


def main(fetch: bool) -> None:
    prices = data.fetch_meme_monthly(fetch=fetch)
    mentions = data.mention_counts()
    sig = st.mention_signal(mentions)

    print(f"Panel: {prices.shape[0]} months x {prices.shape[1]} tickers")
    print(f"Range: {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"Fingerprint (last row): {data.fingerprint(prices)}\n")

    res = st.half_returns(sig, prices, q=0.50, min_stocks=6)
    spread = res["spread"].dropna()
    hype_x = (res["hype"] - res["basket"]).dropna()

    s_spread = st.summarize(spread)
    s_hype = st.summarize(res["hype"].dropna())
    s_quiet = st.summarize(res["quiet"].dropna())
    s_basket = st.summarize(res["basket"].dropna())
    s_hype_x = st.summarize(hype_x)

    print("=== Buy loud half, short quiet half, hold 1 month ===")
    print(f"Portfolio months: {len(res)}")
    print(f"Avg names / loud leg: {res['n_total'].mean():.0f} / {res['n_hype'].mean():.1f}")
    print()
    print(f"LOUD  : {s_hype['mean']*12*100:+.1f}%/yr  vol={s_hype['vol']*np.sqrt(12)*100:.0f}%  t={s_hype['tstat']:+.2f}")
    print(f"QUIET : {s_quiet['mean']*12*100:+.1f}%/yr  vol={s_quiet['vol']*np.sqrt(12)*100:.0f}%  t={s_quiet['tstat']:+.2f}")
    print(f"BASKET: {s_basket['mean']*12*100:+.1f}%/yr")
    print(f"SPREAD: {s_spread['mean']*12*100:+.1f}%/yr = {s_spread['mean']*10000:+.1f}bps/mo  t={s_spread['tstat']:+.2f}  hit={s_spread['hit_rate']:.3f}")
    print(f"LOUD excess-basket: {s_hype_x['mean']*12*100:+.1f}%/yr  t={s_hype_x['tstat']:+.2f}")

    if _HAS_QUANTLAB:
        ci = sharpe_ci_bootstrap(spread, n_boot=2000, periods_per_year=12, seed=254)
        print(f"\nSpread Sharpe (ann) 95% CI: [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]  ({ci['frac_negative']*100:.0f}% of resamples negative)")

    # Pooled rank-IC
    fwd = prices.pct_change().shift(-1)
    X, Y = [], []
    for t in sig.index:
        if t not in fwd.index:
            continue
        s = sig.loc[t].dropna()
        f = fwd.loc[t].dropna()
        c = s.index.intersection(f.index)
        if len(c) < 6:
            continue
        X.append(s.loc[c].rank())
        Y.append(f.loc[c].rank())
    if X:
        import scipy.stats as ss
        Xs, Ys = pd.concat(X), pd.concat(Y)
        rho, pval = ss.spearmanr(Xs, Ys)
        print(f"\nPooled rank-IC (Spearman): {rho:+.3f}  p={pval:.3f}  over {len(Xs)} pairs")

    # Sub-periods
    print("\n=== Calendar-year sub-periods (loud-minus-quiet spread) ===")
    for label in ["2021", "2022", "2023"]:
        sub = spread[label:label]
        s = st.summarize(sub)
        print(f"  {label}: mean={s['mean']*12*100:+.1f}%/yr  t={s['tstat']:+.2f}  n={s['n']}")

    # Reversal sign
    s_rev = st.summarize(-spread)
    print(f"\nReversal ('short the loud names'): {s_rev['mean']*12*100:+.1f}%/yr  t={s_rev['tstat']:+.2f}")

    # Turnover and cost drag
    drag = st.turnover_cost_drag(sig, prices, q=0.50, one_way_bps=30.0)
    avg_turnover = drag.mean() / 0.006 if drag.mean() > 0 else np.nan
    net_spread = spread - drag.reindex(spread.index).fillna(drag.mean())
    s_net = st.summarize(net_spread)
    print(f"\n=== Turnover and cost drag (30 bps one-way: illiquid + borrow) ===")
    print(f"Avg monthly loud-leg turnover: {avg_turnover:.1%}")
    print(f"Avg monthly cost drag: {drag.mean()*10000:.1f} bps")
    print(f"Net spread: {s_net['mean']*12*100:+.1f}%/yr  t={s_net['tstat']:+.2f}")

    # Random control
    rand = st.random_portfolio_returns(sig, prices, q=0.50, n_draws=200, seed=254)
    print(f"\nRandom-half excess vs basket: mean={rand.mean()*10000:+.1f}bps (null centred near zero)")

    print(f"\nFingerprint: {data.fingerprint(prices)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
