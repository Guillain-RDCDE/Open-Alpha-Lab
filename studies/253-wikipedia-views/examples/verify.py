"""Real-tape verification -- Study 253 (Wiki-Views). Regenerates docs/results.md numbers.

Builds the attention-surge signal from the curated Wikipedia page-view panel,
forms drought/surge quintile portfolios, compares against a random-portfolio
control and the equal-weight basket, sweeps sub-periods, and reports a turnover
cost sweep.

By default (cache-only) the returns side is the offline best-effort proxy
(``data.best_effort_real_tape``: curated views joined to realistic prices that
are *independent* of the views -- the honest null). With ``--fetch`` it pulls the
genuine Yahoo monthly tape and joins the curated views to it.

    python studies/253-wikipedia-views/examples/verify.py          # cache-only / offline proxy
    python studies/253-wikipedia-views/examples/verify.py --fetch  # real Yahoo tape
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from wikipedia_views import data, strategy as st  # noqa: E402

try:
    from quantlab.stats import sharpe_ci_bootstrap
    _HAS_QUANTLAB = True
except ImportError:
    _HAS_QUANTLAB = False


def _load(fetch: bool):
    if fetch:
        prices = data.fetch_returns(fetch=True)
        views = data.wiki_views_panel(n_months=prices.shape[0])
        views, prices = data.align_panels(views, prices)
        tag = "real Yahoo tape"
    elif os.path.exists(data.RETURNS_CACHE):
        prices = data.fetch_returns(fetch=False)
        views = data.wiki_views_panel(n_months=prices.shape[0])
        views, prices = data.align_panels(views, prices)
        tag = "cached Yahoo tape"
    else:
        views, prices = data.best_effort_real_tape(n_months=120, seed=data.SEED)
        tag = "offline best-effort proxy (views-independent prices)"
    return views, prices, tag


def main(fetch: bool) -> None:
    views, prices, tag = _load(fetch)
    print(f"Tape: {tag}")
    print(f"Panel: {prices.shape[0]} months x {prices.shape[1]} tickers")
    print(f"Range: {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"Views fingerprint (last row):  {data.fingerprint(views)}")
    print(f"Prices fingerprint (last row): {data.fingerprint(prices)}\n")

    sur = st.attention_surge(views, smooth=1)
    res = st.decile_returns(sur, prices, q=0.20)
    spread = res["spread"].dropna()
    low_x = (res["low"] - res["market"]).dropna()

    s_spread = st.summarize(spread)
    s_low = st.summarize(res["low"].dropna())
    s_high = st.summarize(res["high"].dropna())
    s_market = st.summarize(res["market"].dropna())
    s_low_x = st.summarize(low_x)

    print("=== Attention-surge quintile sort, 1-month hold (low-minus-high) ===")
    print(f"Portfolio months: {len(res)}")
    print(f"Avg stocks / leg: {res['n_low'].mean():.0f}")
    print()
    print(f"LOW  (drought, long) : {s_low['mean']*1200:+.2f}%/yr  t={s_low['tstat']:+.2f}")
    print(f"HIGH (surge, short)  : {s_high['mean']*1200:+.2f}%/yr  t={s_high['tstat']:+.2f}")
    print(f"MARKET (eq-weight)   : {s_market['mean']*1200:+.2f}%/yr")
    print(f"SPREAD (low-high)    : {s_spread['mean']*1200:+.2f}%/yr = {s_spread['mean']*10000:+.1f}bps/mo  t={s_spread['tstat']:+.2f}  hit={s_spread['hit_rate']:.3f}")
    print(f"LOW excess-basket    : {s_low_x['mean']*1200:+.2f}%/yr  t={s_low_x['tstat']:+.2f}")

    if _HAS_QUANTLAB and len(spread) > 12:
        ci = sharpe_ci_bootstrap(spread, n_boot=2000, periods_per_year=12, seed=data.SEED)
        print(f"\nSpread Sharpe (ann) 95% CI: [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]  ({ci['frac_negative']*100:.0f}% of resamples negative)")

    print("\n=== Sub-period analysis (low-high spread) ===")
    for label, a, b in [("2017-2019", "2017", "2019"),
                        ("2020-2022", "2020", "2022"),
                        ("2023-2026", "2023", "2026")]:
        sub = spread[a:b]
        s = st.summarize(sub)
        print(f"  {label}: mean={s['mean']*1200:+.2f}%/yr  t={s['tstat']:+.2f}  n={s['n']}")

    drag = st.turnover_cost_drag(sur, prices, q=0.20, one_way_bps=10.0)
    avg_turnover = drag.mean() / 0.004 if drag.mean() > 0 else np.nan
    avg_drag = drag.mean() * 10000
    net = spread - drag.reindex(spread.index).fillna(drag.mean())
    s_net = st.summarize(net)
    print(f"\n=== Turnover and cost drag (both legs) ===")
    print(f"Avg monthly turnover: {avg_turnover:.2%}")
    print(f"Avg drag @10bps one-way: {avg_drag:.1f} bps/mo")
    print(f"Net spread @10bps: {s_net['mean']*1200:+.2f}%/yr = {s_net['mean']*10000:+.1f}bps/mo  t={s_net['tstat']:+.2f}")

    print("\nComputing random-portfolio null (n_draws=200)...")
    rand = st.random_portfolio_returns(sur, prices, q=0.20, n_draws=200, seed=data.SEED)
    print(f"Random excess: mean={rand.mean()*10000:+.1f}bps  std={rand.std()*10000:.1f}bps  (null centred at zero: {abs(rand.mean()*10000) < 10})")

    print(f"\nFingerprints -- views: {data.fingerprint(views)}  prices: {data.fingerprint(prices)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
