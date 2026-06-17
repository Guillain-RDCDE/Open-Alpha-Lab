"""Real-tape verification -- Study 223 (Same-Month Seasonality). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the monthly large-cap price panel, forms top/bottom decile
portfolios on the same-calendar-month historical return signal (Heston-Sadka 2008), compares
against a random-portfolio control and the equal-weight market, sweeps sub-periods,
and reports a turnover cost sweep.
Network is touched only with --fetch.

    python studies/223-same-month-seasonality/examples/verify.py          # cache-only
    python studies/223-same-month-seasonality/examples/verify.py --fetch  # refresh the panel
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from same_month_seasonality import data, strategy as st  # noqa: E402

try:
    from quantlab.stats import sharpe_ci_bootstrap
    _HAS_QUANTLAB = True
except ImportError:
    _HAS_QUANTLAB = False


def main(fetch: bool) -> None:
    prices = data.fetch_monthly(fetch=fetch)
    print(f"Panel: {prices.shape[0]} months x {prices.shape[1]} tickers")
    print(f"Range: {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"Fingerprint (last row): {data.fingerprint(prices)}\n")

    # ---- Same-month signal, decile sort, 1-month hold ----------------------
    sig = st.same_month_signal(prices, min_years=5)
    res = st.decile_returns(sig, prices, q=0.10)
    spread = res["spread"].dropna()
    high_xret = (res["high"] - res["market"]).dropna()
    low_xret = (res["low"] - res["market"]).dropna()

    s_spread = st.summarize(spread)
    s_high   = st.summarize(res["high"].dropna())
    s_low    = st.summarize(res["low"].dropna())
    s_market = st.summarize(res["market"].dropna())
    s_high_x = st.summarize(high_xret)

    print("=== Same-month signal, decile sort, 1-month hold ===")
    print(f"Portfolio months: {len(res)}")
    print(f"Avg stocks in universe / top decile: {res['n_total'].mean():.0f} / {res['n_high'].mean():.0f}")
    print()
    print(f"HIGH (top decile)  : {s_high['mean']*12*100:+.2f}%/yr  vol={s_high['vol']*np.sqrt(12)*100:.1f}%  t={s_high['tstat']:+.2f}  SR(ann)={s_high['sharpe']*np.sqrt(12):+.2f}")
    print(f"LOW (bot decile)   : {s_low['mean']*12*100:+.2f}%/yr  vol={s_low['vol']*np.sqrt(12)*100:.1f}%  t={s_low['tstat']:+.2f}  SR(ann)={s_low['sharpe']*np.sqrt(12):+.2f}")
    print(f"MARKET (eq-weight) : {s_market['mean']*12*100:+.2f}%/yr")
    print(f"SPREAD (high-low)  : {s_spread['mean']*12*100:+.2f}%/yr = {s_spread['mean']*10000:+.1f}bps/mo  t={s_spread['tstat']:+.2f}  hit={s_spread['hit_rate']:.3f}")
    print(f"HIGH excess-market : {s_high_x['mean']*12*100:+.2f}%/yr  t={s_high_x['tstat']:+.2f}")

    if _HAS_QUANTLAB:
        ci = sharpe_ci_bootstrap(spread, n_boot=2000, periods_per_year=12, seed=223)
        print(f"\nSpread Sharpe (ann) 95% CI: [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]  ({ci['frac_negative']*100:.0f}% of resamples negative)")

    # Sub-period analysis
    print("\n=== Sub-period analysis (high-low spread) ===")
    for label, start, end in [("1999-2008", "1999", "2008"),
                               ("2009-2017", "2009", "2017"),
                               ("2018-2026", "2018", "2026")]:
        sub = spread[start:end]
        s = st.summarize(sub)
        print(f"  {label}: mean={s['mean']*12*100:+.2f}%/yr  t={s['tstat']:+.2f}  n={s['n']}")

    # Monthly turnover and cost drag
    drag = st.turnover_cost_drag(sig, prices, q=0.10, one_way_bps=10.0)
    # drag = turnover * 2 * 10bps
    avg_turnover = drag.mean() / 0.002 if drag.mean() > 0 else np.nan
    avg_drag = drag.mean() * 10000
    net_spread = spread - drag.reindex(spread.index).fillna(drag.mean())
    s_net = st.summarize(net_spread)
    print(f"\n=== Turnover and cost drag ===")
    print(f"Avg monthly portfolio turnover: {avg_turnover:.2%}")
    print(f"Avg monthly cost drag @10bps one-way: {avg_drag:.1f} bps")
    print(f"Net spread: {s_net['mean']*12*100:+.2f}%/yr = {s_net['mean']*10000:+.1f}bps/mo  t={s_net['tstat']:+.2f}")

    # Random portfolio null
    print("\nComputing random-portfolio null (n_draws=200)...")
    rand = st.random_portfolio_returns(sig, prices, q=0.10, n_draws=200, seed=223)
    print(f"Random excess: mean={rand.mean()*10000:+.1f}bps  std={rand.std()*10000:.1f}bps  (null centred at zero: {abs(rand.mean()*10000) < 10})")

    # Per-calendar-month breakdown
    print("\n=== Per-calendar-month spread breakdown ===")
    months_lbl = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m in range(1, 13):
        sub = spread[spread.index.month == m]
        if len(sub) < 5:
            continue
        s = st.summarize(sub)
        print(f"  {months_lbl[m-1]}: {s['mean']*10000:+.1f}bps/mo  t={s['tstat']:+.2f}  n={s['n']}")

    print(f"\nFingerprint: {data.fingerprint(prices)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
