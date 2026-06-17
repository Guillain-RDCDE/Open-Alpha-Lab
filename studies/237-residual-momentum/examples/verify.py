"""Real-tape verification -- Study 237 (Residual-Momentum).

Fetches (or reads from cache) the monthly large-cap price panel + SPY, estimates
rolling CAPM betas, computes residual returns, forms winner/loser quintile
portfolios on trailing 11-month residual returns (skip 1), and compares against
raw momentum and a random-portfolio control.

    python studies/237-residual-momentum/examples/verify.py          # cache-only
    python studies/237-residual-momentum/examples/verify.py --fetch  # refresh the panel
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from residual_momentum import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    prices, spy = data.fetch_monthly(fetch=fetch)
    print(f"Panel: {prices.shape[0]} months x {prices.shape[1]} tickers")
    print(f"Range: {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"SPY range: {spy.index[0].date()} -> {spy.index[-1].date()}")
    print(f"Fingerprint (last row): {data.fingerprint(prices)}\n")

    # ---- Align indices -------------------------------------------------------
    common_idx = prices.index.intersection(spy.index)
    prices = prices.loc[common_idx]
    spy = spy.loc[common_idx]

    # ---- Rolling betas -------------------------------------------------------
    print("Estimating rolling betas (beta_window=36 months)...")
    beta_df = st.rolling_betas(prices, spy, beta_window=36)
    med_beta = float(beta_df.median().median())
    print(f"Median rolling beta: {med_beta:.3f}")

    # ---- Residual signal (11-month formation, skip 1) -------------------------
    print("Computing residual signal (formation=11, skip=1)...")
    sig_resid = st.residual_signal(prices, spy, beta_df, formation=11, skip=1)

    # ---- Raw momentum signal for comparison -----------------------------------
    sig_raw = st.raw_signal(prices, formation=11, skip=1)

    # ---- Quintile portfolios -------------------------------------------------
    res_resid = st.quintile_returns(sig_resid, prices, q=0.20)
    res_raw   = st.quintile_returns(sig_raw,   prices, q=0.20)

    spread_resid = res_resid["spread"].dropna()
    spread_raw   = res_raw["spread"].dropna()
    winner_xret  = (res_resid["winner"] - res_resid["market"]).dropna()

    s_resid = st.summarize(spread_resid)
    s_raw   = st.summarize(spread_raw)
    s_mkt   = st.summarize(res_resid["market"].dropna())
    s_winner = st.summarize(res_resid["winner"].dropna())
    s_loser  = st.summarize(res_resid["loser"].dropna())
    s_winner_x = st.summarize(winner_xret)

    print(f"\n=== Residual momentum (11-1, quintile L/S) ===")
    print(f"Portfolio months: {len(res_resid)}")
    print(f"Avg stocks in universe: {res_resid['n_total'].mean():.0f}  /  winner: {res_resid['n_winner'].mean():.0f}")
    print()
    print(f"WINNER : {s_winner['mean']*12*100:+.2f}%/yr  vol={s_winner['vol']*np.sqrt(12)*100:.1f}%  t={s_winner['tstat']:+.2f}  SR(ann)={s_winner['sharpe']*np.sqrt(12):+.2f}")
    print(f"LOSER  : {s_loser['mean']*12*100:+.2f}%/yr  vol={s_loser['vol']*np.sqrt(12)*100:.1f}%  t={s_loser['tstat']:+.2f}  SR(ann)={s_loser['sharpe']*np.sqrt(12):+.2f}")
    print(f"MARKET : {s_mkt['mean']*12*100:+.2f}%/yr")
    print(f"SPREAD (W-L): {s_resid['mean']*12*100:+.2f}%/yr = {s_resid['mean']*10000:+.1f}bps/mo  t={s_resid['tstat']:+.2f}  hit={s_resid['hit_rate']:.3f}")
    print(f"WINNER excess over market: {s_winner_x['mean']*12*100:+.2f}%/yr  t={s_winner_x['tstat']:+.2f}")

    print(f"\n=== Raw momentum (11-1, quintile L/S) ===")
    print(f"SPREAD: {s_raw['mean']*12*100:+.2f}%/yr = {s_raw['mean']*10000:+.1f}bps/mo  t={s_raw['tstat']:+.2f}")

    # ---- Beta decomposition (vs equal-weight market) -----------------------
    mkt_ew = res_resid["market"].dropna()
    winner_r = res_resid["winner"].reindex(mkt_ew.index).dropna()
    loser_r  = res_resid["loser"].reindex(mkt_ew.index).dropna()

    common2 = winner_r.index.intersection(loser_r.index).intersection(mkt_ew.index)
    wl = winner_r.loc[common2].values
    ll = loser_r.loc[common2].values
    ml = mkt_ew.loc[common2].values

    beta_winner = np.cov(wl, ml)[0, 1] / np.var(ml) if np.var(ml) > 0 else np.nan
    beta_loser  = np.cov(ll, ml)[0, 1] / np.var(ml) if np.var(ml) > 0 else np.nan
    spread_arr  = wl - ll
    beta_spread_vs_mkt = np.cov(spread_arr, ml)[0, 1] / np.var(ml) if np.var(ml) > 0 else np.nan
    alpha_winner = (wl.mean() - beta_winner * ml.mean()) * 12 * 100
    alpha_loser  = (ll.mean() - beta_loser  * ml.mean()) * 12 * 100

    print(f"\nBeta decomposition (vs equal-weight market):")
    print(f"  Winner  beta={beta_winner:.3f}  alpha={alpha_winner:+.2f}%/yr")
    print(f"  Loser   beta={beta_loser:.3f}  alpha={alpha_loser:+.2f}%/yr")
    print(f"  Spread  beta vs market={beta_spread_vs_mkt:.3f}")

    # ---- Sub-period analysis -------------------------------------------------
    print("\n=== Sub-period analysis (residual spread) ===")
    for label, start, end in [("1999-2006", "1999", "2006"),
                               ("2007-2015", "2007", "2015"),
                               ("2016-2026", "2016", "2026")]:
        sub = spread_resid[start:end]
        s = st.summarize(sub)
        print(f"  {label}: mean={s['mean']*12*100:+.2f}%/yr  t={s['tstat']:+.2f}  n={s['n']}")

    # ---- Max-drawdown in crash months ----------------------------------------
    # Momentum crash months: 2009-03 and 2020-03
    crash_months = ["2009-03", "2020-03"]
    print("\n=== Crash-month spread returns ===")
    for cm in crash_months:
        try:
            v = float(spread_resid.loc[cm].values[0]) if isinstance(spread_resid.loc[cm], pd.Series) else float(spread_resid.loc[cm])
            print(f"  {cm}: residual spread = {v*100:+.2f}%")
        except (KeyError, IndexError):
            print(f"  {cm}: not in sample")

    # ---- Turnover and costs --------------------------------------------------
    drag = st.turnover_cost_drag(sig_resid, prices, q=0.20, one_way_bps=10.0)
    avg_turnover = drag.mean() / 0.002
    avg_drag_bps = drag.mean() * 10000
    net_spread = spread_resid - drag.reindex(spread_resid.index).fillna(drag.mean())
    s_net = st.summarize(net_spread)
    print(f"\n=== Turnover and cost drag ===")
    print(f"Avg monthly winner-quintile turnover: {avg_turnover:.2%}")
    print(f"Avg monthly cost drag @10bps one-way: {avg_drag_bps:.1f} bps")
    print(f"Net spread: {s_net['mean']*12*100:+.2f}%/yr = {s_net['mean']*10000:+.1f}bps/mo  t={s_net['tstat']:+.2f}")

    print(f"\nDone. Fingerprint: {data.fingerprint(prices)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
