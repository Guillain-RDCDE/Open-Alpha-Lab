"""Real-tape verification — Study 206 (Dividend-Aristocrats). Regenerates docs/results.md numbers.

Fetches (or reads from cache) NOBL and SPY daily total-return prices, computes CAGR,
Sharpe, max drawdown, the excess-return t-stat, OLS alpha/beta, crash episode table,
and the random-mix control.  Network is touched only with --fetch.

    python studies/206-dividend-aristocrats/examples/verify.py            # cache-only
    python studies/206-dividend-aristocrats/examples/verify.py --fetch    # refresh prices
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dividend_aristocrats import data, strategy as st  # noqa: E402

CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))


def main(fetch: bool) -> None:
    # --- Warm / verify cache ---
    for t in data.TICKERS:
        df = data.fetch_prices(t, fetch=fetch, cache_dir=CACHE)
        fp = data.fingerprint(df)
        print(f"{t:5s} {df.index[0].date()} -> {df.index[-1].date()} n={len(df)} fp={fp}")
    print()

    prices = data.load_aligned(fetch=False, cache_dir=CACHE)
    fp_aligned = data.fingerprint(prices)
    print(f"Aligned: {prices.index[0].date()} -> {prices.index[-1].date()} n={len(prices)} fp={fp_aligned}")
    print()

    # --- Gross returns (auto_adjust=True so these are total returns) ---
    ret = st.daily_returns(prices).dropna()
    print("=== Gross (unadjusted for ER) ===")
    for col in ["NOBL", "SPY"]:
        s = st.summarize(ret[col], label=col)
        print(f"{col}: CAGR={s['cagr_pct']:+.2f}% vol={s['vol_ann_pct']:.1f}% "
              f"Sharpe={s['sharpe_ann']:+.3f} MaxDD={s['max_dd_pct']:+.1f}% "
              f"mean={s['mean_bps']:+.3f}bps t={s['tstat']:+.2f}")

    # --- Excess return ---
    diff = st.excess_return(prices)
    s_diff = st.summarize(diff, label="NOBL-SPY")
    print(f"\nExcess return (NOBL-SPY): mean={s_diff['mean_bps']:+.3f}bps/day "
          f"CAGR_diff={s_diff['cagr_pct']:+.2f}%/yr t={s_diff['tstat']:+.2f}")

    # --- Net of fees ---
    adj = st.net_of_fees(prices, nobl_er=st.NOBL_ER, spy_er=st.SPY_ER)
    ret_adj = st.daily_returns(adj).dropna()
    print("\n=== Net of ER ===")
    for col in ["NOBL", "SPY"]:
        s = st.summarize(ret_adj[col], label=col + " net")
        print(f"{col} net: CAGR={s['cagr_pct']:+.2f}% Sharpe={s['sharpe_ann']:+.3f}")

    # --- OLS alpha/beta ---
    ols = st.beta_alpha_ols(prices)
    print(f"\nOLS alpha: {ols['alpha_daily_bps']:+.3f} bps/day "
          f"({ols['alpha_ann_pct']:+.2f}%/yr) t={ols['t_alpha']:+.2f} "
          f"beta={ols['beta']:.4f} R2={ols['r_squared']:.4f}")

    # --- Total return ---
    nobl_total = prices["NOBL"].iloc[-1] / prices["NOBL"].iloc[0] - 1
    spy_total = prices["SPY"].iloc[-1] / prices["SPY"].iloc[0] - 1
    print(f"\nTotal return since {prices.index[0].date()}: "
          f"NOBL={nobl_total*100:.0f}% SPY={spy_total*100:.0f}%")

    # --- Crash analysis ---
    print("\n=== Crash episodes (SPY drawdown >= 15%) ===")
    crashes = st.crash_analysis(prices, threshold=-0.15)
    print(crashes.to_string())

    # --- Bootstrap Sharpe CI ---
    try:
        from quantlab import stats
        ci_nobl = stats.sharpe_ci_bootstrap(ret["NOBL"], periods_per_year=252, seed=206)
        ci_diff = stats.sharpe_ci_bootstrap(diff, periods_per_year=252, seed=206)
        print(f"\nNOBL Sharpe CI: [{ci_nobl['ci_low']:+.3f}, {ci_nobl['ci_high']:+.3f}] "
              f"frac_neg={ci_nobl['frac_negative']*100:.0f}%")
        print(f"Excess Sharpe CI: [{ci_diff['ci_low']:+.3f}, {ci_diff['ci_high']:+.3f}] "
              f"frac_neg={ci_diff['frac_negative']*100:.0f}%")
    except ImportError:
        print("\n(quantlab not available for bootstrap CI)")

    # --- Random-mix control ---
    ctrl = st.random_mix_control(prices, n_seeds=20, seed=206)
    frac_beat = (ctrl["sharpe_ann"] > ret["NOBL"].mean() / ret["NOBL"].std() * np.sqrt(252)).mean()
    print(f"\nRandom-mix control Sharpe (mean): {ctrl['sharpe_ann'].mean():+.3f} "
          f"frac > NOBL: {frac_beat*100:.0f}%")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
