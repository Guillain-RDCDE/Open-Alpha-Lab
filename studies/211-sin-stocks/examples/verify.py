"""Real-tape verification — Study 211 (Sin-Stocks). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the sin basket and benchmark daily total-return prices,
computes CAGR, Sharpe, max drawdown, the excess-return t-stat, OLS alpha/beta, per-ticker
attribution, and the crash-episode table.  Network is touched only with --fetch.

    python studies/211-sin-stocks/examples/verify.py            # cache-only
    python studies/211-sin-stocks/examples/verify.py --fetch    # refresh prices
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sin_stocks import data, strategy as st  # noqa: E402

CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))


def main(fetch: bool) -> None:
    # --- Warm / verify cache ---
    for t in data.ALL_TICKERS:
        df = data.fetch_prices(t, fetch=fetch, cache_dir=CACHE)
        fp = data.fingerprint(df)
        print(f"{t:5s} {df.index[0].date()} -> {df.index[-1].date()} n={len(df)} fp={fp}")
    print()

    all_prices = data.load_aligned(fetch=False, cache_dir=CACHE)
    fp_aligned = data.fingerprint(all_prices)
    print(f"Aligned: {all_prices.index[0].date()} -> {all_prices.index[-1].date()} "
          f"n={len(all_prices)} fp={fp_aligned}")
    print()

    # --- Build equal-weight basket ---
    basket_prices = all_prices[data.SIN_BASKET]
    ew_ret = st.build_equal_weight(basket_prices)
    spy_ret = np.log(all_prices["SPY"] / all_prices["SPY"].shift(1)).dropna()
    dsi_ret = np.log(all_prices["DSI"] / all_prices["DSI"].shift(1)).dropna()
    idx = ew_ret.index.intersection(spy_ret.index).intersection(dsi_ret.index)
    ew_ret = ew_ret.loc[idx]
    spy_ret = spy_ret.loc[idx]
    dsi_ret = dsi_ret.loc[idx]

    print(f"Common: {idx[0].date()} -> {idx[-1].date()} n={len(idx)}")
    print()

    # --- Gross returns ---
    print("=== Gross returns ===")
    for label, ret in [("SIN_EW", ew_ret), ("SPY", spy_ret), ("DSI", dsi_ret)]:
        s = st.summarize(ret, label=label)
        print(f"{label:8s}: CAGR={s['cagr_pct']:+.2f}% vol={s['vol_ann_pct']:.1f}% "
              f"Sharpe={s['sharpe_ann']:+.3f} MaxDD={s['max_dd_pct']:+.1f}% "
              f"mean={s['mean_bps']:+.3f}bps t={s['tstat']:+.2f}")

    # --- Excess returns ---
    diff_vs_spy = ew_ret - spy_ret
    diff_vs_dsi = ew_ret - dsi_ret
    s_exc_spy = st.summarize(diff_vs_spy, label="SIN-SPY")
    s_exc_dsi = st.summarize(diff_vs_dsi, label="SIN-DSI")
    print(f"\nExcess vs SPY: CAGR={s_exc_spy['cagr_pct']:+.2f}%/yr "
          f"mean={s_exc_spy['mean_bps']:+.3f}bps t={s_exc_spy['tstat']:+.2f}")
    print(f"Excess vs DSI: CAGR={s_exc_dsi['cagr_pct']:+.2f}%/yr "
          f"mean={s_exc_dsi['mean_bps']:+.3f}bps t={s_exc_dsi['tstat']:+.2f}")

    # --- OLS decomposition ---
    ols_spy = st.beta_alpha_ols(ew_ret, spy_ret, label="SIN vs SPY")
    ols_dsi = st.beta_alpha_ols(ew_ret, dsi_ret, label="SIN vs DSI")
    print(f"\nOLS vs SPY: alpha={ols_spy['alpha_daily_bps']:+.3f} bps/day "
          f"({ols_spy['alpha_ann_pct']:+.2f}%/yr) t={ols_spy['t_alpha']:+.2f} "
          f"beta={ols_spy['beta']:.4f} R2={ols_spy['r_squared']:.4f}")
    print(f"OLS vs DSI: alpha={ols_dsi['alpha_daily_bps']:+.3f} bps/day "
          f"({ols_dsi['alpha_ann_pct']:+.2f}%/yr) t={ols_dsi['t_alpha']:+.2f} "
          f"beta={ols_dsi['beta']:.4f} R2={ols_dsi['r_squared']:.4f}")

    # --- Total return ---
    sin_total = (np.exp(np.sum(ew_ret)) - 1) * 100
    spy_total = (np.exp(np.sum(spy_ret)) - 1) * 100
    dsi_total = (np.exp(np.sum(dsi_ret)) - 1) * 100
    print(f"\nTotal return: SIN_EW={sin_total:.0f}%  SPY={spy_total:.0f}%  DSI={dsi_total:.0f}%")

    # --- Per-ticker attribution ---
    print("\n=== Per-ticker attribution ===")
    attr = st.sector_attribution(all_prices, basket=data.SIN_BASKET)
    print(attr[["cagr_pct", "vol_ann_pct", "sharpe_ann", "max_dd_pct"]].round(2))

    # --- Crash analysis ---
    all_ret_df = pd.DataFrame({"SIN_EW": ew_ret, "SPY": spy_ret, "DSI": dsi_ret})
    print("\n=== Crash episodes (SPY drawdown >= 10%) ===")
    crashes = st.crash_analysis(all_ret_df, asset="SIN_EW", benchmark="SPY", threshold=-0.10)
    print(crashes.to_string())

    # --- Bootstrap Sharpe CI ---
    try:
        from quantlab import stats
        ci_diff = stats.sharpe_ci_bootstrap(diff_vs_spy, periods_per_year=252, seed=211)
        print(f"\nExcess Sharpe CI (vs SPY): [{ci_diff['ci_low']:+.3f}, {ci_diff['ci_high']:+.3f}] "
              f"frac_neg={ci_diff['frac_negative']*100:.0f}%")
    except ImportError:
        print("\n(quantlab not available for bootstrap CI)")

    # --- Content fingerprint ---
    fp_final = data.fingerprint(all_prices)
    print(f"\nFingerprint (aligned): {fp_final}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
