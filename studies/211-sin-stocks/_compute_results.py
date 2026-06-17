"""Compute headline numbers for Study 211 sin-stocks."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from sin_stocks import data, strategy as st

CACHE = os.path.join(os.path.dirname(__file__), "_cache")

# Load all tickers; PM starts 2008-03-17 -> use inner join start
print("Loading all tickers...")
all_prices = data.load_aligned(fetch=False, cache_dir=CACHE)
print(f"Aligned: {all_prices.index[0].date()} -> {all_prices.index[-1].date()} n={len(all_prices)}")
print(f"Tickers: {list(all_prices.columns)}")
print()

# Build equal-weight basket
basket_prices = all_prices[data.SIN_BASKET]
ew_ret = st.build_equal_weight(basket_prices)

# Also need SPY and DSI returns
spy_ret = np.log(all_prices["SPY"] / all_prices["SPY"].shift(1)).dropna()
dsi_ret = np.log(all_prices["DSI"] / all_prices["DSI"].shift(1)).dropna()

# Align all
idx = ew_ret.index.intersection(spy_ret.index).intersection(dsi_ret.index)
ew_ret = ew_ret.loc[idx]
spy_ret = spy_ret.loc[idx]
dsi_ret = dsi_ret.loc[idx]

print(f"Common index: {idx[0].date()} -> {idx[-1].date()} n={len(idx)}")
print()

# Summary stats
s_sin = st.summarize(ew_ret, label="SIN_EW")
s_spy = st.summarize(spy_ret, label="SPY")
s_dsi = st.summarize(dsi_ret, label="DSI")

for s in [s_sin, s_spy, s_dsi]:
    print(f"{s['label']:8s}: CAGR={s['cagr_pct']:+.2f}% vol={s['vol_ann_pct']:.1f}% "
          f"Sharpe={s['sharpe_ann']:+.3f} MaxDD={s['max_dd_pct']:+.1f}% "
          f"mean={s['mean_bps']:+.3f}bps t={s['tstat']:+.2f}")

# Excess returns
diff_vs_spy = ew_ret - spy_ret
diff_vs_dsi = ew_ret - dsi_ret

s_exc_spy = st.summarize(diff_vs_spy, label="SIN-SPY")
s_exc_dsi = st.summarize(diff_vs_dsi, label="SIN-DSI")

print()
print(f"Excess vs SPY: {s_exc_spy['cagr_pct']:+.2f}%/yr  t={s_exc_spy['tstat']:+.2f}")
print(f"Excess vs DSI: {s_exc_dsi['cagr_pct']:+.2f}%/yr  t={s_exc_dsi['tstat']:+.2f}")

# OLS
ols_spy = st.beta_alpha_ols(ew_ret, spy_ret, label="SIN vs SPY")
ols_dsi = st.beta_alpha_ols(ew_ret, dsi_ret, label="SIN vs DSI")
print()
print(f"OLS vs SPY: alpha={ols_spy['alpha_daily_bps']:+.3f} bps/day ({ols_spy['alpha_ann_pct']:+.2f}%/yr) "
      f"t={ols_spy['t_alpha']:+.2f} beta={ols_spy['beta']:.4f} R2={ols_spy['r_squared']:.4f}")
print(f"OLS vs DSI: alpha={ols_dsi['alpha_daily_bps']:+.3f} bps/day ({ols_dsi['alpha_ann_pct']:+.2f}%/yr) "
      f"t={ols_dsi['t_alpha']:+.2f} beta={ols_dsi['beta']:.4f} R2={ols_dsi['r_squared']:.4f}")

# Total returns
sin_total = (all_prices[data.SIN_BASKET].iloc[-1] / all_prices[data.SIN_BASKET].iloc[0] - 1).mean() * 100
spy_total = (all_prices["SPY"].iloc[-1] / all_prices["SPY"].iloc[0] - 1) * 100
dsi_total = (all_prices["DSI"].iloc[-1] / all_prices["DSI"].iloc[0] - 1) * 100
print()
print(f"Total return (avg basket): {sin_total:.0f}%")
print(f"Total return SPY: {spy_total:.0f}%")
print(f"Total return DSI: {dsi_total:.0f}%")

# Per-ticker attribution
print()
print("=== Per-ticker attribution ===")
attr = st.sector_attribution(all_prices, basket=data.SIN_BASKET)
print(attr[["cagr_pct", "vol_ann_pct", "sharpe_ann", "max_dd_pct"]].round(2))

# Crash analysis
print()
print("=== Crash episodes (SPY drawdown >= 10%) ===")
all_ret_df = pd.DataFrame({"SIN_EW": ew_ret, "SPY": spy_ret, "DSI": dsi_ret})
crashes = st.crash_analysis(all_ret_df, asset="SIN_EW", benchmark="SPY", threshold=-0.10)
print(crashes.to_string())

# Fingerprint
fp = data.fingerprint(all_prices)
print(f"\nFingerprint: {fp}")

# Bootstrap Sharpe CI (try quantlab)
try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from quantlab import stats
    ci = stats.sharpe_ci_bootstrap(diff_vs_spy, periods_per_year=252, seed=211)
    print(f"\nExcess Sharpe CI (vs SPY): [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}] "
          f"frac_neg={ci['frac_negative']*100:.0f}%")
    ci2 = stats.sharpe_ci_bootstrap(ew_ret, periods_per_year=252, seed=211)
    print(f"SIN_EW Sharpe CI: [{ci2['ci_low']:+.3f}, {ci2['ci_high']:+.3f}]")
except Exception as e:
    print(f"\n(quantlab not available: {e})")
