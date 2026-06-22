"""Reproducible headline run for Study 359 — Farmland.

Prints every number quoted in docs/results.md and frozen into notebooks/build_notebooks.py
(the ``R`` dict). The listed-proxy numbers need the yfinance cache (``_cache/prices_monthly.csv``,
rebuilt with ``farmland.data.fetch_prices``); the NCREIF/CPI series and the synthetic control are
hardcoded/deterministic and run with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from farmland import data, strategy as st

# --------------------------------------------------------------------------- #
print("# Listed farmland proxies — month-end total returns (yfinance, auto-adjusted)")
if data.have_real():
    rets = data.monthly_returns()
    print(f"window         : {rets.index.min().date()} -> {rets.index.max().date()}")
    for c in ["LAND", "FPI", "SPY"]:
        s = rets[c].dropna()
        print(f"  {c:4s} n={len(s):3d}  ann_ret={s.mean()*12*100:5.1f}%  ann_vol={s.std()*np.sqrt(12)*100:5.1f}%  first={s.index.min().date()}")

    print("\n# Signal axis — CAPM market beta vs SPY (HAC, 6 lags)")
    for c in ["LAND", "FPI"]:
        r = st.market_beta(rets[c], rets["SPY"], lags=6)
        print(f"  {c:4s} beta={r['beta']:.2f} (t={r['t_beta']:.2f})  corr={r['corr']:.2f}  "
              f"r2={r['r2']:.2f}  alpha_ann={r['alpha']*12*100:+.1f}% (t={r['t_alpha']:.2f})  n={r['n']}")

    print("\n# Tradability — net Sharpe (15 bps one-way, buy-and-hold)")
    for c in ["LAND", "FPI"]:
        n = st.net_sharpe(rets[c], cost_oneway=0.0015, turnover_per_year=1.0)
        print(f"  {c:4s} sharpe_gross={n['sharpe_gross']:.2f}  sharpe_net={n['sharpe_net']:.2f}  "
              f"ann_ret_net={n['ann_return_net']*100:5.1f}%  ann_vol={n['ann_vol']*100:.1f}%")
    n = st.net_sharpe(rets["SPY"], cost_oneway=0.0001, turnover_per_year=1.0)
    print(f"  SPY  sharpe_gross={n['sharpe_gross']:.2f}  ann_vol={n['ann_vol']*100:.1f}%  (benchmark)")
else:
    print("  (no yfinance cache — rebuild with farmland.data.fetch_prices(); see docs/results.md)")

# --------------------------------------------------------------------------- #
print("\n# The smoothness myth — NCREIF farmland annual TR vs CPI / S&P (hardcoded public series)")
ann = data.ncreif_annual()
print(f"  years          : {ann.index.min()}-{ann.index.max()}  (n={len(ann)})")
print(f"  NCREIF farmland: mean={ann['farmland'].mean()*100:.1f}%  vol={ann['farmland'].std()*100:.1f}%  rho1={st.autocorr(ann['farmland'].values,1):.2f}")
print(f"  S&P 500 annual : mean={ann['spx'].mean()*100:.1f}%  vol={ann['spx'].std()*100:.1f}%  rho1={st.autocorr(ann['spx'].values,1):.2f}")
print(f"  corr(farmland, S&P) = {np.corrcoef(ann['farmland'], ann['spx'])[0,1]:+.2f}")

rf = st.inflation_beta(ann, "farmland", lags=2)
rs = st.inflation_beta(ann, "spx", lags=2)
print(f"\n# Inflation-hedge regression (annual return on CPI, HAC)")
print(f"  farmland : coef={rf['beta']:+.2f} (t={rf['t_beta']:.2f})  corr_CPI={rf['corr_cpi']:+.2f}  mean_real={rf['mean_real']*100:.1f}%")
print(f"  S&P 500  : coef={rs['beta']:+.2f} (t={rs['t_beta']:.2f})  corr_CPI={rs['corr_cpi']:+.2f}")

sm = st.smoothing_summary(ann["farmland"].values)
print(f"\n# Un-smoothing the appraisal index (Geltner first-order)")
print(f"  reported vol={sm['vol_reported']*100:.1f}%  ->  un-smoothed vol={sm['vol_unsmoothed']*100:.1f}%  "
      f"(ratio {sm['vol_ratio']:.2f}; appraisal hides ~{(1-sm['vol_ratio'])*100:.0f}% of the vol)")

# --------------------------------------------------------------------------- #
print("\n# Synthetic positive control — appraisal smoothing, measured vs closed form")
for a in (0.10, 0.20, 0.40):
    df = data.synthetic_smoothing(n=4000, a=a, beta=0.30, seed=359)
    vr_meas = np.var(df["app"], ddof=1) / np.var(df["true"], ddof=1)
    rho_meas = st.autocorr(df["app"].values, 1)
    bt = st.ols_hac(df["true"].values, df["mkt"].values, lags=4)["beta"]
    ba = st.ols_hac(df["app"].values, df["mkt"].values, lags=4)["beta"]
    print(f"  a={a:.2f}: var-ratio meas={vr_meas:.3f} exact={st.expected_var_ratio(a):.3f}  |  "
          f"rho1 meas={rho_meas:.3f} exact={st.expected_rho1(a):.3f}  |  beta true={bt:.2f} -> appraised={ba:.2f}")
