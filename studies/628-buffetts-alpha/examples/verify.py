"""Reproducible headline run for Study 628 — Buffett's Alpha.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached monthly frame under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from buffetts_alpha import data, strategy as st

try:
    from quantlab import repro
except Exception:                                  # pragma: no cover
    repro = None

print("# Buffett's Alpha — BRK-A vs the US market, monthly, 1980-2026 (yfinance + Shiller)")
if data.have_real():
    df = data.load_real()
    if repro is not None:
        print(repro.data_stamp("ba_monthly", df, cols=["brk", "mkt", "rf"],
                               asof=data.AS_OF))
    w_brk = float((1 + df["brk"]).prod())
    w_mkt = float((1 + df["mkt"]).prod())
    n = len(df)
    print(f"window          : {df.index.min().date()} -> {df.index.max().date()}  "
          f"({n} months, {n/12:.1f} years)")
    print(f"market splice   : {(df['mkt_src'] == 'gspc+shiller_dy').sum()} months "
          f"^GSPC+Shiller-div-yield, {(df['mkt_src'] == 'spy_tr').sum()} months SPY total return")
    print(f"$1 grows to     : BRK ${w_brk:,.0f}  vs  market ${w_mkt:,.0f}  "
          f"(BRK {(w_brk**(12/n)-1)*100:.2f}%/yr vs mkt {(w_mkt**(12/n)-1)*100:.2f}%/yr, total return)")

    print("\n# Full-sample CAPM (excess-vs-excess, Newey-West HAC)")
    c = st.capm(df)
    print(f"  alpha         : {c['alpha_ann_pct']:+.2f}%/yr   HAC t = {c['t_alpha']:+.2f}  "
          f"(NW lags={c['lags']}, n={c['n']})")
    print(f"  beta          : {c['beta']:.2f}  (t={c['t_beta']:.1f})   R^2={c['r2']:.2f}   "
          f"IR={c['ir']:.2f}")
    print(f"  excess return : BRK {c['brk_exc_ann_pct']:+.2f}%/yr vs mkt {c['mkt_exc_ann_pct']:+.2f}%/yr")
    print(f"  Sharpe (excess-vs-excess): BRK {c['brk_sharpe']:.2f} vs mkt {c['mkt_sharpe']:.2f}  "
          f"(BRK vol {c['brk_vol_ann_pct']:.1f}%/yr)")

    print("\n# FKP-era subsample (our tape start -> 2011-12, the paper's end date)")
    cf = st.capm(df, end="2011-12-31")
    print(f"  alpha {cf['alpha_ann_pct']:+.2f}%/yr  HAC t = {cf['t_alpha']:+.2f}   beta {cf['beta']:.2f}  "
          f"(n={cf['n']})  Sharpe BRK {cf['brk_sharpe']:.2f} vs mkt {cf['mkt_sharpe']:.2f}")

    print("\n# Decade table (CAPM per decade; 2020s is a partial decade)")
    dt = st.decade_table(df)
    for dec, r in dt.iterrows():
        print(f"  {dec}: alpha {r['alpha_ann_pct']:+6.2f}%/yr  t={r['t_alpha']:+5.2f}  "
              f"beta {r['beta']:.2f}  (BRK exc {r['brk_exc_ann_pct']:+.2f}% vs "
              f"mkt {r['mkt_exc_ann_pct']:+.2f}%, {int(r['months'])}m)")

    print("\n# Rolling 10-year CAPM alpha (the decay curve)")
    ra = st.rolling_alpha(df, window=120)
    last_sig = ra[ra["t_alpha"] >= 2].index.max()
    print(f"  windows       : {len(ra)} (ends {ra.index.min().date()} -> {ra.index.max().date()})")
    print(f"  peak          : {ra['alpha_ann_pct'].max():+.2f}%/yr at "
          f"{ra['alpha_ann_pct'].idxmax().date()}")
    print(f"  latest        : {ra['alpha_ann_pct'].iloc[-1]:+.2f}%/yr  "
          f"t={ra['t_alpha'].iloc[-1]:+.2f}  beta={ra['beta'].iloc[-1]:.2f}")
    print(f"  last window with t>=2 ends: {last_sig.date()}   "
          f"(share of windows with t>=2: {(ra['t_alpha'] >= 2).mean()*100:.1f}%)")

    print("\n# Has the alpha faded? (early vs post-2010 contrast, HAC t on the DIFFERENCE)")
    fd = st.subperiod_fade(df, split="2010-12-31")
    print(f"  early alpha   : {fd['alpha_early_ann_pct']:+.2f}%/yr  (t={fd['t_alpha_early']:+.2f})")
    print(f"  alpha change  : {fd['d_alpha_ann_pct']:+.2f} pp/yr after 2010   "
          f"HAC t = {fd['t_d_alpha']:+.2f}")
    print(f"  beta change   : {fd['d_beta']:+.2f}  (t={fd['t_d_beta']:+.2f})")

    print("\n# Third axis — any alpha left in the last 15 years? (2011-07 -> 2026-06)")
    c15 = st.capm(df, start="2011-07-01")
    d15 = df[df.index >= "2011-07-01"]
    w15b = float((1 + d15["brk"]).prod())
    w15m = float((1 + d15["mkt"]).prod())
    print(f"  alpha         : {c15['alpha_ann_pct']:+.2f}%/yr   HAC t = {c15['t_alpha']:+.2f}  "
          f"(n={c15['n']})")
    print(f"  beta          : {c15['beta']:.2f}   Sharpe BRK {c15['brk_sharpe']:.2f} "
          f"vs mkt {c15['mkt_sharpe']:.2f}")
    print(f"  $1 grows to   : BRK ${w15b:.2f} vs market ${w15m:.2f} — Berkshire LAGGED the index")

    print("\n# Factor-lite attribution (investable ETF proxies QUAL + USMV, 2013-08 ->; "
          "NOT the academic QMJ/BAB factors)")
    fl = st.factor_lite(df)
    print(f"  window        : {fl['start']} -> {fl['end']}  (n={fl['n']})")
    print(f"  CAPM alone    : alpha {fl['capm_alpha_ann_pct']:+.2f}%/yr (t={fl['capm_t_alpha']:+.2f})  "
          f"beta {fl['capm_beta']:.2f}  R^2={fl['r2_capm']:.2f}")
    print(f"  + QUAL + USMV : alpha {fl['full_alpha_ann_pct']:+.2f}%/yr (t={fl['full_t_alpha']:+.2f})  "
          f"R^2={fl['r2_full']:.2f}")
    print(f"  loadings      : mkt {fl['b_mkt']:+.2f} (t={fl['t_mkt']:+.2f})  "
          f"QUAL {fl['b_qual']:+.2f} (t={fl['t_qual']:+.2f})  "
          f"USMV {fl['b_usmv']:+.2f} (t={fl['t_usmv']:+.2f})")
else:
    print("(no _cache/ba_monthly.csv — run data.fetch_all() once to build the cache)")

print("\n# Synthetic control — deterministic, no network, averaged over 20 seeds")
print("  the HAC-alpha machinery must recover a PLANTED alpha and must NOT manufacture")
print("  significance when the true alpha is 0. (Machinery proof only — never market evidence.)")
for a in (0.0, 0.005):
    ts, als = [], []
    for s in range(628, 648):
        c = st.capm(data.synthetic_world(alpha_m=a, seed=s))
        ts.append(c["t_alpha"])
        als.append(c["alpha_ann_pct"])
    ts, als = np.array(ts), np.array(als)
    print(f"  planted alpha={a*12*100:5.1f}%/yr: mean recovered {als.mean():+.2f}%/yr  "
          f"mean HAC t {ts.mean():+.2f}  share of seeds |t|>=2: {np.mean(np.abs(ts) >= 2)*100:.0f}%")
