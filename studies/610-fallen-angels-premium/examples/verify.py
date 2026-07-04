"""Reproducible headline run for Study 610 — Fallen-Angels-Premium.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached ETF panel under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))  # repo root (quantlab)

from fallen_angels_premium import data, strategy as st
from quantlab import repro


def show(tag: str, r: dict, factors: list[str] = ()) -> None:
    line = (f"  {tag}: alpha {r['alpha_bps']:+.2f} bps/mo (se {r['alpha_se_bps']:.2f})  "
            f"HAC t = {r['t_alpha']:+.2f}  (n={r['n']}, lags={r['lags']})")
    for f in factors:
        line += f"  beta_{f}={r[f'beta_{f}']:+.3f} (t={r[f't_{f}']:+.1f})"
    print(line)


print("# Fallen-Angels-Premium — ANGL vs HYG/JNK (+FALN confirmation), yfinance total return")
if data.have_real():
    px = data.load_prices()                      # sliced to the frozen as-of
    print(repro.data_stamp("ETF panel (total-return closes)", px, asof=data.AS_OF))
    m = st.monthly_returns(px)

    common = st.align_common(m, ["ANGL", "HYG", "JNK", "IEF", "LQD", "BIL"])
    print(f"common monthly sample (ANGL era): {len(common)} months, "
          f"{common.index.min():%Y-%m} -> {common.index.max():%Y-%m}")
    mc = common  # everything ANGL-era runs on the common sample

    print("\n# Headline — monthly excess spread ANGL - HYG (raw, total return)")
    diff = (mc["ANGL"] - mc["HYG"]).dropna()
    h = st.hac_mean(diff)
    print(f"  ANGL - HYG: {h['alpha_bps']:+.2f} bps/mo  HAC t = {h['t_alpha']:+.2f}  "
          f"(n={h['n']}, lags={h['lags']})")
    print(f"  annualised: ANGL {st.ann_return(mc['ANGL']):.2f}%/yr  vs  "
          f"HYG {st.ann_return(mc['HYG']):.2f}%/yr  "
          f"(spread {st.ann_return(mc['ANGL']) - st.ann_return(mc['HYG']):+.2f} pp/yr)")

    print("\n# CAPM-style alpha — ANGL excess on HYG excess (rf = BIL, excess-vs-excess)")
    c1 = st.capm_alpha(mc, "ANGL", "HYG")
    show("ANGL ~ HYG", c1, ["HYG"])

    print("\n# Duration/quality honesty check — add the rate factor (IEF), then IG credit (LQD)")
    f2 = st.factor_alpha(mc, "ANGL", ["HYG", "IEF"])
    show("ANGL ~ HYG + IEF", f2, ["HYG", "IEF"])
    f3 = st.factor_alpha(mc, "ANGL", ["HYG", "IEF", "LQD"])
    show("ANGL ~ HYG + IEF + LQD", f3, ["HYG", "IEF", "LQD"])

    print("\n# Robustness — JNK instead of HYG as the benchmark")
    cj = st.capm_alpha(mc, "ANGL", "JNK")
    show("ANGL ~ JNK", cj, ["JNK"])
    fj = st.factor_alpha(mc, "ANGL", ["JNK", "IEF"])
    show("ANGL ~ JNK + IEF", fj, ["JNK", "IEF"])

    print("\n# Confirmation — FALN (independent issuer/index, 2016->)")
    mf = st.align_common(m, ["FALN", "HYG", "IEF", "BIL"])
    print(f"  FALN sample: {len(mf)} months, {mf.index.min():%Y-%m} -> {mf.index.max():%Y-%m}")
    dfn = (mf["FALN"] - mf["HYG"]).dropna()
    hf = st.hac_mean(dfn)
    print(f"  FALN - HYG: {hf['alpha_bps']:+.2f} bps/mo  HAC t = {hf['t_alpha']:+.2f}  (n={hf['n']})")
    cf = st.capm_alpha(mf, "FALN", "HYG")
    show("FALN ~ HYG", cf, ["HYG"])
    ff = st.factor_alpha(mf, "FALN", ["HYG", "IEF"])
    show("FALN ~ HYG + IEF", ff, ["HYG", "IEF"])

    print("\n# Sub-periods — the premium by half (split at 2019-06, the sample midpoint)")
    h1, h2 = diff.loc[:"2019-06-30"], diff.loc["2019-07-01":]
    for tag, sl in (("2012-05 -> 2019-06", h1), ("2019-07 -> 2026-06", h2)):
        hs = st.hac_mean(sl)
        print(f"  {tag}: {hs['alpha_bps']:+.2f} bps/mo  HAC t = {hs['t_alpha']:+.2f}  (n={hs['n']})")
    wt = st.welch_t(h1, h2)
    print(f"  decay test (half-1 minus half-2): diff {wt['diff_bps']:+.2f} bps/mo  "
          f"Welch t = {wt['t']:+.2f}  -> decay NOT certified" if abs(wt['t']) < 2
          else f"  decay test: diff {wt['diff_bps']:+.2f} bps/mo  Welch t = {wt['t']:+.2f}")

    print("\n# Sharpe race — excess-vs-excess (rf = BIL), common ANGL-era sample")
    for tk in ("ANGL", "HYG", "JNK"):
        print(f"  {tk}: excess Sharpe {st.sharpe_excess(mc, tk):.2f}  "
              f"(ann total return {st.ann_return(mc[tk]):.2f}%/yr)")

    print("\n# Drawdowns — daily total-return, ANGL era (2012-04-11 ->)")
    for tk in ("ANGL", "HYG"):
        dd = st.max_drawdown(px[tk].loc["2012-04-11":])
        print(f"  {tk}: max DD {dd['depth_pct']:.1f}%  (peak {dd['peak']} -> trough {dd['trough']})")

    print("\n# Costs — the trade is ONE switch (sell HYG, buy ANGL), long-only, no borrow")
    yrs = len(diff) / 12.0
    for sp in (2.0, 5.0, 10.0):
        drag = st.switch_cost_drag(sp, yrs)
        net = h["alpha_bps"] * 12.0 - drag
        print(f"  {sp:>4.1f} bps one-way x 2 legs over {yrs:.1f} yrs: drag {drag:.1f} bps/yr  "
              f"-> net spread {net:+.0f} bps/yr (gross {h['alpha_bps']*12:+.0f})")
    print("  (ETF fees already inside the tape: ANGL ER 0.35%, HYG 0.49%, FALN 0.25% - "
          "net-of-fee total returns)")

else:
    print("(no _cache/fap_prices.csv — run data.fetch_prices() once to build the cache)")

print("\n# Synthetic control — machinery proof (deterministic, no network; never market evidence)")
print("  the HAC-alpha pipeline must recover a PLANTED alpha and must NOT manufacture one from 0.")
for planted in (0.0, 20.0):
    w = data.synthetic_world(alpha_bps=planted, seed=610)
    r1 = st.nw_ols(w["FA"], w[["HY"]])
    r2 = st.nw_ols(w["FA"], w[["HY", "RATE"]])
    print(f"  planted alpha={planted:+5.1f} bps/mo: 1-factor alpha {r1['alpha_bps']:+.2f} "
          f"(t={r1['t_alpha']:+.2f})  2-factor alpha {r2['alpha_bps']:+.2f} (t={r2['t_alpha']:+.2f})")
