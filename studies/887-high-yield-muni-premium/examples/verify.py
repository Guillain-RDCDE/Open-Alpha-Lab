"""Reproducible headline run for Study 887 — High-Yield Muni Premium.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached ETF panels under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))  # repo root

from hy_muni import data, strategy as st
from quantlab import repro

RATE = data.TOP_MARGINAL_RATE

print("# High-Yield Muni Premium — HYD vs MUB/TFI, taxable-HY (HYG), rf=BIL; yfinance total return")

if data.have_real():
    px = data.load_prices()          # total-return closes, sliced to the frozen as-of
    pr = data.load_price_only()      # price-only closes (for income decomposition)
    print(repro.data_stamp("ETF panel (total-return closes)", px, asof=data.AS_OF))

    m = st.monthly_returns(px, asof=data.AS_OF)
    inc = st.monthly_income(px, pr, asof=data.AS_OF)
    mc = st.align_common(m, ["HYD", "MUB", "TFI", "HYG", "BIL"])
    print(f"common monthly sample (HYD era): {len(mc)} months, "
          f"{mc.index.min():%Y-%m} -> {mc.index.max():%Y-%m} ({len(mc)/12:.1f} yrs)")

    print("\n# Headline — the HY-muni credit spread HYD - MUB (monthly total return)")
    sp = st.premium_series(mc)
    h = st.hac_mean(sp.values)
    boot = st.bootstrap_mean_ci(sp.values)
    print(f"  HYD - MUB: {h['mean_bps']:+.2f} bps/mo  HAC t = {h['tstat']:+.2f}  "
          f"(n={h['n']}, lags={h['lags']})  ~ {h['mean_bps']*12/100:+.2f} pp/yr")
    print(f"  bootstrap mean 95% CI: [{boot['ci_low_bps']:+.2f}, {boot['ci_high_bps']:+.2f}] "
          f"bps/mo  (frac<0 = {boot['frac_negative']:.3f}, block={boot['block_size']})")
    print(f"  annualised total return: HYD {st.ann_return(mc['HYD']):.2f}%  "
          f"MUB {st.ann_return(mc['MUB']):.2f}%  TFI {st.ann_return(mc['TFI']):.2f}%  "
          f"HYG {st.ann_return(mc['HYG']):.2f}%  BIL {st.ann_return(mc['BIL']):.2f}%")

    print("\n# Excess-vs-excess Sharpe race (rf = BIL)")
    for tk in ("HYD", "MUB", "TFI", "HYG"):
        print(f"  {tk}: excess Sharpe {st.sharpe_excess(mc, tk):+.3f}")
    print(f"  --> HYD advantage over MUB: "
          f"{st.sharpe_excess(mc,'HYD') - st.sharpe_excess(mc,'MUB'):+.3f}")

    print("\n# Era cut + the crisis windows (where illiquidity bites)")
    eras = st.era_table(sp, [
        ("2009-03", "2016-12", "2009-2016"),
        ("2017-01", "2026-06", "2017-2026"),
        ("2020-01", "2020-12", "2020 COVID"),
        ("2022-01", "2022-12", "2022 rate shock"),
    ])
    for e in eras:
        print(f"  {e['label']:16s}: {e['mean_bps']:+7.1f} bps/mo  HAC t = {e['tstat']:+.2f}  (n={e['n']})")

    print("\n# Sharpe advantage HYD-MUB by half (is the edge persistent across eras?)")
    for lo, hi, lbl in (("2009-03", "2017-12", "H1 2009-2017"),
                        ("2018-01", "2026-06", "H2 2018-2026")):
        mh = mc.loc[lo:hi]
        d = st.sharpe_excess(mh, "HYD") - st.sharpe_excess(mh, "MUB")
        print(f"  {lbl}: HYD {st.sharpe_excess(mh,'HYD'):+.2f}  MUB {st.sharpe_excess(mh,'MUB'):+.2f}"
              f"  diff {d:+.2f}")

    print("\n# The tax wrapper — income yields, tax-equivalent yield, after-tax race")
    iy = st.income_yields(inc.loc[mc.index], ["HYD", "MUB", "HYG"])
    print(f"  income (distribution) yield: HYD {iy['HYD']:.2f}%  MUB {iy['MUB']:.2f}%  HYG {iy['HYG']:.2f}%")
    tey = st.tax_equivalent_yield(iy["HYD"], RATE)
    print(f"  HYD tax-equivalent yield @ {RATE*100:.1f}% = {tey:.2f}%  vs  taxable HYG {iy['HYG']:.2f}%")
    at_hyd = st.after_tax_returns(mc, inc.loc[mc.index], "HYD", RATE, tax_exempt=True)
    at_mub = st.after_tax_returns(mc, inc.loc[mc.index], "MUB", RATE, tax_exempt=True)
    at_hyg = st.after_tax_returns(mc, inc.loc[mc.index], "HYG", RATE, tax_exempt=False)
    print(f"  after-tax annualised: HYD {st.ann_return(at_hyd):.2f}%  MUB {st.ann_return(at_mub):.2f}%  "
          f"HYG {st.ann_return(at_hyg):.2f}%")
    print(f"  after-tax excess Sharpe: HYD {st.after_tax_sharpe(at_hyd, mc, RATE):+.2f}  "
          f"MUB {st.after_tax_sharpe(at_mub, mc, RATE):+.2f}  HYG {st.after_tax_sharpe(at_hyg, mc, RATE):+.2f}")

    print("\n# Risk — daily total-return max drawdown (HYD era, 2009-02 ->)")
    for tk in ("HYD", "MUB", "HYG"):
        dd = st.max_drawdown(px[tk].loc["2009-02-01":])
        print(f"  {tk}: max DD {dd['depth_pct']:.1f}%  (peak {dd['peak']} -> trough {dd['trough']})")

    print("\n# Costs — the trade is ONE switch (sell MUB, buy HYD), long-only, no borrow")
    yrs = len(sp) / 12.0
    gross = h["mean_bps"] * 12.0
    for spx in (5.0, 15.0, 30.0):
        drag = st.switch_cost_drag(spx, yrs)
        print(f"  {spx:>5.1f} bps one-way x 2 legs over {yrs:.1f} yrs: drag {drag:.2f} bps/yr  "
              f"-> net {gross - drag:+.0f} bps/yr (gross {gross:+.0f})")
    print("  (ETF fees already inside the tape: HYD ER 0.35%, MUB 0.05%, TFI 0.23%, HYG 0.49%)")

else:
    print("(no _cache/hym_prices.csv - run data.fetch() once to build the caches)")

print("\n# Synthetic control - machinery proof (deterministic, no network; never market evidence)")
print("  the HAC/bootstrap pipeline must recover a PLANTED premium and NOT manufacture one from 0.")
for planted in (0.0, 0.03):
    w = data.synthetic_world(premium_annual=planted, seed=887)
    d = st.synthetic_detect(w)
    print(f"  planted {planted*100:+.1f}%/yr: mean {d['mean_bps']:+.2f} bps/mo  HAC t = {d['tstat']:+.2f}  "
          f"CI [{d['ci_low_bps']:+.1f}, {d['ci_high_bps']:+.1f}]  frac<0 {d['frac_negative']:.3f}")
