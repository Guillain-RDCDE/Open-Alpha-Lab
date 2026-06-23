"""Reproducible headline run for Study 396 — Reshoring-Basket.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached ETF prices under ``_cache/`` if
present (the real-tape numbers), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reshoring_basket import data, strategy as st

print("# Reshoring-Basket — equal-weight XLI (US industrials) / EWW (Mexico) / ROK (automation)")
if data.have_real():
    f = data.load_real()
    span = (f.index.max() - f.index.min()).days / 365.25
    print(f"basket days    : {len(f)}  ({f.index.min().date()} -> {f.index.max().date()}, "
          f"{span:.1f} years)")

    print("\n# Full sample — raw excess vs beta-adjusted CAPM alpha (HAC = Newey-West)")
    print(f"  {'bench':>6} {'excess/yr':>10} {'t_ex':>6} {'hac_ex':>7} {'alpha/yr':>9} "
          f"{'beta':>5} {'hac_alpha':>10} {'Sh_bask':>8} {'Sh_bench':>9} {'p_plac':>7}")
    for bench, sub in (("SPY", f), ("ACWI", f.dropna(subset=["acwi"]))):
        col = "mkt" if bench == "SPY" else "acwi"
        s = st.summarize(sub, col)
        print(f"  {bench:>6} {s['excess_ann']*100:>9.2f}% {s['t_excess']:>6.2f} "
              f"{s['t_excess_hac']:>7.2f} {s['alpha_ann']*100:>8.2f}% {s['beta']:>5.2f} "
              f"{s['t_alpha_hac']:>10.2f} {s['sharpe_basket']:>8.2f} {s['sharpe_bench']:>9.2f} "
              f"{s['p_placebo']:>7.3f}")

    print("\n# The kill shot — pre/post-2018 reshoring-narrative split (vs SPY)")
    split = data.narrative_start()
    for name, sub in (("pre-2018", f[f.index < split]),
                      ("post-2018 (narrative)", f[f.index >= split])):
        s = st.summarize(sub, "mkt")
        print(f"  {name:22} n={s['n']:>5}  excess={s['excess_ann']*100:>6.2f}%/yr  "
              f"alpha={s['alpha_ann']*100:>6.2f}%/yr  hac_t(alpha)={s['t_alpha_hac']:>5.2f}  "
              f"beta={s['beta']:.2f}  Sh_bask={s['sharpe_basket']:.2f} Sh_bench={s['sharpe_bench']:.2f}")

    print("\n# It's one stock, not a theme — per-sleeve excess vs SPY")
    rets = data.load_prices().pct_change()
    spy = rets["SPY"]
    for nm in data.RESHORING_NAMES:
        x = (rets[nm] - spy).dropna()
        print(f"  {nm:>4}  excess={x.mean()*252*100:>6.2f}%/yr  hac_t={st.hac_t(x.values):>5.2f}")

    print("\n# Costs — long-basket / short-SPY, annual rebalance (one-way × turnover)")
    for cb in (5, 10, 20):
        c = st.net_of_costs(f, "mkt", cost_bps=cb)
        print(f"  {cb:>2}bps  gross={c['gross_excess_ann']*100:>5.2f}%/yr  "
              f"net={c['net_excess_ann']*100:>5.2f}%/yr  annual_cost={c['annual_cost']*100:.2f}%")
else:
    print("(no _cache/reshoring_prices.csv — run data.fetch_prices() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  basket = beta*market + planted_alpha + noise; alpha=0 must NOT manufacture")
print("  significance (despite high beta); a large planted alpha must light up.")
for a in (0.0, 0.0004):
    syn = data.synthetic_reshoring(alpha_daily=a, seed=396)
    s = st.summarize(syn, "mkt")
    print(f"  planted alpha={a*252*100:>+5.0f}%/yr: excess={s['excess_ann']*100:>6.2f}%/yr  "
          f"alpha={s['alpha_ann']*100:>6.2f}%/yr  beta={s['beta']:.2f}  "
          f"hac_t(alpha)={s['t_alpha_hac']:>5.2f}")
