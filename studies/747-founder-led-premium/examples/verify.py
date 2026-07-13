"""Reproducible headline run for Study 747 — Founder-Led-Premium.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached monthly closes under
``_cache/`` if present (the real-tape long/short sort), and always runs the synthetic
positive control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from founder_led_premium import data, strategy as st

print("# Founder-Led-Premium — founder basket (long) vs professional-CEO basket (short)")
if data.have_real():
    rets, firms = data.load_real()
    priced_f = [t for t in data.FOUNDER_TICKERS if t in rets.columns]
    priced_p = [t for t in data.PRO_TICKERS if t in rets.columns]
    dropped = [t for t in data.FOUNDER_TICKERS + data.PRO_TICKERS if t not in rets.columns]
    print(f"months               : {len(rets)}  ({rets.index.min().date()} .. {rets.index.max().date()})")
    print(f"founder priced        : {len(priced_f)} of {len(data.FOUNDER_TICKERS)}  {priced_f}")
    print(f"professional priced   : {len(priced_p)} of {len(data.PRO_TICKERS)}  {priced_p}")
    print(f"dropped (no price)    : {dropped}  <- literal survivorship (delisted/reticker)")
    print(f"membership fingerprint: {data.fingerprint(data.FIRMS)}")

    s = st.summarize(rets, data.FOUNDER_TICKERS, data.PRO_TICKERS)
    print("\n# Raw basket performance (total-return proxy: adjusted closes)")
    print(f"  founder basket  : {s['founder_ann']*100:+.1f}% / yr")
    print(f"  professional    : {s['pro_ann']*100:+.1f}% / yr")
    print(f"  long/short (F-P): {s['ls_ann']*100:+.1f}% / yr  |  {s['ls_mean_bps']:+.1f} bps/mo  "
          f"|  Sharpe {s['ls_sharpe']:.2f}")
    print(f"  L/S raw-mean Newey-West HAC t = {s['ls_hac']['t']:+.2f} (lags {s['ls_hac']['lags']})")

    c = s["ls_capm"]
    mkt_mean_bps = rets["SPY"].reindex(st.long_short(rets, data.FOUNDER_TICKERS,
                                                     data.PRO_TICKERS).index).mean() * 1e4
    beta_part = c["beta"] * mkt_mean_bps
    print("\n# The abnormal return — CAPM (market-model) alpha with a Newey-West HAC t")
    print(f"  L/S alpha = {c['alpha_bps']:+.1f} bps/mo   beta = {c['beta']:.2f}   "
          f"HAC t(alpha) = {c['t_alpha']:+.2f}   R2 = {c['r2']:.2f}")
    print(f"  decomposition: raw spread {s['ls_mean_bps']:+.1f} bps = alpha {c['alpha_bps']:+.1f} "
          f"+ beta*mkt {beta_part:+.1f}  ->  beta explains {100*beta_part/s['ls_mean_bps']:.0f}%")
    fc, pc = s["founder_capm"], s["pro_capm"]
    print(f"  founder long-only alpha = {fc['alpha_bps']:+.1f} bps/mo  beta {fc['beta']:.2f}  "
          f"HAC t = {fc['t_alpha']:+.2f}")
    print(f"  professional  alpha     = {pc['alpha_bps']:+.1f} bps/mo  beta {pc['beta']:.2f}  "
          f"HAC t = {pc['t_alpha']:+.2f}")

    print("\n# Leave-one-out (jackknife) — drop each founder name, recompute L/S alpha")
    jk = st.jackknife_alpha(rets, data.FOUNDER_TICKERS, data.PRO_TICKERS)
    for _, row in jk.iterrows():
        print(f"  drop {row['dropped']:>5}: alpha {row['alpha_bps']:+7.1f} bps  t {row['t_alpha']:+.2f}")

    print("\n# Placebo — random long/short labels on the SAME names (isolates the founder tag)")
    pool = data.FOUNDER_TICKERS + data.PRO_TICKERS
    null = st.placebo_alpha_dist(rets, pool, k_long=len(priced_f), k_short=len(priced_p),
                                 n_draws=4000)
    p = st.placebo_pvalue(c["alpha_bps"], null)
    print(f"  observed founder alpha {c['alpha_bps']:+.1f} bps   null mean {null.mean():+.1f} bps   "
          f"two-sided p = {p:.3f}   (random label beats it {(null>=c['alpha_bps']).mean()*100:.0f}% of the time)")

    print("\n# Costs + short borrow — one-way turnover both legs + borrow on the short")
    nc = st.net_of_costs(st.long_short(rets, data.FOUNDER_TICKERS, data.PRO_TICKERS)["ls"].mean(),
                         len(priced_f), len(priced_p))
    print(f"  gross {nc['gross_bps']:+.1f} bps/mo  ->  net {nc['net_bps']:+.1f} bps/mo  "
          f"(drag {nc['monthly_drag_bps']:.1f} bps) — costs are NOT the binding constraint")
else:
    print("(no _cache — run data.fetch_prices() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  engine must recover a PLANTED founder alpha and must NOT manufacture a")
print("  significant long/short alpha when the true edge is 0.")
for edge in (0.0, 200.0):
    syn = data.synthetic_baskets(alpha_bps=edge, seed=747)
    cc = st.capm_alpha(syn["ls"], syn["mkt"])
    print(f"  planted {edge:+6.0f} bps/mo: L/S alpha {cc['alpha_bps']:+7.1f} bps  "
          f"HAC t {cc['t_alpha']:+.2f}  beta {cc['beta']:.2f}")
