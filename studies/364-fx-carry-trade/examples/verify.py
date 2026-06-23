"""Reproducible headline run for Study 364 — FX Carry Trade.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached FX spot under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fx_carry_trade import data, strategy as st

print("# FX Carry Trade - long high-yield / short low-yield G10, carry proxy + FX spot")
if data.have_real():
    b = data.load_real()
    spot = b["spot"]
    tot = b["total_ret"]
    span = (spot.index.max() - spot.index.min()).days / 365.25
    print(f"spot days      : {len(spot)}  ({spot.index.min().date()} -> "
          f"{spot.index.max().date()}, {span:.1f} years)")
    print(f"months         : {len(tot)}")
    print(f"carry proxy    : annualised short-rate diff vs USD (PROXY), high -> low:")
    for c, v in st.carry_ranks(b["carry"], tot.columns).items():
        print(f"    {c}: {v:+.2f}%")

    w = st.carry_weights(b["carry"], tot.columns, k=3)
    longs = [c for c in w.index if w[c] > 0]
    shorts = [c for c in w.index if w[c] < 0]
    print(f"\n# Basket (k=3): long {longs}  /  short {shorts}  (dollar-neutral)")

    s = st.summarize(tot, b["carry"], k=3)
    print(f"  n months          : {s['n_months']}")
    print(f"  ann mean (gross)  : {s['ann_mean']*100:+.2f}%")
    print(f"  ann mean (net)    : {s['ann_mean_net']*100:+.2f}%")
    print(f"  annualised Sharpe : {s['sharpe']:.2f}  (net {s['sharpe_net']:.2f})")
    print(f"  HAC t (mean)      : {s['hac_t']:.2f}")
    print(f"  skewness          : {s['skew']:.2f}")
    print(f"  worst month       : {s['worst_month']*100:.2f}%")
    print(f"  max drawdown      : {s['max_dd']*100:.1f}%")
    cs = s["crash"]
    print(f"  calm-month mean   : {cs['calm_mean']*12*100:+.1f}%/yr")
    print(f"  risk-off mean     : {cs['off_mean']*12*100:+.1f}%/yr  (worst 10% of months)")
    print(f"  tail share        : {cs['tail_share']:+.2f}x cumulative return given back in the tail")

    g = st.basket_returns(tot, b["carry"], k=3)
    print("\n# The carry-crash months (worst 6)")
    for d, r in g.sort_values().head(6).items():
        print(f"    {d.date()}  {r*100:+.2f}%")

    print("\n# Cost sweep - borrow on the short leg (bps/yr)")
    for r in st.cost_sweep(tot, b["carry"], k=3):
        print(f"    borrow={r['borrow_bps']:>5.0f}bps  net={r['ann_mean_net']*100:+.2f}%/yr"
              f"  Sharpe={r['sharpe_net']:.2f}")

    print("\n# Robustness - basket width k (currencies per leg)")
    for k in (1, 2, 3, 4):
        sk = st.summarize(tot, b["carry"], k=k)
        print(f"    k={k}: ann={sk['ann_mean']*100:+5.2f}%  Sharpe={sk['sharpe']:.2f}  "
              f"HAC t={sk['hac_t']:.2f}  skew={sk['skew']:+.2f}  maxDD={sk['max_dd']*100:.1f}%")
else:
    print("(no _cache/fx_spot.csv — run data.fetch_fx() once to build it)")

print("\n# Synthetic positive control - deterministic, no network")
print("  null must NOT manufacture significance; a planted carry premium must light up;")
print("  the planted crash skew must surface as deep negative skewness.")
configs = [
    ("null  (no carry, no crash)", dict(carry_spread=0.0, carry_edge=0.0, crash_skew=0.0)),
    ("carry (premium, no crash)  ", dict(carry_spread=0.07, carry_edge=0.0, crash_skew=0.0)),
    ("carry + crash skew         ", dict(carry_spread=0.07, carry_edge=0.0, crash_skew=0.06)),
    ("big edge + crash           ", dict(carry_spread=0.07, carry_edge=0.10, crash_skew=0.06)),
]
for name, cfg in configs:
    syn = data.synthetic_fx(n_months=240, n_ccy=8, seed=364, **cfg)
    s = st.summarize(syn["total_ret"], syn["carry"], k=3)
    print(f"  {name}: ann={s['ann_mean']*100:+6.2f}%  Sharpe={s['sharpe']:+5.2f}  "
          f"HAC t={s['hac_t']:+5.2f}  skew={s['skew']:+6.2f}  maxDD={s['max_dd']*100:6.1f}%")
