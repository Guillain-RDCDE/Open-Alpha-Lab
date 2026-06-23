"""Reproducible headline run for Study 365 — the Lottery / MAX effect.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached basket prices under ``_cache/``
if present (the real-tape numbers), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lottery_max_effect import data, strategy as st

ASOF = "2026-05-31"   # pinned as-of (last complete month-end on the cached tape)

print("# Lottery / MAX effect — quintile sort on a fixed S&P-100-style large-cap basket")
if data.have_real():
    prices = data.load_prices()
    prices = prices[prices.index <= ASOF]          # drop the partial 2026-06 bar
    yrs = (prices.index.max() - prices.index.min()).days / 365.25
    print(f"basket prices  : {prices.shape[1]} names, {len(prices)} days "
          f"({prices.index.min().date()} -> {prices.index.max().date()}, {yrs:.1f} years)")
    print(f"fingerprint    : {data.fingerprint(prices)}")

    panel = data.build_panel(prices)
    qret = st.quintile_returns(panel)
    print(f"quintile months: {len(qret)}  ({qret.index.min().date()} -> {qret.index.max().date()})")

    print("\n# MAX quintiles — next-month return, equal-weight (Q1 = LOW max … Q5 = HIGH max)")
    qs = st.quintile_summary(qret)
    print(f"  {'':>3} {'mean/yr':>9} {'vol':>7} {'sharpe':>7}")
    for q in qs.index:
        print(f"  {q:>3} {qs.loc[q,'mean_ann']*100:>8.1f}% {qs.loc[q,'vol_ann']*100:>6.1f}% "
              f"{qs.loc[q,'sharpe']:>7.2f}")

    print("\n# The trade — long LOW-MAX (Q1), short HIGH-MAX (Q5), monthly")
    ls = st.long_short(qret)
    ss = st.spread_stats(ls)
    print(f"  GROSS : mean={ss['mean_ann']*100:+.1f}%/yr  Sharpe={ss['sharpe']:+.2f}  "
          f"HAC t={ss['tstat']:+.2f}  win={ss['win']*100:.0f}%  placebo p={ss['p_placebo']:.3f}")
    lsn = st.long_short(qret, cost_bps=20.0, borrow_ann_bps=50.0, turnover=1.0)
    ssn = st.spread_stats(lsn)
    print(f"  NET   : mean={ssn['mean_ann']*100:+.1f}%/yr  Sharpe={ssn['sharpe']:+.2f}  "
          f"HAC t={ssn['tstat']:+.2f}  win={ssn['win']*100:.0f}%  (20bps/leg + 50bps/yr borrow)")

    print("\n# Robustness — the sign holds across cut granularity")
    for nq in (3, 5, 10):
        q = st.quintile_returns(panel, n_q=nq)
        s = st.spread_stats(st.long_short(q, low="Q1", high=f"Q{nq}"))
        print(f"  n_q={nq:>2}: mean={s['mean_ann']*100:>6.1f}%/yr  t={s['tstat']:>6.2f}  "
              f"win={s['win']*100:.0f}%")

    print("\n# Sub-periods — same direction in both halves")
    for lab, a, b in (("2005-2015", "2005-01-01", "2015-12-31"),
                      ("2016-2026", "2016-01-01", "2026-12-31")):
        s = st.spread_stats(st.long_short(qret.loc[a:b]))
        print(f"  {lab}: n={s['n']:>3}  mean={s['mean_ann']*100:>6.1f}%/yr  t={s['tstat']:>6.2f}")
else:
    print("(no _cache/basket_prices.csv — run data.fetch_basket() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  edge=0 must NOT manufacture significance; a planted lottery penalty must light up.")
for edge in (0.0, 0.002):
    syn = data.synthetic_panel(edge=edge, seed=365)
    q = st.quintile_returns(syn)
    s = st.spread_stats(st.long_short(q))
    print(f"  planted edge={edge:+.3f}: mean={s['mean_ann']*100:>7.2f}%/yr  t={s['tstat']:>6.2f}  "
          f"win={s['win']*100:>3.0f}%  placebo p={s['p_placebo']:.3f}")
