"""Reproducible headline run for Study 795 — Corporate-Bond-Momentum.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached bond-ETF tape under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from bond_momentum import data, strategy as st  # noqa: E402

print("# Corporate-Bond-Momentum — does ranking bond ETFs on trailing return pay?")

if not data.have_real():
    print("(cache miss — fetching the bond-ETF basket once)")
    data.fetch_panel(fetch=True)

prices = data.load_real()
if prices.empty:
    print("!! no real cache — synthetic control only")
else:
    prices = data.drop_partial_last_month(prices)
    fp = data.fingerprint(prices)
    print(f"[data] bond-ETF panel: {prices.shape[0]:,} rows x {prices.shape[1]} names "
          f"{prices.index.min().date()} -> {prices.index.max().date()}  as-of {data.AS_OF}"
          f"  fingerprint={fp}")
    print("  names:", ", ".join(prices.columns))

    mp = st.to_monthly(prices)
    print(f"  month-ends: {len(mp)}")

    print("\n# THE HEADLINE — winners-minus-losers, 6-month formation, hold 1m, monthly rebal")
    bk = st.wml_book(mp, cost_bps=0.0)
    s = st.summary(bk["wml_gross"])
    print(f"  n months held      : {s['n']}")
    print(f"  WML mean (ann)     : {s['mean']*100:+.2f}%/yr")
    print(f"  WML vol  (ann)     : {s['vol']*100:.2f}%/yr")
    print(f"  Sharpe (gross)     : {s['sharpe']:.2f}")
    print(f"  HAC t (Newey-West) : {s['tstat']:+.2f}   plain t = {s['t_iid']:+.2f}")
    print(f"  hit rate           : {s['hit_rate']*100:.1f}%  "
          f"Wilson [{s['hit_lo']*100:.1f}%, {s['hit_hi']*100:.1f}%]")
    print(f"  max drawdown       : {s['max_dd']*100:.1f}%   worst month {s['worst']*100:+.2f}%")
    print(f"  avg names / leg    : {bk['n_leg'].mean():.1f} of {bk['n_names'].mean():.1f} ranked")
    print(f"  avg one-way turnover: {bk['turnover'].mean()*100:.1f}% / month")

    print("\n# Rank-shuffle placebo (2,000 perms of the monthly momentum ranking)")
    pl = st.placebo_pvalue(mp, n_perm=2000)
    print(f"  real WML {pl['real_mean_ann']*100:+.2f}%/yr vs placebo mean "
          f"{pl['placebo_mean_ann']*100:+.2f}%/yr (sd {pl['placebo_sd_ann']*100:.2f}) -> "
          f"p = {pl['p_value']:.4f}")

    print("\n# Subperiods — Jostova-era (<=2013) vs post-publication (2014-2026)")
    for lo, hi, tag in (("2007-01-01", "2013-12-31", "2007-2013 (GFC + Jostova era)"),
                        ("2014-01-01", "2026-06-30", "2014-2026 (post-publication)")):
        sub = bk["wml_gross"][(bk.index >= lo) & (bk.index <= hi)]
        ss = st.summary(sub)
        print(f"  {tag:<30}: {ss['mean']*100:+.2f}%/yr, Sharpe {ss['sharpe']:.2f}, "
              f"HAC t = {ss['tstat']:+.2f} (n={ss['n']})")

    print("\n# Lookback robustness — 6m vs 12m formation, and a 12-1 skip")
    for lb, sk, tag in ((6, 0, "6m, no skip (headline)"), (12, 0, "12m, no skip"),
                        (12, 1, "12m, skip 1m (12-1)"), (3, 0, "3m, no skip")):
        b = st.wml_book(mp, lookback=lb, skip=sk)
        ss = st.summary(b["wml_gross"])
        print(f"  {tag:<24}: {ss['mean']*100:+.2f}%/yr, Sharpe {ss['sharpe']:.2f}, "
              f"HAC t = {ss['tstat']:+.2f} (n={ss['n']})")

    print("\n# THE TIMER — net of one-way costs x NAV, short leg pays borrow")
    for cb, bb in ((5.0, 50.0), (10.0, 50.0), (20.0, 100.0)):
        b = st.wml_book(mp, cost_bps=cb, borrow_ann_bps=bb)
        ss = st.summary(b["wml_net"])
        print(f"  cost={cb:>4.1f}bps borrow={bb:>5.1f}bps/yr: net {ss['mean']*100:+.2f}%/yr, "
              f"Sharpe {ss['sharpe']:.2f}, HAC t = {ss['tstat']:+.2f}")

    print("\n# Long-only credit vs the WML spread (is the 'momentum' just being long credit?)")
    hy = mp["HYG"].pct_change().dropna() if "HYG" in mp.columns else None
    if hy is not None:
        shy = st.summary(hy)
        print(f"  HYG buy&hold: {shy['mean']*100:+.2f}%/yr, Sharpe {shy['sharpe']:.2f}, "
              f"HAC t = {shy['tstat']:+.2f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the WML engine must NOT fire on a null panel (mom_strength=0) and must recover a")
print("  planted cross-sectional momentum. Null checked over 20 seeds.")
sc = st.synthetic_control()
for _, row in sc.iterrows():
    print(f"  mom_strength={row['mom_strength']:.2f}: WML {row['mean_ann']*100:+.2f}%/yr, "
          f"HAC t = {row['tstat']:+.2f} (n={int(row['n'])})")
nr = st.synthetic_null_robustness()
print(f"  null over {nr['n_seeds']} seeds: mean HAC t = {nr['mean_null_t']:+.2f}, "
      f"|t|>2 in {nr['frac_abs_t_gt2']*100:.0f}% of seeds")
