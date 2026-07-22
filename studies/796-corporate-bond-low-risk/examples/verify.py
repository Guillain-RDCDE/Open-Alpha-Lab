"""Reproducible headline run for Study 796 — Corporate-Bond-Low-Risk (BAB in bonds).

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

from bond_low_risk import data, strategy as st  # noqa: E402

print("# Corporate-Bond-Low-Risk — do low-vol bond ETFs earn a higher risk-adjusted return?")

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

    print("\n# THE HEADLINE — vol-scaled low-minus-high (BAB), 1y trailing-vol sort, hold 1m")
    bk = st.bab_book(prices, cost_bps=0.0)
    s = st.summary(bk["bab_gross"])
    print(f"  n months held      : {s['n']}")
    print(f"  BAB mean (ann)     : {s['mean']*100:+.2f}%/yr")
    print(f"  BAB vol  (ann)     : {s['vol']*100:.2f}%/yr")
    print(f"  Sharpe (gross)     : {s['sharpe']:.2f}")
    print(f"  HAC t (Newey-West) : {s['tstat']:+.2f}   plain t = {s['t_iid']:+.2f}")
    print(f"  hit rate           : {s['hit_rate']*100:.1f}%  "
          f"Wilson [{s['hit_lo']*100:.1f}%, {s['hit_hi']*100:.1f}%]")
    print(f"  max drawdown       : {s['max_dd']*100:.1f}%   worst month {s['worst']*100:+.2f}%")
    print(f"  avg leverage       : low {bk['k_low'].mean():.2f}x / high {bk['k_high'].mean():.2f}x")
    print(f"  avg names / leg    : {bk['n_leg'].mean():.1f} of {bk['n_names'].mean():.1f} ranked")
    print(f"  avg one-way turnover: {bk['turnover'].mean()*100:.1f}% / month")
    print(f"  break-even one-way : {st.break_even_bps(bk):.1f} bps")

    print("\n# The claim's plain form — Sharpe of the LOW-vol vs HIGH-vol basket (unlevered)")
    legs = st.leg_returns(prices)
    sl = st.summary(legs["low"]); sh = st.summary(legs["high"])
    print(f"  low-vol basket : {sl['mean']*100:+.2f}%/yr, vol {sl['vol']*100:.2f}%, "
          f"Sharpe {sl['sharpe']:.2f}, HAC t = {sl['tstat']:+.2f}")
    print(f"  high-vol basket: {sh['mean']*100:+.2f}%/yr, vol {sh['vol']*100:.2f}%, "
          f"Sharpe {sh['sharpe']:.2f}, HAC t = {sh['tstat']:+.2f}")

    print("\n# Vol-rank-shuffle placebo (2,000 perms of the monthly volatility assignment)")
    pl = st.placebo_pvalue(prices, n_perm=2000)
    print(f"  real BAB {pl['real_mean_ann']*100:+.2f}%/yr vs placebo mean "
          f"{pl['placebo_mean_ann']*100:+.2f}%/yr (sd {pl['placebo_sd_ann']*100:.2f}) -> "
          f"p = {pl['p_value']:.4f}")

    print("\n# Robustness — the trailing-vol window (63d / 126d / 252d)")
    for vw, tag in ((63, "63d (~3m)"), (126, "126d (~6m)"), (252, "252d (~1y, headline)")):
        b = st.bab_book(prices, vol_window=vw)
        ss = st.summary(b["bab_gross"])
        print(f"  {tag:<22}: {ss['mean']*100:+.2f}%/yr, Sharpe {ss['sharpe']:.2f}, "
              f"HAC t = {ss['tstat']:+.2f} (n={ss['n']})")

    print("\n# Myth-check — is the low-risk edge just the 2022 duration crash?")
    full = st.summary(bk["bab_gross"])
    ex22 = st.summary(bk["bab_gross"][(bk.index.year != 2022)])
    print(f"  full sample   : {full['mean']*100:+.2f}%/yr, HAC t = {full['tstat']:+.2f} (n={full['n']})")
    print(f"  ex-2022       : {ex22['mean']*100:+.2f}%/yr, HAC t = {ex22['tstat']:+.2f} (n={ex22['n']})")

    print("\n# Subperiods — pre-2016 vs 2016-2026")
    for lo, hi, tag in (("2008-01-01", "2015-12-31", "2008-2015"),
                        ("2016-01-01", "2026-06-30", "2016-2026")):
        sub = bk["bab_gross"][(bk.index >= lo) & (bk.index <= hi)]
        ss = st.summary(sub)
        print(f"  {tag:<12}: {ss['mean']*100:+.2f}%/yr, Sharpe {ss['sharpe']:.2f}, "
              f"HAC t = {ss['tstat']:+.2f} (n={ss['n']})")

    print("\n# THE TIMER — net of one-way costs x NAV, plus financing on the levered notional")
    for cb, fb in ((5.0, 50.0), (10.0, 75.0), (20.0, 100.0)):
        b = st.bab_book(prices, cost_bps=cb, fin_ann_bps=fb)
        ss = st.summary(b["bab_net"])
        print(f"  cost={cb:>4.1f}bps fin={fb:>5.1f}bps/yr: net {ss['mean']*100:+.2f}%/yr, "
              f"Sharpe {ss['sharpe']:.2f}, HAC t = {ss['tstat']:+.2f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the BAB engine must NOT fire on a null panel (lowrisk_strength=0, one Sharpe line)")
print("  and must recover a planted low-risk tilt. Null checked over 20 seeds.")
sc = st.synthetic_control()
for _, row in sc.iterrows():
    print(f"  lowrisk_strength={row['lowrisk_strength']:.2f}: BAB {row['mean_ann']*100:+.2f}%/yr, "
          f"HAC t = {row['tstat']:+.2f} (n={int(row['n'])})")
nr = st.synthetic_null_robustness()
print(f"  null over {nr['n_seeds']} seeds: mean HAC t = {nr['mean_null_t']:+.2f}, "
      f"|t|>2 in {nr['frac_abs_t_gt2']*100:.0f}% of seeds")
