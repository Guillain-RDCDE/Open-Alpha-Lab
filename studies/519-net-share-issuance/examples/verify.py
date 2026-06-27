"""Reproducible headline run for Study 519 — Net-Share-Issuance.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic given the cache; uses the cached year-end prices &
split-adjusted shares under ``_cache/`` (the real-tape numbers), and always runs the synthetic
control with no network.

    python examples/verify.py

Build the cache once (network) with:

    python -c "import sys; sys.path.insert(0,'.'); from net_share_issuance import data; data.fetch_real()"
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from net_share_issuance import data, strategy as st

print("# Net-Share-Issuance - annual cross-sectional sort: long LOW issuance (buybacks),")
print("#                       short HIGH issuance (dilution); hold next year (1-yr lag).")
print(f"# basket: {len(data.BASKET)} large-cap survivors; signal = YoY change in SPLIT-ADJUSTED shares")

if data.have_real():
    px, sh = data.load_real()
    px, sh = data.drop_partial_last_year(px, sh)
    fp = data.fingerprint(px, sh)
    print(f"panel      : year-ends {px.index.min().year} -> {px.index.max().year} "
          f"({len(px)} year-ends, {sh.shape[1]} names) | fingerprint {fp}")

    ls = st.long_short_series(px, sh, frac=0.3)
    print(f"\n# Long-short panel (formation year-end -> next-year return), {len(ls)} usable years")
    print(f"  {'year':>6} {'long':>8} {'short':>8} {'L-S':>8}")
    for t, row in ls.iterrows():
        print(f"  {t.year:>6} {row['long_ret']*100:>7.1f}% {row['short_ret']*100:>7.1f}% "
              f"{row['ls_ret']*100:>7.1f}%")

    s = st.summarize(px, sh, frac=0.3)
    print("\n# Headline long-short (frac = 0.3 tails)")
    print(f"  n_years      : {s['n_years']}")
    print(f"  long  mean   : {s['long_mean']*100:>6.2f}% / yr")
    print(f"  short mean   : {s['short_mean']*100:>6.2f}% / yr")
    print(f"  L-S   mean   : {s['ls_mean']*100:>6.2f}% / yr   (median {s['ls_median']*100:.2f}%)")
    print(f"  win-rate     : {s['win']*100:>4.0f}%   annual Sharpe {s['sharpe']:>5.2f}")
    print(f"  one-sample t : {s['t']:>5.2f}   (need |t| >= 2)")
    print(f"  placebo p    : {s['p_placebo']:.3f}   (label-shuffle: P[shuffled L-S >= real])")

    print("\n# Costs - one-way bps x both legs x turnover + short borrow")
    c = st.net_of_costs(px, sh, frac=0.3, cost_bps=10.0, borrow_ann_bps=50.0)
    print(f"  gross L-S    : {c['gross_mean']*100:>6.2f}% / yr")
    print(f"  net @ {c['cost_bps']:.0f}bps + {c['borrow_bps']:.0f}bps borrow : "
          f"{c['net_mean']*100:>6.2f}% / yr  (net t {c['net_t']:.2f})")

    print("\n# Robustness - quantile width sweep")
    print(f"  {'frac':>6} {'n':>3} {'L-S mean':>9} {'t':>7} {'Sharpe':>7}")
    for r in st.width_sweep(px, sh, fracs=(0.2, 0.3, 0.4)):
        print(f"  {r['frac']:>6.1f} {r['n_years']:>3} {r['ls_mean']*100:>8.2f}% "
              f"{r['t']:>7.2f} {r['sharpe']:>7.2f}")

    print("\n# Robustness - split-half by year")
    x = ls["ls_ret"].to_numpy()
    half = len(x) // 2
    for lab, seg in (("first half ", x[:half + 1]), ("second half", x[half + 1:])):
        print(f"  {lab}: L-S mean {seg.mean()*100:>6.2f}%  t {st.one_sample_t(seg):>5.2f}  n {len(seg)}")
else:
    print("(no _cache/prices.csv|shares.csv - run data.fetch_real() once to build them)")

print("\n# Synthetic positive control - deterministic, no network, averaged over 25 seeds")
print("  the engine must recover a PLANTED issuance edge and must NOT manufacture a t>=2")
print("  from ~40 names over 9 years when the true edge is zero.")
print(f"  {'planted edge':>13} {'mean t (25 seeds)':>18} {'L-S mean':>10}")
for r in st.synthetic_control(data.synthetic_panel, edges=(0.0, 0.10), n_seeds=25):
    print(f"  {r['edge']:>+13.2f} {r['mean_t']:>18.2f} {r['ls_mean']*100:>9.1f}%")
