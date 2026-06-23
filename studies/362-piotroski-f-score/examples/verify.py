"""Reproducible headline run for Study 362 — Piotroski F-Score.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached EDGAR fundamentals + yfinance
yearly returns under ``_cache/`` if present (the real-tape numbers), and always runs the
synthetic positive control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from piotroski_f_score import data, strategy as st

# The in-progress calendar year is dropped before stats (as-of 2026-06-22).
AS_OF_LAST_RETURN_YEAR = 2025
LAST_FISCAL_YEAR = 2024

print("# Piotroski F-Score — 9-point fundamental health from EDGAR companyfacts + yfinance")

if data.have_real():
    P = data.load_real()
    ret = P["returns"]
    ret = ret[ret.index <= AS_OF_LAST_RETURN_YEAR]
    fs = st.compute_fscore(P)
    fs = fs[fs.index <= LAST_FISCAL_YEAR]

    vals = fs.to_numpy().flatten()
    vals = vals[~np.isnan(vals)]
    print(f"panel firms     : {fs.shape[1]}  (survived the all-9-concepts intersection)")
    print(f"fiscal years    : {int(fs.index.min())} -> {int(fs.index.max())}  "
          f"(returns {int(ret.index.min())} -> {int(ret.index.max())})")
    print(f"firm-years      : {len(vals)}   mean F-score = {np.nanmean(vals):.2f} / 9")
    print(f"fingerprints    : fscore {data.fingerprint(fs)}  returns {data.fingerprint(ret)}")

    ls = st.long_short(fs, ret)
    print("\n# High-minus-low F-score portfolio (long >=7, short <=4; fiscal y -> ret y+1)")
    print(f"  usable years  : {len(ls)}")
    print(f"  high leg      : {ls['high'].mean()*100:>6.2f}% / yr")
    print(f"  low leg       : {ls['low'].mean()*100:>6.2f}% / yr")
    print(f"  market (EW)   : {ls['market'].mean()*100:>6.2f}% / yr")
    print(f"  high - market : {(ls['high']-ls['market']).mean()*100:>6.2f}% / yr  "
          f"(HAC t = {st.hac_tstat(ls['high']-ls['market'])['tstat']:.2f})")

    s = st.summarize(fs, ret)
    print(f"\n  spread mean   : {s['spread_mean']*100:>6.2f}% / yr")
    print(f"  spread HAC t  : {s['spread_t']:>6.2f}")
    print(f"  placebo p     : {s['placebo_p']:>6.3f}")
    print(f"  spread hit    : {s['spread_hit']*100:>5.0f}%  ({int(round(s['spread_hit']*len(ls)))}/{len(ls)} years)")
    print(f"  net mean      : {s['net_mean']*100:>6.2f}% / yr   net HAC t = {s['net_t']:.2f}")

    print("\n# Monotonicity — mean next-year return by F-score bucket")
    bk = st.score_buckets(fs, ret)
    for label, row in bk.iterrows():
        print(f"  {label:>4} : {row['mean_ret']*100:>6.2f}%   (n={int(row['n'])})")

    print("\n# Robustness — shift the high/low thresholds")
    for hi, lo in [(8, 3), (7, 3), (7, 4), (6, 4), (6, 5)]:
        ss = st.summarize(fs, ret, hi=hi, lo=lo)
        print(f"  hi>={hi} lo<={lo}: years={ss['n_years']:>2}  spread={ss['spread_mean']*100:>6.2f}%  "
              f"t={ss['spread_t']:>5.2f}  p={ss['placebo_p']:.3f}")
else:
    print("(no _cache — run data.fetch_fundamentals() and data.fetch_returns() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the engine must recover a PLANTED F-score->return edge and must NOT manufacture")
print("  significance when the true edge is zero.")
for edge in (0.0, 0.04, 0.10):
    syn = data.synthetic_panel(planted_edge=edge, seed=362)
    s = st.summarize(syn["fscore"], syn["returns"])
    print(f"  planted edge={edge:+.2f}: years={s['n_years']:>2}  high={s['high_mean']*100:>5.1f}%  "
          f"low={s['low_mean']*100:>5.1f}%  spread={s['spread_mean']*100:>6.2f}%  "
          f"t={s['spread_t']:>6.2f}  p={s['placebo_p']:.3f}")
