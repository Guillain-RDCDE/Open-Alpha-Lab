"""Reproducible headline run for Study 387 — Economic-Surprise-Index.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached FRED panel + SPY under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from economic_surprise_index import data, strategy as st

print("# Economic-Surprise-Index — transparent CESI proxy from 6 FRED series + SPY")
if data.have_real():
    f = data.build_real()
    span_years = (f.index.max() - f.index.min()).days / 365.25
    print(f"monthly frame  : {len(f)} months  ({f.index.min().date()} -> "
          f"{f.index.max().date()}, {span_years:.1f} years)")
    print(f"ESI            : mean {f['esi'].mean():+.3f}  std {f['esi'].std():.3f}  "
          f"frac>0 {(f['esi'] > 0).mean():.3f}  (PROXY for Citi CESI)")

    print("\n# Forward SPY returns after a 'beat' month (ESI>0) vs unconditional "
          "(1-month entry lag)")
    print(f"  {'H':>3} {'n':>4} {'cond_mean':>10} {'cond_win':>9} {'base_mean':>10} "
          f"{'base_win':>9} {'Welch_t':>8} {'p_placebo':>10}")
    for h in (1, 3, 6, 12):
        s = st.summarize(f, h)
        print(f"  {str(h)+'m':>3} {s['n']:>4} {s['cond_mean']*100:>9.2f}% "
              f"{s['cond_win']*100:>8.0f}% {s['base_mean']*100:>9.2f}% "
              f"{s['base_win']*100:>8.0f}% {s['t']:>8.2f} {s['p_placebo']:>10.3f}")

    print("\n# Timing backtest — hold SPY when ESI>0 (1-month lag, price-only)")
    for short in (False, True):
        b = st.timing_backtest(f, cost_bps=10.0, allow_short=short)
        tag = "long/short" if short else "long/flat "
        print(f"  {tag}: exposure={b['exposure']:.2f}  turns={b['n_turns']:.0f}  "
              f"net Sharpe={b['net']['sharpe']:.2f}  net ann={b['net']['ann_ret']*100:.1f}%  "
              f"(buy&hold Sharpe={b['buy_hold']['sharpe']:.2f})")

    print("\n# Robustness — shift the 'beat' threshold (12-month horizon)")
    for thr in (-0.5, 0.0, 0.5):
        s = st.summarize(f, 12, thr=thr)
        print(f"  thr={thr:+.1f}: n={s['n']:>3}  cond12={s['cond_mean']*100:>5.1f}%  "
              f"t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
else:
    print("(no _cache — run data.fetch_macro() and data.fetch_spy() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  inference must recover a PLANTED edge and must NOT manufacture significance when")
print("  the true edge is 0 (the ESI carries no forward information).")
for edge in (0.0, 0.04):
    syn = data.synthetic(n_months=360, edge=edge, seed=387)
    s6 = st.summarize(syn, 6)
    print(f"  planted edge={edge:+.2f}: n={s6['n']:>3}  cond6={s6['cond_mean']*100:>6.2f}%  "
          f"base6={s6['base_mean']*100:>5.2f}%  t={s6['t']:>5.2f}  "
          f"p_placebo={s6['p_placebo']:.3f}")
