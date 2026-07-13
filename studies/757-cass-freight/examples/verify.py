"""Reproducible headline run for Study 757 — Cass-Freight.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY + IYT month-end closes
under ``_cache/`` if present (the real-tape numbers) with the hardcoded Cass Freight PROXY,
and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from cass_freight import data, strategy as st  # noqa: E402

print("# Cass-Freight — hardcoded Cass shipments PROXY vs real SPY and IYT (yfinance)")
if data.have_real():
    F = data.build_real()
    Fi = F.dropna(subset=["iyt"])
    span = (F.index.max() - F.index.min()).days / 365.25
    yoy = st.freight_yoy(F).dropna()
    print(f"monthly frame  : {len(F)} months  ({F.index.min().date()} -> "
          f"{F.index.max().date()}, {span:.1f} years); IYT from {Fi.index.min().date()} "
          f"({len(Fi)} months)")
    print(f"Cass YoY       : mean {yoy.mean():+.3f}  std {yoy.std():.3f}  "
          f"frac expanding {(yoy > 0).mean():.2f}  (PROXY; execution lag "
          f"{data.DEFAULT_LAG}m = publication + trade)")

    for col in ("spy", "iyt"):
        Fx = F if col == "spy" else Fi
        print(f"\n# Forward {col.upper()} returns after 'freight expanding' (YoY>0) vs "
              f"unconditional (lag {data.DEFAULT_LAG}m)")
        print(f"  {'H':>3} {'n':>4} {'cond_mean':>10} {'cond_win':>9} {'base_mean':>10} "
              f"{'base_win':>9} {'Welch_t':>8} {'p_placebo':>10}")
        for h in (1, 3, 6, 12):
            s = st.summarize(Fx, col, h)
            print(f"  {str(h)+'m':>3} {s['n']:>4} {s['cond_mean']*100:>9.2f}% "
                  f"{s['cond_win']*100:>8.0f}% {s['base_mean']*100:>9.2f}% "
                  f"{s['base_win']*100:>8.0f}% {s['t']:>8.2f} {s['p_placebo']:>10.3f}")

    print("\n# Lead-lag cross-correlation: dFreight(t) vs equity return(t+k)")
    print("  k>0 => freight LEADS stocks (leading indicator); k<0 => stocks lead freight")
    for col in ("spy", "iyt"):
        Fx = F if col == "spy" else Fi
        ll = st.lead_lag_corr(Fx, col, max_lag=12)
        neg = np.nanmean([abs(c) for k, c in zip(ll["lags"], ll["corr"]) if k < 0])
        pos = np.nanmean([abs(c) for k, c in zip(ll["lags"], ll["corr"]) if k > 0])
        print(f"  {col.upper():>3}: peak at lag={ll['peak_lag']:+d} (corr={ll['peak_corr']:+.3f})"
              f"  | mean|corr| stocks-lead={neg:.3f}  freight-lead={pos:.3f}")

    print("\n# Timing overlay - hold when freight expanding (lag 2m, 10bps/turn, price-only)")
    for col in ("spy", "iyt"):
        Fx = F if col == "spy" else Fi
        for short in (False, True):
            b = st.timing_backtest(Fx, col, cost_bps=10.0, allow_short=short)
            tag = "long/short" if short else "long/flat "
            print(f"  {col.upper()} {tag}: exposure={b['exposure']:.2f}  turns={b['n_turns']:.0f}"
                  f"  net Sharpe={b['net']['sharpe']:.2f}  net ann={b['net']['ann_ret']*100:.1f}%"
                  f"  (buy&hold Sharpe={b['buy_hold']['sharpe']:.2f})")

    print("\n# Robustness - shift the 'expanding' threshold (SPY, 12-month horizon)")
    for thr in (-0.02, 0.0, 0.02):
        s = st.summarize(F, "spy", 12, thr=thr)
        print(f"  thr={thr:+.2f}: n={s['n']:>3}  cond12={s['cond_mean']*100:>5.1f}%  "
              f"t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
else:
    print("(no _cache — run data.fetch_spy() and data.fetch_iyt() once to build it)")

print("\n# Synthetic positive control - deterministic, no network")
print("  inference must recover a PLANTED edge and must NOT manufacture significance when")
print("  the true edge is 0 (freight carries no forward information).")
for edge in (0.0, 0.05):
    syn = data.synthetic(n_months=312, edge=edge, seed=757)
    s6 = st.summarize(syn, "spy", 6)
    print(f"  planted edge={edge:+.2f}: n={s6['n']:>3}  cond6={s6['cond_mean']*100:>6.2f}%  "
          f"base6={s6['base_mean']*100:>5.2f}%  t={s6['t']:>5.2f}  "
          f"p_placebo={s6['p_placebo']:.3f}")
