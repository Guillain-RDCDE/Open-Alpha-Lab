"""Reproducible headline run for Study 381 — TIPS-Breakeven.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached ETF prices under ``_cache/`` if
present (the real-tape numbers), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tips_breakeven import data, strategy as st

print("# TIPS-Breakeven — breakeven proxy log(TIP/IEF) vs forward TIP / SPY / GLD returns")
if data.have_real():
    prices = data.load_prices()
    span = (prices.index.max() - prices.index.min()).days / 365.25
    m = data.load_real()
    be = st.breakeven_series(m)
    print(f"daily tape    : {len(prices)} days  ({prices.index.min().date()} -> "
          f"{prices.index.max().date()}, {span:.1f} years)")
    print(f"signal cadence: {len(m)} month-ends; breakeven proxy = log(TIP/IEF) (PROXY for FRED T10YIE)")

    sigs = {"level-z": st.level_z(be), "3m-change": st.change_3m(be)}
    print("\n# HAC (Newey-West) predictive-regression scan: signal x asset x horizon")
    print(f"  {'signal':9s} {'asset':5s} {'h':>3} {'slope':>9} {'NW_t':>7} {'placebo_p':>10} {'n':>4}")
    n_cross = 0
    for asset in ("TIP", "SPY", "GLD"):
        for sname, sig in sigs.items():
            for h in (3, 6, 12):
                fwd = st.forward_return(m[asset], h)
                reg = st.hac_regression(sig, fwd, lags=h)
                p = st.placebo_pvalue(sig, fwd, lags=h, n_draws=5_000)
                flag = "  <-- |t|>=2" if abs(reg["t"]) >= 2 else ""
                if abs(reg["t"]) >= 2:
                    n_cross += 1
                print(f"  {sname:9s} {asset:5s} {h:>3d} {reg['slope']:>+9.4f} "
                      f"{reg['t']:>+7.2f} {p:>10.3f} {reg['n']:>4d}{flag}")
    print(f"\n  -> {n_cross} of 18 tests cross |t|>=2 (expect ~0.9 by chance at the 5% level)")

    print("\n# The ONE |t|>=2 on a genuinely independent asset, and the self-prediction artifact")
    fwd_be = be.shift(-6) - be          # breakeven level predicting its OWN forward change
    reg = st.hac_regression(st.level_z(be), fwd_be, lags=6)
    print(f"  breakeven level -> its own forward 6m change: slope={reg['slope']:+.4f} "
          f"t={reg['t']:+.2f}  (the 'TIP edge' is just proxy mean reversion)")

    print("\n# Directional win-rate vs the unconditional base rate")
    for asset, sname, h in (("TIP", "level-z", 12), ("SPY", "level-z", 12), ("GLD", "3m-change", 3)):
        w = st.directional_winrate(sigs[sname], st.forward_return(m[asset], h))
        print(f"  {asset:4s} {sname:9s} {h:>2d}m: cond_win={w['cond_win']*100:>3.0f}%  "
              f"base_win={w['base_win']*100:>3.0f}%  n={w['n']}")

    print("\n# Tradable sign-timer (1-month execution lag, 10 bps one-way x turnover)")
    g = st.sign_timer(m["GLD"], st.change_3m(be), contrarian=True, cost_bps=10.0)
    print(f"  GLD contrarian (3m-change): gross={g['gross_ann']*100:+.1f}%/yr  "
          f"net={g['net_ann']*100:+.1f}%/yr  net_sharpe={g['net_sharpe']:+.2f}  "
          f"| buy&hold={g['bh_ann']*100:.1f}%/yr sharpe={g['bh_sharpe']:.2f}")
    t = st.sign_timer(m["TIP"], st.level_z(be), contrarian=True, cost_bps=10.0)
    print(f"  TIP contrarian (level-z) : gross={t['gross_ann']*100:+.1f}%/yr  "
          f"net={t['net_ann']*100:+.1f}%/yr  net_sharpe={t['net_sharpe']:+.2f}  "
          f"| buy&hold={t['bh_ann']*100:.1f}%/yr sharpe={t['bh_sharpe']:.2f}")
else:
    print("(no _cache/proxy_prices.csv — run data.fetch_assets() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  planted slope of next-month asset return on the breakeven z-score:")
print("  edge=0 must NOT manufacture significance; a large edge must light up |t|>=2.")
for edge in (0.0, 0.03):
    syn = data.synthetic_macro(n_months=240, edge=edge, seed=381)
    sbe = st.breakeven_series(syn)
    sz = st.level_z(sbe)
    fwd1 = st.forward_return(syn["ASSET"], 1)
    reg = st.hac_regression(sz, fwd1, lags=1)
    p = st.placebo_pvalue(sz, fwd1, lags=1, n_draws=5_000)
    print(f"  planted edge={edge:+.2f}: slope={reg['slope']:+.4f}  t={reg['t']:+.2f}  "
          f"placebo_p={p:.3f}  n={reg['n']}")
