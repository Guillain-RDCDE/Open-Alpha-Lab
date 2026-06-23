"""Reproducible headline run for Study 374 — Vol-of-Vol (VVIX vs VIX).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached tape under ``_cache/`` if
present (the real-tape numbers), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vol_of_vol import data, strategy as st

print("# Vol-of-Vol — does VVIX time equity risk better than the VIX itself?")
if data.have_real():
    F = data.load_real()
    span = (F.index.max() - F.index.min()).days / 365.25
    print(f"tape           : {len(F)} days  ({F.index.min().date()} -> {F.index.max().date()}, "
          f"{span:.1f} years)  cols: vvix, vix, spy")
    state = st.vvix_state(F)
    print(f"high-VVIX state: {int(state.sum())} days "
          f"({state.mean()*100:.1f}% — VVIX > trailing 252d 80th pctile)")

    print("\n# Forward SPY after a high-VVIX day vs unconditional (1-day entry lag)")
    print(f"  {'H':>4} {'n':>5} {'cond':>8} {'cwin':>6} {'base':>8} {'bwin':>6} "
          f"{'Welch_t':>8} {'p_plac':>7} {'cond_dd':>8} {'base_dd':>8}")
    for m in (1, 3):
        s = st.summarize(F, state, m)
        print(f"  {str(m)+'m':>4} {s['n']:>5} {s['cond_mean']*100:>7.2f}% "
              f"{s['cond_win']*100:>5.0f}% {s['base_mean']*100:>7.2f}% "
              f"{s['base_win']*100:>5.0f}% {s['t']:>8.2f} {s['p_placebo']:>7.3f} "
              f"{s['cond_dd']*100:>7.2f}% {s['base_dd']*100:>7.2f}%")

    print("\n# THE decisive test — incremental over VIX (Newey-West / HAC regression)")
    print("  forward return ~ a + b*z(VIX) + c*z(VVIX); does VVIX add over VIX?")
    for m in (1, 3):
        r = st.incremental_regression(F, m)
        print(f"  {m}m: corr(VIX,VVIX)={r['corr_vix_vvix']:.2f}")
        print(f"     VIX-only   : b={r['vix_only_beta']*100:>6.2f}%  HAC t={r['vix_only_t']:>5.2f}")
        print(f"     VVIX-only  : b={r['vvix_only_beta']*100:>6.2f}%  HAC t={r['vvix_only_t']:>5.2f}")
        print(f"     VIX + VVIX : VIX  b={r['vix_beta']*100:>6.2f}%  t={r['vix_t']:>5.2f} | "
              f"VVIX b={r['vvix_beta']*100:>6.2f}%  t={r['vvix_t']:>5.2f}  <- decisive")

    print("\n# Robustness — shift the VVIX state quantile (1-month horizon)")
    for q in (0.70, 0.80, 0.90):
        s = st.summarize(F, st.vvix_state(F, q=q), 1)
        print(f"  q={q:.2f}: n={s['n']:>4}  cond={s['cond_mean']*100:>5.2f}%  "
              f"base={s['base_mean']*100:>5.2f}%  Welch t={s['t']:>5.2f}  "
              f"cond_dd={s['cond_dd']*100:>6.2f}%")

    print("\n# Tradability — long/flat timing rule (flat when high-VVIX), 1-day lag, 2 bps")
    bt = st.timing_backtest(F, state)
    print(f"  strategy net: ann={bt['net_ann']*100:>5.1f}%  vol={bt['net_vol']*100:>4.1f}%  "
          f"Sharpe={bt['net_sharpe']:.2f}  ({bt['n_trades']} trades, {bt['pct_flat']*100:.0f}% flat)")
    print(f"  buy & hold  : ann={bt['bh_ann']*100:>5.1f}%  vol={bt['bh_vol']*100:>4.1f}%  "
          f"Sharpe={bt['bh_sharpe']:.2f}")
else:
    print("(no _cache/vol_tape.csv — run data.fetch_tape() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the engine must isolate INCREMENTAL vol-of-vol: with edge=0 the VVIX coefficient")
print("  (over VIX) must NOT clear |t|>=2; a large planted edge must light it up.")
for edge in (0.0, 0.04, 0.10):
    syn = data.synthetic_tape(edge=edge, seed=374)
    r = st.incremental_regression(syn, 1)
    print(f"  planted edge={edge:+.2f}: corr={r['corr_vix_vvix']:.2f}  "
          f"VVIX-only t={r['vvix_only_t']:>6.2f}  ->  VVIX|VIX t={r['vvix_t']:>6.2f}  "
          f"(b={r['vvix_beta']*100:>6.2f}%)")
