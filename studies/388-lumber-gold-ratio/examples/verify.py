"""Reproducible headline run for Study 388 — Lumber-Gold-Ratio.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached real tape under ``_cache/``
if present (the WOOD/GLD/SPY/TLT/^IRX parquets), and always runs the synthetic null +
positive control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lumber_gold_ratio import data, strategy as st


def _fmt_ci(ci):
    lo, hi = ci
    return f"[{lo*1e4:+.3f}, {hi*1e4:+.3f}] bps/day"


print("# Lumber-Gold-Ratio rotation switch — WOOD (lumber proxy) / GLD / SPY / TLT / ^IRX")
if data.have_real():
    f = data.load_real()
    span_years = (f.index.max() - f.index.min()).days / 365.25
    ratio = st.lumber_gold_ratio(f)
    print(f"real days      : {len(f)}  ({f.index.min().date()} -> {f.index.max().date()}, "
          f"{span_years:.1f} years)")
    print(f"fingerprint    : {data.fingerprint(f)}")
    print(f"lumber proxy   : WOOD timber-equity ETF (LBS=F futures discontinued May 2023)")
    print(f"ratio WOOD/GLD : last={ratio.iloc[-1]:.4f}  min={ratio.min():.4f}  max={ratio.max():.4f}")

    print("\n# The race — lumber/gold rotation switch vs passive yardsticks")
    print("#   (excess-of-cash basis, 1-day execution lag, 5 bps one-way cost on each rotation)")
    rc = st.race(f, lookback=60, enter_z=0.0, cost_bps=5.0, rand_seed=388)
    print(f"  lookback={rc['lookback']}  enter_z={rc['enter_z']}  cost_bps={rc['cost_bps']}  "
          f"n_switches~={rc['n_switches']}")
    hdr = f"  {'book':>16} {'ann_ret':>9} {'sharpe_ex':>10} {'max_dd':>8} {'%stocks':>8} {'t_excess':>9}"
    print(hdr)
    for key, label in (("switch", "lumber/gold sw"), ("buy_hold", "buy&hold SPY"),
                       ("blend_6040", "static 60/40"), ("random", "random-timing")):
        s = rc[key]
        print(f"  {label:>16} {s['ann_return']*100:>8.2f}% {s['sharpe_excess']:>10.2f} "
              f"{s['max_dd']*100:>7.1f}% {s['pct_stocks']*100:>7.1f}% {s['t_excess']:>9.2f}")

    print("\n# Head-to-head — HAC t-stat that the switch's NET excess-of-cash daily return")
    print("#   exceeds the benchmark's (the decisive number), plus block-bootstrap CIs")
    print(f"  t  vs buy&hold SPY : {rc['t_vs_bh']:>6.2f}")
    print(f"  t  vs static 60/40 : {rc['t_vs_6040']:>6.2f}   <-- the decisive number")
    print(f"  t  vs random-time  : {rc['t_vs_random']:>6.2f}")
    print(f"  CI vs 60/40        : {_fmt_ci(rc['ci_vs_6040'])}")
    print(f"  CI vs random       : {_fmt_ci(rc['ci_vs_random'])}")

    print("\n# Robustness — vary the lookback and entry threshold")
    print(f"  {'cfg':>20} {'sharpe_sw':>10} {'t_vs_6040':>10} {'t_vs_rand':>10}")
    for lb, ez in ((40, 0.0), (60, 0.0), (90, 0.0), (60, 0.5), (60, -0.5)):
        r2 = st.race(f, lookback=lb, enter_z=ez, cost_bps=5.0, rand_seed=388)
        cfg = f"lb={lb},z>{ez:+.1f}"
        print(f"  {cfg:>20} {r2['switch']['sharpe_excess']:>10.2f} "
              f"{r2['t_vs_6040']:>10.2f} {r2['t_vs_random']:>10.2f}")
else:
    print("(no _cache/ parquets — run data.load_real(fetch=True) once to build it)")

print("\n# Synthetic null + positive control — deterministic, no network")
print("  pred_r=0 (null) must NOT manufacture significance; pred_r>0 (planted edge) must be recovered.")
print(f"  {'planted pred_r':>16} {'sharpe_sw':>10} {'t_vs_6040':>10} {'t_vs_rand':>10} {'verdict':>14}")
for pr in (0.0, 0.15):
    syn, truth = data.synthetic_daily(n_days=4000, pred_r=pr, seed=388)
    rs = st.race(syn, lookback=60, enter_z=0.0, cost_bps=5.0, rand_seed=388)
    sig = "significant" if (rs['t_vs_6040'] is not None and abs(rs['t_vs_6040']) >= 2.0) else "not signif."
    print(f"  {pr:>+16.2f} {rs['switch']['sharpe_excess']:>10.2f} "
          f"{rs['t_vs_6040']:>10.2f} {rs['t_vs_random']:>10.2f} {sig:>14}")
