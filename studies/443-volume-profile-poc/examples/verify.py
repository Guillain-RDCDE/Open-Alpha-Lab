"""Reproducible headline run for Study 443 — Volume Profile POC.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached 5m event tables under ``_cache/``
if present (the real-tape numbers) and always runs the synthetic control with no network.

The pinned real run DROPS the in-progress current session (partial bar) — the as-of is the
last *complete* trading day in the cache.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from volume_profile_poc import data, strategy as st

AS_OF = pd.Timestamp("2026-06-23")          # last COMPLETE session in the pinned cache
NDRAWS = 5000


def _real():
    ev = data.load_real()
    return ev[ev["date"] <= AS_OF].reset_index(drop=True)


print("# Volume Profile POC — does price return to the point of control? (yfinance 5m)")
if data.have_real():
    ev = _real()
    span_days = (ev["date"].max() - ev["date"].min()).days
    print(f"events         : {len(ev)}  ({ev['ticker'].nunique()} names, "
          f"{ev['date'].min().date()} -> {ev['date'].max().date()}, ~{span_days} cal-days)")
    print(f"fingerprint    : {data.fingerprint(ev)}")
    print("  ** SHORT SPAN: ~60 calendar days of 5m bars (Yahoo's cap). "
          "Capacity caveat: intraday levels die to the spread. **")

    rs = st.reversion_stats(ev)
    print("\n# Reversion test — POC touch-rate vs a distance-matched RANDOM control level")
    print(f"  events kept (gap>=5bps): {rs['n']}")
    print(f"  POC  touch-rate : {rs['poc_rate']*100:5.1f}%  "
          f"(95% Wilson [{rs['poc_ci'][0]*100:.1f}, {rs['poc_ci'][1]*100:.1f}])")
    print(f"  ctrl touch-rate : {rs['ctrl_rate']*100:5.1f}%  "
          f"(95% Wilson [{rs['ctrl_ci'][0]*100:.1f}, {rs['ctrl_ci'][1]*100:.1f}])")
    print(f"  edge (POC-ctrl) : {rs['edge']*100:+5.1f} pp   "
          f"paired t={rs['t_paired']:+.2f}  HAC t={rs['t_hac']:+.2f}")

    pl = st.placebo_pvalue(ev, n_draws=NDRAWS)
    print(f"\n# Shuffle-the-POC placebo: p(random pairing >= true POC) = {pl['p_value']:.3f}  "
          f"(placebo mean touch-rate {pl['placebo_mean']*100:.1f}%)")

    print("\n# Fade the open toward the POC — gross vs net (1 bp/leg one-way)")
    fd = st.fade_trade(ev, cost_bps=1.0)
    print(f"  trades={fd['n']}  target-hit={fd['hit_rate']*100:.1f}%  "
          f"gross={fd['gross_mean']*100:+.3f}%  net={fd['net_mean']*100:+.3f}%  "
          f"gross t={fd['gross_t']:+.2f}  net t={fd['net_t']:+.2f}")
    print(f"  break-even one-way cost: {fd['breakeven_bps']:+.1f} bps "
          "(negative = no edge to pay any spread from)")

    print("\n# Robustness — touch tolerance sweep (a touch within tol x price counts)")
    print(f"  {'tol(bps)':>8} {'POC%':>6} {'ctrl%':>6} {'edge_pp':>8} {'HAC t':>7}")
    for tol_bps in (0.0, 5.0, 10.0, 20.0):
        r = st.reversion_stats(ev, tol=tol_bps / 1e4)
        print(f"  {tol_bps:>8.0f} {r['poc_rate']*100:>6.1f} {r['ctrl_rate']*100:>6.1f} "
              f"{r['edge']*100:>+8.1f} {r['t_hac']:>+7.2f}")

    print("\n# Per-ticker reversion edge (POC - control, pp)")
    for tk in sorted(ev["ticker"].unique()):
        sub = ev[ev["ticker"] == tk]
        r = st.reversion_stats(sub)
        print(f"  {tk:>5}: n={r['n']:>3}  POC={r['poc_rate']*100:5.1f}%  "
              f"ctrl={r['ctrl_rate']*100:5.1f}%  edge={r['edge']*100:+5.1f}pp")
else:
    print("(no _cache/vp_*_5m.csv — run data.fetch_panel(tk) once per ticker to build the cache)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the detector must NOT manufacture an edge at pull=0 and MUST light up at pull>0.")
for pull in (0.0, 0.7):
    sev, truth = data.synthetic_panel(pull=pull, n_sessions=60, n_names=6)
    r = st.reversion_stats(sev)
    p = st.placebo_pvalue(sev, n_draws=3000)
    print(f"  planted pull={pull:+.1f}: n={r['n']:>3}  POC={r['poc_rate']*100:5.1f}%  "
          f"ctrl={r['ctrl_rate']*100:5.1f}%  edge={r['edge']*100:+5.1f}pp  "
          f"HAC t={r['t_hac']:+5.2f}  placebo p={p['p_value']:.3f}")
