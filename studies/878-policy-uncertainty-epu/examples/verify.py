"""Reproducible headline run for Study 878 — Economic Policy Uncertainty (EPU).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached real tape under ``_cache/``
(fetching once on a cache miss), and always runs the synthetic control with no network.

    python examples/verify.py

Data honesty: the intended signal is the Baker-Bloom-Davis newspaper EPU index. When that
feed is unreachable, ``data.load_uncertainty`` falls back to a documented VIX proxy
(``source == "vix_proxy"``) — a market-based stand-in, never the newspaper index.
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import fingerprint  # noqa: E402

from epu import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Economic Policy Uncertainty — does high EPU predict forward vol AND forward return?")

if not data.have_real_market():
    print("(cache miss — fetching SPY/VIX + best-effort EPU once)")
    data.fetch()

frame, source = data.build_real()
tag = "REAL Baker-Bloom-Davis EPU" if source == "epu" else \
    "VIX PROXY (real newspaper EPU feed unreachable in-environment — labelled proxy)"
print(f"[data] signal source = {source}  ({tag})")
print(f"[data] n={len(frame)} months  {frame.index.min().date()} -> {frame.index.max().date()}  "
      f"as-of {data.AS_OF}  fingerprint={fingerprint(frame)}")
print(f"       uncertainty level range {frame['unc'].min():.1f}..{frame['unc'].max():.1f} "
      f"(mean {frame['unc'].mean():.1f})")

print("\n# LEG 1 — FORWARD REALIZED VOL on uncertainty (the 'vol story')")
for on in ("level", "change"):
    tb = st.regress_forward(frame, "rv", horizons=(1, 3, 6, 12), on=on)
    print(f"  on {on}:")
    for h, r in tb.iterrows():
        print(f"    h={h:>2}: slope {r['slope']:+.5f}  HAC t = {r['t']:+.2f}  "
              f"R2 = {r['r2']:.3f}  beta_sd = {r['beta_sd']:+.2f}  n={int(r['n'])}")

print("\n# LEG 2 — FORWARD SPY RETURN on uncertainty (the 'risk-premium story')")
for on in ("level", "change"):
    tb = st.regress_forward(frame, "ret", horizons=(1, 3, 6, 12), on=on)
    print(f"  on {on}:")
    for h, r in tb.iterrows():
        print(f"    h={h:>2}: slope {r['slope']:+.5f}  HAC t = {r['t']:+.2f}  "
              f"R2 = {r['r2']:.3f}  beta_sd = {r['beta_sd']:+.2f}  n={int(r['n'])}")

print("\n# ROBUSTNESS — two eras (split 2009-01-01), horizon 3m")
for lo, hi, lbl in [("1900", "2009-01-01", "1993-2008"), ("2009-01-01", "2100", "2009-2026")]:
    sub = frame[(frame.index >= lo) & (frame.index < hi)]
    rr = st.predictive_reg(sub["unc"].reindex(sub.index),
                           st.forward_return(sub, 3).reindex(sub.index))
    vv = st.predictive_reg(sub["unc"].reindex(sub.index),
                           st.forward_rv(sub, 3).reindex(sub.index))
    print(f"  {lbl} (n={len(sub)}): RET t = {rr['t']:+.2f} (slope {rr['slope']:+.5f}) | "
          f"RV t = {vv['t']:+.2f}")

print("\n# PLACEBO — block-shuffle the regressor (broken link), 1,000 draws")
for out, h in [("ret", 3), ("ret", 6), ("rv", 3)]:
    p = st.placebo_pvalue(frame, out, h, on="level", n_draws=1000)
    print(f"  {out} h={h}: obs slope {p['obs_slope']:+.5f}  p = {p['p_value']:.3f}  "
          f"(placebo sd {p['placebo_sd']:.5f})")

print("\n# THE TIMER — lean INTO high uncertainty (risk-premium bet) vs buy-and-hold")
for lean in (True, False):
    tm = st.timer_stats(frame, thr_q=0.66, cost_bps=10.0, lean_in=lean)
    lab = "lean-in " if lean else "de-risk "
    print(f"  {lab}: net ann {tm['net']['ann_ret'] * 100:+.1f}%  vol "
          f"{tm['net']['ann_vol'] * 100:.1f}%  Sharpe {tm['net']['sharpe']:.2f}  "
          f"(exposure {tm['exposure']:.2f}, {int(tm['n_turns'])} turns)")
bh = st.timer_stats(frame, cost_bps=10.0)["buy_hold"]
print(f"  buy-and-hold: ann {bh['ann_ret'] * 100:+.1f}%  vol {bh['ann_vol'] * 100:.1f}%  "
      f"Sharpe {bh['sharpe']:.2f}")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
d0 = st.synthetic_detect(*data.synthetic(360, 0.0, 0.0, 878), horizon=3)
d1 = st.synthetic_detect(*data.synthetic(360, 0.02, 0.6, 878), horizon=3)
print(f"  null (edge=0)          : ret_t {d0['ret_t']:+.2f}, rv_t {d0['rv_t']:+.2f}")
print(f"  planted (0.02 / 0.6)   : ret_t {d1['ret_t']:+.2f}, rv_t {d1['rv_t']:+.2f}")
null_rt = np.array([st.synthetic_detect(*data.synthetic(300, 0.0, 0.0, 878 + s), 3)["ret_t"]
                    for s in range(10)])
print(f"  null 10 seeds: ret_t mean {null_rt.mean():+.2f} (sd {null_rt.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_rt) >= 2).sum()}/10")
