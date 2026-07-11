"""Reproducible headline run for Study 660 — Carry-Everywhere.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached wide daily-close tape
under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from carry_everywhere import data as dt, strategy as st  # noqa: E402

print("# Carry-Everywhere — does a carry signal pay off in every asset class, and does "
      "a diversified basket earn a robust premium?")

if not dt.have_real():
    print("(cache miss — fetching FX/bond/equity/commodity closes once)")
    dt.fetch()

closes = dt.load_real()
print(data_stamp("wide daily closes (11 tickers)", closes, asof=dt.AS_OF))

mret = dt.monthly_returns(closes)
sl = st.all_sleeves(mret)
print(f"monthly sample: {len(sl)} months  {sl.index.min()} -> {sl.index.max()}  "
      f"(as-of {dt.AS_OF}, last complete month)")

print("\n# THE FOUR SLEEVES — gross monthly carry legs")
print(f"  {'sleeve':6s} {'ann %':>8s} {'vol %':>7s} {'Sharpe':>7s} {'HAC t':>7s} "
      f"{'skew':>7s} {'maxDD %':>8s}")
sleeve_out = {}
for c in sl.columns:
    r = sl[c]
    h = st.hac_mean(r)
    row = {"ann_pct": r.mean() * 12 * 100, "vol_pct": r.std(ddof=1) * np.sqrt(12) * 100,
           "sharpe": st.sharpe(r), "t": h["t"], "skew": st.skewness(r),
           "maxdd_pct": st.max_drawdown(r) * 100}
    sleeve_out[c] = row
    print(f"  {c:6s} {row['ann_pct']:8.2f} {row['vol_pct']:7.2f} {row['sharpe']:7.2f} "
          f"{row['t']:7.2f} {row['skew']:7.2f} {row['maxdd_pct']:8.2f}")

print("\n# Correlation matrix (monthly gross sleeve returns)")
corr = sl.corr()
print(corr.round(2).to_string())

print("\n# THE HEADLINE — equal-weight (1/4 each) cross-asset combo")
cb = st.combo(sl)
h = st.hac_mean(cb)
lo, hi = st.block_bootstrap_sharpe_ci(cb)
print(f"  ann return {cb.mean()*12*100:+.2f}%   ann vol {cb.std(ddof=1)*np.sqrt(12)*100:.2f}%   "
      f"Sharpe {st.sharpe(cb):+.2f}   HAC t = {h['t']:+.2f}")
print(f"  skew {st.skewness(cb):+.2f}   max drawdown {st.max_drawdown(cb)*100:.2f}%")
print(f"  block-bootstrap 95% Sharpe CI: [{lo:+.2f}, {hi:+.2f}]  (block=6mo, 2000 draws)")

print("\n# Robustness — alternate combo weightings")
ivw = st.inv_vol_weights(sl)
cb_iv = st.combo(sl, ivw)
h_iv = st.hac_mean(cb_iv)
print(f"  inverse-vol weights {({k: round(v,3) for k,v in ivw.items()})}")
print(f"  inv-vol combo: ann {cb_iv.mean()*12*100:+.2f}%  Sharpe {st.sharpe(cb_iv):+.2f}  "
      f"HAC t = {h_iv['t']:+.2f}")
cb3 = st.combo(sl[["FX", "BOND", "CMD"]])
h3 = st.hac_mean(cb3)
print(f"  ex-EQ 3-sleeve combo (FX+BOND+CMD, equal weight): ann {cb3.mean()*12*100:+.2f}%  "
      f"Sharpe {st.sharpe(cb3):+.2f}  HAC t = {h3['t']:+.2f}")

print("\n# Costs — one-way rebalance cost x turnover + ETF short-leg borrow (EQ/CMD only)")
to = st.all_turnover(mret)
print(f"  avg monthly one-way turnover per sleeve (frac of NAV): "
      f"{ {k: round(v,3) for k,v in to.mean().items()} }")
for cbps in (5.0, 10.0):
    n = st.combo_net(mret, cbps)
    hn = st.hac_mean(n)
    print(f"  cost={cbps:>4.1f} bps: net ann {n.mean()*12*100:+.2f}%   "
          f"net Sharpe {st.sharpe(n):+.2f}   net HAC t = {hn['t']:+.2f}")

print("\n# THIRD AXIS — does carry crash everywhere at once (2008 GFC / 2020 COVID)?")
for name, window in dt.CRISIS_WINDOWS.items():
    cs = st.crisis_stats(cb, window)
    print(f"  {name} ({window[0]} -> {window[1]}, n={cs['n_in']} months): "
          f"combo cumulative {cs['cum_return_pct']:+.2f}%  vs other-months mean "
          f"{cs['mean_other_mo_pct']:+.3f}%/mo")
    for c in sl.columns:
        cs_c = st.crisis_stats(sl[c], window)
        print(f"    {c:6s} cumulative {cs_c['cum_return_pct']:+.2f}%")

print("\n# Synthetic positive control — deterministic, no network")
print("  the HAC detector must NOT fire on a null world (no carry, no crash) across")
print("  >= 10 seeds, and must recover a planted carry premium.")
null_ts = []
for s_ in range(20):
    r = st.synthetic_detect(carry_bps_mo=0.0, crash_beta=0.0, seed=660 + s_)
    null_ts.append(r["t"])
null_ts = np.asarray(null_ts)
print(f"  null (no carry), 20 seeds: mean HAC t = {null_ts.mean():+.2f} "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
planted = st.synthetic_detect(carry_bps_mo=40.0, crash_beta=0.0, seed=660)
print(f"  planted carry = +40 bps/mo, no crash (seed 660): ann {planted['mean_bps']*12/100:+.2f}%  "
      f"Sharpe {planted['sharpe']:+.2f}  HAC t = {planted['t']:+.2f}")
planted_c = st.synthetic_detect(carry_bps_mo=40.0, crash_beta=1.0, seed=660)
print(f"  same +40 bps/mo carry WITH a planted synchronized crash factor: "
      f"Sharpe {planted_c['sharpe']:+.2f}  HAC t = {planted_c['t']:+.2f}  "
      f"skew {planted_c['skew']:+.2f}  (a real premium, swamped by tail risk — the honest "
      "reason a genuine carry factor can still fail to certify)")
