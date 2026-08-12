"""Reproducible headline run for Study 883 — Mid-Cap Sweet Spot.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic once the parquet cache is present (fetched
once on a cache miss); always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import fingerprint  # noqa: E402

from midcap import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Mid-Cap Sweet Spot — does the 'forgotten middle' beat BOTH large and small?")

if not data.have_real():
    print("(cache miss — fetching the ETF total-return tape once)")
    data.fetch()

px = data.load_prices()
ret = st.daily_returns(px)
print(f"[data] {px.shape[1]} tickers, {len(px)} rows  "
      f"{px.index.min().date()} -> {px.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint(Close)={fingerprint(px)}")
print("  SHORT HISTORY: BIL (cash leg) lists only from 2007-05, fixing the "
      "excess-of-cash common window; the pairwise mid-minus-benchmark difference is "
      "cash-independent, so the era cut reaches the full IJH/MDY tape. Named on Signal.")

print("\n# THE RACE — excess-of-cash Sharpe on the common (BIL-anchored) window")
r = st.race(ret, ["IJH", "MDY", "SPY", "IWM"], "BIL")
common_n = int(r["n"].iloc[0])
for a in ["IJH", "MDY", "SPY", "IWM"]:
    row = r.loc[a]
    print(f"  {a}: annret {row['ann_ret_pct']:6.2f}%  vol {row['ann_vol_pct']:5.1f}%  "
          f"exSharpe {row['ex_sharpe']:.3f}  maxDD {row['max_dd_pct']:6.1f}%")

print("\n# THE ADVANTAGE — mid (IJH) excess-Sharpe minus each neighbour, paired-bootstrap CI")
for other in ["SPY", "IWM"]:
    bs = st.sharpe_adv_bootstrap(ret, "IJH", other, "BIL", n_boot=2000, seed=883)
    print(f"  IJH - {other}: adv {bs['adv']:+.3f}  95% CI [{bs['ci_lo']:+.3f}, "
          f"{bs['ci_hi']:+.3f}]  {'CLEARS 0' if bs['clears_zero'] else 'spans 0'}")

print("\n# PAIRWISE RETURN DIFFERENCE (cash-independent -> full tape)")
for mid in ["IJH", "MDY"]:
    for other in ["SPY", "IWM"]:
        d = st.pairwise_diff(ret, mid, other)
        print(f"  {mid} - {other}: {d['ann_diff_pct']:+6.2f}%/yr  HAC t = {d['t_nw']:+.2f}  "
              f"(n={d['n']}, {d['start']}->{d['end']})")

print("\n# ROBUSTNESS — MDY - SPY by era (the longest mid-vs-large tape)")
tab = st.era_table(ret, "MDY", "SPY",
                   ["1995-01-01", "2003-01-01", "2010-01-01", "2017-01-01", "2026-07-01"])
for _, row in tab.iterrows():
    print(f"  {row['era']}: {row['ann_diff_pct']:+6.2f}%/yr  HAC t = {row['t_nw']:+.2f}  "
          f"(n={int(row['n'])})")

print("\n# THE COSTED SPREAD — long mid / short neighbour, dollar-neutral")
print("  charge = 50 bps/yr borrow + 2 sides × 3 bps × 4 rebals/yr")
for other in ["SPY", "IWM"]:
    c = st.costed_spread(ret, "IJH", other, cost_bps_oneway=3.0, borrow_bps_yr=50.0,
                         rebalances_per_year=4.0)
    print(f"  long IJH / short {other}: gross {c['gross_ann_pct']:+.2f}%/yr -> "
          f"net {c['net_ann_pct']:+.2f}%/yr (charge {c['charge_ann_pct']:.2f}, "
          f"t_net = {c['t_net_nw']:+.2f})")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_beats = 0
null_advL = []
for s_ in range(20):
    w0 = data.synthetic_world(n_days=3000, edge=0.0, seed=883 + s_)
    sig0 = st.synthetic_detect(w0)
    null_beats += int(sig0["beats_both"] and abs(sig0["t_large"]) >= 2 and abs(sig0["t_small"]) >= 2)
    null_advL.append(sig0["adv_large"])
w1 = data.synthetic_world(n_days=3000, edge=0.0006, seed=883)
sig1 = st.synthetic_detect(w1)
print(f"  null (edge=0), 20 seeds: mean adv-vs-large {np.mean(null_advL):+.3f}, "
      f"significant-beats-both in {null_beats}/20 seeds")
print(f"  planted (edge=0.0006, seed 883): adv-vs-large {sig1['adv_large']:+.3f} "
      f"(t={sig1['t_large']:+.2f}), adv-vs-small {sig1['adv_small']:+.3f} "
      f"(t={sig1['t_small']:+.2f}), beats_both={sig1['beats_both']}")
