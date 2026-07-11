"""Reproducible headline run for Study 662 — EM-Local-Bonds.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached EBND/LEMB/EMB/AGG/UUP/BIL tape
under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from em_local_bonds import data, strategy as st  # noqa: E402

print("# EM-Local-Bonds — does local-currency EM debt pay a better carry than USD EM debt?")

if not data.have_real():
    print("(cache miss — fetching EBND/LEMB/EMB/AGG/UUP/BIL once)")
    data.fetch()

px = data.load_real()
print(data_stamp("EBND/LEMB/EMB/AGG/UUP/BIL adjusted close", px, asof=data.AS_OF))

ret_full = st.monthly_returns(px)
ret = ret_full.loc[ret_full.index >= data.COMMON_START].dropna()
print(f"common monthly sample: {ret.index.min().date()} -> {ret.index.max().date()} "
      f"({len(ret)} months, all six tickers co-exist from LEMB's 2011-10-20 inception)")

print("\n# THE HEADLINE — Local (avg EBND/LEMB) minus EMB (USD EM), excess-of-cash (BIL)")
h = st.headline_spread(ret, versus="EMB")
print(f"  mean monthly gap: {h['mean_diff_mo']*1e4:+.1f} bps/mo  ({h['mean_diff_ann']*100:+.2f}%/yr)")
print(f"  paired t (no HAC)            : {h['t_paired']:+.2f}")
print(f"  Newey-West t  (lags 3/6/12)  : {h['nw_t_lag3']:+.2f} / {h['nw_t_lag6']:+.2f} / "
      f"{h['nw_t_lag12']:+.2f}")
print(f"  circular block-bootstrap 95% CI on the annualized gap: "
      f"[{h['boot_lo']*100:+.2f}%, {h['boot_hi']*100:+.2f}%]/yr  (5,000 draws, 6-month blocks)")
print(f"  hit rate (Local beats EMB, month-to-month): {h['hits']}/{h['n_pairs']} = "
      f"{h['hit_rate']*100:.1f}%  (Wilson 95% [{h['wilson_lo']*100:.1f}%, {h['wilson_hi']*100:.1f}%])")

print("\n# Sharpe race — excess of cash (BIL), full sample")
print(f"  Local (avg EBND/LEMB): Sharpe {h['sharpe_local']:+.3f}")
print(f"  EMB   (USD EM)       : Sharpe {h['sharpe_versus']:+.3f}")
print(f"  AGG   (US aggregate) : Sharpe {h['sharpe_agg']:+.3f}")

print("\n# One-time entry cost (buy-and-hold — no periodic turnover to charge)")
n_months = h["n"]
for cb in (5.0, 10.0):
    net = st.apply_entry_cost(h["mean_diff_ann"], n_months, cb)
    print(f"  cost={cb:>4.1f} bps one-way, one-time entry: gross {h['mean_diff_ann']*100:+.2f}%/yr "
          f"-> net {net*100:+.2f}%/yr  (amortized over {n_months/12:.1f} years — negligible)")

print("\n# THE FX-DRAG REGRESSION — each leg's monthly return vs UUP (dollar-strength proxy)")
fx = st.fx_beta_table(ret, lags=6)
for name in ("Local", "EMB", "AGG"):
    r = fx[name]
    print(f"  {name:>4s} ~ UUP: beta={r['beta']:+.3f}  NW t={r['t_beta']:+.2f}  "
          f"corr={r['corr']:+.3f}  (n={r['n']})")
diff_fx = fx["Local-minus-EMB"]
print(f"  isolated channel: (Local-EMB) ~ UUP: beta={diff_fx['beta']:+.3f}  "
      f"NW t={diff_fx['t_beta']:+.2f}  corr={diff_fx['corr']:+.3f}")
print("  (removes the EM-credit-cycle component both legs share; what's left is the")
print("   incremental FX channel unique to holding the local currency)")

print("\n# NAMED CRISIS WINDOWS — cumulative return inside each episode")
cw = st.crisis_window_table(ret, data.CRISIS_WINDOWS)
for label, row in cw.iterrows():
    print(f"  {label:>22s} (n={int(row['n_months'])}mo): Local {row['local']*100:+6.2f}%  "
          f"EMB {row['emb']*100:+6.2f}%  AGG {row['agg']*100:+6.2f}%  "
          f"dollar(UUP) {row['dollar']*100:+6.2f}%")

print("\n# Max drawdown (peak-to-trough, full sample)")
local = st.local_basket(ret)
for name, series in (("Local", local), ("EMB", ret["EMB"]), ("AGG", ret["AGG"])):
    dd, dt = st.max_drawdown(series)
    print(f"  {name:>4s}: {dd*100:6.2f}%  bottomed {dt.date()}")

print("\n# Synthetic control — deterministic, no network")
print("  the Welch/NW detector must NOT fire on a null world (yield_pickup=0) and must")
print("  recover a planted extra-yield world. Null checked over 20 seeds (never one stream).")
null_ts = []
for s_ in range(20):
    loc, usd = data.synthetic_world(seed=662 + s_, yield_pickup=0.0, drag=0.0)
    null_ts.append(st.synthetic_detect(loc, usd)["nw_t"])
null_ts = np.asarray(null_ts)
print(f"  null (yield_pickup=0), 20 seeds: mean NW t = {null_ts.mean():+.2f} "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")

loc, usd = data.synthetic_world(seed=662, yield_pickup=0.0025, drag=0.0)
sy = st.synthetic_detect(loc, usd)
print(f"  planted +2.5bps/mo (+3.0%/yr) undragged pickup (seed 662): "
      f"mean gap {sy['mean_diff_ann']*100:+.2f}%/yr  NW t = {sy['nw_t']:+.2f}")

loc, usd = data.synthetic_world(seed=662, yield_pickup=0.0025, drag=1.0)
sy2 = st.synthetic_detect(loc, usd)
print(f"  illustrative: same pickup + a full-strength dollar drag (drag=1.0, matched to the "
      f"real UUP secular drift) that eats it: "
      f"mean gap {sy2['mean_diff_ann']*100:+.2f}%/yr  NW t = {sy2['nw_t']:+.2f}")
print("  (machinery/narrative illustration only — never cited in support of the real-tape stamp)")
