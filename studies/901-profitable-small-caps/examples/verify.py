"""Reproducible headline run for Study 901 — Profitable Small-Caps.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; cache-first on the real tape
(_cache/psc_prices.csv, built once from yfinance) and always-offline on the synthetic control.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from profitable_small import data, strategy as st  # noqa: E402
from quantlab import repro  # noqa: E402

# Display name -> excess-return column, and -> raw-return column.
XLEGS = {"CALF": "x_CALF", "XSHQ": "x_XSHQ", "IWM": "x_IWM", "IJR": "x_IJR", "SPY": "x_SPY"}
RLEGS = {"CALF": "r_CALF", "XSHQ": "r_XSHQ", "IWM": "r_IWM", "IJR": "r_IJR", "SPY": "r_SPY"}

print("# Profitable Small-Caps — CALF/XSHQ vs IWM/IJR vs SPY (yfinance, total-return, excess-of-cash)")
prices = data.load_prices()
frame = data.daily_frame(prices, asof=data.AS_OF)

stamp_df = frame.dropna(subset=["x_SPY", "rf"])
print(repro.data_stamp("daily excess frame", stamp_df,
                       cols=["x_CALF", "x_XSHQ", "x_IWM", "x_IJR", "x_SPY", "rf"],
                       asof=data.AS_OF))
print(f"tape: {prices.index.min().date()} -> {prices.index.max().date()} daily; "
      f"as-of {data.AS_OF}. ERs: {data.EXPENSE_RATIOS}")

print("\n# Sharpe race — COMMON window (CALF is the youngest; all legs sliced to overlap)")
r = st.race(frame, XLEGS)
print(f"  window {r['start']} -> {r['end']}  n={r['n']} days")
print(f"  {'leg':<6s} {'ann_exc%':>9s} {'ann_vol%':>9s} {'Sharpe':>7s} {'maxDD%':>8s}")
for name, s in r["legs"].items():
    print(f"  {name:<6s} {s['ann_excess_pct']:>+9.2f} {s['ann_vol_pct']:>9.2f} "
          f"{s['sharpe']:>+7.3f} {s['maxdd_pct']:>+8.1f}")

print("\n# Bootstrap Sharpe CI (excess-of-cash, common window)")
for name, col in XLEGS.items():
    win = st.common_window(frame, list(XLEGS.values()))
    b = st.sharpe_ci_bootstrap(win[col].to_numpy())
    print(f"  {name:<6s} Sharpe {b['sharpe']:+.3f}  95% CI [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}]  "
          f"P(Sharpe<0)={b['frac_negative']:.3f}")

print("\n# Head-to-head — quality vs plain and vs large (paired Sharpe-diff bootstrap + HAC t)")
PAIRS = [("CALF", "x_CALF", "IWM", "x_IWM"), ("CALF", "x_CALF", "IJR", "x_IJR"),
         ("CALF", "x_CALF", "SPY", "x_SPY"), ("XSHQ", "x_XSHQ", "IWM", "x_IWM"),
         ("XSHQ", "x_XSHQ", "IJR", "x_IJR"), ("XSHQ", "x_XSHQ", "SPY", "x_SPY")]
for an, ac, bn, bc in PAIRS:
    p = st.pair_test(frame, ac, bc)
    print(f"  {an} vs {bn:<4s} [{p['start']}->{p['end']} n={p['n']}]  "
          f"SR {p['sharpe_a']:+.3f} vs {p['sharpe_b']:+.3f}  diff {p['sharpe_diff']:+.3f} "
          f"CI[{p['diff_ci_low']:+.3f},{p['diff_ci_high']:+.3f}] P(diff<=0)={p['diff_frac_le0']:.3f}  "
          f"| daily diff {p['mean_diff_bps']:+.3f} bps HAC t={p['t_nw_diff']:+.2f}")

print("\n# Size/value beta decomposition — is the quality edge just a small-cap/market tilt?")
print("#   y = alpha + b1*x_IWM (size/small-cap) + b2*x_SPY (market); alpha = the 'cleaned' bit")
for name, col in [("CALF", "x_CALF"), ("XSHQ", "x_XSHQ")]:
    d = st.beta_decomp(frame, col, ["x_IWM", "x_SPY"])
    print(f"  {name}: alpha {d['alpha_ann_pct']:+.2f}%/yr (HAC t={d['t_alpha']:+.2f})  "
          f"b_IWM={d['betas']['x_IWM']:+.3f} (t={d['t_betas']['x_IWM']:+.1f})  "
          f"b_SPY={d['betas']['x_SPY']:+.3f} (t={d['t_betas']['x_SPY']:+.1f})  R2={d['r2']:.3f}")

print("\n# Era cut — pre-2021 (incl. COVID) vs 2021+ (era-robustness of the CALF vs IWM Sharpe)")
eras = st.era_races(frame, {"CALF": "x_CALF", "IWM": "x_IWM"}, split=data.ERA_SPLIT)
for tag, rr in eras.items():
    q, p = rr["legs"]["CALF"], rr["legs"]["IWM"]
    print(f"  {tag:<4s} [{rr['start']}->{rr['end']} n={rr['n']}]  "
          f"CALF SR {q['sharpe']:+.3f}  IWM SR {p['sharpe']:+.3f}  gap {q['sharpe']-p['sharpe']:+.3f}")

print("\n# Calendar-year total returns (%) — common window")
cy = st.calendar_years(frame, RLEGS)
print(cy.round(1).to_string())

print("\n# Tradability — costed net Sharpe race + long-quality/short-plain isolation trade")
for qn, qc, er in [("CALF", "x_CALF", data.EXPENSE_RATIOS["CALF"]),
                   ("XSHQ", "x_XSHQ", data.EXPENSE_RATIOS["XSHQ"])]:
    c = st.costed_race(frame, qc, "x_IWM", er_quality=er,
                       er_plain=data.EXPENSE_RATIOS["IWM"])
    print(f"  {qn} vs IWM: charge {c['charge_ann_pct']:.2f}%/yr  "
          f"Sharpe gross {c['sharpe_q_gross']:+.3f} -> net {c['sharpe_q_net']:+.3f}  "
          f"IWM {c['sharpe_plain']:+.3f}  net gap {c['net_gap']:+.3f}")
for qn, qc in [("CALF", "x_CALF"), ("XSHQ", "x_XSHQ")]:
    it = st.isolation_trade(frame, qc, "x_IWM")
    print(f"  {qn}-IWM isolation: gross {it['gross_ann_pct']:+.2f}%/yr (t={it['t_nw_gross']:+.2f})  "
          f"net {it['net_ann_pct']:+.2f}%/yr (t={it['t_nw_net']:+.2f})  charge {it['charge_ann_pct']:.2f}%/yr")

print("\n# Synthetic control — deterministic, no network (machinery proof, never market evidence)")
print("  planted quality Sharpe edge: quality-minus-plain must recover the knob; null must NOT fire.")
for edge in (0.0, 0.4):
    w = data.synthetic_world(edge=edge, seed=901)
    p = st.pair_test(w.assign().rename(columns=str), "x_quality", "x_plain")
    # pair_test expects columns present; build a frame with those names
    import pandas as pd  # noqa: E402
    fr = pd.DataFrame({"x_quality": w["x_quality"], "x_plain": w["x_plain"]}, index=w.index)
    p = st.pair_test(fr, "x_quality", "x_plain")
    print(f"  planted edge {edge:+.2f}: quality SR {p['sharpe_a']:+.3f}  plain SR {p['sharpe_b']:+.3f}  "
          f"diff {p['sharpe_diff']:+.3f} CI[{p['diff_ci_low']:+.3f},{p['diff_ci_high']:+.3f}]  "
          f"daily diff HAC t={p['t_nw_diff']:+.2f}")

fp = repro.fingerprint(stamp_df, cols=["x_CALF", "x_XSHQ", "x_IWM", "x_IJR", "x_SPY", "rf"])
print(f"\nFingerprint {fp}")
