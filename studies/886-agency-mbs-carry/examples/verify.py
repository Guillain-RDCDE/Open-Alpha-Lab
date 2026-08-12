"""Reproducible headline run for Study 886 — Agency MBS Carry.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; cache-first on the real tape
(_cache/mbs_prices.parquet, built once from yfinance) and always-offline on the
synthetic control.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mbs_carry import data, strategy as st
from quantlab import repro

SPLITS = ["2014-01-01", "2020-01-01"]     # QE grind / mid-cycle / rate-vol eras
HIKE_START = "2022-01-01"                 # the Fed hiking + convexity-blowup era


def line(tag: str, s: dict) -> None:
    print(f"  {tag:<11s} n={s['n']:>3d} {s['start']}->{s['end']}  "
          f"beta {s['beta']:.3f} (reg t={s['t_reg_beta']:+.1f}, R2={s['r2']:.2f})  "
          f"carry {s['carry_ann_pct']:+.2f}%/yr  HAC t={s['t_hac']:+.2f}  "
          f"Sharpe {s['sharpe']:+.2f}  maxDD {s['max_dd_pct']:+.1f}%")


print("# Agency MBS Carry — MBB/VMBS excess over duration-matched IEF (yfinance, total-return)")
prices = data.load_prices()
panel = data.monthly_panel(prices, asof=data.AS_OF)

stamp = panel.dropna(subset=["MBB", "IEF", "BIL"])
print(repro.data_stamp("monthly panel", stamp,
                       cols=["MBB", "VMBS", "IEF", "AGG", "BIL"], asof=data.AS_OF))
print(f"tape: {prices.index.min().date()} -> {prices.index.max().date()} daily; monthly, "
      f"as-of {data.AS_OF} (last complete month)")

print("\n# Full sample — duration-neutral carry = (MBS-cash) - beta*(IEF-cash), both excess of BIL")
print("#  empirical beta = realized rate sensitivity (prices in negative convexity)")
for mbs in ["MBB", "VMBS"]:
    ef = data.excess_frame(panel, mbs)
    s = st.carry_stats(ef)                                   # empirical beta
    line(f"{mbs} (emp)", s)
    ci = st.mean_ci_bootstrap((ef["mbs"] - s["beta"] * ef["ief"]).values)
    print(f"     bootstrap 95% CI on carry: [{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}] %/yr, "
          f"P(mean<0)={ci['frac_negative']:.2f}")
    ss = st.carry_stats(ef, beta=data.OAD[mbs] / data.OAD["IEF"])   # static OAD-ratio hedge
    line(f"{mbs} (static)", ss)

print("\n# Excess-vs-excess Sharpe race (both legs minus BIL cash)")
for mbs in ["MBB", "VMBS"]:
    r = st.sharpe_race(data.excess_frame(panel, mbs))
    print(f"  {mbs}: MBS excess Sharpe {r['mbs_sharpe']:+.3f} "
          f"({r['mbs_ann_pct']:+.2f}%/yr, vol {r['mbs_vol_pct']:.2f}%) vs "
          f"IEF {r['ief_sharpe']:+.3f} ({r['ief_ann_pct']:+.2f}%/yr, vol {r['ief_vol_pct']:.2f}%) "
          f"-> adv {r['sharpe_adv']:+.3f}, raw Welch t={r['welch_t_raw']:+.2f}")

print("\n# Era cut (empirical hedge) — does the carry hold across regimes?")
ef = data.excess_frame(panel, "MBB")
for e in st.era_cut(ef, SPLITS):
    print(f"  {e['start']}->{e['end']}  n={e['n']:>3d}  carry {e['carry_ann_pct']:+.2f}%/yr  "
          f"HAC t={e['t_hac']:+.2f}")
hike = data.excess_frame(panel, "MBB", start=HIKE_START)
sh = st.carry_stats(hike)
print(f"  hiking era {HIKE_START}->{sh['end']}  n={sh['n']}  carry {sh['carry_ann_pct']:+.2f}%/yr  "
      f"HAC t={sh['t_hac']:+.2f}")

print("\n# HAC-lag sensitivity (MBB empirical hedge, full sample)")
c, _ = st.carry_series(ef)
for lags in (3, 6, 12):
    print(f"  NW lags={lags:>2d}: t={st.newey_west_t(c.values, lags):+.2f}")

print("\n# Tradability — costs on both legs + borrow on the short Treasury leg")
for mbs in ["MBB", "VMBS"]:
    cc = st.costed_carry(data.excess_frame(panel, mbs))
    print(f"  {mbs}: gross {cc['gross_ann_pct']:+.2f}%/yr - charge {cc['charge_ann_pct']:.2f} "
          f"-> net {cc['net_ann_pct']:+.2f}%/yr (HAC t={cc['t_net_hac']:+.2f})")

print("\n# Calendar-year duration-neutral carry (MBB empirical hedge, %)")
cy = st.calendar_year_table(c) * 100
print("  " + "  ".join(f"{y}:{v:+.1f}" for y, v in cy.round(1).items()))

print("\n# Synthetic control — the machinery is unbiased (never market evidence)")
for tag, carry in [("null (0%/yr)", 0.0), ("planted (+2%/yr)", 0.02)]:
    s = st.synthetic_detect(data.synthetic_world(carry_annual=carry, seed=886))
    print(f"  {tag:<18s} recovered carry {s['carry_ann_pct']:+.2f}%/yr  HAC t={s['t_hac']:+.2f}  "
          f"beta {s['beta']:.3f}")

print("\n# Fingerprint (MBB/VMBS/IEF/AGG/BIL monthly panel)")
print(f"  {repro.fingerprint(stamp, cols=['MBB', 'VMBS', 'IEF', 'AGG', 'BIL'])}")
