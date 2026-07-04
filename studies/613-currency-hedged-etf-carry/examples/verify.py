"""Reproducible headline run for Study 613 — Currency-Hedged ETF Carry.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; cache-first on the real tape
(_cache/chc_prices.csv, built once from yfinance) and always-offline on the synthetic control.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from currency_hedged_etf_carry import data, strategy as st
from quantlab import repro

HIGH_START = "2022-07-01"          # the Fed's 2022 hiking cycle makes the differential fat
# Both US and JP short rates pinned near zero; starts 2010-04 because DXJ only became a HEDGED
# fund on 2010-04-01 (before that it was the unhedged Japan Total Dividend Fund — its hedge
# beta on 2006-07..2010-03 is -0.15, i.e. no hedge at all).
ZIRP = ("2010-04-01", "2015-12-31")
PAIR_START = {
    "DXJ/EWJ": "2010-04-01",   # DXJ ran an UNHEDGED dividend mandate before Apr-2010
    "HEDJ/VGK": "2012-02-01",  # HEDJ switched to a hedged mandate in Jan-2012
}


def show(tag: str, s: dict) -> None:
    print(f"  {tag:<10s} n={s['n']:>3d} {s['start']}->{s['end']}  "
          f"diff {s['diff_bps']:+7.2f} bps/mo (t={s['t_diff']:+.2f})  "
          f"fx {s['fx_bps']:+7.2f}  carry {s['carry_bps']:+7.2f} bps/mo "
          f"[{s['carry_ann_pct']:+.2f}%/yr]  HAC t={s['t_carry']:+.2f}  "
          f"| rate-diff {s['rate_diff_ann_pct']:+.2f}%/yr  "
          f"| reg: alpha {s['alpha_bps']:+.2f} bps (t={s['t_alpha']:+.2f}) "
          f"beta {s['beta']:+.3f} (t={s['t_beta']:+.1f}) R2={s['r2']:.3f}")


print("# Currency-Hedged ETF Carry — DXJ/EWJ, HEWJ/EWJ, HEDJ/VGK (yfinance, total-return)")
prices = data.load_prices()
panel = data.monthly_panel(prices, asof=data.AS_OF)

stamp_df = panel.dropna(subset=["EWJ", "fx_JPY", "us_rate"])
print(repro.data_stamp("monthly panel", stamp_df,
                       cols=["DXJ", "EWJ", "HEWJ", "HEDJ", "VGK", "fx_JPY", "fx_EUR", "us_rate"],
                       asof=data.AS_OF))
print(f"tape: {prices.index.min().date()} -> {prices.index.max().date()} daily; monthly stats "
      f"sliced to as-of {data.AS_OF} (last complete month)")

print("\n# Full sample — the identity: hedged - unhedged = carry - fx  =>  carry_hat = diff + fx")
print("  (HAC/Newey-West t, 6 lags; regression is diff on -fx: beta ~ 1 = a full currency hedge)")
FULL = {}
for name, (h, u, ccy, label) in data.PAIRS.items():
    pf = st.pair_frame(panel, h, u, ccy, start=PAIR_START.get(name))
    FULL[name] = st.pair_stats(pf)
    print(f"  -- {name}: {label}")
    show(name, FULL[name])

print("\n# High-differential era (2022-07 -> 2026-06, 48 months; Fed ~5% vs BOJ ~0)")
for name, (h, u, ccy, label) in data.PAIRS.items():
    pf = st.pair_frame(panel, h, u, ccy, start=HIGH_START)
    show(name, st.pair_stats(pf))

print("\n# Near-zero-carry era (2010-04 -> 2015-12; US and JP rates both ~0,")
print("#  and DXJ hedged from 2010-04 only) — third axis")
pf_z = st.pair_frame(panel, "DXJ", "EWJ", "JPY", start=ZIRP[0], end=ZIRP[1])
show("DXJ/EWJ", st.pair_stats(pf_z))
sz = st.pair_stats(pf_z)
print(f"  => with nothing to pocket (rate-diff {sz['rate_diff_ann_pct']:+.2f}%/yr) the carry "
      f"component is {sz['carry_ann_pct']:+.2f}%/yr (t={sz['t_carry']:+.2f}) — yet hedged still "
      f"beat unhedged by {sz['diff_bps']:+.2f} bps/mo, ALL of it the falling yen "
      f"(fx {sz['fx_bps']:+.2f} bps/mo), none of it carry.")

print("\n# Does the pocketed carry TRACK the observable rate differential? (12m rolling)")
for name, (h, u, ccy, label) in data.PAIRS.items():
    pf = st.pair_frame(panel, h, u, ccy, start=PAIR_START.get(name))
    r = st.rolling_carry_track(pf)
    print(f"  {name:<10s} corr(rolling carry, rolling rate-diff) = {r['corr']:+.3f}  "
          f"pass-through slope = {r['slope']:+.3f}  (n={r['n']} rolling months)")

print("\n# HAC-lag sensitivity — HEWJ/EWJ carry t (the headline number)")
pf_c = st.pair_frame(panel, "HEWJ", "EWJ", "JPY")
pf_ch = st.pair_frame(panel, "HEWJ", "EWJ", "JPY", start=HIGH_START)
for lags in (3, 6, 12):
    _, t_full = st.nw_mean_t(pf_c["carry_hat"].values, lags)
    _, t_high = st.nw_mean_t(pf_ch["carry_hat"].values, lags)
    print(f"  lags={lags:>2d}: full-sample t={t_full:+.2f}   2022+ t={t_high:+.2f}")

print("\n# Tradability")
print("  (a) The zero-extra-cost version: hold the hedged share class instead of the unhedged")
print("      one. ERs (fund pages, 2026): DXJ 0.48% ~ EWJ 0.50%; HEWJ 0.50% = EWJ 0.50%;")
print("      but HEDJ 0.58% vs VGK 0.09% — the EUR pair charges ~0.5%/yr for the wrapper.")
print("  (b) Isolating the carry: long hedged / short unhedged, borrow 50 bps/yr on the short")
print("      leg + 5 bps one-way x ~2 rebalances/yr. NOTE: the raw spread is also SHORT the")
print("      foreign currency; net_carry strips the fx leg.")
for name, (h, u, ccy, label) in data.PAIRS.items():
    pf = st.pair_frame(panel, h, u, ccy, start=HIGH_START)
    sp = st.spread_trade(pf)
    print(f"  {name:<10s} 2022+: raw spread net {sp['net_diff_ann_pct']:+.2f}%/yr "
          f"(t={sp['t_net_diff']:+.2f})  fx-stripped carry net {sp['net_carry_ann_pct']:+.2f}%/yr "
          f"(t={sp['t_net_carry']:+.2f})  charges {sp['charge_ann_pct']:.2f}%/yr")

print("\n  (c) Share-class switch: hold HEDGED when the PRIOR month-end rate differential")
print("      > 1%/yr, else unhedged (one-month lag, 5 bps one-way per leg per switch):")
for name, (h, u, ccy, label) in data.PAIRS.items():
    pf = st.pair_frame(panel, h, u, ccy, start=PAIR_START.get(name))
    sw = st.switch_strategy(pf, (panel[h], panel[u]))
    print(f"  {name:<10s} net {sw['strat_net_ann_pct']:+.2f}%/yr  vs always-hedged "
          f"{sw['hedged_ann_pct']:+.2f}%  vs always-unhedged {sw['unhedged_ann_pct']:+.2f}%  "
          f"({sw['switches']} switches in {sw['n']} months; hedged {sw['share_hedged']*100:.0f}% "
          f"of the time)")

print("\n# Synthetic control — deterministic, no network (machinery proof, never market evidence)")
print("  planted-carry world: carry_hat must recover the knob; the zero-carry null must NOT fire.")
import pandas as pd  # noqa: E402  (only used to shape the synthetic frame)

for carry in (0.0, 0.05):
    world = data.synthetic_world(carry_annual=carry, seed=613)
    pfx = pd.DataFrame({"diff": world["hedged"] - world["unhedged"], "fx": world["fx"],
                        "rate_diff": world["rate_diff"]}, index=world.index)
    pfx["carry_hat"] = pfx["diff"] + pfx["fx"]
    s = st.pair_stats(pfx)
    print(f"  planted carry {carry*100:+5.1f}%/yr: carry_hat {s['carry_ann_pct']:+.3f}%/yr "
          f"(HAC t={s['t_carry']:+.2f})  beta {s['beta']:+.3f} (t={s['t_beta']:+.1f})")
