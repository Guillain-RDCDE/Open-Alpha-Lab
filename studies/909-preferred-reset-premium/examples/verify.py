"""Reproducible headline run for Study 909 — Preferred Reset Premium.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached total-return panel under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control offline.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from pref_reset import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Preferred Reset Premium — do variable-rate preferreds out-carry fixed in high rates?")

if not data.have_real():
    print("(cache miss — fetching the preferred-ETF total-return panel once)")
    data.fetch()

prices = data.load_prices()
monthly = data.monthly_returns(prices)
sleeves = data.sleeve_returns(monthly)
print(f"[data] {prices.shape[1]} tickers, {len(prices)} daily rows  "
      f"{prices.index.min().date()} -> {prices.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint(prices)={data.fingerprint(prices)}")
print("  SHORT HISTORY: VRP 2014-05, PFFV 2020-06 — the thesis leans on the single 2022 "
      "hiking cycle. Named on the Signal axis.")

# ---- Flagship pair: VRP (variable) vs PFF (fixed), full common window -------
rf = st.race_frame(sleeves, var_col="VRP", fix_col="PFF")
h = st.race_stats(rf)
print(f"\n# FLAGSHIP — VRP vs PFF, excess-of-cash, {h['start']} -> {h['end']} (n={h['n']} mo)")
print(f"  VRP excess Sharpe = {h['var_ex_sharpe']:+.2f}   PFF excess Sharpe = {h['fix_ex_sharpe']:+.2f}"
      f"   advantage = {h['sharpe_adv']:+.2f}")
print(f"  (VRP - PFF) spread = {h['spread_ann_pct']:+.2f}%/yr  NW t = {h['t_nw']:+.2f}  "
      f"1s t = {h['t_1s']:+.2f}")

era = st.era_cut(rf, data.HIGH_RATE_SPLIT)
lo, hi = era["low_rate"], era["high_rate"]
print(f"\n# ERA CUT at {data.HIGH_RATE_SPLIT}")
print(f"  low-rate  {lo['start']}->{lo['end']} (n={lo['n']}): spread {lo['spread_ann_pct']:+.2f}%/yr  "
      f"NW t={lo['t_nw']:+.2f}  advantage {lo['sharpe_adv']:+.2f}")
print(f"  high-rate {hi['start']}->{hi['end']} (n={hi['n']}): spread {hi['spread_ann_pct']:+.2f}%/yr  "
      f"NW t={hi['t_nw']:+.2f}  advantage {hi['sharpe_adv']:+.2f}")

# ---- Bootstrap CIs on the full-window spread and Sharpe advantage ----------
sp = rf["spread"].to_numpy(float)
pt, blo, bhi = st.block_bootstrap_ci(sp, lambda a: float(np.mean(a) * st.MONTHS * 100))
print(f"\n# BOOTSTRAP (block=6, 2000 draws) full-window spread mean")
print(f"  spread {pt:+.2f}%/yr  95% CI [{blo:+.2f}, {bhi:+.2f}]  (clears 0: {blo > 0})")

def _adv(a_idx):
    ve = rf["var_ex"].to_numpy(float); fe = rf["fix_ex"].to_numpy(float)
    return st.ann_sharpe(ve) - st.ann_sharpe(fe)
# Sharpe-advantage bootstrap: resample the row index jointly.
rng = np.random.default_rng(909)
ve = rf["var_ex"].to_numpy(float); fe = rf["fix_ex"].to_numpy(float)
n = len(rf); block = 6; advs = []
for _ in range(2000):
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    sel = ((starts[:, None] + np.arange(block)[None, :]) % n).reshape(-1)[:n]
    advs.append(st.ann_sharpe(ve[sel]) - st.ann_sharpe(fe[sel]))
alo, ahi = np.percentile(advs, [2.5, 97.5])
print(f"  Sharpe advantage {h['sharpe_adv']:+.2f}  95% CI [{alo:+.2f}, {ahi:+.2f}]  (clears 0: {alo > 0})")

# ---- Sleeve summaries (drawdowns) ------------------------------------------
print("\n# SLEEVE SUMMARIES (full common window, total return)")
for c in ("variable", "fixed", "VRP", "PFF"):
    s = st.sleeve_summary(sleeves, c)
    print(f"  {c:9s}: ann {s['ann_ret_pct']:+.2f}%  vol {s['vol_ann']*100:4.1f}%  "
          f"maxDD {s['max_dd']*100:+.1f}%  excess Sharpe {s['ex_sharpe']:+.2f}  (n={s['n']})")

# ---- Multi-name sleeve (adds PFFV from 2020-07) ----------------------------
rf_sl = st.race_frame(sleeves, start=data.PFFV_INCEPTION)
hs = st.race_stats(rf_sl)
print(f"\n# MULTI-NAME SLEEVE variable(VRP,PFFV) vs fixed(PFF,PGX,PGF), {hs['start']}->{hs['end']} (n={hs['n']})")
print(f"  variable excess Sharpe {hs['var_ex_sharpe']:+.2f}  fixed excess Sharpe {hs['fix_ex_sharpe']:+.2f}"
      f"  advantage {hs['sharpe_adv']:+.2f}")
print(f"  spread {hs['spread_ann_pct']:+.2f}%/yr  NW t {hs['t_nw']:+.2f}")

# ---- Calendar-year table ---------------------------------------------------
print("\n# CALENDAR-YEAR TOTAL RETURN (variable / fixed / cash)")
cyt = st.calendar_year_table(sleeves)
for yr, row in cyt.iterrows():
    print(f"  {yr}: variable {row['variable']*100:+6.1f}%  fixed {row['fixed']*100:+6.1f}%  "
          f"cash {row['cash']*100:+5.1f}%")

# ---- Costed isolation spread ----------------------------------------------
print("\n# COSTED long-variable / short-fixed isolation spread (full window)")
for cb in (4.0, 8.0):
    cs = st.costed_spread(rf, cost_bps_oneway=cb)
    print(f"  {cb:.0f} bps one-way: gross {cs['gross_ann_pct']:+.2f}%  charge {cs['charge_ann_pct']:.2f}%  "
          f"net {cs['net_ann_pct']:+.2f}%/yr  net NW t {cs['t_net']:+.2f}  net Sharpe {cs['net_sharpe']:+.2f}")

# ---- Regime-switch strategy ------------------------------------------------
sw = st.switch_strategy(sleeves)
print(f"\n# REGIME-SWITCH (hold variable when rising-rate signal>0, else fixed; n={sw['n']}, "
      f"{sw['switches']} switches, {sw['share_variable']*100:.0f}% in variable)")
print(f"  switch net excess Sharpe {sw['switch_ex_sharpe']:+.2f}  ({sw['switch_net_ann_pct']:+.2f}%/yr)")
print(f"  always-fixed excess Sharpe {sw['always_fixed_ex_sharpe']:+.2f}  "
      f"always-variable excess Sharpe {sw['always_var_ex_sharpe']:+.2f}")

# ---- Synthetic control -----------------------------------------------------
print("\n# SYNTHETIC CONTROL")
planted = st.synthetic_detect(data.synthetic_world(edge=0.0030, seed=909))
print(f"  planted (edge=0.30%/mo): spread NW t = {planted['t_nw']:+.2f}  "
      f"high-regime {planted['spread_high_ann_pct']:+.2f}%/yr  low-regime {planted['spread_low_ann_pct']:+.2f}%/yr")
null_ts = []
for s in range(20):
    w = data.synthetic_world(edge=0.0, dur_hit=0.0, seed=909 + s)
    null_ts.append(st.synthetic_detect(w)["t_nw"])
null_ts = np.array(null_ts)
print(f"  null (edge=0), 20 seeds: mean t {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f})  "
      f"|t|>=2 in {int((np.abs(null_ts) >= 2).sum())}/20 seeds")

print(f"\nfingerprint(prices) = {data.fingerprint(prices)}")
