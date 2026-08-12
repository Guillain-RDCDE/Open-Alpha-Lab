"""Reproducible headline run for Study 895 — Defensive Momentum.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the study's cached yfinance tape under
``_cache/`` (the real-tape numbers) and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from def_momentum import data, strategy as st  # noqa: E402

print("# Defensive Momentum — MTUM+USMV blend vs MTUM / USMV / QUAL / SPY (excess-of-cash)")
print("# (Dedup: 508 grades the momentum-CRASH mechanism, 330 the low-vol anomaly, 601 the")
print("#  live single wrappers, 237 residual momentum; this study asks whether BLENDING the")
print("#  two shipped wrappers buys a better risk-adjusted deal than momentum alone.)")

if not data.have_real():
    print("(missing cache: _cache/defmom_prices.parquet — run data.fetch() once)")
    sys.exit(0)

try:
    from quantlab import repro
except Exception:
    repro = None

prices = data.load_prices()
mret = data.monthly_total_returns(prices)
cash = mret[data.CASH]

# Common window across the two sleeves (blend starts when both exist = MTUM inception).
sleeves = mret[data.SLEEVES].dropna()
common = sleeves.index
fifty = st.fixed_blend(sleeves, 0.5)
volw = st.vol_weighted_blend(sleeves, lookback=12, lag=1)

print("\n# Data stamp")
print(f"tape        : {prices.shape[0]} days x {prices.shape[1]} tickers, "
      f"{prices.index.min().date()} -> {prices.index.max().date()} "
      f"(yfinance auto-adjusted closes = TOTAL RETURN)")
print(f"as-of       : monthly stats sliced to {data.AS_OF} (last complete month)")
print(f"blend window: {common.min().date()} -> {common.max().date()}  ({len(common)} months, "
      f"MTUM-inception limited)")
if repro is not None:
    panel = pd.concat([mret[data.TICKERS]], axis=1).loc[common]
    print(f"fingerprint : monthly_panel={repro.fingerprint(panel)}")
for tk in data.TICKERS:
    s = mret[tk].dropna()
    print(f"  {tk}: {s.index.min().date()} -> {s.index.max().date()}  ({len(s)} months)")

# --------------------------------------------------------------------------- #
def line(name, r):
    a = st.ann_stats(r.reindex(common).dropna(), cash)
    print(f"  {name:16s} CAGR {a['cagr']*100:6.2f}%  vol {a['vol']*100:5.1f}%  "
          f"exSharpe {a['sharpe']:.3f}  maxDD {a['maxdd']*100:6.1f}%  ${a['wealth']:.2f}")


print("\n# Per-strategy stats (same MTUM-inception window; excess = minus BIL cash)")
line("MTUM", mret["MTUM"])
line("USMV", mret["USMV"])
line("QUAL", mret["QUAL"])
line("SPY", mret["SPY"])
line("50/50 blend", fifty["gross"])
line("50/50 net", st.apply_costs(fifty, 3.0))
line("inv-vol blend", volw["gross"])
line("inv-vol net", st.apply_costs(volw, 3.0))

# --------------------------------------------------------------------------- #
print("\n# HEADLINE — excess-vs-excess Sharpe race: does the blend beat MTUM alone?")
mt_ex = st.excess(mret, "MTUM").reindex(common).dropna()
for label, blend in (("50/50", fifty), ("inv-vol", volw)):
    bl_ex = (blend["gross"] - cash).reindex(common).dropna()
    idx = bl_ex.index.intersection(mt_ex.index)
    race = st.sharpe_advantage(bl_ex.loc[idx], mt_ex.loc[idx])
    boot = st.bootstrap_sharpe_adv(bl_ex.loc[idx], mt_ex.loc[idx], n_draws=2000, seed=895)
    # net version
    net_ex = (st.apply_costs(blend, 3.0) - cash).reindex(common).dropna()
    idxn = net_ex.index.intersection(mt_ex.index)
    race_n = st.sharpe_advantage(net_ex.loc[idxn], mt_ex.loc[idxn])
    print(f"\n  == {label} blend vs MTUM ({race['n']} months)")
    print(f"  exSharpe blend {race['sharpe_a']:.3f} vs MTUM {race['sharpe_b']:.3f}  "
          f"(advantage {race['sharpe_adv']:+.3f})")
    print(f"  monthly return diff {race['diff_bps']:+.1f} bps/mo   NW t = {race['t_nw']:+.2f}"
          f"   (one-sample t {race['t_1s']:+.2f})")
    print(f"  bootstrap Sharpe-adv 95% CI [{boot['lo']:+.3f}, {boot['hi']:+.3f}]  "
          f"P(adv>0) = {boot['p_gt0']:.3f}")
    print(f"  NET of 3bps/side costs: exSharpe adv {race_n['sharpe_adv']:+.3f}  "
          f"diff {race_n['diff_bps']:+.1f} bps  NW t {race_n['t_nw']:+.2f}")

# --------------------------------------------------------------------------- #
print("\n# Crash geometry — the whole point ('without the crashes')")
print(f"  NOTE: MTUM the ETF launched 2013-04 — the classic 2008-09 momentum crash is")
print(f"  OUT of sample. The testable crashes are 2020-COVID and the 2022 bear.")
mtd = st.drawdown_curve(mret["MTUM"].reindex(common).dropna())
bld = st.drawdown_curve(fifty["gross"].reindex(common).dropna())
usd = st.drawdown_curve(mret["USMV"].reindex(common).dropna())
print(f"  full-window max drawdown: MTUM {mtd.min()*100:.1f}%  50/50 {bld.min()*100:.1f}%  "
      f"USMV {usd.min()*100:.1f}%")
for lab, s, e in (("2020 COVID", "2020-01-01", "2020-06-30"),
                  ("2022 bear", "2022-01-01", "2022-12-31"),
                  ("2018 Q4", "2018-09-01", "2018-12-31")):
    m = st.window_drawdown(mret["MTUM"].reindex(common), s, e)
    b = st.window_drawdown(fifty["gross"].reindex(common), s, e)
    u = st.window_drawdown(mret["USMV"].reindex(common), s, e)
    print(f"  {lab:11s} drawdown: MTUM {m*100:6.1f}%   50/50 {b*100:6.1f}%   USMV {u*100:6.1f}%")

# --------------------------------------------------------------------------- #
print("\n# Era cut — has the blend's edge held across sub-periods?")
bl_ex = (fifty["gross"] - cash).reindex(common).dropna()
idx = bl_ex.index.intersection(mt_ex.index)
eras = st.era_split(bl_ex.loc[idx], mt_ex.loc[idx])
for name in ("early", "late"):
    e = eras[name]
    print(f"  {name:5s} {e['start']}..{e['end']} ({e['n']}m): Sharpe adv "
          f"{e['sharpe_adv']:+.3f}  diff {e['diff_bps']:+.1f} bps  NW t {e['t_nw']:+.2f}")

# --------------------------------------------------------------------------- #
print("\n# Calendar-year total returns (%)")
cal = pd.DataFrame({
    "MTUM": st.calendar_year_returns(mret["MTUM"].reindex(common)) * 100,
    "USMV": st.calendar_year_returns(mret["USMV"].reindex(common)) * 100,
    "50/50": st.calendar_year_returns(fifty["gross"].reindex(common)) * 100,
    "SPY": st.calendar_year_returns(mret["SPY"].reindex(common)) * 100,
}).round(1)
print(cal.to_string())

# --------------------------------------------------------------------------- #
print("\n# Turnover / cost check")
print(f"  50/50 mean monthly turnover  {fifty['turnover'].iloc[1:].mean()*100:.2f}% of NAV")
print(f"  inv-vol mean monthly turnover {volw['turnover'].dropna().iloc[1:].mean()*100:.2f}% of NAV")

# --------------------------------------------------------------------------- #
print("\n# Synthetic control — machinery proof only, never market evidence")
for label, edge in [("null  (edge=0.0)", 0.0), ("planted (edge=1.0)", 1.0)]:
    w = data.synthetic_sleeves(edge=edge, seed=895, n_months=160)
    d = st.synthetic_detect(w, 0.5)
    print(f"  {label}: Sharpe adv {d['sharpe_adv']:+.3f}  NW t {d['t_nw']:+.2f}  "
          f"diff {d['diff_bps']:+.1f} bps  blend maxDD {d['blend_maxdd']*100:.1f}% "
          f"vs MTUM {d['mtum_maxdd']*100:.1f}%")
