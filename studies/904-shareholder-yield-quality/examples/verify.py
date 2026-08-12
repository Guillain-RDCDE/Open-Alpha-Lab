"""Reproducible headline run for Study 904 — Shareholder-Yield + Quality.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached ETF panel under ``_cache/``
if present (the real-tape numbers), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))  # repo root (quantlab)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sy_quality import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

try:
    from quantlab.repro import fingerprint  # noqa: E402
except Exception:  # pragma: no cover
    def fingerprint(df, cols=None):
        import hashlib
        v = df.to_numpy(dtype="float64", na_value=np.nan).tobytes()
        return hashlib.sha1(v).hexdigest()[:12]


print("# Shareholder-Yield + Quality — quality-screened buybacks (PKW+QUAL) vs raw buyback (PKW) vs SPY")

if not data.have_real():
    print("(cache miss — run data.fetch() once to build _cache/syq_prices.parquet)")
    sys.exit(0)

px = data.load_prices()
for tk in data.TICKERS:
    s = px[tk].dropna()
    print(f"    {tk:5} lists {s.index.min().date()}  n={len(s)}")

m = data.monthly_total_returns(px)
cash = st.cash_returns(m, data.CASH)

qsy = st.sleeve_returns(m, data.QSY)     # PKW + QUAL
raw = st.sleeve_returns(m, data.RAW)     # PKW
spy = m[data.BENCH].rename("SPY")

# common QSY-era window (bound by QUAL 2013-07)
common = pd.concat([qsy, raw, spy, cash], axis=1).dropna()
common.columns = ["QSY", "RAW", "SPY", "CASH"]
print(f"\n[data] common QSY-era monthly frame: {len(common)} months  "
      f"{common.index.min():%Y-%m} -> {common.index.max():%Y-%m}  as-of {data.AS_OF}  "
      f"fingerprint={fingerprint(common)}")
qsy_c = common.iloc[:, 0]; raw_c = common.iloc[:, 1]
spy_c = common.iloc[:, 2]; cash_c = common.iloc[:, 3]

print("\n# Excess-of-cash annualised stats (rf = BIL, common window)")
for name, s in [("QSY (PKW+QUAL)", qsy_c), ("RAW (PKW)", raw_c), ("SPY", spy_c)]:
    a = st.ann_stats(s, cash_c)
    print(f"  {name:16}: CAGR {a['cagr']*100:6.2f}%  vol {a['vol']*100:5.2f}%  "
          f"exSharpe {a['sharpe']:+.3f}  maxDD {a['maxdd']*100:6.1f}%  $1->${a['wealth']:.2f}")

print("\n# Race 1 — QSY vs SPY (does quality-screened shareholder yield BEAT THE MARKET?)")
g1 = st.sharpe_gap_test(qsy_c, spy_c, cash_c)
print(f"  QSY-SPY: {g1['diff_ann_pct']:+.2f} pp/yr ({g1['diff_mean_bps']:+.2f} bps/mo)  "
      f"exSharpe {g1['sharpe_a']:+.3f} vs {g1['sharpe_b']:+.3f} (gap {g1['sharpe_gap']:+.3f})  "
      f"NW t = {g1['t_nw']:+.2f}  1s t = {g1['t_1s']:+.2f}  (n={g1['n_months']})")
b1 = st.sharpe_gap_bootstrap(qsy_c, spy_c, cash_c)
print(f"  Sharpe-gap bootstrap 95% CI [{b1['lo']:+.3f}, {b1['hi']:+.3f}]  "
      f"P(gap<0)={b1['frac_negative']:.2f}  (n_draws={b1['n_draws']})")

print("\n# Race 2 — QSY vs RAW (does the QUALITY OVERLAY add value over raw buybacks?)")
g2 = st.sharpe_gap_test(qsy_c, raw_c, cash_c)
print(f"  QSY-RAW: {g2['diff_ann_pct']:+.2f} pp/yr ({g2['diff_mean_bps']:+.2f} bps/mo)  "
      f"exSharpe {g2['sharpe_a']:+.3f} vs {g2['sharpe_b']:+.3f} (gap {g2['sharpe_gap']:+.3f})  "
      f"NW t = {g2['t_nw']:+.2f}  1s t = {g2['t_1s']:+.2f}  (n={g2['n_months']})")
b2 = st.sharpe_gap_bootstrap(qsy_c, raw_c, cash_c)
print(f"  Sharpe-gap bootstrap 95% CI [{b2['lo']:+.3f}, {b2['hi']:+.3f}]  "
      f"P(gap<0)={b2['frac_negative']:.2f}  (n_draws={b2['n_draws']})")

print("\n# Also — RAW vs SPY (does raw buyback alone beat the market?)")
g3 = st.sharpe_gap_test(raw_c, spy_c, cash_c)
print(f"  RAW-SPY: {g3['diff_ann_pct']:+.2f} pp/yr  exSharpe gap {g3['sharpe_gap']:+.3f}  "
      f"NW t = {g3['t_nw']:+.2f}  (n={g3['n_months']})")

print("\n# Era cut — QSY vs SPY, split 2020-01")
ec = st.era_cut(qsy_c, spy_c, cash_c, split="2020-01-01")
for lbl in ("early", "late"):
    e = ec[lbl]
    if "t_nw" in e:
        print(f"  {lbl:5}: {e['diff_ann_pct']:+.2f} pp/yr  gap {e['sharpe_gap']:+.3f}  "
              f"NW t = {e['t_nw']:+.2f}  (n={e['n_months']})")

print("\n# Era cut — QSY vs RAW, split 2020-01")
ec2 = st.era_cut(qsy_c, raw_c, cash_c, split="2020-01-01")
for lbl in ("early", "late"):
    e = ec2[lbl]
    if "t_nw" in e:
        print(f"  {lbl:5}: {e['diff_ann_pct']:+.2f} pp/yr  gap {e['sharpe_gap']:+.3f}  "
              f"NW t = {e['t_nw']:+.2f}  (n={e['n_months']})")

print("\n# Costs — monthly rebalance to equal weight, one-way spread x turnover")
for name, mem in [("QSY (PKW+QUAL)", data.QSY), ("RAW (PKW)", data.RAW)]:
    c = st.costed_sleeve(m.loc[common.index], mem, cash_c, one_way_bps=3.0)
    print(f"  {name:16}: turnover {c['avg_turnover_pct']:.2f}%/mo  drag {c['cost_drag_bps_yr']:.1f} bps/yr  "
          f"gross exSharpe {c['gross_sharpe']:+.3f} -> net {c['net_sharpe']:+.3f}")

print("\n# Context — SPYD (raw dividend yield) and BUYB (too young), common-with-QSY where available")
spyd = m[data.DIV].rename("SPYD")
dfd = pd.concat([qsy, spyd, cash], axis=1).dropna()
if len(dfd) > 6:
    gd = st.sharpe_gap_test(dfd.iloc[:, 0], dfd.iloc[:, 1], dfd.iloc[:, 2])
    print(f"  QSY vs SPYD (from {dfd.index.min():%Y-%m}, n={gd['n_months']}): "
          f"exSharpe {gd['sharpe_a']:+.3f} vs {gd['sharpe_b']:+.3f}  gap {gd['sharpe_gap']:+.3f}  NW t = {gd['t_nw']:+.2f}")
byoung = px[data.YOUNG].dropna()
print(f"  BUYB lists {byoung.index.min().date()} (n={len(byoung)} days) -> too young to race; named only")

print("\n# Calendar-year total returns (common window)")
cal = st.calendar_year_table({"QSY": qsy_c, "RAW": raw_c, "SPY": spy_c})
print((cal * 100).round(1).to_string())

print("\n# Drawdowns (monthly, common window)")
for name, s in [("QSY", qsy_c), ("RAW", raw_c), ("SPY", spy_c)]:
    print(f"  {name}: max DD {st.max_drawdown(s)*100:.1f}%")

print("\n# Synthetic control — machinery proof (deterministic, no network; never market evidence)")
print("  the Sharpe-gap detector must recover a PLANTED quality-over-raw edge and must NOT manufacture one from 0.")
for planted in (0.0, 3.0):
    ts = np.array([st.synthetic_detect(data.synthetic_world(edge=planted, seed=904 + s))["t_nw"]
                   for s in range(20)])
    fire = int((np.abs(ts) >= 2).sum())
    one = st.synthetic_detect(data.synthetic_world(edge=planted, seed=904))
    print(f"  planted edge={planted:.1f}%/yr: gap {one['sharpe_gap']:+.3f}  NW t = {one['t_nw']:+.2f} "
          f"(seed 904);  20-seed |t|>=2 in {fire}/20")
