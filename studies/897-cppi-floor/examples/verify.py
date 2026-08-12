"""Reproducible headline run for Study 897 — CPPI Floor.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; reads the cached total-return closes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no network.

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

from cppi import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")
try:  # keep the unicode headers legible on a Windows cp1252 console
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

MULT, FLOOR = 5.0, 0.80

print("# CPPI Floor — does mechanical portfolio insurance protect the floor, and what does "
      "it cost?")

if not data.have_real():
    print("(cache miss — fetching SPY/IEF/BIL total-return closes once)")
    data.fetch()

px = data.load_prices()
ret = st.to_returns(px)
cash = ret[data.CASH_TICKER]
print(f"[data] {px.shape[1]} tickers, {len(ret)} return rows  "
      f"{ret.index.min().date()} -> {ret.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint(prices)={fingerprint(px)}")
print("  SHORT HISTORY: BIL (cash / floor-accretion leg) lists 2007-05-30 and bounds the joint "
      "window — a single-cycle caveat named on the Signal axis (but it puts 2008 in-sample).")

r = st.race(ret, multiplier=MULT, floor_frac=FLOOR)
print(f"\n# THE RACE — CPPI (m={MULT:.0f}, floor {FLOOR:.0%}, accreting at cash) vs buy-and-hold "
      f"SPY vs a matched-average-exposure static mix (all excess-of-cash, gross)")
for tag, s in [("CPPI       ", r["cppi"]), ("buy-and-hold", r["buy_hold"]),
               ("static@avgw", r["static"])]:
    print(f"  {tag}: excess-Sharpe {s['sharpe']:.3f}  CAGR {s['cagr']*100:5.2f}%  "
          f"vol {s['vol']*100:5.2f}%  maxDD {s['max_dd']*100:6.1f}%")
print(f"  excess-Sharpe vs buy-and-hold : {r['sharpe_vs_bh']:+.3f}")
print(f"  excess-Sharpe vs static mix   : {r['sharpe_vs_static']:+.3f}  "
      f"(a constant fraction of SPY has the SAME excess-Sharpe as SPY — so this equals vs-BH)")
print(f"  re-timing spanning alpha (CPPI on static): {r['alpha_ann']*100:+.2f}%/yr  "
      f"HAC t = {r['t_alpha']:+.2f}  (beta {r['alpha_beta']:.3f})")
print(f"  raw excess-diff HAC t = {r['diff_t_nw']:+.2f}")
print(f"  avg weight {r['avg_weight']:.2f}  (min {r['min_weight']:.2f} / max {r['max_weight']:.2f})  "
      f"overlay turnover {r['turnover_ann']:.1f}x/yr  floor breaches {r['n_breach']}  "
      f"min cushion {r['min_cushion']:.4f}")

bs = st.bootstrap_sharpe_diff(r["cppi_net"], r["bh_net"], cash, n_boot=2000)
print(f"\n# BOOTSTRAP — circular block CI on the excess-Sharpe difference CPPI − buy-and-hold "
      f"(2,000 resamples)")
print(f"  gain {bs['point']:+.3f}  95% CI [{bs['ci95'][0]:+.3f}, {bs['ci95'][1]:+.3f}]  "
      f"P(CPPI wins) = {bs['frac_a_wins']*100:.1f}%")

print("\n# CRASH YEARS — does the floor bite? (calendar-year total return & worst intra-year DD)")
cy_c = st.calendar_year_returns(r["cppi_net"]); cy_b = st.calendar_year_returns(r["bh_net"])
dd_c = st.calendar_year_drawdowns(r["cppi_net"]); dd_b = st.calendar_year_drawdowns(r["bh_net"])
for yr in (2008, 2020, 2022):
    print(f"  {yr}: buy-and-hold {cy_b.loc[yr]*100:+6.1f}% (DD {dd_b.loc[yr]*100:6.1f}%)  ->  "
          f"CPPI {cy_c.loc[yr]*100:+6.1f}% (DD {dd_c.loc[yr]*100:6.1f}%)")

print("\n# ROBUSTNESS — two eras (split 2016-01-01)")
for lo, hi, lbl in [("2007-05-30", "2016-01-01", "2007-2015"),
                    ("2016-01-01", "2026-07-01", "2016-2026")]:
    sub = ret[(ret.index >= lo) & (ret.index < hi)]
    rr = st.race(sub, multiplier=MULT, floor_frac=FLOOR)
    print(f"  {lbl}: n={rr['n_days']}  CPPI-Sh {rr['cppi']['sharpe']:+.3f}  BH-Sh "
          f"{rr['buy_hold']['sharpe']:+.3f}  vs-BH {rr['sharpe_vs_bh']:+.3f}  "
          f"alpha-t {rr['t_alpha']:+.2f}  maxDD {rr['dd_cppi']*100:.1f}% vs {rr['dd_bh']*100:.1f}%")

print("\n# MULTIPLIER SWEEP — the protection/return frontier (gross); breach needs a 1-day drop > 1/m")
ms = st.multiplier_sweep(ret, floor_frac=FLOOR)
for m, row in ms.iterrows():
    print(f"  m={m:.0f} (breach>{row['breach_thresh']*100:.0f}%): avg-w {row['avg_weight']:.2f}  "
          f"CPPI-Sh {row['cppi_sharpe']:+.3f}  maxDD {row['dd_cppi']*100:.1f}%  "
          f"breaches {int(row['n_breach'])}  turn {row['turnover_ann']:.1f}x")

print("\n# FLOOR ROBUSTNESS — higher floor buys more protection at more upside give-up (m=5, gross)")
for ff in (0.70, 0.80, 0.90):
    rr = st.race(ret, multiplier=MULT, floor_frac=ff)
    print(f"  floor {ff:.0%}: CPPI-Sh {rr['cppi']['sharpe']:+.3f}  maxDD {rr['dd_cppi']*100:.1f}%  "
          f"avg-w {rr['avg_weight']:.2f}  CAGR {rr['cppi']['cagr']*100:.2f}%")

print("\n# COSTED — one-way bps on the CPPI overlay's daily turnover (m=5)")
sweep = st.cost_sweep(ret, multiplier=MULT, floor_frac=FLOOR)
for c, row in sweep.iterrows():
    print(f"  {c:>4.0f} bp: CPPI {row['cppi_sharpe']:+.3f}  BH {row['bh_sharpe']:.3f}  "
          f"vs-BH {row['sharpe_vs_bh']:+.3f}  maxDD(CPPI) {row['dd_cppi']*100:.1f}%")

print("\n# GAP RISK — splice a single overnight crash at the thinnest-cushion day (m=5, thresh 20%)")
gs = st.gap_stress(ret, multiplier=MULT, floor_frac=FLOOR)
for g, row in gs.iterrows():
    print(f"  gap {g*100:.0f}%: breaches floor = {bool(row['breaches_floor'])}  "
          f"(cushion after {row['cushion_after']:+.4f})")

print("\n# SYNTHETIC CONTROL — the machinery is unbiased (offline, no network)")
bear = [st.synthetic_detect(data.synthetic_prices(seed=897 + s, n_days=2500,
                            drift=-0.0004, sigma=0.016)[0]) for s in range(30)]
calm = [st.synthetic_detect(data.synthetic_prices(seed=897 + s, n_days=2500,
                            drift=0.0004, sigma=0.006)[0]) for s in range(30)]
bp = np.array([d["dd_protection"] for d in bear])
bb = np.array([d["n_breach"] for d in bear])
cp = np.array([d["dd_protection"] for d in calm])
print(f"  BEAR (planted), 30 seeds: drawdown protection mean {bp.mean():+.3f} (min {bp.min():+.3f}), "
      f"floor breaches in {(bb > 0).sum()}/30 seeds  -> the floor holds mechanically")
print(f"  CALM (null),    30 seeds: drawdown protection mean {cp.mean():+.3f} (max {cp.max():+.3f})  "
      f"-> nothing to protect")
gap = [data.synthetic_prices(seed=897 + s, n_days=4000, drift=0.0003, sigma=0.011,
                             gap_prob=0.01, gap_size=0.25)[0] for s in range(20)]
nog = [data.synthetic_prices(seed=897 + s, n_days=4000, drift=0.0003, sigma=0.011,
                             gap_prob=0.01, gap_size=0.15)[0] for s in range(20)]
gb = np.array([st.synthetic_detect(f, multiplier=5.0)["n_breach"] for f in gap])
ng = np.array([st.synthetic_detect(f, multiplier=5.0)["n_breach"] for f in nog])
print(f"  GAP 25% (> 1/5 = 20%): floor breach fired in {(gb > 0).sum()}/20 seeds")
print(f"  GAP 15% (< 1/5 = 20%): floor breach fired in {(ng > 0).sum()}/20 seeds")
print("  (A faithful-engine / power check only — never cited in support of the real-tape stamp.)")
