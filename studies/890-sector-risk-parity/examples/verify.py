"""Reproducible headline run for Study 890 — Sector Risk-Parity.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached sector panel under ``_cache/``
(fetching once on a cache miss), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantlab.repro import fingerprint  # noqa: E402

from sector_rp import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")


def _fmt(s):
    return (f"Sharpe {s['sharpe']:.3f}  ann {s['ann']*100:+.2f}%  vol {s['vol']*100:.2f}%  "
            f"maxDD {s['max_drawdown']*100:.1f}%")


print("# Sector Risk-Parity — does equal-RISK-weighting the GICS sectors beat cap-weight SPY?")

if not data.have_real():
    print("(cache miss — fetching the 13-ticker sector panel once)")
    data.fetch()

for label, secs in [("ELEVEN-SECTOR (headline, 2018-)", data.SECTORS_11),
                    ("NINE-SECTOR (long history, 2007-)", data.SECTORS_9)]:
    p = data.daily_panel(sectors=secs)
    fp = fingerprint(p["prices"])
    print("\n" + "=" * 78)
    print(f"{label}  |  {len(p['prices'])} rows  {p['prices'].index.min().date()} -> "
          f"{p['prices'].index.max().date()}  as-of {data.AS_OF}  fingerprint={fp}")
    for sch in ("invvol", "erc"):
        r = st.race(p["sector_ret"], p["bench_ret"], p["cash_ret"], scheme=sch, cost_bps=3.0)
        print(f"\n-- {sch.upper()}  {r['start']}..{r['end']}  n={r['n_days']}  "
              f"rebalances={r['n_rebalances']}")
        print(f"   RP (net)  : {_fmt(r['strat_net'])}")
        print(f"   SPY       : {_fmt(r['bench'])}")
        print(f"   Sharpe diff (RP-SPY) net {r['sharpe_diff_net']:+.3f}  "
              f"95% CI [{r['ci_net'][0]:+.3f}, {r['ci_net'][1]:+.3f}]  "
              f"P(diff<0)={r['frac_neg_net']:.2f}  |  NW t (mean ret diff) = {r['nw_t_diff']:+.2f}")
        print(f"   turnover ~{r['ann_turnover']*100:.0f}%/yr  ->  cost drag "
              f"{r['cost_drag_bps_yr']:.1f} bps/yr at 3 bps one-way")

# ---- deep dive on the long 9-sector inverse-vol book (the fairest, longest sample) ------
p9 = data.daily_panel(sectors=data.SECTORS_9)
r9 = st.race(p9["sector_ret"], p9["bench_ret"], p9["cash_ret"], scheme="invvol", cost_bps=3.0)
exn, exb = r9["_ex_net"], r9["_ex_bench"]

print("\n" + "=" * 78)
print("# ERA CUT (9-sector inverse-vol) — does the Sharpe advantage hold across sub-eras?")
for lo, hi, lbl in [("2007-01-01", "2016-01-01", "2007-2015"),
                    ("2016-01-01", "2026-07-01", "2016-2026")]:
    a = exn[(exn.index >= lo) & (exn.index < hi)]
    b = exb[(exb.index >= lo) & (exb.index < hi)]
    sa, sb = st.annualized_sharpe(a.to_numpy()), st.annualized_sharpe(b.to_numpy())
    print(f"  {lbl}: RP Sharpe {sa:.3f} vs SPY {sb:.3f}  ->  diff {sa - sb:+.3f}  (n={len(a)})")

print("\n# LEVERED-TO-SPY-VOL TIMER — turn the (lower-vol) RP book into a return edge?")
lev = st.levered_to_bench_vol(exn, exb, p9["cash_ret"])
print(f"  leverage {lev['leverage']:.2f}x  ->  levered Sharpe {lev['sharpe_lev']:.3f} vs "
      f"SPY {lev['sharpe_bench']:.3f}  |  levered ann {lev['ann_lev_pct']:+.2f}% vs SPY "
      f"{lev['ann_bench_pct']:+.2f}%  (financing drag {lev['finance_drag_bps_yr']:.1f} bps/yr, "
      f"levered maxDD {lev['max_dd_lev']*100:.1f}%)")

print("\n# CALENDAR-YEAR TOTAL RETURN (9-sector inverse-vol vs SPY, %)")
ct = st.calendar_year_table(r9["_net"], r9["_bench"])
print(ct.round(1).to_string())
print(f"  RP wins {int((ct['diff'] > 0).sum())}/{len(ct)} years; "
      f"in the three worst SPY years RP beat it by "
      f"{ct.sort_values('SPY').head(3)['diff'].round(1).tolist()} pp")

print("\n" + "=" * 78)
print("# SYNTHETIC POSITIVE CONTROL — the machinery is unbiased (no network)")
nulls = np.array([st.synthetic_detect(data.synthetic_world(vol_spread=0.0, seed=890 + s))["sharpe_advantage"]
                  for s in range(20)])
print(f"  null (vol_spread=0), 20 seeds: mean advantage {nulls.mean():+.4f} "
      f"(sd {nulls.std(ddof=1):.4f}), |adv|>0.10 in {(np.abs(nulls) > 0.1).sum()}/20 seeds")
pl = st.synthetic_detect(data.synthetic_world(vol_spread=0.02, seed=890))
print(f"  planted (vol_spread=0.02, seed 890): advantage {pl['sharpe_advantage']:+.3f} "
      f"(RP Sharpe {pl['sr_rp']:.2f} vs cap-weight {pl['sr_bench']:.2f})")
