"""Reproducible headline run for Study 866 — Flight-to-Quality Beta.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached cross-section panel and the
cached TLT/SPY closes under ``_cache/`` (fetching once on a cache miss through the
quantlab.universe survivorship guard / yfinance), and always runs the synthetic control
with no network.

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

from ftq_beta import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Flight-to-Quality Beta — which stocks truly rally with bonds on risk-off days,")
print("#   do the hedges under-earn (pay-for-protection), and do they really cushion crashes?")

if not data.have_real():
    print("(cache miss — fetching the cross-section panel once, through the survivorship guard)")
    data.fetch()
if not data.have_market():
    print("(cache miss — fetching TLT + SPY closes once via yfinance)")
    data.fetch_market()

panel = data.load_panel()
mk = data.load_market()
closes = pd.DataFrame({s: panel[s]["Close"] for s in data.UNIVERSE if s in panel})
print(f"[data] {len(panel)} names, {len(closes)} rows  "
      f"{closes.index.min().date()} -> {closes.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint(Close)={fingerprint(closes)}")
print(f"[cond] TLT + SPY: {len(mk)} rows  {mk.index.min().date()} -> {mk.index.max().date()}")
print("  SURVIVORSHIP: current-membership mega-cap panel — magnitudes are an upper bound. "
      "Named on the Signal axis.")

ret = st.close_returns(panel)
tlt, spy = st.market_returns(mk)

sp = st.ftq_spreads(ret, tlt, spy)
h = st.ftq_stats(sp)
print(f"\nsort: trailing-252d FTQ beta (TLT beta on down-SPY days), monthly rebalance, "
      f"long bottom20% / short top20%, {h['n_months']} months (median {int(sp['n'].median())} names/mo)")
print("# THE HEADLINE — long-low-FTQ / short-high-FTQ spread (the pay-for-the-hedge premium)")
print(f"  spread: {h['spread_pct']:+.4f} %/mo  ({h['spread_ann_pct']:+.2f} %/yr)  "
      f"NW(6) t = {h['t_nw']:+.2f}  one-sample t = {h['t_1s']:+.2f}")
print(f"  books : low-FTQ {h['lo_pct']:+.4f} vs high-FTQ {h['hi_pct']:+.4f} %/mo  "
      f"(Welch t = {h['welch_t']:+.2f})")

print("\n# PLACEBO — permute the forward returns within each month (1,000 draws)")
pl = st.placebo_pvalue(ret, tlt, spy, n_draws=1000)
sigma = (pl["obs_pct"] - pl["placebo_mean_pct"]) / pl["placebo_sd_pct"]
print(f"  observed {pl['obs_pct']:+.4f} %/mo vs placebo mean {pl['placebo_mean_pct']:+.4f} "
      f"(sd {pl['placebo_sd_pct']:.4f}) over {pl['n_draws']:,} draws -> right-tail p = {pl['p_value']:.4f}  "
      f"(~{sigma:+.2f}σ)")

print("\n# ROBUSTNESS — two eras (split 2018-01-01)")
for lo, hi, lbl in [("2010-01-01", "2018-01-01", "2011-2017"),
                    ("2018-01-01", "2026-07-01", "2018-2026")]:
    sub = sp[(sp.index >= lo) & (sp.index < hi)]
    ts = st.ftq_stats(sub)
    print(f"  {lbl}: n={ts['n_months']}  spread {ts['spread_pct']:+.4f} %/mo "
          f"(NW t={ts['t_nw']:+.2f})")

print("\n# CRASH PROTECTION — the *other* half of the claim (worst 5% of SPY days)")
cp = st.crash_protection(ret, tlt, spy)
print(f"  {cp['n_crash_days']} crash days, mean SPY {cp['spy_crash_pct']:+.2f}%: "
      f"low-FTQ book {cp['lo_book_crash_pct']:+.3f}% vs high-FTQ book {cp['hi_book_crash_pct']:+.3f}%")
print(f"  high-minus-low cushion on crash days: {cp['hi_minus_lo_crash_pct']:+.3f}%/day "
      f"(Welch t = {cp['crash_welch_t']:+.2f})   [all days: {cp['all_days_hi_minus_lo_pct']:+.3f}%/day]")

print("\n# THE TIMER — can you get paid the pay-for-the-hedge premium?")
print("  2 legs x round-trip x one-way x NAV per month; short book pays 50 bps/yr borrow")
for cb in (1.0, 10.0):
    tm = st.timer_stats(sp, one_way_bps=cb, borrow_bps_yr=50.0)
    print(f"  cost={cb:>4.1f} bps/side: gross {tm['gross_pct']:+.4f} -> net {tm['net_pct']:+.4f} "
          f"%/mo (cost {tm['cost_pct_per_month']:.4f}/mo, t = {tm['t_net']:+.2f}, "
          f"Sharpe {tm['sharpe_net']:.2f}, ~{tm['ann_net_pct']:+.2f}%/yr)")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_t = []
for s_ in range(20):
    w0 = data.synthetic_panel(edge=0.0, seed=866 + s_, n_assets=40, n_days=1400)
    null_t.append(st.synthetic_detect(w0, min_stocks=10)["t_nw"])
null_t = np.asarray(null_t)
print(f"  null (edge=0), 20 seeds: spread NW t mean {null_t.mean():+.2f} "
      f"(sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t) >= 2).sum()}/20 seeds")
w1 = data.synthetic_panel(edge=0.004, seed=866, n_assets=40, n_days=1500)
sy = st.synthetic_detect(w1, min_stocks=10)
print(f"  planted (edge=0.004, seed 866): spread NW t = {sy['t_nw']:+.2f}, "
      f"Welch t = {sy['welch_t']:+.2f}")
