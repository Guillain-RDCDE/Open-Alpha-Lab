"""Reproducible headline run for Study 641 — Sell in May (the Halloween indicator).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached ^GSPC / SPY / ^SP500TR / ^IRX
tapes under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from sell_in_may import data, strategy as st  # noqa: E402

print("# Sell in May — does the Nov->Apr half really trounce May->Oct?")

if not data.have_real():
    print("(cache miss — fetching ^GSPC / SPY / ^SP500TR / ^IRX once)")
    data.fetch()

raw = data.load_real()
for name, df in raw.items():
    print(data_stamp(name, df, asof=data.AS_OF))

mm = data.load_monthly()
irx = mm["irx_pct"]
print(f"\nmonthly points: ^GSPC {len(mm['gspc'])} (1950->), SPY {len(mm['spy'])} (1993->), "
      f"^SP500TR {'absent' if 'sp500tr' not in mm else len(mm['sp500tr'])} (1988->), "
      f"^IRX cash proxy {len(irx)} (1960->)")

print("\n# THE HEADLINE — winter (Nov->Apr) vs summer (May->Oct) monthly log return")
for key, label in (("gspc", "^GSPC (price-only, 1950-2026, deep history)"),
                   ("sp500tr", "^SP500TR (total-return, 1988-2026)"),
                   ("spy", "SPY (dividend-adjusted, 1993-2026)")):
    if key not in mm:
        continue
    s = st.headline_split(mm[key])
    print(f"  {label}")
    print(f"    winter {s['winter_pct']:+.3f}%/mo  vs summer {s['summer_pct']:+.3f}%/mo   "
          f"gap {s['gap_pct']:+.3f}%/mo")
    print(f"    6-month cumulative: winter {s['winter_half_pct']:+.2f}%  vs summer "
          f"{s['summer_half_pct']:+.2f}%")
    print(f"    Welch t = {s['welch_t']:+.2f}   Newey-West(3) t = {s['nw_t']:+.2f}   "
          f"(n_winter={s['n_winter']}, n_summer={s['n_summer']})")

print("\n# Year-block pairing — one (summer, winter) point per Halloween year")
for key, label in (("gspc", "^GSPC"), ("sp500tr", "^SP500TR"), ("spy", "SPY")):
    if key not in mm:
        continue
    pairs = st.halloween_year_pairs(mm[key])
    sgn = st.sign_test_stats(pairs)
    bs = st.year_block_bootstrap(pairs)
    print(f"  {label}: {sgn['n']} Halloween years, winter beat summer in "
          f"{sgn['k_winter_wins']}/{sgn['n']} = {sgn['hit_rate']*100:.1f}% "
          f"(Wilson [{sgn['hit_lo']*100:.1f}%, {sgn['hit_hi']*100:.1f}%], "
          f"sign-test p = {sgn['p_value']:.4f})")
    print(f"    year-block bootstrap (10,000 draws): mean gap {bs['mean_gap_pct']:+.2f}% "
          f"(t = {bs['t_analytic']:+.2f}), 95% CI [{bs['boot_lo_pct']:+.2f}%, "
          f"{bs['boot_hi_pct']:+.2f}%], P(gap<=0) = {bs['boot_p_le0']:.4f}")

print("\n# By calendar month — ^GSPC 1950-2026 (mean %, n)")
byc = st.by_calendar_month(mm["gspc"])
names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
for i, row in enumerate(byc.itertuples(), start=1):
    print(f"  {names[i-1]}: {row.mean_pct:+.3f}%  (n={int(row.count)})")

print("\n# 'A handful of bad autumns' — drop the 5 worst Septembers + 5 worst Octobers")
gx = st.gap_excluding_worst_months(mm["gspc"], k=5)
print(f"  full sample gap {gx['full_gap_pct']:+.3f}%/mo  Welch t = {gx['full_welch_t']:+.2f}")
print(f"  after dropping {gx['n_dropped']} months ({gx['n_dropped']}/917 = "
      f"{gx['n_dropped']/917*100:.1f}% of the sample): gap {gx['trimmed_gap_pct']:+.3f}%/mo  "
      f"Welch t = {gx['trimmed_welch_t']:+.2f}")
print(f"  dropped: {', '.join(gx['dropped_dates'])}")

print("\n# THE HALLOWEEN TIMER — long Nov->Apr, cash (^IRX) May->Oct, vs buy & hold")
print("  (2 x one-way cost x NAV per switch, 2 switches/yr; Sharpe is EXCESS-of-cash both legs)")
for key, label in (("sp500tr", "^SP500TR"), ("spy", "SPY"), ("gspc", "^GSPC price-only (caveat below)")):
    if key not in mm:
        continue
    for cb in (5.0, 10.0):
        r = st.halloween_timer(mm[key], irx, cost_bps=cb)
        print(f"  {label} @ {cb:.0f} bps: timer CAGR {r['cagr_pct']:+.2f}% vs buy&hold "
              f"{r['bh_cagr_pct']:+.2f}%   excess-Sharpe {r['sharpe_excess']:.2f} vs "
              f"{r['bh_sharpe_excess']:.2f}   maxDD {r['max_dd_pct']:.1f}% vs "
              f"{r['bh_max_dd_pct']:.1f}%   ({r['one_way_trades_per_yr']:.1f} one-way "
              f"trades/yr, {r['n_years']:.1f}y)")
rev = st.halloween_timer(mm["sp500tr"], irx, cost_bps=5.0, reverse=True)
print(f"  reverse strawman (long May->Oct, cash Nov->Apr), ^SP500TR: CAGR "
      f"{rev['cagr_pct']:+.2f}% vs buy&hold {rev['bh_cagr_pct']:+.2f}%   "
      f"excess-Sharpe {rev['sharpe_excess']:.2f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the Welch detector must NOT systematically fire on a null world (premium=0) and must")
print("  recover a planted winter premium. Null checked over 20 seeds.")
null_ts = []
for s_ in range(20):
    ret = data.synthetic_world(premium_bp=0.0, seed=641 + s_)
    null_ts.append(st.synthetic_detect(ret)["welch_t"])
null_ts = np.asarray(null_ts)
print(f"  null (premium=0), 20 seeds: mean Welch t = {null_ts.mean():+.2f} "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
ret = data.synthetic_world(premium_bp=40.0, seed=641)
sy = st.synthetic_detect(ret)
print(f"  planted premium=+40 bps/mo (seed 641): winter {sy['winter_pct']:+.3f}% vs summer "
      f"{sy['summer_pct']:+.3f}%  Welch t = {sy['welch_t']:+.2f}")
