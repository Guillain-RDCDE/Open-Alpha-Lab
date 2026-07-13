"""Reproducible headline run for Study 744 — Tetraphobia.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached raw local-currency closes
(clustering) and total-return ETF closes (calendar) under ``_cache/`` (fetching once on
a cache miss), and always runs the two synthetic controls with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from tetraphobia import data as dt, strategy as st  # noqa: E402

print("# Tetraphobia — does the fear of '4' leave a footprint in prices AND in returns?")

if not dt.have_real():
    print("(cache miss — fetching raw clustering basket + calendar ETFs once)")
    dt.fetch()

cl = dt.load_cluster()
ca = dt.load_calendar()
panel = dt.calendar_panel()
print(data_stamp("Tetraphobia calendar panel (6 China-sphere ETFs)", panel, asof=dt.AS_OF))

# ---- A · price clustering -------------------------------------------------- #
asia = {t: s for t, s in cl.items() if t in dt.ASIA_CLUSTER}
usc = {t: s for t, s in cl.items() if t in dt.US_CONTROL}
sa = st.tetraphobia_stats(st.trailing_digit_counts(asia))
su = st.tetraphobia_stats(st.trailing_digit_counts(usc))
print("\n# A · TRAILING-DIGIT CLUSTERING (raw local-currency closes; round digits 0/5 excluded)")
print(f"  Asia basket ({sa['total']:,} closes): ending-4 {sa['n4']:,}  ending-8 {sa['n8']:,}  "
      f"8-share {sa['share8']*100:.1f}%  z(8>4)={sa['z8_gt_4']:+.2f}  z(digit4 vs non-round)={sa['z4']:+.2f}")
print(f"  US control  ({su['total']:,} closes): ending-4 {su['n4']:,}  ending-8 {su['n8']:,}  "
      f"8-share {su['share8']*100:.1f}%  z(8>4)={su['z8_gt_4']:+.2f}  z(digit4 vs non-round)={su['z4']:+.2f}")
print("  by region:")
for r, d in st.region_tetraphobia(cl).items():
    if np.isfinite(d["z8_gt_4"]):
        print(f"    {r:9s} 8-share {d['share8']*100:5.1f}%  z(8>4)={d['z8_gt_4']:+.2f}  (N={d['total']:,})")
    else:
        print(f"    {r:9s} (no sub-unit trailing digit — won-priced; excluded)")

# ---- B · the 4/4 calendar-returns test ------------------------------------- #
print("\n# B · THE 4/4 CALENDAR-RETURNS TEST (one-sample t across years, total-return)")
m44 = st.market_date_stats(ca, 4, 4, tickers=dt.CALENDAR_CORE)
for t in dt.CALENDAR_CORE + ["POOLED"]:
    d = m44[t]
    print(f"  {t:7s} n={d['n']:2d}  mean={d['mean']*1e4:+6.1f} bps  t={d['t']:+.2f}  "
          f"hit {d['k']}/{d['n']} (Wilson [{d['lo']*100:.0f}%, {d['hi']*100:.0f}%])")
m44e = st.market_date_stats(ca, 4, 4, tickers=dt.CALENDAR_ALL)["POOLED"]
print(f"  pooled all 6 ETFs: n={m44e['n']}  mean={m44e['mean']*1e4:+.1f} bps  t={m44e['t']:+.2f}")

obs = m44["POOLED"]["mean"]
n_per = {t: m44[t]["n"] for t in dt.CALENDAR_CORE}
pl = st.placebo_pvalue(ca, obs, n_per, tickers=dt.CALENDAR_CORE)
print(f"\n  random-calendar placebo (20 seeds x 250 draws, left tail = predicted underperformance):")
print(f"    observed {obs*1e4:+.1f} bps  vs placebo mean {pl['placebo_mean']*1e4:+.1f} bps "
      f"(sd {pl['placebo_sd']*1e4:.1f} bps) over {pl['n_draws']:,} draws  ->  p = {pl['p_value']:.3f}")

print("\n# The 8/8 'lucky date' contrast")
m88 = st.market_date_stats(ca, 8, 8, tickers=dt.CALENDAR_CORE)["POOLED"]
w = st.welch_t(st.pooled_returns(ca, 8, 8), st.pooled_returns(ca, 4, 4))
print(f"  4/4 pooled {obs*1e4:+.1f} bps (t={m44['POOLED']['t']:+.2f})   "
      f"8/8 pooled {m88['mean']*1e4:+.1f} bps (t={m88['t']:+.2f})   Welch (8/8-4/4) = {w:+.2f}")

# ---- Tradability ----------------------------------------------------------- #
print("\n# TRADABILITY — short the 4/4 session, net of 2x5bps + 1bp borrow")
tr = st.short_the_unlucky_day(ca, tickers=dt.CALENDAR_CORE)
print(f"  n={tr['n']}  gross {tr['gross_mean']*1e4:+.1f} bps/event  net {tr['mean']*1e4:+.1f} bps/event  "
      f"t={tr['t']:+.2f}  (negative = the short bled money; the day tended to rise)")

# ---- synthetic controls ---------------------------------------------------- #
print("\n# Synthetic positive controls — deterministic, no network")
nd = np.array([st.synthetic_digit_detect(0.0, seed=744 + s)["z8_gt_4"] for s in range(20)])
print(f"  digit detector null (bias=0), 20 seeds: mean z(8>4)={nd.mean():+.2f} (sd {nd.std(ddof=1):.2f}), "
      f"|z|>=2 in {(abs(nd) >= 2).sum()}/20")
print(f"    planted bias=0.03 -> z={st.synthetic_digit_detect(0.03)['z8_gt_4']:+.2f}   "
      f"bias=0.05 -> z={st.synthetic_digit_detect(0.05)['z8_gt_4']:+.2f}")
nc = np.array([st.synthetic_calendar_detect(0.0, seed=744 + s)["t"] for s in range(20)])
print(f"  4/4 detector null (dip=0), 20 seeds: mean t={nc.mean():+.2f} (sd {nc.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(nc) >= 2).sum()}/20")
print(f"    planted dip=-1% -> t={st.synthetic_calendar_detect(-0.01)['t']:+.2f}   "
      f"dip=-2% -> t={st.synthetic_calendar_detect(-0.02)['t']:+.2f}")

print("\n# VERDICT")
print("  Signal:      NONE   -- 4/4 does not underperform (pooled +16.2bps, t=+0.69, wrong sign,")
print("                         placebo p=0.748); no calendar-return footprint of tetraphobia.")
print("  Tradability: MIRAGE -- shorting 4/4 loses -27.2 bps/event net (t=-1.16).")
print("  Do Asian prices dodge the 4? CONFIRMED -- Asia z(8>4)=+4.73 (Taiwan +5.25, China +3.62),")
print("                         US control flat (-0.84). The superstition shapes prices, not returns.")
