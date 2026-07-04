"""Reproducible headline run for Study 629 — Congress Trading.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached Senate Stock Watcher purchase
events + yfinance price panel under ``_cache/`` if present (the real-tape numbers), and
always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from congress_trading import data, strategy as st  # noqa: E402
from quantlab import repro  # noqa: E402

print("# Congress Trading — do senators' disclosed stock purchases beat the market?")
if data.have_real():
    px, spy, ev = data.load_real()
    full = __import__("pandas").read_parquet(data.PRICES_CACHE)
    n_raw = full.shape[1] - 1  # minus SPY
    dropped = sorted(set(full.columns) - set(px.columns) - {"SPY"})
    lag = (ev["disclosure_date"] - ev["tx_date"]).dt.days

    print(repro.data_stamp("senate purchases + price panel", px, asof=None))
    print(f"events          : {len(ev):,} disclosed stock purchases, "
          f"{ev['disclosure_date'].min().date()} -> {ev['disclosure_date'].max().date()} "
          f"(disclosure dates), {ev['senator'].nunique()} senators")
    print(f"party mix       : R {int((ev['party'] == 'R').sum()):,} / D {int((ev['party'] == 'D').sum()):,} "
          f"| famous subset (Perdue, Loeffler, Inhofe): {int(ev['famous'].sum()):,} events "
          f"({ev['famous'].mean() * 100:.1f}%)")
    print(f"top senator     : David Perdue = {(ev['senator'] == 'David A Perdue , Jr').mean() * 100:.1f}% of all events")
    print(f"reporting lag   : median {lag.median():.0f} days, mean {lag.mean():.1f} days (trade -> public disclosure)")
    print(f"price panel     : {n_raw} tickers resolved on Yahoo; {len(dropped)} dropped by the "
          f"recycled-ticker integrity guard ({', '.join(dropped)}); {px.shape[1]} used + SPY")

    abn = st.event_abnormal(px, spy, ev)
    cov = len(abn) / len(ev)
    print(f"event coverage  : {len(abn):,} of {len(ev):,} events ({cov * 100:.1f}%) have a resolvable "
          f"ticker + a full 12-month forward window (the rest = delisted/renamed tickers -> "
          f"SURVIVORSHIP, named on the Signal axis)")

    print("\n# Event study (color only — windows overlap, the naive t OVERSTATES)")
    for row in st.pooled_table(abn):
        print(f"  {row['h']:>3} td (~{row['h'] // 21:>2} mo): mean abnormal {row['mean_pct']:+.2f}% "
              f"vs SPY   naive t = {row['naive_t']:+.2f}   n = {row['n']:,}")

    print("\n# Calendar-time portfolio (PRIMARY — the verdict statistic, Newey-West t, 10 lags)")
    print("  buy at the first close AFTER disclosure (the one execution lag), hold `hold` td, "
          "equal-weight, excess vs SPY (both total-return), gross:")
    for hold in (63, 126, 252):
        r = st.calendar_time(px, spy, ev, hold=hold)
        print(f"  hold={hold:>3}: alpha {r['alpha_ann_pct']:+.2f}%/yr   NW t = {r['nw_t']:+.2f}   "
              f"({r['n_days']:,} days, avg {r['avg_names']:.0f} names held)")

    print("\n# Random-dates placebo (same tickers, random disclosure dates; 20 seeds — the")
    print("# 'is it the TIMING or just their stock list?' test, hold=126)")
    pl = st.placebo_random_dates(px, spy, ev, hold=126, n_seeds=20)
    r126 = st.calendar_time(px, spy, ev, hold=126)
    z = (r126["alpha_ann_pct"] - pl["mean_alpha_ann_pct"]) / pl["sd_alpha_ann_pct"]
    print(f"  placebo alpha : {pl['mean_alpha_ann_pct']:+.2f}%/yr (sd {pl['sd_alpha_ann_pct']:.2f}) "
          f"| placebo NW t: mean {pl['mean_t']:+.2f} (sd {pl['sd_t']:.2f})")
    print(f"  real - placebo: {r126['alpha_ann_pct'] - pl['mean_alpha_ann_pct']:+.2f}%/yr "
          f"= {z:+.2f} placebo-sd -> the disclosure DATE adds nothing beyond the stock list")

    print("\n# Costs (one-way bps x 2 legs spread over the 126-td hold; long-only, no borrow)")
    for cb in (0.0, 5.0, 10.0):
        r = st.calendar_time(px, spy, ev, hold=126, cost_bps_oneway=cb)
        print(f"  cost={cb:>4.1f} bps one-way: net alpha {r['alpha_ann_pct']:+.2f}%/yr   NW t = {r['nw_t']:+.2f}")

    print("\n# Splits (Welch t on per-event abnormal returns — color)")
    for h in (126, 252):
        for label, mask in [("famous vs rest", abn["famous"]),
                            ("R vs D", abn["party"] == "R"),
                            ("big (>=$15k) vs small", ~abn["amount"].str.startswith("$1,001"))]:
            r = st.split_welch(abn, mask, h=h)
            print(f"  h={h} {label:22s}: {r['mean_a_pct']:+.2f}% (n={r['n_a']:,}) vs "
                  f"{r['mean_b_pct']:+.2f}% (n={r['n_b']:,})  diff {r['diff_pct']:+.2f} pp  "
                  f"Welch t = {r['welch_t']:+.2f}")

    print("\n# Third axis — the famous names (the meme) vs the boring aggregate, calendar-time")
    for label, sub in [("FAMOUS (Perdue/Loeffler/Inhofe)", ev[ev["famous"]]),
                       ("BORING (the other 23 senators) ", ev[~ev["famous"]])]:
        r = st.calendar_time(px, spy, sub, hold=126)
        print(f"  {label}: alpha {r['alpha_ann_pct']:+.2f}%/yr   NW t = {r['nw_t']:+.2f}   "
              f"(avg {r['avg_names']:.0f} names)")
    rEx = st.calendar_time(px, spy, ev[ev["senator"] != "David A Perdue , Jr"], hold=126)
    print(f"  ex-Perdue aggregate            : alpha {rEx['alpha_ann_pct']:+.2f}%/yr   NW t = {rEx['nw_t']:+.2f}")
else:
    print("(no _cache — run data.fetch_events() then data.fetch_prices() once to build it)")

print("\n# Synthetic control — deterministic, no network (machinery proof only, never evidence)")
ts = []
for s in range(20):
    spx, sspy, sev = data.synthetic_world(edge_daily=0.0, seed=629 + s)
    ts.append(st.calendar_time(spx, sspy, sev, hold=126)["nw_t"])
print(f"  null (edge=0), 20 seeds : mean NW t = {np.mean(ts):+.2f}  sd = {np.std(ts):.2f}  "
      f"max |t| = {np.max(np.abs(ts)):.2f}  (no manufactured significance)")
spx, sspy, sev = data.synthetic_world(edge_daily=0.0006, seed=629)
r = st.calendar_time(spx, sspy, sev, hold=126)
print(f"  planted +6 bps/day drift: alpha {r['alpha_ann_pct']:+.2f}%/yr  NW t = {r['nw_t']:+.2f}  (lights up)")
