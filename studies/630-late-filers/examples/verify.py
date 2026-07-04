"""Reproducible headline run for Study 630 — Late Filers (Form NT).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached EDGAR NT-filing panel + yfinance
prices under ``_cache/`` if present (the real-tape numbers), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd

from late_filers import data, strategy as st

try:
    from quantlab.repro import fingerprint
except Exception:                                                  # pragma: no cover
    fingerprint = None

print("# Late Filers (Form NT) — EDGAR NT 10-K / NT 10-Q event study vs SPY")
if data.have_real():
    nt = data.load_nt_filings()
    tick = data.load_cik_tickers()
    events, prices = data.load_real()
    cars = st.build_event_cars(events, prices)

    in_win = nt[(nt["date_filed"] >= data.EVENT_START) & (nt["date_filed"] <= data.EVENT_END)]
    mapped = in_win.merge(tick[["cik"]], on="cik", how="inner")
    print(f"NT filings (index) : {len(nt):,} rows {nt['date_filed'].min().date()} -> "
          f"{nt['date_filed'].max().date()}  "
          f"(NT 10-K {int((nt['form'] == 'NT 10-K').sum()):,} / NT 10-Q {int((nt['form'] == 'NT 10-Q').sum()):,})")
    print(f"event window       : {data.EVENT_START} -> {data.EVENT_END}: {len(in_win):,} filings; "
          f"{len(mapped):,} ({len(mapped) / len(in_win) * 100:.1f}%) map to a CURRENTLY-registered ticker "
          f"(the survivorship gate — delisted offenders drop out)")
    print(f"panel (capped)     : {len(events):,} events, {events['ticker'].nunique()} tickers "
          f"(seeded caps {data.MAX_TICKERS} tickers / {data.MAX_EVENTS} events)")
    print(f"usable events      : {len(cars):,} after the stated screens (entry price >= $1, "
          f"no |daily move| > 100% in-window, >= 80% coverage) "
          f"({int((cars['form'] == 'NT 10-K').sum())} NT 10-K / {int((cars['form'] == 'NT 10-Q').sum())} NT 10-Q; "
          f"{int(cars['repeat'].sum())} repeat offenders)")
    print(f"prices             : {prices.shape[0]:,} days x {prices.shape[1]} names, "
          f"{prices.index.min().date()} -> {prices.index.max().date()}  as-of {data.AS_OF}")
    if fingerprint is not None:
        ev_fp = events.assign(d=events["date_filed"]).set_index("d")[["cik"]]
        print(f"fingerprint        : events {fingerprint(ev_fp)}  "
              f"spy {fingerprint(prices[['SPY']].dropna())}")

    print("\n# Announcement reaction — filing day to entry close (NOT tradable, shown for context)")
    cs = st.cross_sectional_stats(cars)
    print(f"  mean reaction  : {cs['reaction_bps']:+.1f} bps  (naive t = {cs['t_reaction']:+.2f}, "
          f"n = {cs['n']:,})")

    print("\n# Post-filing drift — short at next close, hold 60 trading days, vs SPY")
    ct = st.calendar_time_portfolio(cars, prices)
    print(f"  mean CAR[+1,+60td]        : {cs['car_bps']:+.1f} bps  (naive cross-sectional "
          f"t = {cs['t_naive']:+.2f}; windows overlap -> over-counted, not the primary test)")
    print(f"  calendar-time (PRIMARY)   : {ct['mean_bps_day']:+.2f} bps/day vs SPY, "
          f"HAC t = {ct['hac_t']:+.2f}  ({ct['ann_pct']:+.1f}%/yr; {ct['n_days']:,} active days, "
          f"avg {ct['avg_names']:.1f} names/day)")

    print("\n# Matched self-control — same firms, 252 trading days earlier (NT-free windows)")
    pse = st.pseudo_events(events, prices)
    pcars = st.build_event_cars(pse, prices)
    wt, m_ev, m_ps = st.welch_t(cars["car"].to_numpy(), pcars["car"].to_numpy())
    print(f"  event CAR {m_ev * 1e4:+.1f} bps (n={len(cars):,}) vs pseudo CAR {m_ps * 1e4:+.1f} bps "
          f"(n={len(pcars):,}):  Welch t = {wt:+.2f}")

    print("\n# Random-dates placebo — same tickers, uniform random dates, averaged over 25 seeds")
    rp = st.random_dates_placebo(events, prices, n_seeds=25)
    print(f"  mean CAR across {rp['n_seeds']} seeds: {rp['mean_bps']:+.1f} bps "
          f"(across-seed std {rp['std_bps']:.1f} bps)")

    print("\n# Split by form — is the 10-K miss worse than the 10-Q miss?")
    for form in ("NT 10-K", "NT 10-Q"):
        sub = cars[cars["form"] == form]
        c = st.cross_sectional_stats(sub)
        k = st.calendar_time_portfolio(sub, prices)
        print(f"  {form:8s}: CAR {c['car_bps']:+.1f} bps (n={c['n']:,}, naive t={c['t_naive']:+.2f})  "
              f"cal-time {k['mean_bps_day']:+.2f} bps/d, HAC t = {k['hac_t']:+.2f}")

    print("\n# Third axis — is the SECOND consecutive NT the real kill signal?")
    rep = cars[cars["repeat"]]
    fst = cars[~cars["repeat"]]
    wt2, m_rep, m_fst = st.welch_t(rep["car"].to_numpy(), fst["car"].to_numpy())
    kr = st.calendar_time_portfolio(rep, prices)
    kf = st.calendar_time_portfolio(fst, prices)
    print(f"  repeat offenders : CAR {m_rep * 1e4:+.1f} bps (n={len(rep):,})  "
          f"cal-time HAC t = {kr['hac_t']:+.2f}")
    print(f"  first offenses   : CAR {m_fst * 1e4:+.1f} bps (n={len(fst):,})  "
          f"cal-time HAC t = {kf['hac_t']:+.2f}")
    print(f"  repeat - first   : {(m_rep - m_fst) * 1e4:+.1f} bps  Welch t = {wt2:+.2f}")

    print("\n# Robustness — window length (calendar-time HAC t)")
    for w in (20, 40, 60):
        cw = st.build_event_cars(events, prices, window=w)
        kw = st.calendar_time_portfolio(cw, prices, window=w)
        c = st.cross_sectional_stats(cw)
        print(f"  window={w:2d} td: CAR {c['car_bps']:+.1f} bps (n={c['n']:,})  "
              f"cal-time {kw['mean_bps_day']:+.2f} bps/d, HAC t = {kw['hac_t']:+.2f}")

    print("\n# Tradability — the short pays the spread twice AND the borrow (60-td hold)")
    for cost in (10.0, 20.0):
        for borrow in (3.0, 10.0):
            net = st.short_net(cs["car_bps"], cost, borrow)
            print(f"  cost {cost:4.0f} bps one-way, borrow {borrow:4.1f}%/yr: "
                  f"short gross {-cs['car_bps']:+.1f} bps -> net {net:+.1f} bps per event")
else:
    print("(caches missing — run data.fetch_nt_filings(), data.fetch_cik_tickers(),")
    print(" then data.fetch_prices(build_events()['ticker'].unique().tolist()) once)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the CAR / calendar-time machinery must recover a PLANTED post-NT drift and must NOT")
print("  manufacture significance when the planted drift is 0. (Machinery proof only — never")
print("  cited as market evidence.)")
for edge in (0.0, -20.0):
    ev, px = data.synthetic_world(edge_bps_day=edge, seed=630)
    cs2 = st.cross_sectional_stats(st.build_event_cars(ev, px))
    ct2 = st.calendar_time_portfolio(st.build_event_cars(ev, px), px)
    print(f"  planted drift {edge:+6.1f} bps/day: CAR {cs2['car_bps']:+8.1f} bps "
          f"(naive t {cs2['t_naive']:+6.2f})  cal-time HAC t = {ct2['hac_t']:+6.2f}")
