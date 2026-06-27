"""Reproducible headline run for Study 517 — Pre-FOMC-Drift.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY + basket prices under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control with
no network.

    python examples/verify.py

The stamped window is capped at the last COMPLETE month (no partial months in a stamped run);
as-of <= 2026-06-26.
"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pre_fomc_drift import data, strategy as st

NDRAWS = 20_000
ASOF = "2026-05-31"          # last complete month (no partial-month bias); <= 2026-06-26

print("# Pre-FOMC drift (Lucca-Moench) — SPY + 30-name survivor basket (yfinance)")

if data.have_real():
    prices, ohlc = data.load_real()
    prices = prices[prices.index <= ASOF]
    ohlc = ohlc[ohlc.index <= ASOF]

    spy = prices["SPY"].dropna()
    ret = st.daily_returns(spy)
    mask = data.tag_pre_fomc(ret.index, shift=0)

    span = (ret.index.max() - ret.index.min()).days / 365.25
    print(f"SPY tape       : {len(ret)} daily returns  "
          f"{ret.index.min().date()} -> {ret.index.max().date()}  ({span:.1f} years)")
    print(f"FOMC meetings  : {len(data.fomc_calendar())} scheduled (hardcoded calendar)")
    print(f"pre-FOMC days  : {int(mask.sum())} tagged sessions "
          f"({mask.mean()*100:.1f}% of all sessions)")

    fp = hashlib.sha1(
        np.ascontiguousarray(spy.round(4).values).tobytes()
        + str(len(mask)).encode()).hexdigest()[:12]
    print(f"fingerprint    : {fp}")

    print("\n# Pre-FOMC day vs all other days — full sample (SPY close-to-close)")
    full = st.event_vs_rest(ret.values, mask)
    print(f"  pre-FOMC mean  : {full['ev_mean']*100:+.4f}%/day  (one-sample t0={full['ev_t0']:.2f})")
    print(f"  other-day mean : {full['rest_mean']*100:+.4f}%/day")
    print(f"  Welch t (ev-rest): {full['welch_t']:.2f}")
    print(f"  share of total cumulative return on {full['frac_days']*100:.1f}% of days: "
          f"{full['share_of_total']*100:.1f}%")

    pl = st.placebo_pvalue(ret.values, mask, n_draws=NDRAWS)
    print(f"  placebo p (random {full['n_event']}-day calendars >= obs): {pl['p_value']:.3f}")

    print("\n# Publication-decay split (Lucca-Moench published ~2015; split at "
          f"{st.split_summary(ret, mask)['split']})")
    sp = st.split_summary(ret, mask)
    for key in ("pre", "post"):
        s = sp[key]
        lab = "pre-2012 " if key == "pre" else "post-2012"
        print(f"  {lab}: ev={s['ev_mean']*100:+.4f}%/day  other={s['rest_mean']*100:+.4f}%/day  "
              f"Welch t={s['welch_t']:.2f}  share={s['share_of_total']*100:.1f}%  "
              f"(n_ev={s['n_event']})")

    print("\n# Overnight vs intraday decomposition (the 14:15 run-up question)")
    oi = st.overnight_intraday(ohlc)
    oi = oi[oi.index.isin(ret.index)]
    m2 = data.tag_pre_fomc(oi.index, shift=0)
    for comp in ("overnight", "intraday", "daily"):
        s = st.event_vs_rest(oi[comp].values, m2)
        print(f"  {comp:>9}: pre-FOMC={s['ev_mean']*100:+.4f}%  other={s['rest_mean']*100:+.4f}%  "
              f"Welch t={s['welch_t']:.2f}")

    print("\n# Tradability — hold SPY only on pre-FOMC days, 1 bp one-way x 2 legs/meeting")
    for cb in (0.5, 1.0, 2.0):
        tr = st.timing_strategy(ret.values, mask, cost_bps=cb)
        print(f"  cost={cb:>3} bp: gross={tr['gross_mean']*100:+.4f}%  net={tr['net_mean']*100:+.4f}%  "
              f"net t0={tr['net_t0']:.2f}")

    print("\n# Robustness — tag the session STRICTLY BEFORE the announcement (shift=-1)")
    m_shift = data.tag_pre_fomc(ret.index, shift=-1)
    s = st.event_vs_rest(ret.values, m_shift)
    print(f"  shift=-1: ev={s['ev_mean']*100:+.4f}%/day  Welch t={s['welch_t']:.2f}  "
          f"share={s['share_of_total']*100:.1f}%")
else:
    print("(no _cache/pfd_prices.csv — run data.fetch_panel() once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network (seed-averaged, >=20 seeds)")
print("  the pre-vs-other test must recover a PLANTED pre-meeting drift and must NOT")
print("  manufacture significance when the true edge is 0.")
for edge in (0.0, 0.004):
    ts = []
    for seed in range(40):
        r, m = data.synthetic_world(edge=edge, seed=seed)
        ts.append(st.event_vs_rest(r.values, m)["welch_t"])
    ts = np.array(ts)
    print(f"  planted edge={edge*100:+.2f}%: mean Welch t={ts.mean():+.2f}  "
          f"frac |t|>2 = {np.mean(np.abs(ts) > 2):.2f}  (40 seeds)")
