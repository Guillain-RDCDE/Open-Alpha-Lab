"""Reproducible headline run for Study 602 — Macro-Announcement-Day Premium.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY + TLT closes under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control with
no network.

    python examples/verify.py

The stamped window is 1997-01-02 -> 2026-06-30 (as-of the last complete month; the partial
current month is dropped).
"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from macro_announcement_premium import data, strategy as st

NDRAWS = 20_000
START = "1997-01-01"
ASOF = "2026-06-30"          # last complete month; daily study, no partial-month bias

print("# Macro-announcement-day premium (Savor-Wilson) — SPY + TLT, hardcoded FOMC/CPI/NFP calendar")

if data.have_real():
    px = data.load_real()
    px = px[(px.index >= "1996-12-01") & (px.index <= ASOF)]
    spy = px["SPY"].dropna()
    ret = st.daily_returns(spy)
    ret = ret[ret.index >= START]
    masks = data.announcement_masks(ret.index)
    any_m, fomc, cpi, nfp = masks["ANY"], masks["FOMC"], masks["CPI"], masks["NFP"]
    pure_rest = ~any_m

    span = (ret.index.max() - ret.index.min()).days / 365.25
    cals = data.calendars()
    in_win = {k: int(((v >= ret.index.min()) & (v <= ret.index.max())).sum())
              for k, v in cals.items()}
    fp = hashlib.sha1(np.ascontiguousarray(spy.round(4).values).tobytes()
                      + str(len(ret)).encode()).hexdigest()[:12]

    print(f"SPY tape       : {len(ret)} daily returns  "
          f"{ret.index.min().date()} -> {ret.index.max().date()}  ({span:.1f} years)")
    print(f"fingerprint    : {fp}")
    print(f"calendar (in-window releases): FOMC {in_win['FOMC']}  CPI {in_win['CPI']}  "
          f"NFP {in_win['NFP']}  (total {sum(in_win.values())})")
    print(f"A-days         : {int(any_m.sum())} distinct announcement sessions "
          f"({any_m.mean()*100:.1f}% of all sessions); "
          f"{masks['n_overlap']} carry 2 announcement types; "
          f"{masks['n_mapped']} holiday releases mapped to the next session")

    print("\n# Calendar spot-checks (construction sanity)")
    nfp_dates = cals["NFP"][(cals["NFP"] >= ret.index.min()) & (cals["NFP"] <= ret.index.max())]
    cpi_dates = cals["CPI"][(cals["CPI"] >= ret.index.min()) & (cals["CPI"] <= ret.index.max())]
    fri = float((nfp_dates.dayofweek == 4).mean())
    nonfri = [d.date().isoformat() for d in nfp_dates if d.dayofweek != 4]
    print(f"  NFP on Fridays : {fri*100:.1f}%  ({len(nonfri)} documented exceptions: "
          f"{', '.join(nonfri)})")
    print(f"  CPI day-of-month range: {int(cpi_dates.day.min())} -> {int(cpi_dates.day.max())} "
          f"(mid-month schedule; >15th = shutdown-delayed releases)")
    wk = int((cpi_dates.dayofweek >= 5).sum() + (nfp_dates.dayofweek >= 5).sum())
    print(f"  weekend release dates: {wk} (must be 0)")
    print("  cross-source check: BLS archive-index dates vs official histreleasedates.pdf "
          "agree on 19/19 overlapping 1999-2000 releases (see data.py docstring)")

    print("\n# Headline — announcement days vs all other days (SPY close-to-close, total-return)")
    full = st.event_vs_rest(ret.values, any_m)
    print(f"  A-day mean     : {full['ev_mean_bps']:+.2f} bps/day  (n={full['n_ev']:,}, "
          f"one-sample t0={full['ev_t0']:.2f})")
    print(f"  other-day mean : {full['base_mean_bps']:+.2f} bps/day  (n={full['n_base']:,})")
    print(f"  premium        : {full['diff_bps']:+.2f} bps/day   Welch t = {full['welch_t']:+.2f}")
    print(f"  {full['frac_days']*100:.1f}% of sessions carry {full['share_of_total']*100:.1f}% "
          f"of SPY's cumulative log return")
    pl = st.placebo_pvalue(ret.values, any_m, n_draws=NDRAWS)
    print(f"  placebo p      : {pl['p_value']:.4f}  ({NDRAWS:,} random same-density calendars)")

    print("\n# Per-type split — each announcement type vs PURE non-announcement days")
    for name, m in (("FOMC", fomc), ("CPI", cpi), ("NFP", nfp)):
        s = st.event_vs_rest(ret.values, m, base=pure_rest)
        print(f"  {name:<5}: mean {s['ev_mean_bps']:+7.2f} bps/day  (n={s['n_ev']:>3})  "
              f"vs rest {s['base_mean_bps']:+.2f}  diff {s['diff_bps']:+7.2f}  "
              f"Welch t={s['welch_t']:+.2f}")

    print("\n# Sub-periods (decades)")
    splits = [("1997-2006", "1997-01-01", "2006-12-31"),
              ("2007-2016", "2007-01-01", "2016-12-31"),
              ("2017-2026", "2017-01-01", "2026-06-30")]
    for s in st.subperiod_stats(ret, any_m, splits):
        print(f"  {s['label']}: A-mean {s['ev_mean_bps']:+7.2f}  rest {s['base_mean_bps']:+6.2f}  "
              f"diff {s['diff_bps']:+7.2f} bps/day  Welch t={s['welch_t']:+.2f}  "
              f"(n_A={s['n_ev']})")

    print("\n# TLT (long Treasuries) — same calendar, 2002-08 onward")
    tlt = st.daily_returns(px["TLT"].dropna())
    tlt = tlt[tlt.index >= "2002-08-01"]
    tmasks = data.announcement_masks(tlt.index)
    t_full = st.event_vs_rest(tlt.values, tmasks["ANY"])
    print(f"  A-day mean {t_full['ev_mean_bps']:+.2f} vs other {t_full['base_mean_bps']:+.2f} "
          f"bps/day  diff {t_full['diff_bps']:+.2f}  Welch t={t_full['welch_t']:+.2f}  "
          f"(n_A={t_full['n_ev']}, {len(tlt)} days)")

    print("\n# Tradable overlay — long SPY ONLY on announcement days (enter prior close, exit A-day close)")
    for cb in (1.0, 2.0, 5.0):
        ov = st.overlay_stats(ret, any_m, cost_bps=cb)
        print(f"  cost={cb:>4.1f} bps/leg: gross {ov['gross_bps_per_aday']:+.2f} -> net "
              f"{ov['net_bps_per_aday']:+.2f} bps/A-day (net t0={ov['net_t0']:+.2f}) | "
              f"net ann {ov['net_ann_pct']:+.2f}% vs B&H {ov['bh_ann_pct']:+.2f}%  "
              f"({ov['n_rt_per_yr']:.1f} round trips/yr, {ov['time_in_market_pct']:.1f}% time in market)")

    print("\n# Third axis — is it all just FOMC? (the pooled claim vs the FOMC-only story)")
    ex_fomc = any_m & ~fomc          # CPI/NFP sessions carrying no FOMC decision
    s_ex = st.event_vs_rest(ret.values, ex_fomc, base=pure_rest)
    s_fo = st.event_vs_rest(ret.values, fomc, base=pure_rest)
    pl_ex = st.placebo_pvalue(ret.values, ex_fomc, n_draws=NDRAWS)
    print(f"  CPI/NFP-only A-days (no FOMC): mean {s_ex['ev_mean_bps']:+.2f} bps/day  "
          f"diff {s_ex['diff_bps']:+.2f}  Welch t={s_ex['welch_t']:+.2f}  (n={s_ex['n_ev']})  "
          f"placebo p={pl_ex['p_value']:.4f}")
    print(f"  FOMC decision days           : mean {s_fo['ev_mean_bps']:+.2f} bps/day  "
          f"diff {s_fo['diff_bps']:+.2f}  Welch t={s_fo['welch_t']:+.2f}  (n={s_fo['n_ev']})")
    print("  FOMC-only engine by decade (vs all other days in the same decade):")
    for s in st.subperiod_stats(ret, fomc, splits):
        print(f"    {s['label']}: diff {s['diff_bps']:+7.2f} bps/day  Welch t={s['welch_t']:+.2f}  "
              f"(n_FOMC={s['n_ev']})")
else:
    print("(no _cache/map_prices.csv — run data.fetch_panel() once to build the cache)")

print("\n# Synthetic control — seed-averaged (100 seeds), deterministic, no network")
print("  the A-vs-rest detector must recover a PLANTED premium and must NOT manufacture")
print("  significance when the true premium is 0. Machinery proof only — never market evidence.")
for edge in (0.0, 0.0020):
    sw = st.synthetic_sweep(edge=edge, n_seeds=100)
    print(f"  planted edge={sw['edge_bps']:+6.1f} bps/day: mean diff={sw['mean_diff_bps']:+7.2f} "
          f"bps/day  mean Welch t={sw['mean_t']:+6.2f}  share |t|>=2: {sw['share_t_ge2']*100:.0f}%")
