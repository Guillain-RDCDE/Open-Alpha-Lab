"""Reproducible headline run for Study 634 — US-Leads-the-World ("America sneezes").

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY + overseas-index bars
under ``_cache/`` if present (the real-tape numbers), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np

from us_leads_the_world import data, strategy as st

ERAS = [("1997-2003", "1997", "2003"), ("2004-2010", "2004", "2010"),
        ("2011-2017", "2011", "2017"), ("2018-2026", "2018", "2026")]

print("# US-Leads-the-World — SPY day t vs the NEXT Tokyo/Frankfurt/London/Sydney session (yfinance)")
if data.have_real():
    bars = data.load_real()
    # data stamp: as-of pin + content fingerprint of the wide close matrix
    wide = bars.pivot(index="date", columns="ticker", values="close").sort_index()
    try:
        from quantlab.repro import fingerprint
        fp = fingerprint(wide)
    except Exception:  # pragma: no cover — stdlib fallback if quantlab is not importable
        import hashlib
        fp = hashlib.sha256(wide.round(6).to_csv().encode()).hexdigest()[:12]
    pairs = data.build_all_pairs(bars)
    bk = data.basket_pairs(pairs)
    print(f"as-of          : {data.AS_OF}  (sample pinned; partial July 2026 dropped)")
    print(f"fingerprint    : {fp}")
    print(f"tape           : {wide.index.min().date()} -> {wide.index.max().date()}, "
          f"{wide.shape[0]} dates x {wide.shape[1]} tickers (SPY total-return; indices price-only)")
    print(f"aligned pairs  : " + ", ".join(f"{tk} n={len(p)}" for tk, p in pairs.items())
          + f" | GLOBAL-4 basket n={len(bk)}")

    print("\n# Signal — next-session close-to-close on the US day-t close (OLS slope, Newey-West HAC t)")
    summaries = {}
    for tk, p in pairs.items():
        s = st.market_summary(p)
        summaries[tk] = s
        print(f"  {tk:7s} ({data.LABELS[tk]:22s}): slope={s['cc']['slope']:+.3f}  "
              f"HAC t={s['cc']['t']:+.2f}  R2={s['cc']['r2']*100:.1f}%  n={s['cc']['n']}")
    rb = st.ols_hac(bk["us"], bk["basket"])
    print(f"  GLOBAL-4 equal-weight basket : slope={rb['slope']:+.3f}  HAC t={rb['t']:+.2f}  "
          f"R2={rb['r2']*100:.1f}%  n={rb['n']}")
    pl = st.shuffle_placebo(bk)
    print(f"  shuffle placebo ({pl['n_seeds']} seeds): mean |slope|={pl['mean_abs_slope']:.4f}, "
          f"mean |t|={pl['mean_abs_t']:.2f}  (the time link, not the marginals, carries it)")

    print("\n# Mechanism — where the spillover lands: overnight GAP vs tradable OPEN->CLOSE")
    print("  (open-based legs on LIVE-open days only; Yahoo index opens are stale in stretches)")
    for tk, s in summaries.items():
        gs = st.gap_share(s)
        print(f"  {tk:7s}: gap slope={s['gap']['slope']:+.3f} (t={s['gap']['t']:+.2f}, "
              f"R2={s['gap']['r2']*100:.1f}%) | oc slope={s['oc']['slope']:+.3f} "
              f"(t={s['oc']['t']:+.2f}, R2={s['oc']['r2']*100:.1f}%) | "
              f"gap share={gs*100:.0f}% | live opens {s['live_share']*100:.0f}%")

    print("\n# Decay by era — GLOBAL-4 basket slope + HAC t")
    bk_p = bk.rename(columns={"basket": "cc"})
    for r in st.era_slopes(bk_p, ERAS):
        print(f"  {r['label']}: slope={r['slope']:+.3f}  HAC t={r['t']:+.2f}  n={r['n']}")

    print("\n# Conditional on the size of the US move — |SPY day-t| quintiles, GLOBAL-4")
    for r in st.size_conditional(bk):
        print(f"  {r['label']} (mean |US| {r['mean_abs_us_pct']:.2f}%): slope={r['slope']:+.3f}  "
              f"HAC t={r['t']:+.2f}  n={r['n']}")

    print("\n# Third axis — does the spillover survive the post-2010 algo era? (pre vs post split)")
    for tk, p in pairs.items():
        r = st.prepost(p)
        print(f"  {tk:7s}: pre-2010 slope={r['pre']['slope']:+.3f} (t={r['pre']['t']:+.1f})  "
              f"post-2010 slope={r['post']['slope']:+.3f} (t={r['post']['t']:+.1f})  "
              f"decay t_diff={r['t_diff']:+.2f}")
    rp = st.prepost(bk_p)
    print(f"  GLOBAL-4: pre={rp['pre']['slope']:+.3f} (t={rp['pre']['t']:+.1f})  "
          f"post={rp['post']['slope']:+.3f} (t={rp['post']['t']:+.1f})  "
          f"decay t_diff={rp['t_diff']:+.2f}")

    print("\n# Tradability — the only executable trade: follow the US sign at the next foreign OPEN")
    print("  (gap forfeited by construction; one round trip per day = 2 x one-way cost;")
    print("   live-open days only — ^FTSE excluded: 88% of its Yahoo opens are stale)")
    for tk in ["^N225", "^GDAXI", "^AXJO"]:
        ft = st.feasible_trade(pairs[tk], cost_bps_oneway=5.0)
        ft10 = st.feasible_trade(pairs[tk], cost_bps_oneway=10.0)
        print(f"  {tk:7s}: gross {ft['gross_bps']:+.1f} bps/day (HAC t={ft['t']:+.2f}, n={ft['n']}) "
              f"-> net {ft['net_bps']:+.1f} @5bps | net {ft10['net_bps']:+.1f} @10bps one-way")

    print("\n# The phantom backtest — close-to-close on the US sign (INFEASIBLE: the foreign close")
    print("  of day t prints BEFORE the US day-t close exists; quoted only as the time-machine gap)")
    for tk, p in pairs.items():
        ph = st.phantom_trade(p)
        print(f"  {tk:7s}: gross {ph['gross_bps']:+.1f} bps/day (HAC t={ph['t']:+.2f})")
else:
    print("(no _cache/uslw_bars.csv — run data.fetch_panel() once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network (machinery proof only)")
print("  planted overnight-spillover beta -> the same OLS/HAC pipeline must find exactly it,")
print("  and must find NOTHING when beta = 0.")
for beta in (0.0, 0.35):
    syn = data.synthetic_world(beta=beta, seed=634)
    for tk, p in syn.items():
        s = st.market_summary(p)
        print(f"  beta={beta:+.2f} {tk}: cc slope={s['cc']['slope']:+.3f} (HAC t={s['cc']['t']:+.2f})  "
              f"gap slope={s['gap']['slope']:+.3f} (t={s['gap']['t']:+.2f})  "
              f"oc slope={s['oc']['slope']:+.3f} (t={s['oc']['t']:+.2f})")
