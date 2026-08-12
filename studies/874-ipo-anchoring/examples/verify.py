"""Reproducible headline run for Study 874 — IPO-Price Anchoring.

Prints EVERY number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; the curated anchor table is hardcoded and
always available offline; the live tape reads the cache under ``_cache/`` (fetched once with
``data.fetch()``); the synthetic control runs anywhere with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from ipo_anchor import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")
LAGS = 6

print("# IPO-Price Anchoring — do investors anchor on the offer price?")
print(f"as-of {data.AS_OF} (last complete month; the partial current month is dropped)")

tbl = data.ipo_table(include_direct=True)
tbl_ipo = data.ipo_table(include_direct=False)
print(f"\n# The curated anchor table: {len(tbl)} well-known recent US listings "
      f"({len(tbl_ipo)} underwritten IPOs + {len(tbl) - len(tbl_ipo)} direct listings), "
      "offer/reference prices from the public record (SEC 424B4 / exchange reference notices)")

if not data.have_real():
    print("\n(no _cache/ipo_anchor_prices.csv — run data.fetch() once to build the cache)")
    print("  synthetic control still runs below.")
else:
    px = data.load_prices()
    P = st.build_panel(px, tbl, bench=data.BENCH, asof=data.AS_OF)
    cov = st.panel_coverage(P)
    try:
        from quantlab.repro import fingerprint
        names = [t for t in tbl.index if t in px.columns]
        fp = fingerprint(px[names + [data.BENCH]].resample("ME").last().round(6))
    except Exception:
        fp = "n/a"
    print(f"\n# The live tape: yfinance daily closes vs SPY (market leg), "
          f"{P['months'].min().date()} -> {P['months'].max().date()}")
    print(f"  {cov['n_names']} curated names, {cov['n_active_months']} active months "
          f"(>= {st.MIN_NAMES} live names), {cov['n_obs']} name-months, "
          f"avg {cov['avg_names_per_active_month']:.1f} names/active month, "
          f"below-offer share {cov['below_offer_share']*100:.1f}%  fingerprint {fp}")

    print("\n# TEST 1 — the anchoring pull (Fama-MacBeth cross-sectional slope, HAC t)")
    print("  forward market-adjusted return regressed on gap = log(price/offer); "
          "anchoring => negative slope")
    for lbl, Pn in (("all listings", P),
                    ("IPOs only", st.build_panel(px, tbl_ipo, data.BENCH, data.AS_OF))):
        a = st.anchoring_stats(Pn, lags=LAGS)
        print(f"  {lbl:12s}: mean slope {a['mean_slope']:+.4f} "
              f"({a['slope_bps_per_10pct']:+.1f} bps/mo per +10% above offer)  "
              f"NW t = {a['t_nw']:+.2f}  (1-sample t = {a['t_1s']:+.2f}, "
              f"{a['share_negative']*100:.0f}% of months negative, n={a['n_months']})")

    print("\n# TEST 2 — the below-offer drag (below-offer minus above-offer basket spread, HAC t)")
    sp = st.below_offer_spreads(P)
    b = st.below_offer_stats(sp, lags=LAGS)
    print(f"  all listings: spread {b['spread_bps']:+.2f} bps/mo ({b['ann_pct']:+.2f}%/yr)  "
          f"NW t = {b['t_nw']:+.2f}  (Welch t = {b['welch_t']:+.2f}, n={b['n_months']})")
    print(f"    below-offer basket {b['below_bps']:+.2f} vs above-offer basket "
          f"{b['above_bps']:+.2f} bps/mo")
    sp_i = st.below_offer_spreads(st.build_panel(px, tbl_ipo, data.BENCH, data.AS_OF))
    bi = st.below_offer_stats(sp_i, lags=LAGS)
    print(f"  IPOs only   : spread {bi['spread_bps']:+.2f} bps/mo  NW t = {bi['t_nw']:+.2f} "
          f"(n={bi['n_months']})")

    print("\n# PLACEBO — shuffle gap->forward-return within each month (1,000 permutations)")
    pl = st.placebo_pvalue(P, n_seeds=20, n_draws_per_seed=50)
    print(f"  observed slope {pl['obs_slope']:+.5f} vs placebo mean {pl['placebo_mean']:+.5f} "
          f"(sd {pl['placebo_sd']:.5f}) over {pl['n_draws']:,} draws")
    print(f"  left-tail p = {pl['p_left']:.3f}  two-sided p = {pl['p_two_sided']:.3f}")

    print("\n# ROBUSTNESS — below-offer spread, two eras (split 2022-07)")
    es = st.era_split(sp, "2022-07", lags=LAGS)
    for lbl, e in (("pre-2022-07 ", es["early"]), ("2022-07 on  ", es["late"])):
        print(f"  {lbl}: spread {e['spread_bps']:+.2f} bps/mo  NW t = {e['t_nw']:+.2f} "
              f"(n={e['n_months']})")

    print("\n# TRADABILITY — SHORT below-offer / LONG above-offer, net of borrow + costs")
    print("  monthly re-hedge (2 x one-way x NAV per leg); short (below-offer) leg pays borrow")
    for cb, br in ((10.0, 3.0), (20.0, 5.0)):
        tm = st.timer_stats(sp, cost_bps=cb, borrow_ann_pct=br, lags=LAGS)
        print(f"  cost={cb:>4.1f} bps, borrow={br:.0f}%/yr: gross {tm['gross_bps']:+.2f} "
              f"(t={tm['gross_t']:+.2f}) -> net {tm['net_bps']:+.2f} bps/mo "
              f"({tm['net_ann_pct']:+.2f}%/yr, t={tm['net_t']:+.2f})  "
              f"max drawdown {tm['worst_drawdown_pct']:.1f}%")

print("\n# SYNTHETIC CONTROL — deterministic, no network, 20 seeds (desk law)")
print("  the FM anchoring-slope detector must NOT fire on the null and must recover a planted pull")
for edge in (0.0, 0.15, 0.30):
    r = st.synthetic_control(edge, n_seeds=20, lags=LAGS)
    print(f"  planted edge {edge:.2f}: mean slope {r['mean_slope']:+.4f}  "
          f"mean HAC t = {r['mean_t']:+.2f}  |t|>=2 rejection rate = {r['reject_rate']*100:.0f}%")
print("  (power note: with ~45 names in one dominant cohort the real tape is honest but "
      "very modestly powered — the honest prior is None)")
