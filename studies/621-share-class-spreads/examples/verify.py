"""Reproducible headline run for Study 621 — Share-Class Spreads.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached four-ticker tape under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))  # repo root (quantlab)

from quantlab import repro
from share_class_spreads import data, strategy as st

AS_OF = "2026-06-30"          # last complete month before the 2026-07-03 run date

print("# Share-Class Spreads — BRK-A/BRK-B one-way conversion bound vs GOOG/GOOGL free spread")
if data.have_real():
    px = repro.as_of(data.load_real(), AS_OF)
    fp = repro.fingerprint(px)
    brk = st.gap_series(px, "BRK-A", "BRK-B", data.BRK_PARITY, start=data.BRKB_START)
    goo = st.gap_series(px, "GOOG", "GOOGL", 1.0, start=data.GOOG_START)
    # GOOGL premium = GOOGL/GOOG - 1  (senior voting class over junior);
    # note gap_series(a="GOOG", b="GOOGL", parity=1) = GOOGL/GOOG - 1. Positive = voting premium.
    print(f"tape           : {px.shape[0]} days x {px.shape[1]} tickers, "
          f"{px.index.min().date()} -> {px.index.max().date()}  as-of {AS_OF}  fingerprint {fp}")
    print(f"BRK pair       : {len(brk)} days ({brk.index.min().date()} -> {brk.index.max().date()}), "
          f"split-adjusted parity {data.BRK_PARITY:.0f} B per A across the whole tape")
    print(f"GOOG pair      : {len(goo)} days ({goo.index.min().date()} -> {goo.index.max().date()}), "
          f"no conversion bridge in either direction")

    print("\n# 1 - Bound enforcement: does BRK-B ever cost MORE than A/1500?")
    vb = st.violation_stats(brk)
    vg = st.violation_stats(goo)
    for thr in (0, 10, 50, 100):
        print(f"  B above parity by > {thr:>3d} bps : {vb[f'above_{thr}bps_share']*100:5.2f}% of days "
              f"({vb[f'above_{thr}bps_days']} days)")
    print(f"  worst upward excursion        : {brk.max()*1e4:+.0f} bps on {brk.idxmax().date()} "
          f"(COVID liquidity chaos)")
    print(f"  worst downward excursion      : {brk.min()*1e4:+.0f} bps on {brk.idxmin().date()}")
    print(f"  tail asymmetry (days beyond 50 bps): below {vb['dn50_share']*100:.2f}% vs above "
          f"{vb['up50_share']*100:.2f}%  ->  ratio {vb['asym_ratio']:.1f}x  (one-way bound signature)")
    print(f"  GOOG contrast (no bound)      : above {vg['up50_share']*100:.2f}% vs below "
          f"{vg['dn50_share']*100:.2f}%  ->  ratio {vg['dn50_share']/max(vg['up50_share'],1e-12):.1f}x "
          f"(symmetric, unbounded)")

    print("\n# 2 - The structural B discount (gap = 1500*B/A - 1, bps; negative = B cheap)")
    db = st.dist_stats(brk)
    print(f"  mean {db['mean_bps']:+.1f} bps  HAC t = {db['hac_t']:+.2f} (NW lags={db['hac_lags']}, "
          f"n={db['n']:,})   median {db['median_bps']:+.1f}  std {db['std_bps']:.0f}")
    print(f"  p1 {db['p1_bps']:+.1f}  p99 {db['p99_bps']:+.1f}  min {db['min_bps']:+.0f}  "
          f"max {db['max_bps']:+.0f}")
    for lbl, s in [("1996-2009", brk[:"2010-01-20"]), ("2010-2019", brk["2010-01-21":"2019"]),
                   ("2020-2026", brk["2020":])]:
        d2 = st.dist_stats(s)
        print(f"  era {lbl}: mean {d2['mean_bps']:+.1f} bps  HAC t = {d2['hac_t']:+.2f}  "
              f"(days B above parity: {(s > 0).mean()*100:.1f}%)")

    print("\n# 3 - GOOGL voting premium (GOOGL/GOOG - 1, bps; no bound)")
    dg = st.dist_stats(goo)
    print(f"  mean {dg['mean_bps']:+.1f} bps  HAC t = {dg['hac_t']:+.2f} (NW lags={dg['hac_lags']}, "
          f"n={dg['n']:,})   std {dg['std_bps']:.0f}  min {dg['min_bps']:+.0f}  max {dg['max_bps']:+.0f}")
    for lbl, s in [("2014-2017", goo[:"2017"]), ("2018-2021", goo["2018":"2021"]),
                   ("2022-2026", goo["2022":])]:
        d2 = st.dist_stats(s)
        print(f"  era {lbl}: mean {d2['mean_bps']:+.1f} bps  HAC t = {d2['hac_t']:+.2f}  "
              f"(days GOOGL below GOOG: {(s < 0).mean()*100:.0f}%)")

    print("\n# 4 - Mean-reversion half-lives (AR(1) on the daily gap)")
    rho_b, hl_b = st.half_life(brk)
    rho_g, hl_g = st.half_life(goo)
    print(f"  BRK gap : rho = {rho_b:.4f}  half-life = {hl_b:.1f} trading days (the bound + arb pull it back)")
    print(f"  GOOG gap: rho = {rho_g:.4f}  half-life = {hl_g:.1f} trading days (nothing pulls it back)")

    print("\n# 5 - Tradability: long-cheap/short-rich z-score pairs trade on the BRK gap")
    print("  (entry |z|>1.5 on a 252d window, exit |z|<0.25; one execution lag = signal close t,")
    print("   filled close t+1; costs one-way x NAV per leg, short leg pays 50 bps/yr borrow)")
    ps = st.pairs_trade(px, "BRK-A", "BRK-B", data.BRK_PARITY, fill="same", cost_bps=0.0,
                        borrow_bps_yr=0.0)
    print(f"  DIAGNOSTIC fill-at-signal-print, gross: {ps['gross_ann_pct']:+.2f}%/yr  "
          f"HAC t = {ps['gross_t']:+.2f}  (in market {ps['in_market_pct']:.0f}% of days)")
    for cb in (2.0, 5.0, 10.0):
        pn = st.pairs_trade(px, "BRK-A", "BRK-B", data.BRK_PARITY, fill="next", cost_bps=cb)
        print(f"  HONEST next-close fill, net @ {cb:>4.1f} bps: {pn['net_ann_pct']:+.2f}%/yr  "
              f"HAC t = {pn['net_t']:+.2f}")

    print("\n# 6 - Third axis: has the B-discount ever been a tradable signal for A-holders?")
    print("  (switch overlay: hold B while >= thr cheap, back to A at parity; excess vs holding A,")
    print("   net @ 5 bps one-way x 2 legs per switch)")
    sw_same = st.switch_overlay(px, "BRK-A", "BRK-B", data.BRK_PARITY, thr=0.01,
                                cost_bps=5.0, fill="same")
    print(f"  DIAGNOSTIC fill-at-signal-print (thr=1%): {sw_same['net_ann_pct']:+.2f}%/yr  "
          f"HAC t = {sw_same['hac_t']:+.2f}  ({sw_same['switches']} switches, "
          f"{sw_same['time_in_b_pct']:.0f}% of days in B)")
    for thr in (0.005, 0.01, 0.02, 0.03):
        sw = st.switch_overlay(px, "BRK-A", "BRK-B", data.BRK_PARITY, thr=thr,
                               cost_bps=5.0, fill="next")
        exc = sw["excess"]
        pre = exc[:"2009"]
        post = exc["2010":]
        t_pre, _, _ = st.hac_t(pre.values)
        t_post, _, _ = st.hac_t(post.values)
        print(f"  HONEST thr={thr*100:>3.1f}%: net {sw['net_ann_pct']:+.2f}%/yr  "
              f"HAC t = {sw['hac_t']:+.2f}  | pre-2010 {pre.mean()*252*100:+.2f}%/yr t={t_pre:+.2f}"
              f"  | 2010+ {post.mean()*252*100:+.2f}%/yr t={t_post:+.2f}")
else:
    print("(no _cache/scs_prices.csv — run data.fetch_pairs() once to build the cache)")

print("\n# 7 - Synthetic positive control — deterministic, no network")
print("  the bound detector (tail asymmetry + mean-gap HAC t) must recover a PLANTED one-way")
print("  bound + discount, and must NOT manufacture one from a symmetric unbounded null.")
for lbl, kw in [("PLANTED bound + -40 bps discount", dict(mean_discount=-0.004, bound=True)),
                ("NULL: no bound, zero discount   ", dict(mean_discount=0.0, bound=False))]:
    spx = data.synthetic_pair(seed=621, **kw)
    det = st.bound_detector(spx, data.BRK_PARITY)
    print(f"  {lbl}: mean {det['mean_bps']:+.1f} bps  HAC t = {det['hac_t']:+.2f}  "
          f"days beyond +/-50bps: above {det['up50_share']*100:.2f}% / below {det['dn50_share']*100:.2f}%"
          f"  asym {det['asym_ratio']:.1f}x")
