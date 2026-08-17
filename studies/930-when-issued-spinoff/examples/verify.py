"""Real-tape verification — Study 930 (When-Issued Window). Regenerates docs/results.md.

Reads the shared cache, builds the per-event alpha panel for all three legs of a spin-off
(parent announcement to distribution, child's first 5/10/21 regular-way sessions, and the
parent+child combination), and prints the cross-sectional means, hit rates, t-stats,
bootstrap CIs, the era cut, the benchmark cross-checks, the proxy sweeps, the short-side
economics and the daily hedged sleeve. Network only on ``--fetch``.

    python studies/930-when-issued-spinoff/examples/verify.py            # cache-only
    python studies/930-when-issued-spinoff/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from when_issued import data, strategy as st  # noqa: E402

COST_BPS = 10.0        # one-way, x NAV, on the child/parent side
BENCH_COST_BPS = 1.0   # one-way, x NAV, on the SPY side
BORROW_BPS = 40.0      # annualised borrow on the SHORT SPY leg of every alpha
ALT_BENCH = ("IWM", "MDY")


def synthetic_demo() -> None:
    """Offline fallback when the shared cache is absent — SYNTHETIC, never a real number.

    Runs the identical estimator on a planted world and on the null, so the script is
    runnable and exits 0 on a bare checkout. Nothing printed here belongs in results.md.
    """
    print("=" * 78)
    print("NO SHARED CACHE FOUND — running the SYNTHETIC demo instead.")
    print("Everything below is simulated. Run with --fetch to populate studies/_cache")
    print("and reproduce the real-tape numbers in docs/results.md.")
    print("=" * 78)
    for ss, label in ((1.0, "planted world"), (0.0, "null world")):
        prices, events, truth = data.synthetic_daily(signal_strength=ss, seed=930)
        panel = st.event_panel(prices, events, cost_bps=COST_BPS,
                               bench_cost_bps=BENCH_COST_BPS, borrow_bps=BORROW_BPS)
        print(f"\n[SYNTHETIC] {label} — {len(panel)} events, "
              f"planted child 21d alpha {truth['planted_child_alpha_21d']:+.2%}")
        for leg, r in st.leg_table(panel).iterrows():
            print(f"  {leg:22s} mean {r['mean_pct']:+6.2f}%  t {r['t']:+.2f}  "
                  f"hit {r['hit_rate']:.2f}")


def main(fetch: bool) -> None:
    if fetch:
        missing = data.fetch()
        if missing:
            print("could not retrieve:", missing)

    if not data.have_real():
        synthetic_demo()
        return

    px = data.load_prices()
    ev = data.SPINOFF_EVENTS
    panel = st.event_panel(px, ev, cost_bps=COST_BPS, bench_cost_bps=BENCH_COST_BPS,
                           borrow_bps=BORROW_BPS)

    used = sorted({e["parent"] for e in ev} | {e["child"] for e in ev} | {"SPY", "BIL"})
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px[used])}")
    print(f"events in table: {len(ev)}  usable: {len(panel)}  "
          f"({panel['rw0'].min().date()} -> {panel['rw0'].max().date()}, "
          f"{len(panel) / ((panel['rw0'].max() - panel['rw0'].min()).days / 365.25):.1f}/yr)")
    print(f"dropped for a delisted child: {len(data.DELISTED_EVENTS)} "
          f"({', '.join(e['child'] for e in data.DELISTED_EVENTS)})")
    print(f"tickers: {len(data.TICKERS)}   costs {COST_BPS:.0f} bps one-way x NAV, "
          f"SPY {BENCH_COST_BPS:.0f} bps + {BORROW_BPS:.0f} bp/yr borrow on the short leg")

    print("\n=== the load-bearing anchor: is each regular-way date on the right side? ===")
    tails, exact, snapped = [], 0, []
    for e in ev:
        cs = px[e["child"]].dropna()
        i = int(cs.index.searchsorted(pd.Timestamp(e["regular_way"]), side="left"))
        tails.append(i)
        if i < len(cs) and cs.index[i] == pd.Timestamp(e["regular_way"]):
            exact += 1
        else:
            snapped.append(f"{e['child']}->{cs.index[i].date()}")
    pre = [t for t in tails if t > 0]
    print(f"  stated date is itself a session on the child's tape: {exact}/{len(ev)}"
          + (f"   snapped: {', '.join(snapped)}" if snapped else ""))
    print(f"  when-issued tail before the anchor: {len(pre)}/{len(ev)} events have one "
          f"({min(pre)}-{max(pre)} sessions); {len(ev) - len(pre)} start AT the anchor "
          f"({', '.join(e['child'] for e, t in zip(ev, tails) if t == 0)})")

    print("\n=== the three legs, GROSS (the headline: a negative alpha must not be")
    print("    inflated by charging a long position's costs to it) ===")
    for leg, r in st.leg_table(panel, net=False).iterrows():
        print(f"  {leg:22s} n={int(r['n']):2d}  mean {r['mean_pct']:+6.2f}%  "
              f"med {r['median_pct']:+6.2f}%  sd {r['sd_pct']:5.2f}  "
              f"hit {r['hit_rate']:.2f} [{r['hit_lo']:.2f},{r['hit_hi']:.2f}]  "
              f"t {r['t']:+.2f}  HACt {r['t_hac']:+.2f}  "
              f"CI [{r['ci_low_pct']:+.2f}, {r['ci_high_pct']:+.2f}]%")

    print("\n=== same, NET of all frictions (the long version — quoted, never headlined) ===")
    tbl = st.leg_table(panel, net=True)
    for leg, r in tbl.iterrows():
        print(f"  {leg:22s} n={int(r['n']):2d}  mean {r['mean_pct']:+6.2f}%  "
              f"med {r['median_pct']:+6.2f}%  hit {r['hit_rate']:.2f} "
              f"[{r['hit_lo']:.2f},{r['hit_hi']:.2f}]  t {r['t']:+.2f}  HACt {r['t_hac']:+.2f}  "
              f"CI [{r['ci_low_pct']:+.2f}, {r['ci_high_pct']:+.2f}]%")

    print("\n=== era cut (split 2020-01-01, by distribution date) ===")
    for col in ("a_child_5_net", "a_child_10_net", "a_child_21_net", "a_parent_net", "a_comb_net"):
        e = st.era_cut(panel, col)
        bits = []
        for tag in ("early", "late"):
            s = e[tag]
            bits.append(f"{tag} n={s.get('n', 0)} {s.get('mean_pct', np.nan):+6.2f}% "
                        f"(t={s.get('t', np.nan):+.2f})")
        print(f"  {col:16s} " + " | ".join(bits))

    print("\n=== the five pre-specified legs are a FAMILY (Westfall-Young max-|t|) ===")
    legs = ["a_parent_gross"] + [f"a_child_{h}_gross" for h in (5, 10, 21)] + ["a_comb_gross"]
    fw = st.family_wise_p(panel, legs)
    print("  " + "  ".join(f"{c.replace('a_', '').replace('_gross', '')} t={t:+.2f}"
                           for c, t in fw["t_obs"].items()))
    print(f"  strongest leg {fw['best_leg']} (t={fw['t_best']:+.2f}): "
          f"single-test bootstrap p = {fw['p_single']:.4f}, "
          f"FAMILY-WISE p over {fw['n_legs']} legs = {fw['p_fwe']:.4f}")

    print("\n=== benchmark cross-check (child leg vs a size-matched index) ===")
    print("    (unit-beta dollar hedge — no beta is estimated; this is the check on that)")
    try:
        px_alt = data.load_prices(tickers=tuple(list(data.TICKERS) + list(ALT_BENCH)))
        benches = ("SPY",) + ALT_BENCH
    except FileNotFoundError as exc:
        print(f"  IWM/MDY not in the shared cache — cross-check skipped ({exc.args[0][:60]}...)")
        px_alt, benches = px, ("SPY",)
    for b in benches:
        p_b = st.event_panel(px_alt, ev, bench=b, cost_bps=COST_BPS,
                             bench_cost_bps=BENCH_COST_BPS, borrow_bps=BORROW_BPS)
        s5 = st.leg_stats(p_b, "a_child_5_gross")
        s10 = st.leg_stats(p_b, "a_child_10_gross")
        s21 = st.leg_stats(p_b, "a_child_21_gross")
        print(f"  vs {b:4s}: +5d {s5['mean_pct']:+6.2f}% (t={s5['t']:+.2f})   "
              f"+10d {s10['mean_pct']:+6.2f}% (t={s10['t']:+.2f})   "
              f"+21d {s21['mean_pct']:+6.2f}% (t={s21['t']:+.2f})")

    print("\n=== robustness on the 5-session child leg (GROSS — the headline number) ===")
    for tag, col in (("gross", "a_child_5_gross"), ("net-of-frictions long", "a_child_5_net")):
        g = panel[col].to_numpy(dtype=float)
        jk = [st.one_sample_t(np.delete(g, i)) for i in range(len(g))]
        e = st.era_cut(panel, col)
        dec = panel.assign(q=panel["rw0"].dt.to_period("Q")).drop_duplicates("q")
        s = st.leg_stats(dec, col)
        print(f"  [{tag}] leave-one-out t {min(jk):+.2f} .. {max(jk):+.2f}  |  "
              f"eras {e['early']['mean_pct']:+.2f}% (t={e['early']['t']:+.2f}) / "
              f"{e['late']['mean_pct']:+.2f}% (t={e['late']['t']:+.2f})  |  "
              f"de-clustered n={s['n']} {s['mean_pct']:+.2f}% (t={s['t']:+.2f})")

    print("\n=== cost / borrow sweep (LONG child vs short SPY = the folklore trade) ===")
    for _, r in st.cost_sweep(px, ev, "a_child_5_net", cost_grid=(0.0, 10.0, 25.0, 50.0),
                              borrow_grid=(0.0, 40.0, 100.0)).iterrows():
        print(f"  cost {r['cost_bps']:5.1f} bps  SPY borrow {r['borrow_bps']:5.1f} bp/yr : "
              f"{r['mean_pct']:+6.2f}%  t {r['t']:+.2f}  hit {r['hit_rate']:.2f}")

    print("\n=== the trade that actually harvests it: SHORT the child, 5 sessions ===")
    print("    (child borrow is an ASSUMPTION - a fresh spin-off is the textbook hard-to-borrow)")
    for b, r in st.short_child_economics(panel, horizon=5, cost_bps=COST_BPS,
                                         bench_cost_bps=BENCH_COST_BPS).iterrows():
        print(f"  borrow {b:5.0f}%/yr : {r['mean_pct']:+6.2f}%  t {r['t']:+.2f}  "
              f"hit {r['hit_rate']:.2f} [{r['hit_lo']:.2f},{r['hit_hi']:.2f}]  "
              f"CI [{r['ci_low_pct']:+.2f}, {r['ci_high_pct']:+.2f}]%")

    print("\n=== proxy sweeps (the three non-tape inputs) ===")
    print("  announcement date (parent leg):")
    for _, r in st.announce_shift_sweep(px, ev, cost_bps=COST_BPS).iterrows():
        print(f"    shift {int(r['shift_sessions']):+d} sessions: {r['mean_pct']:+6.2f}% "
              f"(t={r['t']:+.2f}), per-21d {r['per21_mean_pct']:+.2f}%")
    print("  first regular-way session (child leg):")
    for sh, r in st.rw_shift_sweep(px, ev, cost_bps=COST_BPS).iterrows():
        print(f"    shift {sh:+d} sessions: n={int(r['n'])}  {r['mean_pct']:+6.2f}% "
              f"(t={r['t']:+.2f})  hit {r['hit_rate']:.2f}")
    print("  distribution ratio (combined leg):")
    for v, r in st.ratio_sweep(px, ev, cost_bps=COST_BPS).iterrows():
        print(f"    {v:18s}: {r['mean_pct']:+6.2f}% (t={r['t']:+.2f})")

    print("\n=== event-time shape (EXPLORATORY - only 5/10/21 were pre-specified) ===")
    caar = st.caar_path(px, ev, pre=3, post=21)
    for k in (-3, -1, 0, 1, 2, 3, 5, 10, 21):
        if k in caar.index:
            r = caar.loc[k]
            print(f"  session {k:+3d}: CAAR {r['caar_pct']:+6.2f}%  median {r['median_pct']:+6.2f}%  n={int(r['n'])}")
    p_ex = st.event_panel(px, ev, horizons=(1, 2, 3, 5, 10, 21, 63), cost_bps=COST_BPS,
                          bench_cost_bps=BENCH_COST_BPS, borrow_bps=BORROW_BPS)
    print("  horizon grid (gross):")
    for h, r in st.horizon_shape(p_ex).iterrows():
        print(f"    +{h:2d}d: {r['mean_pct']:+6.2f}%  med {r['median_pct']:+6.2f}%  "
              f"hit {r['hit_rate']:.2f}  t {r['t']:+.2f}  HACt {r['t_hac']:+.2f}")

    print("\n=== the daily hedged sleeve (long the children, short SPY, costed) ===")
    print("    (calendar = first distribution .. last exit, NOT whatever pre-history the")
    print("     shared cache happens to hold for SPY)")
    for w in (5, 10, 21):
        s = st.sleeve_stats(st.hedged_sleeve(px, ev, window=w, cost_bps=COST_BPS,
                                             bench_cost_bps=BENCH_COST_BPS,
                                             borrow_bps=BORROW_BPS))
        print(f"  window {w:2d}d: {s['n_active_days']:4d} of {s['n_days']} sessions "
              f"({s['active_frac']:.1%}), "
              f"spread {s['spread_mean_bps']:+7.2f} bps/day  HACt {s['spread_t_hac']:+.2f}  "
              f"block-boot CI [{s['spread_ci_lo_bps']:+.1f}, {s['spread_ci_hi_bps']:+.1f}] bps")
    print("  (the sleeve is idle ~96% of its own window, so its fund-level Sharpe is a")
    print("   capacity statement about 2 events a year, not evidence for or against the effect)")

    print("\n=== per-event detail (net, vs SPY) ===")
    for _, r in panel.iterrows():
        print(f"  {r['name']:32s} rw {r['rw0'].date()}  parent {r['a_parent_net']:+7.2%}  "
              f"+5d {r['a_child_5_net']:+7.2%}  +21d {r['a_child_21_net']:+7.2%}  "
              f"comb {r['a_comb_net']:+7.2%}")

    print("\n=== synthetic control (machinery proof - never supports the stamp) ===")
    pl_px, pl_ev, pl_truth = data.synthetic_daily(signal_strength=1.0, seed=930)
    pl = st.synthetic_detect(pl_px, pl_ev)
    print(f"  planted (child 21d alpha {pl_truth['planted_child_alpha_21d']:.2%}): "
          f"parent {pl['parent_mean_pct']:+.2f}% (t={pl['parent_t']:+.2f}), "
          f"child21 {pl['child21_mean_pct']:+.2f}% (t={pl['child21_t']:+.2f})")
    nulls = []
    for s_ in range(8):
        n_px, n_ev, _ = data.synthetic_daily(signal_strength=0.0, seed=930 + s_)
        nulls.append(st.synthetic_detect(n_px, n_ev))
    pt = np.array([d["parent_t"] for d in nulls])
    ct = np.array([d["child21_t"] for d in nulls])
    print(f"  null x8: parent t mean {pt.mean():+.2f} (|t|>=2 in {int((abs(pt) >= 2).sum())}/8), "
          f"child21 t mean {ct.mean():+.2f} (|t|>=2 in {int((abs(ct) >= 2).sum())}/8)")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main(fetch="--fetch" in sys.argv)
