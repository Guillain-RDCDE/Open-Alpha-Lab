"""Real-tape verification — Study 918 (Creation Halt). Regenerates docs/results.md.

Reads the shared cache, builds one signed fund-versus-uncapped-twin spread per
hardcoded suspension event, and prints: the announcement event study (K = 5/10/20 with
placebo-standardised z), the suspended-regime drift with HAC *t*, the resumption fade,
the leave-one-event-out jackknife, the ruler-quality split, the calendar era cut, the
resumption-date sweep, the costed and borrow-charged trade with its cost x borrow sweep,
and the synthetic control. Network only on ``--fetch``.

    python studies/918-etn-creation-halt/examples/verify.py            # cache-only
    python studies/918-etn-creation-halt/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from creation_halt import data, strategy as st  # noqa: E402

COST_BPS = 10.0
BORROW_ANN = 3.0


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    spreads = st.build_spreads(px)

    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    print(f"{len(data.EVENTS)} hardcoded events, {sum(e['announcement'] for e in data.EVENTS)} "
          f"with an announcement day\n")

    print("=== the pairs ===")
    for ev in data.EVENTS:
        s = spreads[ev["key"]]
        print(f"  {ev['key']:10s} {ev['fund']:>8s} vs {ev['proxy']:<8s} "
              f"dir {ev['direction']:+d}  ruler {ev['ruler']:<16s} {ev['confidence']:<6s} "
              f"n={len(s):5d}  {s.index[0].date()} -> {s.index[-1].date()}  "
              f"sd={s.std()*100:.2f}%/d")

    print("\n=== excess-of-cash IDENTITY check (algebraic, not an empirical finding) ===")
    for ev in data.EVENTS:
        d = st.excess_of_cash_check(px[ev["fund"]], px[ev["proxy"]], px["BIL"],
                                    ev["direction"])
        print(f"  {ev['key']:10s} max |raw spread - excess-of-cash spread| = {d:.2e}")
    print("  (this cancels algebraically and cannot come out non-zero; it proves the code")
    print("   path, not the economics. The short rebate is charged separately as borrow.)")

    n_looks = st.family_size(sum(e["announcement"] for e in data.EVENTS))
    print(f"\n=== multiplicity: {n_looks} placebo percentiles are looked at "
          f"(5 events x 3 horizons x 2 legs) ===")
    print(f"  family-wise 5% bar for a single one-sided look: percentile "
          f">= {st.bonferroni_percentile(n_looks):.5f}")

    print("\n=== 1. announcement event study (window starts t+1) ===")
    for k in (5, 10, 20):
        tbl = st.event_study(spreads, data.EVENTS, k=k)
        pooled = st.pooled_z(tbl)
        print(f"\n  K = {k}")
        for key, row in tbl.iterrows():
            print(f"    {key:10s} CAR {row['car']*100:+7.2f}%  z {row['z']:+7.2f}  "
                  f"placebo pct {row['pct']:.3f} (n {int(row['n_placebo'])} overlapping)  "
                  f"pct_indep {row['pct_indep']:.3f} (n {int(row['n_placebo_indep'])})")
        print(f"    pooled: mean z {pooled['mean_z']:+.2f}  cross-event t {pooled['t']:+.2f}  "
              f"event-resample CI [{pooled['ci_low']:+.2f}, {pooled['ci_high']:+.2f}]  "
              f"positive {pooled['n_positive']}/{pooled['n']}")

    tbl20 = st.event_study(spreads, data.EVENTS, k=20)

    print("\n=== 2. suspended-regime drift (daily signed spread, in vs out) ===")
    print("  control window EXCLUDES the 20 sessions after resumption (the fade); fee_bps")
    print("  is the expense-ratio difference already baked into the spread (ASSUMPTION).")
    reg = st.regime_table(spreads, data.EVENTS)
    for key, row in reg.iterrows():
        print(f"  {key:10s} in {int(row['n_in']):5d}d {row['in_bps']:+9.2f} bps/d "
              f"(HAC t {row['t_in']:+5.2f})  total {row['in_total_pct']:+8.2f}%   "
              f"fee {row['fee_bps']:+5.2f} -> net-of-fee {row['in_bps_net_fee']:+9.2f}   "
              f"out {row['out_bps']:+6.2f} bps/d (t {row['t_out']:+5.2f}, n {int(row['n_out'])})  "
              f"Welch t(diff) {row['t_diff']:+5.2f}")

    print("\n=== 3. resumption fade (window starts resume+1, K = 20) ===")
    fad = st.fade_study(spreads, data.EVENTS, k=20)
    for key, row in fad.iterrows():
        print(f"  {key:10s} CAR {row['car']*100:+7.2f}%  z {row['z']:+7.2f}  "
              f"placebo pct {row['pct']:.3f}  pct_indep {row['pct_indep']:.3f} "
              f"(n {int(row['n_placebo_indep'])})")
    pf = st.pooled_z(fad)
    print(f"  mean fade z {pf['mean_z']:+.2f}  cross-event t {pf['t']:+.2f}  "
          f"event-resample CI [{pf['ci_low']:+.2f}, {pf['ci_high']:+.2f}]  "
          f"negative {(fad['z'] < 0).sum()}/{len(fad)}")

    print("\n=== 4a. jackknife (leave one announcement event out, K = 20) ===")
    for key, row in st.jackknife(tbl20).iterrows():
        print(f"  drop {key:10s} -> mean z {row['mean_z']:+6.2f}  t {row['t']:+5.2f}")

    print("\n=== 4b. ruler-quality split (K = 20) ===")
    for tag, p in st.ruler_split(tbl20, data.EVENTS).items():
        if p is None:
            continue
        print(f"  {tag:17s} n={p['n']}  mean z {p['mean_z']:+6.2f}  t {p['t']:+6.2f}  "
              f"positive {p['n_positive']}/{p['n']}")

    print("\n=== 4c. calendar era cut (split 2020-01-01, K = 20) ===")
    for tag, p in st.era_cut(tbl20, data.EVENTS).items():
        if p is None:
            print(f"  {tag:6s} (empty)")
            continue
        print(f"  {tag:6s} n={p['n']}  mean z {p['mean_z']:+6.2f}  t {p['t']}")

    print("\n=== 4d. resumption-date sweep (APPROX/SOFT dates only) ===")
    for sh, row in st.resume_date_sweep(spreads, data.EVENTS).iterrows():
        print(f"  shift {sh:+3d} bd: mean fade z {row['mean_fade_z']:+6.2f}  "
              f"mean in-halt {row['mean_in_bps']:+9.2f} bps/d  "
              f"fades negative {int(row['n_fade_negative'])}/6")

    print("\n=== 5. the trade (long capped fund / short uncapped twin) ===")
    print("  5a. HINDSIGHT EXIT — held to the resumption date, which nobody knew at entry")
    tr = st.trade_table(spreads, data.EVENTS, cost_bps=COST_BPS, borrow_ann=BORROW_ANN)
    for key, row in tr.iterrows():
        print(f"  {key:10s} {int(row['n_days']):5d}d  gross {row['gross_pct']:+8.2f}%  "
              f"cost {row['cost_pct']:.2f}%  rebal {row['rebal_pct']:.2f}%  "
              f"borrow {row['borrow_pct']:.2f}%  net {row['net_pct']:+8.2f}%")
    net = tr["net_pct"]
    print(f"  mean net {net.mean():+.2f}%  median net {net.median():+.2f}%  "
          f"positive {(net > 0).sum()}/{len(net)}  cross-event t {st.one_sample_t(net.to_numpy()):+.2f}")
    ex_gbtc = net.drop(index="GBTC-2024")
    print(f"  ex-GBTC (the 8.7-year regime): mean net {ex_gbtc.mean():+.2f}%  "
          f"positive {(ex_gbtc > 0).sum()}/{len(ex_gbtc)}")

    print(f"\n  5b. BLIND EXIT — the no-look-ahead rule: enter t+1, exit after a fixed "
          f"{st.BLIND_DAYS} sessions")
    trb = st.trade_table(spreads, data.EVENTS, cost_bps=COST_BPS, borrow_ann=BORROW_ANN,
                         hold="blind")
    for key, row in trb.iterrows():
        print(f"  {key:10s} {int(row['n_days']):5d}d  gross {row['gross_pct']:+8.2f}%  "
              f"rebal {row['rebal_pct']:.2f}%  borrow {row['borrow_pct']:.2f}%  "
              f"net {row['net_pct']:+8.2f}%")
    nb = trb["net_pct"].dropna()
    print(f"  mean net {nb.mean():+.2f}%  median net {nb.median():+.2f}%  "
          f"positive {(nb > 0).sum()}/{len(nb)}  cross-event t {st.one_sample_t(nb.to_numpy()):+.2f}")

    print("\n  cost x borrow sweep (mean / median per-event net %)")
    for _, row in st.trade_sweep(spreads, data.EVENTS).iterrows():
        print(f"    cost {row['cost_bps']:5.1f} bps  borrow {row['borrow_ann_pct']:5.1f}%/yr  "
              f"mean {row['mean_net_pct']:+8.2f}%  median {row['median_net_pct']:+7.2f}%  "
              f"positive {int(row['n_positive'])}/{int(row['n'])}  t {row['t_net']:+.2f}")

    print("\n=== 6. the one exact-ruler suspension, close up (VXX vs VIXY) ===")
    ev = data.event("VXX-2022")
    v = spreads["VXX-2022"]
    i0 = v.index.searchsorted(pd.Timestamp(ev["halt"]), side="right")
    i1 = v.index.searchsorted(pd.Timestamp(ev["resume"]), side="right")
    ins = v.iloc[i0:i1]
    ci = st.block_bootstrap_mean_ci(ins, block=10, seed=918)
    print(f"  announcement-day move (excluded by the lag): {v.loc[ev['halt']]*100:+.2f}%")
    print(f"  in-halt: {len(ins)}d  {ins.mean()*1e4:+.2f} bps/d  HAC t {st.newey_west_t(ins.to_numpy()):+.2f}  "
          f"spread Sharpe {st.spread_sharpe(ins):+.2f}")
    print(f"  block-bootstrap CI on the daily mean [{ci['ci_low']*1e4:+.1f}, {ci['ci_high']*1e4:+.1f}] bps/d, "
          f"share<0 {ci['frac_negative']:.3f}")
    print("  the premium ROUND-TRIPS inside the halt - it is not a monotone accretion:")
    for k in (5, 10, 20, 40, 60, 101):
        c, _, _ = st.window_sum(v, ev["halt"], k)
        print(f"    CAR at K={k:3d}: {c*100:+7.2f}%")
    ac = st.event_car(v, ev, k=20)
    fc = st.resumption_fade(v, ev, k=20)
    print(f"  honest (non-overlapping) percentiles: announce {ac['pct_indep']:.4f} and "
          f"fade {fc['pct_indep']:.4f} of only {ac['n_placebo_indep']} independent windows")
    print(f"  -> neither tail alone clears the {n_looks}-look bar "
          f"({st.bonferroni_percentile(n_looks):.5f}); the JOINT pattern does")

    print("\n=== 7. synthetic control (machinery proof — never supports the stamp) ===")
    for tag, ss in [("planted", 1.0), ("half", 0.5), ("null", 0.0)]:
        res = [st.synthetic_detect(ss, seed=918 + 11 * s) for s in range(8)]
        pt = np.array([r["pooled_t"] for r in res])
        ib = np.array([r["mean_in_bps"] for r in res])
        fz = np.array([r["mean_fade_z"] for r in res])
        print(f"  {tag:8s} (8 seeds): pooled t {pt.mean():+.2f} (|t|>=2 in {(abs(pt) >= 2).sum()}/8)  "
              f"in-halt {ib.mean():+.2f} bps/d  fade z {fz.mean():+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
