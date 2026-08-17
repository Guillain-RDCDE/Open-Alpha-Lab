"""Real-tape verification — Study 924 (First Cut). Regenerates docs/results.md numbers.

Reads cached TLT / IEF / SHY / BIL / ^IRX daily total-return closes, runs the first-cut
event study (buy duration at the close of the session after the announcement, hold
1/3/6/12 months, excess of BIL), the all-cuts control, the random-date placebo, the cost
and borrow sweeps and the era split, and prints the per-event tables. Network only on
``--fetch``.

    python studies/924-cut-cycle-duration-extension/examples/verify.py
    python studies/924-cut-cycle-duration-extension/examples/verify.py --fetch
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from first_cut import data, strategy as st  # noqa: E402

DUR = "TLT"
BELLY = "IEF"
FRONT = "SHY"
CASH = "BIL"
COST_BPS = 5.0
HEADLINE_H = 12


def _legs(px, dur_tk, cash_tk):
    d = px[dur_tk].dropna()
    c = px[cash_tk].dropna()
    common = d.index.intersection(c.index)
    return d.loc[common], c.loc[common], common


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    dur, cash, common = _legs(px, DUR, CASH)

    print(f"{DUR} vs {CASH}: {common[0].date()} -> {common[-1].date()}  n={len(common)}  "
          f"fp={data.fingerprint(px[[DUR, CASH]].loc[common])}")
    print(f"as-of {data.AS_OF}")
    print(f"hardcoded calendar: {len(data.FIRST_CUTS)} first cuts, {len(data.ALL_CUTS)} cuts total")

    print("\n=== per-event table, first cuts, 12-month hold (excess of BIL) ===")
    tbl = st.event_table(dur, cash, data.first_cuts(), HEADLINE_H, COST_BPS)
    for ev, r in tbl.iterrows():
        print(f"  {ev.date()}  {r['entry'].date()} -> {r['exit'].date()}  "
              f"TLT {r['dur_pct']:+7.2f}%  BIL {r['cash_pct']:+6.2f}%  "
              f"excess {r['excess_pct']:+7.2f}%  net {r['excess_net_pct']:+7.2f}%")
    dropped = [d for d in data.FIRST_CUTS if pd.Timestamp(d) not in tbl.index]
    print(f"  dropped (no tape / incomplete window): {dropped}")

    print("\n=== horizon sweep — FIRST cuts (TLT) ===")
    hs = st.horizon_sweep(dur, cash, data.first_cuts(), cost_bps=COST_BPS)
    for h, r in hs.iterrows():
        print(f"  {h:2d}m  N={int(r['n_events'])}  mean {r['mean_excess_pct']:+7.2f}%  "
              f"net {r['mean_excess_net_pct']:+7.2f}%  median {r['median_excess_pct']:+7.2f}%  "
              f"t(N) {r['t_events']:+5.2f}  hit {r['hit_rate']:.0%}  "
              f"in {r['in_ann_pct']:+6.2f}%/y vs out {r['out_ann_pct']:+6.2f}%/y  "
              f"t(in-out) {r['t_in_vs_out']:+5.2f}  HAC t {r['t_event_leg_hac']:+5.2f}")

    print("\n=== horizon sweep — ALL cuts control (TLT) ===")
    hs_all = st.horizon_sweep(dur, cash, data.all_cuts(), cost_bps=COST_BPS)
    for h, r in hs_all.iterrows():
        print(f"  {h:2d}m  N={int(r['n_events'])}  mean {r['mean_excess_pct']:+7.2f}%  "
              f"net {r['mean_excess_net_pct']:+7.2f}%  median {r['median_excess_pct']:+7.2f}%  "
              f"t(N) {r['t_events']:+5.2f}  hit {r['hit_rate']:.0%}  "
              f"in {r['in_ann_pct']:+6.2f}%/y vs out {r['out_ann_pct']:+6.2f}%/y  "
              f"t(in-out) {r['t_in_vs_out']:+5.2f}  HAC t {r['t_event_leg_hac']:+5.2f}")

    print("\n=== horizon sweep — FIRST cuts, IEF (belly) cross-check ===")
    belly, cash_b, _ = _legs(px, BELLY, CASH)
    hs_ief = st.horizon_sweep(belly, cash_b, data.first_cuts(), cost_bps=COST_BPS)
    for h, r in hs_ief.iterrows():
        print(f"  {h:2d}m  N={int(r['n_events'])}  mean {r['mean_excess_pct']:+7.2f}%  "
              f"t(N) {r['t_events']:+5.2f}  hit {r['hit_rate']:.0%}  "
              f"HAC t {r['t_event_leg_hac']:+5.2f}")

    print("\n=== random-date placebo (2,000 draws, matched N and horizon) ===")
    for h in st.HORIZONS:
        cmp = st.compare(dur, cash, data.first_cuts(), horizon_months=h, cost_bps=COST_BPS)
        pl = st.placebo(dur, cash, cmp["n_events"], horizon_months=h,
                        n_draws=2000, cost_bps=COST_BPS, seed=924)
        p = st.placebo_pvalue(cmp["mean_excess_net_pct"], pl["means"])
        print(f"  {h:2d}m  observed net {cmp['mean_excess_net_pct']:+7.2f}%  "
              f"placebo mean {pl['means'].mean():+6.2f}%  sd {pl['means'].std(ddof=1):5.2f}  "
              f"one-sided p = {p:.3f}  (all-date mean {pl['all_mean_pct']:+6.2f}%)")

    print("\n=== block-bootstrap CI on the daily conditional leg (12m windows) ===")
    daily = st.conditional_daily(dur, cash, data.first_cuts(), HEADLINE_H, COST_BPS)
    ci = st.block_bootstrap_mean_ci(daily["e_event"], n_boot=2000, block=21,
                                    seed=924, scale=st.TRADING_DAYS * 100.0)
    print(f"  conditional leg: {ci['mean']:+.2f}%/y  95% CI [{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]  "
          f"share<0 {ci['frac_negative']:.1%}  in-window {daily['in_window'].mean():.1%} of days")
    ci_u = st.block_bootstrap_mean_ci(daily["e_dur"], n_boot=2000, block=21,
                                      seed=924, scale=st.TRADING_DAYS * 100.0)
    print(f"  unconditional TLT-BIL: {ci_u['mean']:+.2f}%/y  95% CI [{ci_u['ci_low']:+.2f}, {ci_u['ci_high']:+.2f}]")

    print("\n=== era split (2020-01-01), 12m ===")
    eras = st.era_split(dur, cash, data.first_cuts(), split="2020-01-01",
                        horizon_months=HEADLINE_H, cost_bps=COST_BPS)
    for tag, e in eras.items():
        print(f"  {tag:5s} N={e['n_events']}  mean net {e['mean_excess_net_pct']:+7.2f}%  {e['events']}")

    print("\n=== cost sweep (12m, first cuts) ===")
    cs = st.cost_sweep(dur, cash, data.first_cuts(), horizon_months=HEADLINE_H)
    for c, r in cs.iterrows():
        print(f"  {c:5.1f} bps one-way: mean net {r['mean_excess_net_pct']:+7.2f}%  "
              f"t(N) {r['t_events_net']:+5.2f}  HAC t {r['t_event_leg_hac']:+5.2f}")

    print("\n=== curve expression: long TLT / short SHY, borrow sweep (12m) ===")
    front = px[FRONT].dropna()
    bs = st.borrow_sweep(dur, front, data.first_cuts(), horizon_months=HEADLINE_H)
    for b, r in bs.iterrows():
        print(f"  borrow {b:5.1f} bps/y: N={int(r['n_events'])}  mean net {r['mean_ls_net_pct']:+7.2f}%  "
              f"t(N) {r['t_events']:+5.2f}")

    print("\n=== ^IRX proxy-cash cross-check (PROXY, not a tradable fund) ===")
    proxy = data.cash_index_from_irx(px["^IRX"])
    dur_p = px[DUR].dropna()
    common_p = dur_p.index.intersection(proxy.index)
    tbl_p = st.event_table(dur_p.loc[common_p], proxy.loc[common_p],
                           data.first_cuts(), HEADLINE_H, COST_BPS)
    print(f"  N={len(tbl_p)}  mean net {tbl_p['excess_net_pct'].mean():+.2f}%  "
          f"t(N) {st.one_sample_t(tbl_p['excess_net_pct'].to_numpy()):+.2f}")
    for ev, r in tbl_p.iterrows():
        print(f"    {ev.date()}  excess net {r['excess_net_pct']:+7.2f}%")

    print("\n=== synthetic control (machinery proof only) ===")
    for tag, ss in [("planted", 1.0), ("null   ", 0.0)]:
        pooled, per_world = [], []
        truth = None
        for pprices, pev, t in data.synthetic_panel(n_worlds=12, signal_strength=ss):
            truth = t
            sub = st.event_table(pprices["duration"], pprices["cash"], pev, 6, COST_BPS)
            pooled.extend(sub["excess_net_pct"].tolist())
            per_world.append(st.one_sample_t(sub["excess_net_pct"].to_numpy()))
        pooled = np.array(pooled)
        per_world = np.array(per_world)
        print(f"  {tag} (true 6m effect {truth['planted_6m_excess']*100:+.1f}%): "
              f"pooled N={len(pooled)} mean {pooled.mean():+.2f}%  pooled t {st.one_sample_t(pooled):+.2f}  "
              f"| per-world 5-event t: mean {per_world.mean():+.2f}, "
              f"|t|>=2 in {(np.abs(per_world) > 2).sum()}/12")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
