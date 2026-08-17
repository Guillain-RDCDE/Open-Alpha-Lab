"""Real-tape verification — Study 919 (Methodology Shock). Regenerates docs/results.md.

Reads the cached daily total-return closes of QQQ, SPY, IWM, MDY and BIL from the shared
desk cache, runs the market-model event study over the hardcoded index-methodology-change
calendar (announcement leg and effective leg), prices the pooled CAR against a placebo
randomisation test, bootstraps the CI, sweeps the window, jackknifes the event list, cuts
the eras, and costs the dollar-neutral pair trade with a borrow sweep. Network only on
``--fetch``.

    python studies/919-index-methodology-change/examples/verify.py            # cache-only
    python studies/919-index-methodology-change/examples/verify.py --fetch    # refresh
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from methodology_shock import data, strategy as st  # noqa: E402

WINDOW = (1, 10)      # the tradable window: on at day +1, off at day +10
HOLD = 10
COST_BPS = 5.0
BORROW_BPS = 50.0
FINANCE_BPS = 200.0   # bill rate charged on the residual (1 - beta) dollar exposure
N_PLACEBO = 2000


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    evs = list(data.EVENTS)

    print(f"tape: {px.index[0].date()} -> {px.index[-1].date()}  n={len(px)}  "
          f"fp={data.fingerprint(px[['QQQ', 'SPY', 'IWM', 'MDY']].dropna(how='all'))}")
    print(f"as-of {data.AS_OF}   |  {len(evs)} hardcoded rule changes (PROXY event list)")
    same = [e["event_id"] for e in evs
            if data.leg_date(e, "announce") is not None
            and data.leg_date(e, "announce") == data.leg_date(e, "effective")]
    print(f"  NOT independent legs: {len(same)} rows announce ON their effective date "
          f"({', '.join(same)}) -- those observations are shared by both legs")

    for leg in ("announce", "effective"):
        s = st.run_event_study(px, evs, leg=leg, window=WINDOW, asof=data.AS_OF)
        print(f"\n=== pooled market-model event study — {leg} leg, window "
              f"[{WINDOW[0]:+d},{WINDOW[1]:+d}] ===")
        print(f"  n={s['n_events']}  mean CAR {s['mean_car_bps']:+.1f} bps  "
              f"median {s['median_car_bps']:+.1f}  sd {s['sd_car_bps']:.1f}")
        print(f"  cross-event t {s['t_cross_event']:+.2f}  |  HAC t on daily AR "
              f"{s['t_hac_daily']:+.2f}  |  hit rate {s['hit_rate']:.0%} "
              f"[{s['hit_ci'][0]:.2f}, {s['hit_ci'][1]:.2f}]")
        print(f"  minimum detectable CAR at 5% with this many events: "
              f"{st.minimum_detectable_car(s):.0f} bps")
        for _, r in s["table"].iterrows():
            print(f"    {r['event_id']:38s} {str(r['date'].date()):10s} "
                  f"{r['treated']}/{r['control']}  beta {r['beta']:.2f}  "
                  f"CAR {r['car_bps']:+7.1f} bps (raw {r['car_raw_bps']:+7.1f})  "
                  f"[{r['date_confidence']}]")

        pl = st.placebo_test(px, evs, leg=leg, window=WINDOW,
                             n_draws=N_PLACEBO, seed=919, asof=data.AS_OF)
        print(f"  placebo: observed {pl['observed_bps']:+.1f} bps vs null "
              f"{pl['null_mean_bps']:+.1f} +/- {pl['null_sd_bps']:.1f} "
              f"(95% [{pl['null_q025_bps']:+.0f}, {pl['null_q975_bps']:+.0f}])  "
              f"p = {pl['p_two_sided']:.3f}  ({pl['n_draws']} draws)")

        boot = st.bootstrap_car_ci(s["table"]["car_bps"].to_numpy(), seed=919)
        blk = st.block_bootstrap_ar_ci(s["ar_daily"], window_len=WINDOW[1] - WINDOW[0] + 1,
                                       block=5, seed=919)
        print(f"  event bootstrap CI [{boot['ci_low']:+.0f}, {boot['ci_high']:+.0f}] bps  "
              f"| block bootstrap CI [{blk['ci_low']:+.0f}, {blk['ci_high']:+.0f}] bps")

    print("\n=== window sweep (announcement leg; placebo p and Bonferroni over 9 windows) ===")
    ws = st.window_sweep(px, evs, leg="announce", n_placebo=N_PLACEBO, asof=data.AS_OF)
    for _, r in ws.iterrows():
        tag = "tradable  " if r["tradable"] else "descriptive"
        print(f"  {r['window']:>10s} {tag}  n={r['n_events']}  CAR {r['mean_car_bps']:+7.1f} bps  "
              f"t {r['t_cross_event']:+.2f}  hit {r['hit_rate']:.0%}  "
              f"placebo p {r['placebo_p']:.3f} (Bonf {r['placebo_p_bonferroni']:.3f})")

    print("\n=== drop-one jackknife (announcement leg, [+1,+10]) ===")
    jk = st.jackknife(px, evs, leg="announce", window=WINDOW, asof=data.AS_OF)
    for _, r in jk.iterrows():
        print(f"  without {r['dropped']:38s} CAR {r['mean_car_bps_without']:+7.1f} bps  "
              f"t {r['t_without']:+.2f}   (its own CAR {r['own_car_bps']:+.1f})")

    print("\n=== era cut (split 2015-01-01, announcement leg) ===")
    for tag, e in st.era_cut(px, evs, split="2015-01-01", leg="announce",
                             window=WINDOW, asof=data.AS_OF).items():
        if e is None:
            print(f"  {tag}: no events")
            continue
        print(f"  {tag:5s}: n={e['n_events']}  CAR {e['mean_car_bps']:+7.1f} bps  "
              f"t {e['t_cross_event']:+.2f}  hit {e['hit_rate']:.0%}")

    print(f"\n=== costed pair trade (long treated / short beta x sibling, +1 -> +{HOLD}) ===")
    pt = st.pair_trade(px, evs, leg="announce", hold_days=HOLD, hedge="beta",
                       cost_bps=COST_BPS, borrow_bps_ann=BORROW_BPS,
                       finance_bps_ann=FINANCE_BPS, asof=data.AS_OF)
    print(f"  n={pt['n']}  gross {pt['mean_gross_bps']:+.1f} bps (t {pt['t_gross']:+.2f})  "
          f"net {pt['mean_net_bps']:+.1f} bps (t {pt['t_net']:+.2f})  "
          f"win rate {pt['hit_rate_net']:.0%}")
    for _, r in pt["table"].iterrows():
        print(f"    {r['event_id']:38s} beta {r['beta']:.2f}  net$ {r['net_dollar_exposure']:+.2f}  "
              f"gross {r['gross_bps']:+7.1f}  fin {r['finance_bps_charged']:+5.1f}  "
              f"net {r['net_bps']:+7.1f} bps")
    ptd = st.pair_trade(px, evs, leg="announce", hold_days=HOLD, hedge="dollar",
                        cost_bps=COST_BPS, borrow_bps_ann=BORROW_BPS,
                        finance_bps_ann=FINANCE_BPS, asof=data.AS_OF)
    print(f"  naive 1x/1x dollar-neutral variant: gross {ptd['mean_gross_bps']:+.1f} bps  "
          f"net {ptd['mean_net_bps']:+.1f} bps (t {ptd['t_net']:+.2f})  "
          f"-- carries the unhedged market exposure")
    pte = st.pair_trade(px, evs, leg="effective", hold_days=HOLD, hedge="beta",
                        cost_bps=COST_BPS, borrow_bps_ann=BORROW_BPS,
                        finance_bps_ann=FINANCE_BPS, asof=data.AS_OF)
    print(f"  effective leg: n={pte['n']}  gross {pte['mean_gross_bps']:+.1f} bps  "
          f"net {pte['mean_net_bps']:+.1f} bps (t {pte['t_net']:+.2f})  "
          f"win rate {pte['hit_rate_net']:.0%}")

    print("\n=== cost x borrow sweep (mean net bps) ===")
    sw = st.cost_sweep(px, evs, leg="announce", hold_days=HOLD,
                       finance_bps_ann=FINANCE_BPS, asof=data.AS_OF)
    for _, r in sw.iterrows():
        print(f"  cost {r['cost_bps']:5.1f} bps | borrow {r['borrow_bps_ann']:5.1f} bps/yr -> "
              f"net {r['mean_net_bps']:+7.1f} bps (t {r['t_net']:+.2f})")

    print("\n=== financing sweep (bill rate on the residual (1-beta) exposure) ===")
    fw = st.finance_sweep(px, evs, leg="announce", hold_days=HOLD,
                          cost_bps=COST_BPS, borrow_bps_ann=BORROW_BPS, asof=data.AS_OF)
    for _, r in fw.iterrows():
        print(f"  finance {r['finance_bps_ann']:5.1f} bps/yr -> mean charge "
              f"{r['mean_finance_bps']:+5.2f} bps, net {r['mean_net_bps']:+7.1f} bps "
              f"(t {r['t_net']:+.2f})")

    print("\n=== deployed-capital race (BIL parked, pair overlaid; excess-of-cash) ===")
    dc = st.deployed_capital_race(px, px["BIL"], evs, leg="announce", hold_days=HOLD,
                                  cost_bps=COST_BPS, borrow_bps_ann=BORROW_BPS,
                                  finance_bps_ann=FINANCE_BPS, asof=data.AS_OF)
    if dc["n_live_events"]:
        s = dc["summary"]
        print(f"  {dc['n_live_events']} events inside the BIL window "
              f"({dc['n_dropped_pre_cash']} dropped: pre-BIL, never teleported onto its "
              f"first session), {dc['n_live_days']} live days "
              f"of {s['n_days']}  |  total excess {dc['total_excess_bps']:+.0f} bps")
        print(f"  overlay excess-of-cash Sharpe {s['sharpe']:+.3f} (HAC t {s['tstat']:+.2f}), "
              f"vol {s['vol_ann']:.2%}, max DD {s['max_drawdown']:+.2%}")
    spy_ex = (px["SPY"].pct_change() - px["BIL"].pct_change()).dropna()
    print(f"  reference: SPY excess-of-cash Sharpe {st.sharpe(spy_ex):+.3f} over the same BIL window")

    print("\n=== synthetic control (machinery proof — never supports the stamp) ===")
    p1, e1, t1 = data.synthetic_panel(signal_strength=1.0, seed=919)
    p0, e0, _ = data.synthetic_panel(signal_strength=0.0, seed=919)
    d1 = st.synthetic_detect(p1, e1, window=WINDOW, n_draws=300)
    d0 = st.synthetic_detect(p0, e0, window=WINDOW, n_draws=300)
    print(f"  planted {t1['planted_car_bps']:.0f} bps: recovered {d1['mean_car_bps']:+.1f} bps "
          f"(t {d1['t_cross_event']:+.2f}, placebo p {d1['placebo_p']:.3f}, hit {d1['hit_rate']:.0%})")
    print(f"  null            : recovered {d0['mean_car_bps']:+.1f} bps "
          f"(t {d0['t_cross_event']:+.2f}, placebo p {d0['placebo_p']:.3f})")
    nulls = np.array([
        st.synthetic_detect(*data.synthetic_panel(signal_strength=0.0, seed=919 + 7 * k)[:2],
                            window=WINDOW, n_draws=120)["mean_car_bps"]
        for k in range(6)
    ])
    print(f"  null across 6 seeds: mean {nulls.mean():+.1f} bps, sd {nulls.std(ddof=1):.1f} bps")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
