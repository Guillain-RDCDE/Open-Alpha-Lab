"""Real-tape verification — Study 928 (Odd-Lot Priority). Regenerates docs/results.md.

Reads the shared cache (180 filed modified-Dutch-auction self-tenders, SPY, BIL), builds the
per-event tape table, prices the odd-lot and round-lot round trips under the declared
assumptions, and prints: the tape legs (run-up, offer drift, post-expiry give-back,
permanent reprice), the costed round trips, the premium-free breakeven premium, the
sweeps over every assumption, the era cut, the calendar-time excess-of-cash race, and the
synthetic control. Network only on ``--fetch``.

    python studies/928-odd-lot-tender/examples/verify.py            # cache-only
    python studies/928-odd-lot-tender/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from odd_lot import data, strategy as st  # noqa: E402

PREMIUM = data.PREMIUM_DEFAULT
PRORATION = data.PRORATION_DEFAULT
EXPIRY = data.EXPIRY_DAYS_DEFAULT
FEE = data.TENDER_FEE_USD_DEFAULT
POSITION = data.ODD_LOT_POSITION_USD
COST_BPS = 15.0
BORROW_BPS = 40.0


def main(fetch: bool) -> None:
    if fetch:
        failed = data.fetch()
        print("fetch failed for:", failed)

    px = data.load_prices()
    events = data.usable_events()
    tbl = st.build_event_table(px, events, market_key=data.MARKET, expiry_days=EXPIRY)
    h = st.headline(tbl, premium=PREMIUM, proration=PRORATION, cost_bps=COST_BPS,
                    tender_fee_usd=FEE, position_usd=POSITION, borrow_bps=BORROW_BPS)
    ev = h["events"]

    # Fingerprint the price MARKS the study actually uses, not the whole shared cache:
    # the cache is written by several studies and a re-fetch of an unrelated ticker must
    # not move this study's data stamp.
    marks = tbl[["p_pre", "p_ann", "p_entry", "p_exp", "p_post",
                 "m_pre", "m_entry", "m_exp", "m_post"]]
    issuers = [t for t in data.event_tickers() if t in px]
    print(f"issuer tapes {len(issuers)} of {len(data.event_tickers())} distinct tickers "
          f"(+ {data.MARKET} market, {data.CASH} cash)  |  events in list {len(events)} "
          f"of {len(data.EVENTS)} filed  |  usable windows {h['n_events']}")
    print(f"window {h['first'].date()} -> {h['last'].date()}  "
          f"({h['events_per_year']:.1f} events/yr)  marks fp={data.fingerprint(marks)}")
    print(f"as-of {data.AS_OF}  |  assumptions: premium {PREMIUM:.0%}, proration "
          f"{PRORATION:.2f}, expiry {EXPIRY}td, fee ${FEE:.0f} on ${POSITION:,.0f}, "
          f"cost {COST_BPS:.0f}bps one-way, borrow {BORROW_BPS:.0f}bps")

    print("\n=== the tape legs (no premium assumption in the first four) ===")
    for key, label in [("runup", "run-up  t-6 -> t+1"),
                       ("runup_abn", "run-up, abnormal"),
                       ("drift", "offer-window drift"),
                       ("giveback", "post-expiry give-back"),
                       ("perm", "permanent reprice"),
                       ("perm_abn", "permanent, abnormal")]:
        s = h["legs"][key]
        print(f"  {label:22s} mean {s['mean_pct']:+7.2f}%  med {s['median_pct']:+7.2f}%  "
              f"t_x {s['t_cross']:+6.2f}  t_HAC {s['t_hac']:+6.2f}  hit {s['hit_rate']:.2f} "
              f"[{s['hit_lo']:.2f},{s['hit_hi']:.2f}]  boot CI "
              f"[{s['ci_low_pct']:+.2f},{s['ci_high_pct']:+.2f}]")

    print("\n=== the priced round trips (premium assumption enters here) ===")
    for key, label in [("r_odd_net", "odd lot, net"),
                       ("r_odd_hedged", "odd lot, net + SPY-hedged"),
                       ("r_round_net", "round lot, net (prorated)"),
                       ("priority_value", "value of odd-lot priority")]:
        s = h["legs"][key]
        print(f"  {label:26s} mean {s['mean_pct']:+7.2f}%  t_HAC {s['t_hac']:+6.2f}  "
              f"hit {s['hit_rate']:.2f}  boot CI "
              f"[{s['ci_low_pct']:+.2f},{s['ci_high_pct']:+.2f}]")
    print(f"  share of events with p_post > p_clear (priority worthless): "
          f"{float((ev['p_post'] > ev['p_clear']).mean()):.2f}")
    print(f"  share of events where the run-up alone exceeds the assumed premium: "
          f"{float((ev['runup'] > PREMIUM).mean()):.2f}")

    print("\n=== the PREMIUM-FREE number: the breakeven premium ===")
    print(f"  mean {h['breakeven_mean_pct']:.2f}%  median {h['breakeven_median_pct']:.2f}%  "
          f"p75 {h['breakeven_p75_pct']:.2f}%  |  share of events needing more than the "
          f"assumed {PREMIUM:.0%}: {h['share_premium_below_breakeven']:.2f}")
    print("  free of the PREMIUM assumption, not of every assumption: it still inherits")
    print("  the anchor (see the anchor sweep: 9.18% at 2 sessions, 22.62% at 21), the")
    print("  assumed offer length and the declared frictions (see the cost sweep).")

    print("\n=== premium sweep (the assumption that decides the sign OF BOTH arms) ===")
    print(st.premium_sweep(tbl, grid=data.PREMIUM_GRID, proration=PRORATION,
                           cost_bps=COST_BPS, tender_fee_usd=FEE,
                           position_usd=POSITION, borrow_bps=BORROW_BPS
                           ).to_string(index=False))
    print("  note: priority_pct/priority_t move with the assumption too — the value of")
    print("  odd-lot priority is NOT a tape fact, it is the assumed premium net of the")
    print("  measured post-expiry price, and it changes sign inside this grid.")

    print("\n=== proration sweep (what priority is worth; 1.00 = under-subscribed) ===")
    print(st.proration_sweep(tbl, grid=data.PRORATION_GRID, premium=PREMIUM,
                             cost_bps=COST_BPS, tender_fee_usd=FEE,
                             position_usd=POSITION, borrow_bps=BORROW_BPS
                             ).to_string(index=False))

    print("\n=== cost sweep (flat fee + the bps touch, which is charged 3x) ===")
    print(st.cost_sweep(tbl, fee_grid=data.TENDER_FEE_GRID, bps_grid=data.COST_BPS_GRID,
                        premium=PREMIUM, proration=PRORATION, position_usd=POSITION,
                        borrow_bps=BORROW_BPS).to_string(index=False))

    print("\n=== capacity sweep (a 99-share lot is 99 x the share price, full stop) ===")
    print(st.capacity_sweep(tbl, premium=PREMIUM, proration=PRORATION,
                            cost_bps=COST_BPS, tender_fee_usd=FEE,
                            borrow_bps=BORROW_BPS).to_string(index=False))

    print("\n=== borrow sweep (the short SPY hedge leg) ===")
    print(st.borrow_sweep(tbl, premium=PREMIUM, proration=PRORATION, cost_bps=COST_BPS,
                          tender_fee_usd=FEE, position_usd=POSITION).to_string(index=False))

    print("\n=== expiry sweep (offer length is assumed, not observed) ===")
    print(st.expiry_sweep(px, events, market_key=data.MARKET, premium=PREMIUM,
                          proration=PRORATION, cost_bps=COST_BPS, tender_fee_usd=FEE,
                          position_usd=POSITION, borrow_bps=BORROW_BPS
                          ).to_string(index=False))

    print("\n=== anchor sweep (how far back the 'pre-announcement' price is taken) ===")
    print(st.anchor_sweep(px, events, market_key=data.MARKET, premium=PREMIUM,
                          proration=PRORATION, cost_bps=COST_BPS, tender_fee_usd=FEE,
                          position_usd=POSITION, borrow_bps=BORROW_BPS
                          ).to_string(index=False))

    print("\n=== era cut (split 2018-01-01) ===")
    eras = st.era_cut(tbl, split="2018-01-01", premium=PREMIUM, proration=PRORATION,
                      cost_bps=COST_BPS, tender_fee_usd=FEE, position_usd=POSITION,
                      borrow_bps=BORROW_BPS)
    for tag in ("early", "late"):
        e = eras.get(tag)
        if e is None:
            continue
        print(f"  {tag:5s} n={e['n']:3d}  run-up {e['runup_pct']:+6.2f}%  "
              f"give-back {e['giveback_pct']:+6.2f}%  hedged {e['hedged_pct']:+6.2f}% "
              f"(t={e['t_hac']:+5.2f})  priority {e['priority_pct']:+5.2f}%  "
              f"breakeven {e['breakeven_pct']:5.2f}%")
    if eras.get("diff"):
        print("  the CONTRAST itself, tested (Welch t on late - early, per event):")
        for col, lbl in [("runup", "run-up"), ("giveback", "give-back"),
                         ("r_odd_hedged", "hedged round trip"),
                         ("breakeven_premium", "breakeven premium"),
                         ("priority_value", "priority value")]:
            d = eras["diff"][col]
            flag = "" if abs(d["welch_t"]) >= 2 else "   <- NOT a significant change"
            print(f"    {lbl:20s} {d['early_pct']:+6.2f}% -> {d['late_pct']:+6.2f}%  "
                  f"gap {d['gap_pct']:+6.2f}pp  Welch t {d['welch_t']:+5.2f}{flag}")

    print("\n=== calendar-time sleeve, excess-of-cash (the un-achievable scaled version) ===")
    for tag, kw in [("gross", dict(cost_bps=0.0, tender_fee_usd=0.0)),
                    ("net  ", dict(cost_bps=COST_BPS, tender_fee_usd=FEE))]:
        cal = st.calendar_time_portfolio(px, ev, market_key=data.MARKET,
                                         cash_key=data.CASH, position_usd=POSITION, **kw)
        race = st.portfolio_race(cal)
        print(f"  {tag}: n={race['n_days']}  sleeve exSharpe {race['sharpe_port']:+.3f}  "
              f"SPY exSharpe {race['sharpe_mkt']:+.3f}  adv {race['sharpe_adv']:+.3f}  "
              f"t(diff) {race['t_diff']:+.2f}  vol {race['vol_port']:.1%}  "
              f"invested {race['invested_frac']:.1%}  live names {race['mean_live']:.2f}")

    print("\n=== synthetic control (machinery proof — never supports the stamp) ===")
    p1, e1, t1 = data.synthetic_panel(signal_strength=1.0, seed=928)
    p0, e0, _ = data.synthetic_panel(signal_strength=0.0, seed=928)
    d1 = st.synthetic_detect(p1, e1)
    d0 = st.synthetic_detect(p0, e0)
    print(f"  planted (run-up {t1['runup_planted_bps']:.0f} bps, give-back "
          f"-{t1['giveback_planted_bps']:.0f} bps): recovered run-up "
          f"{d1['runup_bps']:+.0f} bps (t={d1['runup_t']:+.2f}), give-back "
          f"{d1['giveback_bps']:+.0f} bps (t={d1['giveback_t']:+.2f}), "
          f"breakeven {d1['breakeven_pct']:.2f}%")
    print(f"  null                                   : run-up {d0['runup_bps']:+.0f} bps "
          f"(t={d0['runup_t']:+.2f}), give-back {d0['giveback_bps']:+.0f} bps "
          f"(t={d0['giveback_t']:+.2f}), breakeven {d0['breakeven_pct']:.2f}%")
    nulls = [st.synthetic_detect(*data.synthetic_panel(signal_strength=0.0, n_events=80,
                                                       seed=928 + s)[:2]) for s in range(8)]
    ru = np.array([n["runup_bps"] for n in nulls])
    gb = np.array([n["giveback_bps"] for n in nulls])
    print(f"  null x8: run-up mean {ru.mean():+.0f} bps (sd {ru.std(ddof=1):.0f}), "
          f"|t|>=2 in {int(sum(abs(n['runup_t']) >= 2 for n in nulls))}/8;  "
          f"give-back mean {gb.mean():+.0f} bps, "
          f"|t|>=2 in {int(sum(abs(n['giveback_t']) >= 2 for n in nulls))}/8")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
