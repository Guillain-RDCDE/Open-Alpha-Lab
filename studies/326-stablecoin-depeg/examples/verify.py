"""Reproducible headline run for Study 326 (Stablecoin-Depeg).

    python studies/326-stablecoin-depeg/examples/verify.py

Loads the cached real BTC/ETH daily tapes (cache-only — populate once with
``data.load_real(t, fetch=True)``), builds the equal-weight basket, drops the in-progress
month, and prints the event-study headline (CAR around the two market-wide depegs), the
synthetic-control p-value, the tradable rule's two-trade record, and the synthetic positive
control. All numbers and fingerprints land in docs/results.md.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from stablecoin_depeg import data, strategy as st  # noqa: E402

AS_OF = "2026-05-31"  # drop the in-progress month (today 2026-06)
PRE, POST = 5, 10


def main() -> int:
    if not data.have_real_cache():
        print("No real cache present — run data.load_real(t, fetch=True) for", data.TICKERS)
        return 1

    basket = data.real_basket()
    basket = basket[basket.index <= pd.Timestamp(AS_OF)]
    print(f"Basket: {len(basket)} days  {basket.index.min().date()} -> {basket.index.max().date()}")
    print(f"Fingerprint(basket.ret): {data.fingerprint(basket)}")
    print(f"Per-ticker fingerprints:")
    for t in data.TICKERS:
        df = data.load_real(t)
        df = df[df.index <= pd.Timestamp(AS_OF)]
        print(f"  {t:8s} {len(df):5d} days  {df.index.min().date()} -> {df.index.max().date()}"
              f"  fp={data.fingerprint(df)}")

    events = data.depeg_events(major_only=True)
    print(f"\nMajor depeg events (n={len(events)}): {[d.date() for d in events]}")

    # Per-event CARs (event-day-inclusive and forward-only)
    w = st.event_windows(basket, events, pre=PRE, post=POST)
    car_full = st.event_cars(w, from_day=0, pre=PRE)
    car_fwd = st.event_cars(w, from_day=1, pre=PRE)
    print("\nPer-event CAR (event-day inclusive, [-5,+10]):")
    for ev, c0, c1 in zip(events, car_full, car_fwd):
        print(f"  {ev.date()}  full CAR {c0:+.1%}   forward CAR(t+1..) {c1:+.1%}")

    res_full = st.summarize_event_study(basket, events, from_day=0, pre=PRE, post=POST)
    res_fwd = st.summarize_event_study(basket, events, from_day=1, pre=PRE, post=POST)
    print(f"\nMean full CAR    : {res_full['mean_car']:+.1%}  "
          f"HAC t={res_full['tstat']}  p(vs null)={res_full['p_value']:.3f}  "
          f"CI[{res_full['ci_lo']:+.1%},{res_full['ci_hi']:+.1%}]")
    print(f"Mean forward CAR : {res_fwd['mean_car']:+.1%}  "
          f"HAC t={res_fwd['tstat']}  p(vs null)={res_fwd['p_value']:.3f}  "
          f"CI[{res_fwd['ci_lo']:+.1%},{res_fwd['ci_hi']:+.1%}]")

    # Tradable rule (forward, costs)
    print("\nTradable 'sell on depeg' rule (lag=1, hold=10):")
    for cost in (0.0, 10.0, 30.0):
        book = st.sell_on_depeg(basket, events, hold=10, lag=1, cost_bps=cost)
        s_rule = st.equity_stats(book["strat_ret"].to_numpy())
        s_bh = st.equity_stats(book["ret"].to_numpy())
        excess = (book["strat_ret"] - book["ret"]).sum()
        print(f"  cost={cost:4.0f}bps  rule Sharpe {s_rule['sharpe']:+.2f}  "
              f"B&H Sharpe {s_bh['sharpe']:+.2f}  cum excess vs B&H {excess:+.1%}")

    # All-events robustness (n=4 incl. near-misses)
    ev4 = data.depeg_events(major_only=False)
    res4 = st.summarize_event_study(basket, ev4, from_day=0, pre=PRE, post=POST)
    print(f"\nIncluding 2 near-misses (n={res4['n_events']}): mean full CAR "
          f"{res4['mean_car']:+.1%}  HAC t={res4['tstat']}  p={res4['p_value']:.3f}")

    # Synthetic positive control
    print("\nSynthetic positive control (the engine is a faithful detector):")
    for shock in (0.0, -0.05, -0.10, -0.20):
        sb, sev, _ = data.synthetic_tape(n_events=10, shock=shock, seed=326)
        r = st.summarize_event_study(sb, sev, from_day=0)
        print(f"  planted shock {shock:+.2f}: mean CAR {r['mean_car']:+.1%}  "
              f"t={r['tstat']:+.2f}  p={r['p_value']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
