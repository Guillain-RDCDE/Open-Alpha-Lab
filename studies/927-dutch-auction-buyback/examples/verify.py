"""Real-tape verification — Study 927 (Dutch Auction). Regenerates docs/results.md numbers.

Reads the shared cache, screens the hardcoded EDGAR event table, and runs the whole event
study: the announcement-day abnormal return, the (−5, −1) run-up, the tradable tender
window, the post-expiry drift at 1 / 3 / 6 months, the jackknife, the placebo, the
calendar-time portfolio (HAC *t*), the long-only excess-of-cash race, the era cut and the
cost / borrow / expiry / date sweeps. Network only on ``--fetch``.

    python studies/927-dutch-auction-buyback/examples/verify.py            # cache-only
    python studies/927-dutch-auction-buyback/examples/verify.py --fetch    # refresh tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dutch_auction import data, strategy as st  # noqa: E402

COST_BPS = 10.0
BORROW_BPS_YR = 30.0
LIQ_FLOOR = 10e6          # $10m median 60-day dollar volume for the "liquid" subset


def main(fetch: bool) -> None:
    if fetch:
        failed = data.fetch()
        print("fetch failed for:", failed)

    px = data.load_prices()
    dv = data.load_dollar_volume()
    panel = st.EventPanel(px, data.EVENTS, market=data.MARKET,
                          dollar_volume=dv, min_dollar_volume=0.0)
    tbl = st.event_table(panel)

    print(f"EDGAR rows: {len(data.EVENTS)}  |  issuer tapes cached: "
          f"{len(px) - 2}  |  events surviving the screens: {len(tbl)}  "
          f"({tbl['ticker'].nunique()} distinct issuers)")
    print(f"event window {tbl['t0'].min().date()} -> {tbl['t0'].max().date()}  "
          f"as-of {data.AS_OF}  fp={data.study_fingerprint(px)}")
    print(f"dollar-volume cache present for {len(dv)} issuers "
          f"({'liquidity screen available' if dv else 'liquidity screen NOT applied'})")
    print("NAMED BIAS: the screens require 148 sessions of tape AFTER the event, so issuers "
          "acquired or\n            taken private within ~7 months of their own tender are "
          "dropped — and a takeout is exactly\n            the outcome that would print a big "
          "positive 6m abnormal return. The drift below is an\n            upper bound on how "
          "empty it is. Also named: SEC's CURRENT CIK->ticker map, and the\n            "
          "\"modified Dutch auction\" phrase filter (visibility).")

    print("\n=== event windows (abnormal = stock - SPY) ===")
    s = st.summarise(tbl)
    for m, r in s.iterrows():
        print(f"  {m:7s} n={int(r['n']):3d}  mean {r['mean_pct']:+7.2f}%  "
              f"median {r['median_pct']:+7.2f}%  t={r['t']:+6.2f}  hit={r['hit_rate']:.0%}")

    print("\n=== jackknife (leave-one-out t) ===")
    for c in ("ar0", "window", "m6"):
        j = st.jackknife_t(tbl[c])
        print(f"  {c:7s} t={j['t']:+.2f}  LOO range [{j['t_min']:+.2f}, {j['t_max']:+.2f}]")

    print("\n=== bootstrap CI on the mean (5,000 draws, 5-event blocks) ===")
    for c in ("ar0", "window", "m6"):
        b = st.bootstrap_mean_ci(tbl[c], block=5, seed=927)
        print(f"  {c:7s} mean {b['mean']*100:+.2f}%  95% CI "
              f"[{b['ci_low']*100:+.2f}%, {b['ci_high']*100:+.2f}%]  share<0 {b['frac_negative']:.3f}")

    print("\n=== placebo (same names, random dates, 2,000 draws) ===")
    pl = st.placebo_null(panel, metrics=("ar0", "window", "m6"), n_draws=2000, seed=927)
    for c in ("ar0", "window", "m6"):
        obs = float(tbl[c].mean())
        print(f"  {c:7s} obs {obs*100:+.2f}%  placebo mean {pl[c].mean()*100:+.2f}% "
              f"(sd {pl[c].std(ddof=1)*100:.2f}%)  p={st.placebo_p(obs, pl[c]):.4f}")

    print("\n=== calendar-time portfolio (long issuer / short SPY, tender window) ===")
    cal = st.calendar_time_portfolio(panel, tbl, cost_bps=COST_BPS,
                                     borrow_bps_yr=BORROW_BPS_YR)
    for tag in ("gross", "net"):
        c = cal[tag]
        print(f"  {tag:5s}: {c['mean_daily_bps']:+.3f} bps/day  Sharpe {c['sharpe']:+.3f}  "
              f"vol {c['vol_ann']:.2%}  HAC t {c['t_hac']:+.2f}  cum {c['cum']:+.2%}")
    print(f"  live on {cal['live_days']} of {cal['net']['n_days']} sessions, "
          f"{cal['avg_names_live']:.2f} names on a live day, {cal['n_events']} events")
    print(f"  costs are weight-aware: {cal['n_events']} events at their real portfolio weight "
          f"= {cal['nav_round_trips']:.1f} full-NAV round trips\n"
          f"  (charging one per event would overstate the drag "
          f"{cal['n_events'] / cal['nav_round_trips']:.2f}x), drag "
          f"{cal['cost_drag_bps_day']:.3f} bps/day")

    print("\n=== long-only race, excess-of-cash (BIL) vs SPY ===")
    race = st.long_only_race(panel, px[data.MARKET], px[data.CASH], cost_bps=COST_BPS)
    for tag in ("basket_gross", "basket_net", "spy"):
        r = race[tag]
        print(f"  {tag:13s}: exSharpe {r['sharpe_excess_cash']:+.3f}  "
              f"{r['mean_daily_bps']:+.3f} bps/day  vol {r['vol_ann']:.2%}  HAC t {r['t_hac']:+.2f}")
    print(f"  advantage (net basket - SPY): {race['sharpe_adv_net']:+.3f}  "
          f"HAC t on daily diff {race['t_diff_hac']:+.2f}")

    print("\n=== era cut (split 2018-01-01) ===")
    eras = st.era_cut(tbl)
    for tag, e in eras.items():
        print(f"  {tag:5s} n={e['n']:3d}  ar0 {e['ar0_pct']:+.2f}% (t={e['t_ar0']:+.2f})  "
              f"window {e['window_pct']:+.2f}% (t={e['t_window']:+.2f})  "
              f"m6 {e['m6_pct']:+.2f}% (t={e['t_m6']:+.2f})")

    print("\n=== liquidity subset (median 60d dollar volume >= $10m) ===")
    if dv:
        lp = st.EventPanel(px, data.EVENTS, market=data.MARKET,
                           dollar_volume=dv, min_dollar_volume=LIQ_FLOOR)
        lt = st.event_table(lp)
        ls = st.summarise(lt)
        for m in ("ar0", "window", "m6"):
            r = ls.loc[m]
            print(f"  {m:7s} n={int(r['n']):3d}  mean {r['mean_pct']:+7.2f}%  "
                  f"median {r['median_pct']:+7.2f}%  t={r['t']:+6.2f}  hit={r['hit_rate']:.0%}")
        lcal = st.calendar_time_portfolio(lp, lt, cost_bps=COST_BPS,
                                          borrow_bps_yr=BORROW_BPS_YR)
        print(f"  calendar-time net: Sharpe {lcal['net']['sharpe']:+.3f}  "
              f"HAC t {lcal['net']['t_hac']:+.2f}")
    else:
        print("  dollar-volume cache absent — screen NOT applied")

    print("\n=== cost sweep (one-way x NAV, both legs, in and out) ===")
    for row in st.cost_sweep(panel, tbl, borrow_bps_yr=BORROW_BPS_YR):
        print(f"  cost={row['cost_bps']:5.1f}bps  net window {row['net_window_pct']:+.2f}%  "
              f"cal Sharpe {row['sharpe_net']:+.3f} (HAC t {row['t_hac_net']:+.2f})")

    print("\n=== borrow sweep (short-SPY leg — an ASSUMPTION, not a tape input) ===")
    for row in st.borrow_sweep(panel, tbl, cost_bps=COST_BPS):
        print(f"  borrow={row['borrow_bps_yr']:6.1f} bps/yr  cal Sharpe {row['sharpe_net']:+.3f} "
              f"(HAC t {row['t_hac_net']:+.2f})")

    print("\n=== expiry PROXY sweep (trading days after the entry) ===")
    for row in st.expiry_sweep(panel):
        print(f"  expiry=+{row['expiry_td']:2d}td  n={row['n']:3d}  window {row['window_pct']:+.2f}% "
              f"(t={row['t_window']:+.2f})  m1 {row['m1_pct']:+.2f}% (t={row['t_m1']:+.2f})  "
              f"m3 {row['m3_pct']:+.2f}% (t={row['t_m3']:+.2f})  "
              f"m6 {row['m6_pct']:+.2f}% (t={row['t_m6']:+.2f})")

    print("\n=== event-date shift sweep (is the pop date-locked?) ===")
    for row in st.date_shift_sweep(panel):
        print(f"  shift={row['shift']:+2d}td  n={row['n']:3d}  day-0 abnormal "
              f"{row['ar0_pct']:+.2f}% (t={row['t_ar0']:+.2f})")

    print("\n=== synthetic control (machinery proof — never supports the stamp) ===")
    pp, pe, truth = data.synthetic_panel(signal_strength=1.0, seed=927)
    d = st.synthetic_detect(pp, pe)
    print(f"  planted jump {truth['planted_jump']*1e4:.0f} bps -> recovered "
          f"{d['ar0_bps']:+.0f} bps (t={d['t_ar0']:+.2f})")
    print(f"  planted 6m drift {truth['planted_drift_6m_log']*1e4:.0f} bps of LOG return "
          f"-> {truth['expected_simple_drift_6m']*1e4:.0f} bps expected SIMPLE buy-and-hold "
          f"(Jensen) -> recovered {d['m6_bps']:+.0f} bps (t={d['t_m6']:+.2f})")
    n_seeds = 20
    null_ar0, null_m6 = [], []
    for s_ in range(n_seeds):
        np_, ne_, _ = data.synthetic_panel(signal_strength=0.0, seed=927 + s_)
        dn = st.synthetic_detect(np_, ne_)
        null_ar0.append(dn["t_ar0"]); null_m6.append(dn["t_m6"])
    null_ar0 = np.array(null_ar0); null_m6 = np.array(null_m6)
    print(f"  null x{n_seeds}: |t(ar0)| max {np.abs(null_ar0).max():.2f} "
          f"(>=2 on {int((np.abs(null_ar0) >= 2).sum())}/{n_seeds}), "
          f"|t(m6)| max {np.abs(null_m6).max():.2f} "
          f"(>=2 on {int((np.abs(null_m6) >= 2).sum())}/{n_seeds})")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
