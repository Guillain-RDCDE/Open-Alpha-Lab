"""Reproducible headline run for Study 631 — Restatement Shock (Item 4.02).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached EDGAR event list + yfinance
prices under ``_cache/`` if present (the real-tape numbers), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))

from restatement_shock import data, strategy as st  # noqa: E402
from quantlab import repro  # noqa: E402

AS_OF = "2026-06-30"   # last complete month; drift windows must END on/before this


def main() -> None:
    print("# Restatement Shock — Item 4.02 'do not rely' 8-Ks, market-adjusted event study")
    if data.have_real():
        ev, close, dvol, spy = data.load_real()
        spy = spy[spy.index <= pd.Timestamp(AS_OF)]
        close = close[close.index <= pd.Timestamp(AS_OF)]
        dvol = dvol[dvol.index <= pd.Timestamp(AS_OF)]

        n_all = len(ev)
        n_tk = int((ev["ticker"].astype(str) != "").sum())
        print(f"events (EDGAR FTS): {n_all} Item-4.02 8-Ks "
              f"{ev['file_date'].min().date()} -> {ev['file_date'].max().date()}; "
              f"{n_tk} ({n_tk / n_all * 100:.0f}%) map to a ticker")
        print(repro.data_stamp("close panel", close, cols=list(close.columns[:40]),
                               asof=AS_OF))

        tab = st.build_event_table(ev, close, dvol, spy)
        print(f"usable events    : {len(tab)} "
              f"(ticker on tape + {st.PRE_REQ}d history + full 63d post window)")
        print(f"  -> deads-missing bias NAMED: {n_all - len(tab)} events drop out "
              f"(no current ticker / delisted / no tape) — bankruptcies among them, so any "
              f"negative drift here is an UNDERSTATEMENT.")

        s = st.summarize(tab)
        print("\n# Announcement window [0,+1] — the bomb (market-adjusted log return)")
        print(f"  mean {s['ar0_mean_pct']:+.2f}%  median {s['ar0_median_pct']:+.2f}%  "
              f"t={s['ar0_t']:+.2f}  winsorized t={s['ar0_t_wins']:+.2f}  "
              f"month-clustered HAC t={s['ar0_t_hac']:+.2f}")

        print("\n# Drift window [+2,+64] — 63 trading days from the close of day +1 "
              "(ONE execution lag)")
        print(f"  mean {s['car_mean_pct']:+.2f}%  median {s['car_median_pct']:+.2f}%  "
              f"share negative {s['car_neg_share'] * 100:.1f}%")
        print(f"  per-event t={s['car_t']:+.2f}  winsorized t={s['car_t_wins']:+.2f}  "
              f"MONTH-CLUSTERED HAC t={s['car_t_hac']:+.2f}  "
              f"({s['n_months']} event-months) <- headline stat")

        print("\n# Robustness — penny-stock floor on the entry price (day +1)")
        for floor in (1.0, 5.0):
            sub = tab[tab["p_in"] >= floor]
            print(f"  entry >= ${floor:.0f}: n={len(sub)}  "
                  f"CAR {sub['car'].mean() * 100:+.2f}%  "
                  f"t={st.ttest_vs_zero(sub['car'].to_numpy()):+.2f}  "
                  f"HAC t={st.clustered_hac_t(sub)['t_hac']:+.2f}")

        print("\n# THE adversarial control — chronic-decay placebo (same stocks, "
              "same-length window, 1 year BEFORE the event)")
        pl = st.chronic_decay_placebo(ev, close, dvol, spy)
        print(f"  paired events        : {pl['n_pairs']}")
        print(f"  event window CAR     : {pl['event_pct']:+.2f}%  (t={pl['event_t']:+.2f})")
        print(f"  placebo window CAR   : {pl['placebo_pct']:+.2f}%  "
              f"(t={pl['placebo_t']:+.2f})  <- no event, same stocks")
        print(f"  EVENT-SPECIFIC drift : {pl['diff_pct']:+.2f}%  paired t={pl['diff_t']:+.2f}  "
              f"winsorized t={pl['diff_t_wins']:+.2f}  HAC t={pl['diff_t_hac']:+.2f}")
        print("  -> the raw drift is dominated by what these stocks do in ANY window; the "
              "event-specific increment does not clear t>=2.")

        print("\n# Horizon curve — drift CAR by holding period (entry close +1)")
        hz = st.horizon_curve(ev, close, dvol, spy)
        for _, r in hz.iterrows():
            print(f"  +{int(r['days']):>3}d: CAR {r['car_pct']:+.2f}%  "
                  f"t={r['t']:+.2f}  HAC t={r['t_hac']:+.2f}  (n={int(r['n'])})")

        print("\n# Era split — has the effect changed regime? (2004-2012 vs 2013-2026)")
        for lab, m in (("2004-2012", tab["date"] <= "2012-12-31"),
                       ("2013-2026", tab["date"] > "2012-12-31")):
            se = st.summarize(tab[m])
            print(f"  {lab}: n={se['n_events']:>4}  ann {se['ar0_mean_pct']:+.2f}% "
                  f"(HAC t={se['ar0_t_hac']:+.2f})  drift {se['car_mean_pct']:+.2f}% "
                  f"(t={se['car_t']:+.2f}, HAC t={se['car_t_hac']:+.2f})")

        print("\n# Third axis — is the drift concentrated in SMALL names "
              "(limits-to-arbitrage)?")
        sp = st.size_split(tab)
        print(f"  size proxy: median pre-event dollar volume "
              f"(split at ${sp['median_dvol_musd']:.2f}M/day)")
        print(f"  SMALL half: n={sp['n_small']}  drift {sp['small_car_pct']:+.2f}%  "
              f"t={sp['small_t']:+.2f}  HAC t={sp['small_t_hac']:+.2f}")
        print(f"  LARGE half: n={sp['n_large']}  drift {sp['large_car_pct']:+.2f}%  "
              f"t={sp['large_t']:+.2f}  HAC t={sp['large_t_hac']:+.2f}")
        print(f"  Welch t (small - large) = {sp['welch_t_diff']:+.2f}")

        print("\n# Tradability — short at close +1, cover at +64, SPY-hedged; "
              "shorts pay borrow")
        for cb, bw in ((10.0, 3.0), (20.0, 10.0), (50.0, 25.0)):
            ov = st.short_overlay(tab, cost_bps=cb, borrow_ann_pct=bw)
            print(f"  cost={cb:>4.0f}bps borrow={bw:>4.0f}%/yr: gross {ov['gross_pct']:+.2f}% "
                  f"-> net {ov['net_pct']:+.2f}% per event  "
                  f"(drag {ov['drag_pct']:.2f}%)  net t={ov['net_t']:+.2f}  "
                  f"HAC t={ov['net_t_hac']:+.2f}")
    else:
        print("(no cache — run data.fetch_events() + data.fetch_prices() once)")

    print("\n# Synthetic machinery control — deterministic, no network")
    print("  three planted worlds: a null (nothing may fire), a TRUE underreaction (both "
          "detectors fire),")
    print("  and a melting-ice-cube confound (raw drift false-fires; the paired placebo "
          "refuses) — the real-tape situation.")
    for lab, drift, chronic in (("null              ", 0.0, 0.0),
                                ("true underreaction", -0.08, 0.0),
                                ("chronic bleed only", 0.0, -0.40)):
        ev_s, cl_s, dv_s, spy_s = data.synthetic_market(drift=drift, chronic=chronic,
                                                        seed=631)
        tab_s = st.build_event_table(ev_s, cl_s, dv_s, spy_s)
        s = st.summarize(tab_s)
        pl_s = st.chronic_decay_placebo(ev_s, cl_s, dv_s, spy_s)
        print(f"  {lab} (drift={drift * 100:+.0f}%, chronic={chronic * 100:+.0f}%/yr): "
              f"raw drift {s['car_mean_pct']:+.2f}% (HAC t={s['car_t_hac']:+.2f}) | "
              f"event-specific {pl_s['diff_pct']:+.2f}% (paired t={pl_s['diff_t']:+.2f})")


if __name__ == "__main__":
    main()
