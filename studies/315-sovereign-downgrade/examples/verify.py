"""Real-tape verification — Study 315 (Sovereign-Downgrade). Regenerates docs/results.md.

Loads SPY total-return daily bars (cache-first), runs the directional abnormal-return event
study around the three US sovereign downgrades (S&P 2011, Fitch 2023, Moody's 2025), races
the CARs against a synthetic control of random non-event windows, scores the naive
"sell the downgrade" short trade net of costs and borrow, and runs the synthetic positive
control. Network is touched only with --fetch.

    python studies/315-sovereign-downgrade/examples/verify.py            # cache-only
    python studies/315-sovereign-downgrade/examples/verify.py --fetch    # refresh the tape
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sovereign_downgrade import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    bars = data.load_real("SPY", fetch=fetch)
    ev = data.announcements()
    print(f"SPY total-return tape: {bars.index[0].date()} to {bars.index[-1].date()}, "
          f"{len(bars)} days")
    print(f"Fingerprint: {data.fingerprint(bars)}\n")

    print("=== Per-event abnormal returns (lag=1, first tradeable bar) ===")
    ed = st.event_day_abnormal(bars, ev, lag=1)
    for _, row in ed.iterrows():
        print(f"  {row['event_date'].date()}  event-bar abn = {row['abn']*100:+6.2f}%")

    print("\n=== CARs vs synthetic control ===")
    nev = data.synthetic_null_events(bars, n_events=3000, window=25, seed=315)
    locs_null = st.snap_to_sessions(bars, nev, lag=0)
    for win in [(0, 1), (0, 5), (0, 20)]:
        c = st.car(bars, ev, window=win, lag=1)["car"].to_numpy()
        nc = st.window_cars(bars, locs_null, window=win)
        s = st.summarize(c)
        obs, p = st.permutation_pvalue(c, nc, n_perm=20000, seed=315)
        print(f"  CAR{win}: mean={s['mean']*100:+.2f}%  hit={s['hit_rate']*100:.0f}%  "
              f"HAC t={s['tstat']:+.2f}  control={nc.mean()*100:+.2f}%  perm p={p:.3f}")

    abn1 = st.event_day_abnormal(bars, ev, lag=1)["abn"].to_numpy()
    lo, hi = st.block_bootstrap_ci(abn1, n_boot=10000, seed=315)
    print(f"\nEvent-bar abn mean={abn1.mean()*100:+.2f}%  "
          f"95% block-boot CI=[{lo*100:+.2f}%, {hi*100:+.2f}%]")

    print("\n=== 'Sell the downgrade' short trade ===")
    for hold in (5, 20):
        g = st.short_the_downgrade(bars, ev, hold=hold, lag=1, cost_bps=0, borrow_bps_per_day=0)
        n = st.short_the_downgrade(bars, ev, hold=hold, lag=1, cost_bps=2.0,
                                   borrow_bps_per_day=1.0)
        print(f"  hold={hold:2d}: gross mean={g['pnl_gross'].mean()*100:+.2f}%  "
              f"net mean={n['pnl_net'].mean()*100:+.2f}%  "
              f"per-event(gross)={list(np.round(g['pnl_gross'].to_numpy()*100, 2))}")
    # the look-ahead lag=0 short, to expose the timing trap
    la = st.short_the_downgrade(bars, ev, hold=5, lag=0, cost_bps=0, borrow_bps_per_day=0)
    print(f"  (look-ahead lag=0 hold=5: gross mean={la['pnl_gross'].mean()*100:+.2f}% "
          f"— untradeable, shorts the announcement-day close before the post-close news)")

    yrs = (bars.index[-1] - bars.index[0]).days / 365.25
    bh = (bars["close"].iloc[-1] / bars["close"].iloc[0]) ** (1 / yrs) - 1
    print(f"\nSPY total-return buy&hold: {bh*100:+.1f}%/yr over {yrs:.1f}y")

    print("\n=== Synthetic positive control ===")
    for eff in (0.0, 0.02, 0.04, 0.08):
        b, e, _ = data.synthetic_spy(event_effect=eff, n_events=40, seed=315)
        nv = data.synthetic_null_events(b, n_events=2000, window=25, seed=9)
        ln = st.snap_to_sessions(b, nv, lag=0)
        ec = st.car(b, e, window=(0, 1), lag=0)["car"].to_numpy()
        nc = st.window_cars(b, ln, window=(0, 1))
        obs, p = st.permutation_pvalue(ec, nc, n_perm=10000, seed=1)
        print(f"  effect={eff:.2f}: event CAR={ec.mean()*100:+.2f}%  "
              f"t={st.hac_tstat(ec):+.2f}  perm p={p:.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
