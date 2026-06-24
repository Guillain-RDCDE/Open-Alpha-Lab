"""Real-tape verification — Study 429 (Mass Index, Donald Dorsey).

Reproduces every number in docs/results.md: the bulge event study (forward 5/10/20/40-day
return after each bulge vs the base rate, raw and trend-signed, with a one-sample t, a Welch
t vs base, and a rare-event placebo), the post-bulge fade timing race vs buy-and-hold (net
Sharpe, excess-of-cash, 1-day lag, HAC t of the daily difference), the variant & cost sweeps,
the 5-ETF panel, and the synthetic planted-reversal positive control. Cache-first: the
network is touched only with --fetch on a cache miss.

    python studies/429-mass-index/examples/verify.py            # cache-only
    python studies/429-mass-index/examples/verify.py --fetch    # refresh tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mass_index import data, strategy as st  # noqa: E402

ASOF = "2026-06-23"   # drop the partial 2026-06-24 bar (house rule)


def _spy(fetch: bool):
    b = data.load_real("SPY", fetch=fetch)
    return b[b.index <= ASOF]


def main(fetch: bool) -> None:
    if fetch:
        for t in data.PANEL:
            b = data.load_real(t, fetch=True)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} n={len(b)} fp={data.fingerprint(b)}")
        print()

    spy = _spy(fetch)
    nb = int(st.bulge_signal(spy).sum())
    print(f"SPY {spy.index[0].date()}..{spy.index[-1].date()} n={len(spy)} "
          f"fp={data.fingerprint(spy)}  as-of {ASOF}  years={len(spy)/252:.1f}  bulges={nb}")

    # --- event study: raw forward return after a bulge vs base ---
    print("\n=== event study (raw forward return after bulge vs base rate) ===")
    for h in st.HORIZONS:
        e = st.event_study(spy, h, fade=False)
        pl = st.placebo_pvalue(spy, h, fade=False, n_draws=5000)
        print(f"h={h:3d}  n={e['n_bulge']:3d}  bulge={e['bulge_mean']*100:+.2f}%  "
              f"base={e['base_mean']*100:+.2f}%  win={e['win']*100:.0f}%  "
              f"t0={e['t_vs_zero']:+.2f}  t_vs_base={e['t_vs_base']:+.2f}  placebo_p={pl['p_value']:.3f}")

    # --- event study: trend-signed (the reversal/fade test) ---
    print("\n=== event study (FADE: forward return signed against the prevailing trend) ===")
    for h in st.HORIZONS:
        e = st.event_study(spy, h, fade=True)
        pl = st.placebo_pvalue(spy, h, fade=True, n_draws=5000)
        print(f"h={h:3d}  n={e['n_bulge']:3d}  fade={e['bulge_mean']*100:+.2f}%  "
              f"base={e['base_mean']*100:+.2f}%  win={e['win']*100:.0f}%  "
              f"t0={e['t_vs_zero']:+.2f}  t_vs_base={e['t_vs_base']:+.2f}  placebo_p={pl['p_value']:.3f}")

    # --- timing race ---
    r = st.run_experiment(spy, hold=20, long_short=False, cost_bps=1.0)
    print("\n=== timing race (post-bulge fade long/flat, hold 20, 1bp, excess-of-cash) ===")
    print(f"fade      Sharpe={r['fade_sharpe']:+.3f} CAGR={r['fade_cagr']:+.3f} maxDD={r['fade_mdd']:+.3f} "
          f"TiM={r['fade_tim']:.2f} turnover/yr={r['fade_turn_yr']:.1f}")
    print(f"buy&hold  Sharpe={r['asset_sharpe']:+.3f} CAGR={r['asset_cagr']:+.3f} maxDD={r['asset_mdd']:+.3f}")
    print(f"HAC t (fade - buy-hold) = {r['t_vs_hold']:+.2f}")

    # --- variants ---
    print("\n=== timing variants (1bp) ===")
    for hold in (10, 20, 40):
        for ls in (False, True):
            rr = st.run_experiment(spy, hold=hold, long_short=ls, cost_bps=1.0)
            print(f"hold={hold:3d} {'L/S' if ls else 'L/F'}  Sharpe={rr['fade_sharpe']:+.3f}  "
                  f"t_vs_hold={rr['t_vs_hold']:+.2f}")

    # --- cost sweep ---
    print("\n=== cost sweep (hold 20, long/flat) ===")
    for c in (0.0, 1.0, 2.0, 5.0):
        rr = st.run_experiment(spy, hold=20, cost_bps=c)
        print(f"cost={c:4.1f}bps  Sharpe={rr['fade_sharpe']:+.3f}  t_vs_hold={rr['t_vs_hold']:+.2f}")

    # --- panel ---
    print("\n=== panel (fade event study @ h=20, and timing) ===")
    for t in data.PANEL:
        if not data.have_real(t):
            continue
        bb = data.load_real(t)
        bb = bb[bb.index <= ASOF]
        e = st.event_study(bb, 20, fade=True)
        rr = st.run_experiment(bb, hold=20, cost_bps=1.0)
        print(f"{t:5s} bulges={e['n_bulge']:3d}  fade@20={e['bulge_mean']*100:+.2f}%  t0={e['t_vs_zero']:+.2f}  "
              f"| timing Sharpe={rr['fade_sharpe']:+.3f}  buy-hold={rr['asset_sharpe']:+.3f}  t={rr['t_vs_hold']:+.2f}")

    # --- synthetic positive control ---
    print("\n=== synthetic control (fade event study @ h=20, timing L/S hold 25) ===")
    for edge in (0.0, 3.0):
        bars, truth = data.synthetic_panel(n_days=6000, edge=edge, seed=429)
        e = st.event_study(bars, 20, fade=True)
        pl = st.placebo_pvalue(bars, 20, fade=True, n_draws=3000)
        rr = st.run_experiment(bars, hold=25, long_short=True, cost_bps=1.0)
        print(f"edge={edge:.1f}: detected={e['n_bulge']:3d}  fade@20={e['bulge_mean']*100:+.2f}%  "
              f"t0={e['t_vs_zero']:+.2f}  placebo_p={pl['p_value']:.3f}  | timing Sharpe={rr['fade_sharpe']:+.3f}  "
              f"buy-hold={rr['asset_sharpe']:+.3f}  t_vs_hold={rr['t_vs_hold']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
