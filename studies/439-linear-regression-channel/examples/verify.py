"""Real-tape verification — Study 439 (Linear-Regression-Channel).

Reproduces every number in docs/results.md. Cache-first: reads the cached daily tapes
under ../_cache/ and runs entirely offline. Pass --fetch once to (re)populate the cache.

    python studies/439-linear-regression-channel/examples/verify.py            # cache-only
    python studies/439-linear-regression-channel/examples/verify.py --fetch    # refresh tapes

Headline tape: SPY (total-return / auto-adjusted) daily closes. The slope-rule (long when
the 60-day OLS log-price slope > 0) is raced NET, excess-of-buy-and-hold, against the
obvious benchmark (price > 60-day SMA) and against buy-and-hold itself. One execution lag
(enter next close), one-way costs × NAV turnover, HAC t on the daily active return, a
block-permutation placebo, and a synthetic positive control.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from linear_regression_channel import data, strategy as st  # noqa: E402

PANEL = data.PANEL
WINDOW = 60
COST = 1.0
N_DRAWS = 2000


def main(fetch: bool) -> None:
    if fetch:
        for t in PANEL:
            b = data.load_real(t, fetch=True)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} "
                  f"n={len(b)} fp={data.fingerprint(b)}")
        print()

    spy = data.load_real("SPY", fetch=False)
    close = spy["close"]
    print("=== data stamp (SPY) ===")
    print(f"window {close.index[0].date()} -> {close.index[-1].date()}  "
          f"n={len(close)}  fingerprint={data.fingerprint(spy)}")
    yrs = len(close) / st.TRADING_DAYS
    print(f"years ~ {yrs:.1f}\n")

    # --- headline: slope vs SMA vs buy-and-hold, NET, excess-vs-excess (long/flat) ---
    res = st.run_experiment(close, window=WINDOW, cost_bps=COST, mode="long_flat",
                            do_permutation=True, n_draws=N_DRAWS)
    sl, sma = res["slope"], res["sma"]
    print("=== SPY: 60d slope rule vs 60d SMA rule vs buy-and-hold (NET, long/flat) ===")
    print(f"buy&hold   Sharpe(excess)={sl['sharpe_bh']:+.3f}  CAGR={sl['cagr_bh']*100:+.2f}%")
    print(f"SLOPE rule Sharpe(excess)={sl['sharpe_strat']:+.3f}  gap={sl['sharpe_gap']:+.3f}  "
          f"active_t={sl['active_t']:+.2f}  TIM={sl['time_in_mkt']*100:.0f}%  "
          f"turn/yr={sl['turnover_yr']:.2f}  CAGRnet={sl['cagr_net']*100:+.2f}%")
    print(f"SMA   rule Sharpe(excess)={sma['sharpe_strat']:+.3f}  gap={sma['sharpe_gap']:+.3f}  "
          f"active_t={sma['active_t']:+.2f}  TIM={sma['time_in_mkt']*100:.0f}%  "
          f"turn/yr={sma['turnover_yr']:.2f}  CAGRnet={sma['cagr_net']*100:+.2f}%")
    perm = res["slope_perm"]
    print(f"slope active-mean placebo: obs={perm['obs']*1e4:+.3f}bps/day  p={perm['p_value']:.3f}")
    print()

    # --- cost sweep on the slope rule (net active Sharpe gap) ---
    print("=== SPY slope rule: cost sweep (net) ===")
    for c in (0.0, 0.5, 1.0, 2.0, 5.0):
        r = st.run_experiment(close, window=WINDOW, cost_bps=c, mode="long_flat")
        s = r["slope"]
        print(f"cost={c:4.1f}bps  Sharpe(strat)={s['sharpe_strat']:+.3f}  "
              f"gap={s['sharpe_gap']:+.3f}  active_t={s['active_t']:+.2f}")
    print()

    # --- window sweep (is the 60d a cherry pick?) ---
    print("=== SPY slope rule: window sweep (net, cost=1bp) ===")
    for w in (20, 40, 60, 100, 200):
        r = st.run_experiment(close, window=w, cost_bps=COST, mode="long_flat")
        s = r["slope"]
        print(f"window={w:3d}  Sharpe(strat)={s['sharpe_strat']:+.3f}  "
              f"gap={s['sharpe_gap']:+.3f}  active_t={s['active_t']:+.2f}")
    print()

    # --- long/short variant ---
    rls = st.run_experiment(close, window=WINDOW, cost_bps=COST, mode="long_short")
    print("=== SPY slope rule, long/SHORT variant (net) ===")
    print(f"Sharpe(strat)={rls['slope']['sharpe_strat']:+.3f}  "
          f"gap={rls['slope']['sharpe_gap']:+.3f}  active_t={rls['slope']['active_t']:+.2f}")
    print()

    # --- small panel: how often does the slope rule beat buy-and-hold? ---
    print("=== panel: slope-rule active_t vs buy-and-hold (net, 60d, long/flat) ===")
    beat = 0
    for t in PANEL:
        if not data.have_real(t):
            continue
        c = data.load_real(t, fetch=False)["close"]
        s = st.evaluate(c, st.slope_signal(c, window=WINDOW), cost_bps=COST, label=t)
        flag = "BEAT" if s["sharpe_gap"] > 0 else "lost"
        if s["sharpe_gap"] > 0:
            beat += 1
        print(f"{t:5s} Sharpe(strat)={s['sharpe_strat']:+.3f}  bh={s['sharpe_bh']:+.3f}  "
              f"gap={s['sharpe_gap']:+.3f}  active_t={s['active_t']:+.2f}  [{flag}]")
    print(f"slope beat buy-and-hold on {beat}/{len(PANEL)} names")
    print()

    # --- synthetic positive control: machinery proof ---
    print("=== synthetic control: zero edge must NOT light up; planted edge must ===")
    for edge in (0.0, 0.50):
        bars, truth = data.synthetic_panel(edge=edge, seed=439)
        r = st.run_experiment(bars["close"], window=WINDOW, cost_bps=COST,
                              mode="long_flat", do_permutation=True, n_draws=N_DRAWS)
        s = r["slope"]
        print(f"planted edge={edge:+.2f}/yr  Sharpe(strat)={s['sharpe_strat']:+.3f}  "
              f"bh={s['sharpe_bh']:+.3f}  gap={s['sharpe_gap']:+.3f}  "
              f"active_t={s['active_t']:+.2f}  perm_p={r['slope_perm']['p_value']:.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
