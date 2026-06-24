"""Real-tape verification — Study 421 (Williams Alligator). Regenerates docs/results.md numbers.

Loads (cache-first) SPY daily total-return closes back to 1993, computes the Alligator
(13/8/5 SMMA fan), turns it into a long/flat and long/short timing rule, and races the
NET Sharpe vs buy-and-hold and vs the simpler SMA(200) benchmark — with HAC difference
t-stats, a block-permutation placebo, a cost sweep, and a synthetic positive control.

    python studies/421-williams-alligator/examples/verify.py            # cache-first (offline if cached)
    python studies/421-williams-alligator/examples/verify.py --fetch    # force a refresh
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from williams_alligator import data, strategy as st  # noqa: E402

TICKER = "SPY"
SMA_N = 200
TBILL_RATE = 0.04   # flat proxy for cash yield (FRED unavailable; conservative)
TBILL_DAILY = TBILL_RATE / 252.0
COST_BPS = 5.0      # one-way transaction cost on NAV
BORROW_BPS = 50.0   # annual borrow on the short leg
END = "2026-06-23"  # pin to the last complete trading day (drop today's partial bar)


def fmt(s: dict) -> str:
    return (f"CAGR={s['cagr']:+.2%}  Sharpe={s['sharpe']:+.3f}  "
            f"Vol={s['vol_ann']:.2%}  MaxDD={s['max_drawdown']:+.2%}  t={s['tstat']:+.2f}")


def main(fetch: bool) -> None:
    if fetch and os.path.exists(data._cache_path(TICKER, data.DEFAULT_CACHE)):
        os.remove(data._cache_path(TICKER, data.DEFAULT_CACHE))
    bars = data.load_real(TICKER, start="1993-01-01")
    bars = bars.loc[:END]  # pin to last complete trading day (no partial bar in a stamped run)
    print(f"{TICKER}: {bars.index[0].date()} -> {bars.index[-1].date()}  "
          f"n={len(bars)}  fp={data.fingerprint(bars)}")

    # --- long/flat (headline) ----------------------------------------------
    res_lf = st.run_experiment(bars, tbill_daily=TBILL_DAILY, cost_bps=COST_BPS,
                               borrow_bps_ann=BORROW_BPS, mode="long_flat",
                               sma_n=SMA_N, placebo=True, n_draws=2000)
    print("\n=== long/flat ===")
    print(f"In-position fraction: {res_lf['in_pos_frac']:.1%}  |  position changes: {res_lf['n_changes']}")
    print(f"Buy-and-hold:         {fmt(res_lf['bh'])}")
    print(f"Alligator long/flat:  {fmt(res_lf['alligator'])}")
    print(f"SMA(200) benchmark:   {fmt(res_lf['sma'])}")
    print(f"Sharpe-diff t (Allig vs BH):   {res_lf['t_allig_vs_bh']:+.2f}")
    print(f"Sharpe-diff t (Allig vs SMA):  {res_lf['t_allig_vs_sma']:+.2f}")
    print(f"Block-permutation placebo p:   {res_lf['placebo_p']:.3f}  (obs diff {res_lf['placebo_obs']:+.3f})")

    # --- long/short --------------------------------------------------------
    res_ls = st.run_experiment(bars, tbill_daily=TBILL_DAILY, cost_bps=COST_BPS,
                               borrow_bps_ann=BORROW_BPS, mode="long_short",
                               sma_n=SMA_N, placebo=False)
    print("\n=== long/short ===")
    print(f"In-position fraction: {res_ls['in_pos_frac']:.1%}  |  position changes: {res_ls['n_changes']}")
    print(f"Alligator long/short: {fmt(res_ls['alligator'])}")
    print(f"Sharpe-diff t (Allig vs BH):   {res_ls['t_allig_vs_bh']:+.2f}")

    # --- cost sweep (long/flat) -------------------------------------------
    print("\n=== cost sweep (Alligator long/flat, one-way bps) ===")
    sig_a = st.alligator_signal(bars, mode="long_flat")
    for c in (0.0, 1.0, 5.0, 10.0, 25.0):
        bt = st.run_backtest(bars, sig_a, tbill_daily=TBILL_DAILY, cost_bps=c, borrow_bps_ann=BORROW_BPS)
        s = st.summary(bt["r_strategy"])
        print(f"  cost={c:5.1f}bps  Sharpe={s['sharpe']:+.3f}  CAGR={s['cagr']:+.2%}  MaxDD={s['max_drawdown']:+.2%}")

    # --- synthetic positive control (long/short on symmetric planted trends) ----
    print("\n=== synthetic control (planted multi-week trends, long/short) ===")
    for edge, lbl in ((0.0, "null (edge=0)"), (1.5, "trend (edge=1.5)")):
        sb, _ = data.synthetic_panel(n_days=4000, edge=edge, seed=421)
        r = st.run_experiment(sb, tbill_daily=TBILL_DAILY, cost_bps=COST_BPS,
                              borrow_bps_ann=BORROW_BPS, mode="long_short", sma_n=SMA_N)
        print(f"  {lbl:20s}: BH Sharpe={r['bh']['sharpe']:+.3f}  Allig Sharpe={r['alligator']['sharpe']:+.3f}  "
              f"Allig-t={r['alligator']['tstat']:+.2f}  diff-t={r['t_allig_vs_bh']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
