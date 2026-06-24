"""Real-tape verification — Study 435 (Guppy Multiple MA). Regenerates docs/results.md.

Loads (cache-first) SPY daily total-return closes back to 1993, runs the GMMA ribbon-cross
and conviction-gated rules as long/flat timing strategies, races their excess-of-cash Sharpe
against buy-and-hold, a random-timing control and a single 50-day SMA/EMA, and reports HAC
t-stats, a rotation permutation placebo, a cost sweep, and the synthetic positive control.
Network is touched only with --fetch (and only on a cache miss).

    python studies/435-guppy-mma/examples/verify.py            # cache-only (offline)
    python studies/435-guppy-mma/examples/verify.py --fetch    # refresh the SPY tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from guppy_mma import data, strategy as st  # noqa: E402

TICKER = "SPY"
RF = 0.04 / 252.0   # flat 4%/yr cash proxy (FRED unavailable)
COST_BPS = 2.0      # one-way x NAV
SMA_N = 50


def fmt(s: dict) -> str:
    return (f"CAGR={s['cagr']:+.2%}  Sharpe={s['sharpe']:+.3f}  "
            f"Vol={s['vol_ann']:.1%}  MaxDD={s['max_drawdown']:+.1%}  t={s['tstat']:+.2f}")


def main(fetch: bool) -> None:
    df = data.load_real(TICKER, start="1993-01-01", fetch=fetch)
    close = df["close"]
    print(f"{TICKER}: {df.index[0].date()} -> {df.index[-1].date()}  "
          f"n={len(df)}  fp={data.fingerprint(df)}")

    res = st.run_experiment(close, rf_daily=RF, cost_bps=COST_BPS, sma_n=SMA_N, allow_short=False)

    print(f"\nIn-market fraction (ribbon): {res['in_market_frac']:.1%}  |  "
          f"switches: {res['n_switches']}")
    print("\n--- excess-of-cash race (long/flat, 2 bps one-way) ---")
    for k, lbl in [("bh", "Buy-and-hold"), ("ribbon", "GMMA ribbon cross"),
                   ("conviction", "GMMA conviction"), ("sma", "Single 50d SMA"),
                   ("ema", "Single 50d EMA"), ("random", "Random timing (null)")]:
        print(f"  {lbl:22s} {fmt(res[k])}")

    print("\n--- Sharpe-race HAC t-stats (on daily return difference) ---")
    print(f"  ribbon vs buy-and-hold : {res['t_ribbon_vs_bh']:+.2f}")
    print(f"  ribbon vs random timing: {res['t_ribbon_vs_rand']:+.2f}")
    print(f"  ribbon vs single SMA   : {res['t_ribbon_vs_sma']:+.2f}")
    print(f"  conviction vs ribbon   : {res['t_conv_vs_ribbon']:+.2f}")

    print("\n--- rotation permutation placebo (ribbon, gross) ---")
    perm = st.permutation_pvalue(close, st.ribbon_signal, rf_daily=0.0,
                                 cost_bps=0.0, n_perm=2000, seed=435)
    print(f"  real mean excess = {perm['real_mean_bps']:+.2f} bps/day  |  "
          f"placebo mean = {perm['perm_mean_bps']:+.2f}  |  p = {perm['p_value']:.3f}")

    print("\n--- cost sweep (ribbon, excess Sharpe) ---")
    sig = st.ribbon_signal(close)
    for c in (0.0, 2.0, 5.0, 10.0):
        bt = st.run_backtest(close, sig, rf_daily=RF, cost_bps=c)
        s = st.summary(bt["r_strategy_excess"])
        print(f"  cost={c:5.1f}bps  Sharpe={s['sharpe']:+.3f}  CAGR={s['cagr']:+.2%}")

    print("\n--- synthetic positive control (planted edge vs flat null) ---")
    for edge, lbl in [(1.0, "planted edge"), (0.0, "flat null")]:
        d, truth = data.synthetic_panel(
            n_years=35, edge=edge, drift_bull=0.16, drift_bear=-0.55,
            vol_bull=0.10, vol_bear=0.45, p_bull_to_bear=0.12, p_bear_to_bull=0.20, seed=435,
        )
        r = st.run_experiment(d["close"], rf_daily=RF, cost_bps=COST_BPS)
        p = st.permutation_pvalue(d["close"], st.ribbon_signal, n_perm=2000, seed=11)
        print(f"  {lbl:13s}: ribbon Sharpe={r['ribbon']['sharpe']:+.3f}  "
              f"BH Sharpe={r['bh']['sharpe']:+.3f}  t(ribbon-BH)={r['t_ribbon_vs_bh']:+.2f}  "
              f"perm p={p['p_value']:.4f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
