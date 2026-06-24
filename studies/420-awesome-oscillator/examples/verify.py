"""Real-tape verification — Study 420 (Awesome Oscillator). Regenerates docs/results.md numbers.

Reads (or fetches with --fetch) SPY daily OHLC back to 1993, runs the AO(5/34) long/flat
timing rule vs buy-and-hold and an identical MACD(12/26) rule, plus a random-timing control,
on excess-of-cash returns. Reports excess Sharpe, CAGR, max drawdown, HAC difference t-stats,
a rotation permutation placebo, a cost sweep, sub-periods, and the synthetic power control.
Network is touched only with --fetch (or a cache miss).

    python studies/420-awesome-oscillator/examples/verify.py             # cache-only
    python studies/420-awesome-oscillator/examples/verify.py --fetch     # refresh the tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from awesome_oscillator import data, strategy as st  # noqa: E402

TICKER = "SPY"
END = "2026-06-12"      # pin: drop any partial/today bar
TBILL_RATE = 0.04       # flat cash proxy (FRED unavailable)
COST_BPS = 2.0          # one-way


def main(fetch: bool) -> None:
    df = data.load_real(TICKER, start="1993-01-01", fetch=fetch).loc[:END]
    print(f"{TICKER}: {df.index[0].date()} -> {df.index[-1].date()}  "
          f"n={len(df)}  fp={data.fingerprint(df)}")

    tbill_daily = TBILL_RATE / 252.0
    res = st.run_experiment(df, tbill_daily=tbill_daily, cost_bps=COST_BPS)

    def fmt(s: dict) -> str:
        return (f"exCAGR={s['cagr']:+.2%}  exSharpe={s['sharpe']:+.3f}  "
                f"Vol={s['vol_ann']:.2%}  MaxDD={s['max_drawdown']:+.2%}")

    print(f"\nIn-market: AO {res['ao_in_mkt']:.1%}  MACD {res['macd_in_mkt']:.1%}  "
          f"| switches AO {res['ao_switches']}  MACD {res['macd_switches']}")
    print(f"\nBuy-and-hold (excess):  {fmt(res['bh_excess'])}")
    print(f"AO   timing  (excess):  {fmt(res['ao_excess'])}")
    print(f"MACD timing  (excess):  {fmt(res['macd_excess'])}")
    print(f"Random timing(excess):  {fmt(res['random_excess'])}")
    print(f"\n[gross] BH Sharpe {res['bh']['sharpe']:+.3f}  AO {res['ao']['sharpe']:+.3f}  "
          f"MACD {res['macd']['sharpe']:+.3f}")
    print(f"\nHAC difference t-stats (excess):")
    print(f"  AO   vs BH    : {res['t_ao_vs_bh']:+.2f}   (H1: beats index?  needs >= +2)")
    print(f"  MACD vs BH    : {res['t_macd_vs_bh']:+.2f}")
    print(f"  AO   vs MACD  : {res['t_ao_vs_macd']:+.2f}   (H2: beats MACD?  needs >= +2)")
    print(f"  AO   vs random: {res['t_ao_vs_random']:+.2f}   (H3: timing > exposure?)")

    # Rotation permutation placebo (gross, AO vs BH)
    perm = st.permutation_pvalue(df["close"], st.ao_signal(df),
                                 tbill_daily=tbill_daily, cost_bps=0.0,
                                 n_draws=2000, seed=420)
    print(f"\nRotation placebo: obs edge {perm['obs_edge_bps']:+.2f} bps/day  "
          f"p = {perm['p_value']:.3f}")

    # Cost sweep (AO excess Sharpe)
    print("\n=== cost sweep (AO excess Sharpe) ===")
    ao_sig = st.ao_signal(df)
    for c in (0.0, 1.0, 2.0, 5.0, 10.0):
        s = st.summary(st.run_backtest(df["close"], ao_sig, tbill_daily=tbill_daily,
                                       cost_bps=c)["r_strat_excess"])
        print(f"  cost={c:5.1f}bps  exSharpe={s['sharpe']:+.3f}")

    # Sub-periods
    print("\n=== sub-period breakdown (excess Sharpe: AO / MACD / BH) ===")
    for name, s, e in [("Pre-2000", "1993-01-01", "1999-12-31"),
                       ("2000s",    "2000-01-01", "2009-12-31"),
                       ("2010s",    "2010-01-01", "2019-12-31"),
                       ("2020s",    "2020-01-01", END)]:
        sub = df.loc[s:e]
        if len(sub) < 300:
            continue
        r = st.run_experiment(sub, tbill_daily=tbill_daily, cost_bps=COST_BPS)
        print(f"  {name:9s}: AO {r['ao_excess']['sharpe']:+.2f}  "
              f"MACD {r['macd_excess']['sharpe']:+.2f}  BH {r['bh_excess']['sharpe']:+.2f}")

    # Synthetic power control
    print("\n=== synthetic control (seed=123): null vs planted bears ===")
    for edge, lbl in [(0.0, "Null (1 regime)"), (1.5, "Planted bears")]:
        bars, truth = data.synthetic_panel(n_days=10000, edge=edge, annual_drift=0.16,
                                           annual_vol=0.14, seed=123,
                                           p_bull_to_bear=0.05, p_bear_to_bull=0.07)
        r = st.run_experiment(bars, tbill_daily=tbill_daily, cost_bps=COST_BPS)
        print(f"  {lbl:16s}: AO exSh {r['ao_excess']['sharpe']:+.3f}  "
              f"BH exSh {r['bh_excess']['sharpe']:+.3f}  t_ao_vs_bh {r['t_ao_vs_bh']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
