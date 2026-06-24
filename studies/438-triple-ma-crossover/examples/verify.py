"""Real-tape verification — Study 438 (Triple-MA-Crossover). Regenerates docs/results.md.

Reads (or fetches) ^GSPC daily price-only closes back to 1993, runs the triple-MA ribbon
timing rule (5/20/50) vs buy-and-hold and the simpler two-MA cross, and reports net excess
Sharpe, CAGR, max drawdown, HAC t-stats, a block-permutation null, a config sweep and a
cost sweep. The in-progress final bar (today) is dropped so no partial day enters the run.
Network is touched only with --fetch.

    python studies/438-triple-ma-crossover/examples/verify.py             # cache-only
    python studies/438-triple-ma-crossover/examples/verify.py --fetch     # refresh the tape
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from triple_ma_crossover import data, strategy as st  # noqa: E402

TICKER = "^GSPC"
ASOF_EXCL = "2026-06-24"   # drop today's in-progress bar from the stamped run
FAST, MID, SLOW = 5, 20, 50
RF = 0.03                  # ^GSPC is price-only; flat cash proxy -> excess-vs-excess race
COST_BPS = 5.0


def main(fetch: bool) -> None:
    df = data.load_real(TICKER, start="1993-01-01", fetch=fetch)
    df = df[df.index < ASOF_EXCL]
    close = df["close"]
    print(f"{TICKER}: {df.index[0].date()} -> {df.index[-1].date()}  "
          f"n={len(df)}  fp={data.fingerprint(df)}")

    res = st.run_experiment(close, FAST, MID, SLOW, rf_annual=RF, cost_bps=COST_BPS)

    def fmt(s):
        return (f"Sharpe(ex)={s['sharpe']:+.3f}  CAGR={s['cagr']:+.2%}  "
                f"MaxDD={s['max_drawdown']:+.2%}  HAC_t={s['tstat']:+.2f}")

    print(f"\nin-market: triple {res['in_market_triple']:.1%}  dual {res['in_market_dual']:.1%}"
          f"  | triple switches/yr ~{res['switches_triple']/(len(df)/252):.1f}")
    print(f"\nBuy-and-hold:   {fmt(res['bh'])}")
    print(f"Two-MA  {FAST}/{SLOW}:   {fmt(res['dual'])}")
    print(f"Three-MA {FAST}/{MID}/{SLOW}: {fmt(res['triple'])}")
    print(f"\nHAC Sharpe-diff t (triple - BH):   {res['t_triple_vs_bh']:+.2f}")
    print(f"HAC Sharpe-diff t (triple - dual): {res['t_triple_vs_dual']:+.2f}")

    print("\n=== config robustness (triple vs dual vs BH, net excess Sharpe) ===")
    for cfg in [(5, 20, 50), (10, 50, 200), (20, 50, 100), (5, 10, 20)]:
        r = st.run_experiment(close, *cfg, rf_annual=RF, cost_bps=COST_BPS)
        print(f"  {cfg}: triple Sh={r['triple']['sharpe']:+.3f} t={r['triple']['tstat']:+.2f}"
              f" | dual Sh={r['dual']['sharpe']:+.3f} | BH={r['bh']['sharpe']:+.3f}"
              f" | t(tri-dual)={r['t_triple_vs_dual']:+.2f}")

    print("\n=== cost sweep (triple 5/20/50) ===")
    for c in (0.0, 5.0, 10.0, 25.0):
        r = st.run_experiment(close, FAST, MID, SLOW, rf_annual=RF, cost_bps=c)
        print(f"  cost={c:5.1f}bps  Sharpe={r['triple']['sharpe']:+.3f}  CAGR={r['triple']['cagr']:+.2%}")

    print("\n=== block-permutation null (triple net excess Sharpe, 2000 draws) ===")
    pp = st.permutation_pvalue(close, FAST, MID, SLOW, rf_annual=RF, cost_bps=COST_BPS,
                               n_draws=2000, seed=438)
    print(f"  observed Sharpe={pp['obs']:+.3f}  p={pp['p_value']:.2f}"
          f"  null mean={float(np.mean(pp['draws'])):+.3f}  null sd={float(np.std(pp['draws'])):.3f}")

    print("\n=== synthetic positive control (faithful-engine / power) ===")
    for edge in (0.0, 2.0):
        d, t = data.synthetic_panel(n_days=8000, edge=edge, seed=438)
        r = st.run_experiment(d["close"], FAST, MID, SLOW, rf_annual=0.0, cost_bps=COST_BPS)
        print(f"  edge={edge}: triple Sharpe={r['triple']['sharpe']:+.3f} t={r['triple']['tstat']:+.2f}"
              f" (bh={r['bh']['sharpe']:+.3f})")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
