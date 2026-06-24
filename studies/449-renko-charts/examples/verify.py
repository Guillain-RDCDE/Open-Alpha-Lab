"""Real-tape verification — Study 449 (Renko-Charts).

Reproduces every number in docs/results.md.  Reads the cached daily tapes (cache-first;
network only with --fetch), builds the ATR-Renko brick series, runs the Renko-filtered
moving-average crossover head-to-head against the same crossover on raw closes and against
buy-and-hold, sweeps costs, runs the return-permutation placebo, and prints the synthetic
positive control.

    python studies/449-renko-charts/examples/verify.py            # cache-only
    python studies/449-renko-charts/examples/verify.py --fetch    # refresh tapes

The headline run is pinned to as-of 2026-05-31 (the last complete month at build time);
the current partial month is dropped before any statistic.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from renko_charts import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "DIA", "IWM", "EFA", "GLD"]
ASOF = "2026-05-31"
FAST, SLOW, ATR_W, ATR_M = 10, 30, 14, 1.0


def _load(t: str, fetch: bool) -> pd.DataFrame:
    b = data.load_real(t, fetch=fetch)
    return b[b.index <= ASOF]


def _pooled(bars: dict, kind: str, cost: float = 0.0) -> pd.Series:
    cols = []
    for t in TICKERS:
        b = bars[t]
        if kind == "renko":
            cols.append(st.crossover_returns(b, st.renko_from_atr(b, ATR_W, ATR_M),
                                             FAST, SLOW, cost))
        elif kind == "raw":
            cols.append(st.crossover_returns(b, b["close"], FAST, SLOW, cost))
        else:
            cols.append(st.buy_and_hold(b))
    return pd.concat(cols, axis=1).mean(axis=1)


def main(fetch: bool) -> None:
    bars = {}
    for t in TICKERS:
        b = _load(t, fetch)
        bars[t] = b
        print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} "
              f"n={len(b)} fp={data.fingerprint(b)}")
    print()

    print("=== per-instrument (fast=10, slow=30, ATR-brick=1.0xATR14, gross) ===")
    print(f"{'tkr':5s} {'renko_S':>8s} {'renko_t':>8s} {'raw_S':>8s} {'bh_S':>8s} "
          f"{'dS':>7s} {'brick':>8s}")
    for t in TICKERS:
        res = st.run_experiment(bars[t], FAST, SLOW, ATR_W, ATR_M, 0.0)
        print(f"{t:5s} {res['renko']['sharpe']:8.3f} {res['renko']['tstat']:8.2f} "
              f"{res['raw']['sharpe']:8.3f} {res['bh']['sharpe']:8.3f} "
              f"{res['delta_sharpe']:7.3f} {res['brick']:8.3f}")

    print("\n=== pooled equal-weight portfolio (gross) ===")
    for name in ("renko", "raw", "bh"):
        p = st.perf(_pooled(bars, name))
        print(f"{name:6s} S={p['sharpe']:.3f} t={p['tstat']:.2f} "
              f"annret={p['ann_ret']*100:+.2f}% annvol={p['ann_vol']*100:.2f}% "
              f"maxdd={p['max_dd']*100:.1f}% exposure={p['exposure']:.2f}")
    sr = st.perf(_pooled(bars, "renko"))["sharpe"]
    sa = st.perf(_pooled(bars, "raw"))["sharpe"]
    sb = st.perf(_pooled(bars, "bh"))["sharpe"]
    print(f"delta Renko - raw   = {sr - sa:+.3f} Sharpe")
    print(f"delta Renko - BH    = {sr - sb:+.3f} Sharpe")

    print("\n=== cost sweep (pooled, Sharpe) ===")
    print(f"{'cost_bps':>8s} {'renko_S':>8s} {'raw_S':>8s} {'delta':>8s}")
    for c in (0.0, 1.0, 2.0, 5.0, 10.0):
        s_r = st.perf(_pooled(bars, "renko", c))["sharpe"]
        s_a = st.perf(_pooled(bars, "raw", c))["sharpe"]
        print(f"{c:8.1f} {s_r:8.3f} {s_a:8.3f} {s_r - s_a:8.3f}")

    print("\n=== turnover (trades/yr, SPY) ===")
    b = bars["SPY"]
    yrs = (b.index[-1] - b.index[0]).days / 365.25
    pos_r = st.crossover_position(st.renko_from_atr(b, ATR_W, ATR_M), FAST, SLOW)
    pos_a = st.crossover_position(b["close"], FAST, SLOW)
    print(f"renko {pos_r.reindex(b.index).fillna(0).diff().abs().sum() / yrs:.1f}  "
          f"raw {pos_a.reindex(b.index).fillna(0).diff().abs().sum() / yrs:.1f}")

    print("\n=== return-permutation placebo (SPY, Renko-minus-raw Sharpe) ===")
    pl = st.permutation_pvalue(b, FAST, SLOW, ATR_W, ATR_M, n_perm=500, seed=449)
    print(f"observed delta={pl['observed_delta']:+.4f}  p={pl['p_value']:.3f}  "
          f"n_valid={pl['n_valid']}")

    print("\n=== synthetic positive control (crossover Sharpe; renko vs raw vs BH) ===")
    print(f"{'trend':>6s} {'renko_S':>8s} {'raw_S':>8s} {'bh_S':>8s} {'d(R-raw)':>9s}")
    for tr in (0.0, 0.05, 0.10, 0.15):
        d, _ = data.synthetic_panel(n_days=4000, trend=tr, annual_drift=0.0, seed=449)
        res = st.run_experiment(d, FAST, SLOW, ATR_W, ATR_M, 0.0)
        print(f"{tr:6.2f} {res['renko']['sharpe']:8.3f} {res['raw']['sharpe']:8.3f} "
              f"{res['bh']['sharpe']:8.3f} {res['delta_sharpe']:9.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
