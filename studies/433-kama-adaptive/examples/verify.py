"""Real-tape verification — Study 433 (KAMA). Regenerates docs/results.md numbers.

Reads six daily tapes from cache (cache-first), builds the KAMA(10,2,30) long/flat timing
book, races it against the matched SMA(30) book and buy-and-hold, sweeps costs and
parameters, runs the permutation placebo on SPY, and prints the synthetic positive control.
Network is touched only on a cache miss or with --fetch. As-of is pinned to 2026-05-31 so
the partial current month is dropped.

    python studies/433-kama-adaptive/examples/verify.py            # cache-first
    python studies/433-kama-adaptive/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kama_adaptive import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM", "EFA", "GLD", "AAPL"]
CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))
ASOF = "2026-05-31"  # drop the partial current month


def _get(ticker: str, fetch: bool):
    b = data.load_real(ticker, fetch=fetch, cache_dir=CACHE)
    return b[b.index <= ASOF]


def main(fetch: bool) -> None:
    print(f"=== Data stamp (as-of {ASOF}) ===")
    for t in TICKERS:
        b = _get(t, fetch)
        print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} "
              f"n={len(b):5d} fp={data.fingerprint(b)}")

    print("\n=== Headline (SPY, long/flat, net 1 bp): KAMA vs SMA vs buy-hold ===")
    spy = _get("SPY", False)
    r = st.run_experiment(spy["close"], er_period=10, fast=2, slow=30,
                          sma_window=30, cost_bps=1.0, n_perm=2000)
    print(f"  KAMA  Sharpe={r['kama_sharpe']:+.3f} CAGR={r['kama_cagr']:+.4f} "
          f"MDD={r['kama_mdd']:+.3f} turnover={r['kama_turnover']:.0f}")
    print(f"  SMA   Sharpe={r['sma_sharpe']:+.3f} CAGR={r['sma_cagr']:+.4f} "
          f"MDD={r['sma_mdd']:+.3f} turnover={r['sma_turnover']:.0f}")
    print(f"  BUYH  Sharpe={r['bh_sharpe']:+.3f} CAGR={r['bh_cagr']:+.4f} "
          f"MDD={r['bh_mdd']:+.3f}")
    print(f"  KAMA - SMA : {r['diff_ks_mean_bps']:+.3f} bps/day  HAC t={r['diff_ks_t']:+.2f}")
    print(f"  KAMA - BH  : {r['diff_kbh_mean_bps']:+.3f} bps/day  HAC t={r['diff_kbh_t']:+.2f}")
    print(f"  permutation p (KAMA book) = {r['perm_p']:.4f}")

    print("\n=== Panel (long/flat, net 1 bp) ===")
    for t in TICKERS:
        b = _get(t, False)
        rr = st.run_experiment(b["close"], cost_bps=1.0)
        print(f"{t:5s} kama={rr['kama_sharpe']:+.3f} sma={rr['sma_sharpe']:+.3f} "
              f"bh={rr['bh_sharpe']:+.3f} KS_t={rr['diff_ks_t']:+.2f}")

    print("\n=== Cost sweep (SPY, KAMA - SMA) ===")
    for c in (0.0, 0.5, 1.0, 2.0, 5.0):
        rr = st.run_experiment(spy["close"], cost_bps=c)
        print(f"cost={c:.1f}bp  kama={rr['kama_sharpe']:+.3f} sma={rr['sma_sharpe']:+.3f} "
              f"KS_bps={rr['diff_ks_mean_bps']:+.3f} KS_t={rr['diff_ks_t']:+.2f}")

    print("\n=== Robustness grid (SPY, net 1 bp) ===")
    for slow in (20, 30, 50):
        rr = st.run_experiment(spy["close"], slow=slow, sma_window=slow, cost_bps=1.0)
        print(f"slow={slow:3d}/sma={slow:3d}  kama={rr['kama_sharpe']:+.3f} "
              f"sma={rr['sma_sharpe']:+.3f} KS_t={rr['diff_ks_t']:+.2f}")
    for er in (5, 10, 20):
        rr = st.run_experiment(spy["close"], er_period=er, cost_bps=1.0)
        print(f"er={er:2d}        kama={rr['kama_sharpe']:+.3f} "
              f"sma={rr['sma_sharpe']:+.3f} KS_t={rr['diff_ks_t']:+.2f}")
    ls = st.run_experiment(spy["close"], cost_bps=1.0, long_short=True)
    print(f"long/short    kama={ls['kama_sharpe']:+.3f} sma={ls['sma_sharpe']:+.3f} "
          f"bh={ls['bh_sharpe']:+.3f} KS_t={ls['diff_ks_t']:+.2f}")

    print("\n=== Synthetic positive control (cost 2 bp) ===")
    for edge in (0.0, 2.0, 4.0, 8.0):
        d, _ = data.synthetic_panel(n_days=6000, edge=edge, seed=433)
        rr = st.run_experiment(d["close"], cost_bps=2.0)
        print(f"edge={edge:.1f}  kama={rr['kama_sharpe']:+.3f} sma={rr['sma_sharpe']:+.3f} "
              f"KS_bps={rr['diff_ks_mean_bps']:+.3f} KS_t={rr['diff_ks_t']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
