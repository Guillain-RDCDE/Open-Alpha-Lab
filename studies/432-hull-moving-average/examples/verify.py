"""Real-tape verification — Study 432 (Hull Moving Average). Regenerates results.md numbers.

Reads (or fetches) five daily tapes, builds the HMA(16)-slope long/flat timing rule, races
it net of costs against buy-and-hold and against SMA(50) and MACD benchmarks, runs the
position-shuffle permutation placebo, sweeps costs, breaks down per instrument, splits the
sample in two, and runs the synthetic positive control. Network is touched only with --fetch
(or an empty cache).

    python studies/432-hull-moving-average/examples/verify.py            # cache-only
    python studies/432-hull-moving-average/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hull_moving_average import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "XLE"]
CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))
ASOF = "2026-06-13"


def _load(t: str, fetch: bool):
    b = data.load_real(t, fetch=fetch, cache_dir=CACHE)
    return b[b.index <= ASOF]


def main(fetch: bool) -> None:
    if fetch:  # warm the cache + stamp fingerprints
        for t in TICKERS:
            b = _load(t, fetch=True)
            b.to_parquet(data._cache_path(t, CACHE))
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} "
                  f"n={len(b)} fp={data.fingerprint(b)}")
        print()

    spy = _load("SPY", False)
    res = st.run_experiment(spy, hma_period=16, sma_period=50, cost_bps=2.0)

    print("=== SPY headline (long/flat, cost=2bps) ===")
    for k in ("HMA", "SMA", "MACD"):
        r = res[k]
        print(f"{k:5s} Sharpe_net={r['sharpe_net']:+.3f} CAGR={r['cagr_net']*100:+.2f}% "
              f"DD={r['maxdd_net']*100:.1f}% spread={r['mean_spread_bps']:+.2f}bps "
              f"t={r['spread_t']:+.2f} switches/yr={r['switches_per_yr']:.1f} "
              f"tim={r['time_in_market']:.2f}")
    h = res["HMA"]
    print(f"B&H   Sharpe_net={h['bh_sharpe']:+.3f} CAGR={h['bh_cagr']*100:+.2f}% "
          f"DD={h['bh_maxdd']*100:.1f}%")

    print("\n=== Whipsaw count (switches/yr) ===")
    for k in ("HMA", "SMA", "MACD"):
        print(f"{k:5s} {res[k]['switches_per_yr']:.1f} switches/yr")

    print("\n=== Permutation placebo (HMA, gross spread) ===")
    p = res["HMA_permutation"]
    print(f"observed={p['observed_spread_bps']:+.2f}bps placebo_mean="
          f"{p['placebo_mean_bps']:+.2f}bps p={p['p_value']:.4f}")

    print("\n=== SPY HMA cost sweep (net) ===")
    for c in (0.0, 1.0, 2.0, 5.0, 10.0):
        r = st.run_experiment(spy, cost_bps=c)["HMA"]
        print(f"cost={c:4.1f}bps Sharpe_net={r['sharpe_net']:+.3f} "
              f"spread={r['mean_spread_bps']:+.2f}bps t={r['spread_t']:+.2f}")

    print("\n=== Per-instrument (HMA long/flat, cost=2bps) ===")
    for t in TICKERS:
        b = _load(t, False)
        r = st.run_experiment(b, cost_bps=2.0)["HMA"]
        print(f"{t:5s} HMA_Sharpe={r['sharpe_net']:+.3f} B&H={r['bh_sharpe']:+.3f} "
              f"spread={r['mean_spread_bps']:+.2f}bps t={r['spread_t']:+.2f}")

    print("\n=== SPY in/out split (HMA, cost=2bps) ===")
    half = len(spy) // 2
    for lbl, seg in (("H1", spy.iloc[:half]), ("H2", spy.iloc[half:])):
        r = st.run_experiment(seg, cost_bps=2.0)["HMA"]
        print(f"{lbl} HMA_Sharpe={r['sharpe_net']:+.3f} B&H={r['bh_sharpe']:+.3f} "
              f"spread={r['mean_spread_bps']:+.2f}bps t={r['spread_t']:+.2f}")

    print("\n=== SPY HMA long/short (cost=2bps, borrow 50bps/yr) ===")
    r = st.run_experiment(spy, cost_bps=2.0, long_short=True)["HMA"]
    print(f"Sharpe_net={r['sharpe_net']:+.3f} B&H={r['bh_sharpe']:+.3f} "
          f"spread={r['mean_spread_bps']:+.2f}bps t={r['spread_t']:+.2f}")

    print("\n=== Synthetic positive control (HMA spread vs planted edge) ===")
    for e in (0.0, 0.3, 0.6, 1.0):
        b, _ = data.synthetic_panel(n_days=4000, edge=e, seed=432)
        r = st.run_experiment(b, cost_bps=0.0)["HMA"]
        print(f"edge={e:.1f} HMA_spread={r['mean_spread_bps']:+.2f}bps "
              f"t={r['spread_t']:+.2f} Sharpe_net={r['sharpe_net']:+.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
