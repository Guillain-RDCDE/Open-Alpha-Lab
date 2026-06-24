"""Real-tape verification — Study 425 (Detrended Price Oscillator).

Reproduces every number in docs/results.md from the cached daily parquets. Builds the
look-ahead-free DPO long/flat timing rule, races net excess-of-cash Sharpe vs buy-and-hold,
computes the HAC edge t and a circular-shift permutation placebo, sweeps costs and periods,
benchmarks against SMA-cross and MACD, and runs the synthetic planted-cycle positive control.
Network is touched only with --fetch.

    python studies/425-detrended-price-oscillator/examples/verify.py            # cache-only
    python studies/425-detrended-price-oscillator/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detrended_price_oscillator import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM", "EFA", "GLD", "DBC"]
CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))
AS_OF = "2026-05-31"


def main(fetch: bool) -> None:
    if fetch:  # warm the cache + stamp fingerprints
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True, cache_dir=CACHE)
            print(f"{t:4s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")
        print()

    print(f"=== Per-instrument long/flat DPO(20), band_k=1, one-day lag, 1bp, excess-of-cash "
          f"(as-of {AS_OF}) ===")
    for t in TICKERS:
        b = data.load_real(t, cache_dir=CACHE, end=AS_OF)
        s = st.run_experiment(b["close"], period=20, band_k=1.0, cost_bps=1.0, n_perm=2000)
        print(f"{t:4s} | {b.index[0].date()}->{b.index[-1].date()} fp={data.fingerprint(b)} | "
              f"n={s['n_days']} expo={s['exposure']:.2f} tpy={s['trades_per_yr']:.1f} "
              f"DPO_Sh={s['strat_sharpe']:+.2f} BH_Sh={s['mkt_sharpe']:+.2f} dSh={s['sharpe_delta']:+.2f} "
              f"ann_edge={s['ann_edge']*100:+.2f}% edge_t={s['edge_t']:+.2f} perm_p={s['perm_p']:.3f}")

    cl = data.load_real("SPY", cache_dir=CACHE, end=AS_OF)["close"]

    print("\n=== Cost sweep (SPY long/flat, period 20) ===")
    for c in (0.0, 1.0, 2.0, 5.0):
        s = st.run_experiment(cl, period=20, cost_bps=c, n_perm=1)
        print(f"cost={c:>3.1f}bps  dSh={s['sharpe_delta']:+.2f}  "
              f"ann_edge={s['ann_edge']*100:+.2f}%  edge_t={s['edge_t']:+.2f}")

    print("\n=== Period sweep (SPY long/flat, cost 1bp) ===")
    for p in (10, 20, 30, 50):
        s = st.run_experiment(cl, period=p, cost_bps=1.0, n_perm=1)
        print(f"period={p:2d}  dSh={s['sharpe_delta']:+.2f}  "
              f"ann_edge={s['ann_edge']*100:+.2f}%  edge_t={s['edge_t']:+.2f}  expo={s['exposure']:.2f}")

    print("\n=== Benchmark race (SPY long/flat, cost 1bp) ===")
    s = st.run_experiment(cl, period=20, cost_bps=1.0, n_perm=1)
    print(f"DPO        dSh={s['sharpe_delta']:+.2f}  edge_t={s['edge_t']:+.2f}")
    for k, v in s["bench"].items():
        print(f"{k:10s} dSh={v['sharpe_delta']:+.2f}  edge_t={v['edge_t']:+.2f}")

    print("\n=== Look-ahead trap (SPY long/short, cost 0): honest vs centered/peeking ===")
    import numpy as np
    import pandas as pd
    honest = st.dpo_position(cl, period=20, band_k=1.0, long_short=True)
    hb = st.summarize_book(st.book_returns(cl, honest, cost_bps=0.0))
    dpo_c = st.dpo_centered(cl, 20); band = dpo_c.rolling(20).std()
    pp = pd.Series(0.0, index=cl.index); state = 0
    dv, bv = dpo_c.to_numpy(), band.to_numpy()
    for i in range(len(cl)):
        if np.isnan(dv[i]) or np.isnan(bv[i]):
            pp.iloc[i] = 0; continue
        if dv[i] < -bv[i]:
            state = 1
        elif dv[i] > bv[i]:
            state = -1
        pp.iloc[i] = state
    pb = st.summarize_book(st.book_returns(cl, pp, cost_bps=0.0))
    print(f"honest   dSh={hb['sharpe_delta']:+.2f}  ann_edge={hb['ann_edge']*100:+.2f}%  edge_t={hb['edge_t']:+.2f}")
    print(f"peeking  dSh={pb['sharpe_delta']:+.2f}  ann_edge={pb['ann_edge']*100:+.2f}%  edge_t={pb['edge_t']:+.2f}")

    print("\n=== Synthetic positive control (long/short, cost 0, period 20, cycle 20) ===")
    for e in (0.0, 0.02, 0.05, 0.10):
        bb, _ = data.synthetic_panel(n_days=4000, edge=e, cycle_period=20, seed=425)
        s = st.run_experiment(bb["close"], period=20, band_k=1.0, cost_bps=0.0,
                              long_short=True, n_perm=1)
        print(f"edge={e:.2f}  dSh={s['sharpe_delta']:+.2f}  "
              f"ann_edge={s['ann_edge']*100:+.2f}%  edge_t={s['edge_t']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
