"""Real-tape verification — Study 183 (Fisher-Transform). Regenerates docs/results.md numbers.

Fetches (or reads from cache) SPY, QQQ, IWM, AAPL, TSLA, NVDA daily tapes, runs the
Fisher/Trigger crossover on a 5-day forward-return backtest against a random-direction
control, sweeps holding periods and costs, and verifies the monotonicity claim on real data.

    python studies/183-fisher-transform/examples/verify.py            # cache-only
    python studies/183-fisher-transform/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fisher_transform import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA"]

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))


def _pool(
    fetch: bool,
    hold_days: int,
    cost: float,
    period: int = 10,
    seed: int | None = None,
) -> pd.DataFrame:
    frames = []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=fetch, cache_dir=CACHE_DIR)
        ft = st.fisher_transform(bars, period=period)
        ent = st.crossover_entries(ft)
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(bars, ent, hold_days=hold_days, cost_bps=cost,
                                    directions=dirs))
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True, cache_dir=CACHE_DIR)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")
        print()

    print("=== Monotonicity verification (Fisher vs raw-x crossovers) ===")
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=False, cache_dir=CACHE_DIR)
        ft = st.fisher_transform(bars, period=10)
        agree = st.fisher_vs_raw_agreement(ft)
        print(f"{t:5s}  fisher_in_raw={agree['fisher_in_raw']:.4f}  "
              f"raw_in_fisher={agree['raw_in_fisher']:.4f}  "
              f"n_fisher={agree['n_fisher']}  n_raw={agree['n_raw']}")

    print()
    print("=== Primary signal: Fisher/Trigger crossover, 5-day forward return (gross) ===")
    cross = st.summarize(_pool(False, 5, 0), "ret_gross")
    rand = st.summarize(_pool(False, 5, 0, seed=183), "ret_gross")
    print(f"FISHER  n={cross['n_trades']} win={cross['win_rate']:.3f} "
          f"mean={cross['mean_bps']:+.2f}bps t={cross['tstat']:+.2f}")
    print(f"RANDOM  n={rand['n_trades']} win={rand['win_rate']:.3f} "
          f"mean={rand['mean_bps']:+.2f}bps t={rand['tstat']:+.2f}")

    print()
    print("=== Per-instrument breakdown (5-day, gross, Fisher) ===")
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=False, cache_dir=CACHE_DIR)
        ft = st.fisher_transform(bars, period=10)
        ent = st.crossover_entries(ft)
        s = st.summarize(st.run_trades(bars, ent, hold_days=5, cost_bps=0), "ret_gross")
        print(f"{t:5s} n={s['n_trades']} win={s['win_rate']:.3f} "
              f"mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print()
    print("=== Holding-period sweep (pooled, gross, Fisher) ===")
    for hd in (1, 3, 5, 10, 20):
        s = st.summarize(_pool(False, hd, 0), "ret_gross")
        print(f"hold={hd:2d}d  n={s['n_trades']} mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print()
    print("=== Cost sweep (5-day hold, net) ===")
    for c in (0.0, 0.5, 1.0, 2.0, 5.0):
        s = st.summarize(_pool(False, 5, c), "ret_net")
        print(f"cost={c:4.1f}bps net mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
