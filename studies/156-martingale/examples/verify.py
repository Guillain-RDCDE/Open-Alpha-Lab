"""Real-tape verification — Study 156 (Martingale). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the real tickers, runs the martingale strategy against
buy-and-hold, sweeps costs and capital levels, and shows the ruin/skew profile.
Network is touched only with --fetch.

    python studies/156-martingale/examples/verify.py            # cache-only
    python studies/156-martingale/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from martingale import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "AAPL", "GE"]
STEP = 0.05
TP = 0.05
MAX_LEVELS = 6
COST = 5.0  # bps per unit


def _load(ticker: str, fetch: bool) -> pd.DataFrame:
    return data.fetch_daily(ticker, start="1993-01-01", fetch=fetch)


def _run_one(ticker: str, fetch: bool) -> tuple[pd.DataFrame, dict]:
    prices = _load(ticker, fetch)
    ledger = st.run_episodes(
        prices["close"], step_pct=STEP, take_profit_pct=TP, max_levels=MAX_LEVELS, cost_bps=COST
    )
    s = st.summarize(ledger, "excess_ret_net")
    return ledger, s


def main(fetch: bool) -> None:
    if fetch:
        print("=== Fetching real tapes ===")
        for t in TICKERS:
            b = _load(t, fetch=True)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")
        print()

    print(f"=== Martingale vs Buy-and-Hold (step={STEP*100:.0f}%, TP={TP*100:.0f}%, "
          f"max_levels={MAX_LEVELS}, cost={COST}bps) ===\n")
    print(f"{'ticker':>6} | {'episodes':>8} | {'ruin%':>6} | {'win%':>6} | "
          f"{'excess bps':>10} | {'skew':>6} | {'t':>7}")
    print("-" * 65)

    all_ledgers = []
    for t in TICKERS:
        try:
            ledger, s = _run_one(t, False)
            all_ledgers.append(ledger)
            print(f"{t:>6} | {s['n_episodes']:>8d} | "
                  f"{s['ruin_rate'] * 100:>5.1f}% | "
                  f"{s['win_rate'] * 100:>5.1f}% | "
                  f"{s['mean_bps']:>+10.1f} | "
                  f"{s['skew']:>+6.2f} | "
                  f"{s['tstat']:>+7.2f}")
        except FileNotFoundError:
            print(f"{t:>6} | no cache (run with --fetch)")

    if all_ledgers:
        pooled = pd.concat(all_ledgers, ignore_index=True)
        sp = st.summarize(pooled, "excess_ret_net")
        sm = st.summarize(pooled, "ret_net")
        sb = st.summarize(pooled, "bah_ret_net")
        print("-" * 65)
        print(f"{'POOLED':>6} | {sp['n_episodes']:>8d} | "
              f"{sp['ruin_rate'] * 100:>5.1f}% | "
              f"{sp['win_rate'] * 100:>5.1f}% | "
              f"{sp['mean_bps']:>+10.1f} | "
              f"{sp['skew']:>+6.2f} | "
              f"{sp['tstat']:>+7.2f}")

        print(f"\n  Martingale raw net: {sm['mean_bps']:+.1f} bps/episode, "
              f"skew={sm['skew']:+.3f}, t={sm['tstat']:+.2f}")
        print(f"  Buy-and-hold net:   {sb['mean_bps']:+.1f} bps/episode")
        print(f"  Excess (M-BaH):     {sp['mean_bps']:+.1f} bps/episode, t={sp['tstat']:+.2f}")

    print("\n=== Ruin probability sweep (Monte Carlo, 5,000 paths each) ===\n")
    print(f"{'max_levels':>12} | {'capital_units':>14} | {'ruin%':>8} | {'skew':>8}")
    print("-" * 52)
    for lvl in (3, 4, 5, 6, 7, 8):
        res = st.ruin_probability(
            annual_vol=0.18, mu=0.0, step_pct=STEP, take_profit_pct=TP,
            max_levels=lvl, n_paths=5000, seed=156
        )
        cap = 2**lvl - 1
        print(f"{lvl:>12d} | {cap:>14d} | "
              f"{res['ruin_rate'] * 100:>7.1f}% | "
              f"{res['skew']:>+8.3f}")

    print("\n=== Cost sensitivity (pooled, excess_ret_net) ===\n")
    print(f"{'cost_bps':>10} | {'excess bps':>10} | {'t':>7}")
    print("-" * 35)
    if all_ledgers:
        for cost in (0.0, 2.0, 5.0, 10.0):
            frames = []
            for t in TICKERS:
                try:
                    prices = _load(t, False)
                    ledger = st.run_episodes(
                        prices["close"], step_pct=STEP, take_profit_pct=TP,
                        max_levels=MAX_LEVELS, cost_bps=cost
                    )
                    frames.append(ledger)
                except FileNotFoundError:
                    pass
            if frames:
                pool = pd.concat(frames, ignore_index=True)
                sc = st.summarize(pool, "excess_ret_net")
                print(f"{cost:>10.1f} | {sc['mean_bps']:>+10.1f} | {sc['tstat']:>+7.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
