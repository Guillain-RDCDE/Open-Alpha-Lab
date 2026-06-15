"""Offline, deterministic demo — Study 171 (Naive-1-Over-N).

No network. Builds a synthetic multi-asset return matrix and shows the study's spine:
the Markowitz optimiser beats 1/N *only when given an oracle estimate* (large signal,
enough data), and fails to beat it under realistic estimation noise. Run:

    python studies/171-naive-1-over-n/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from naive_1_over_n import data, strategy as st  # noqa: E402


def run(signal_strength: float, lookback: int, n_periods: int, seed: int) -> dict:
    """Run the four-strategy tournament on a synthetic tape.

    ``synthetic_returns`` already produces return observations (not prices) --
    do NOT call .pct_change() on the output.
    """
    df, _ = data.synthetic_returns(
        n_periods=n_periods, signal_strength=signal_strength, seed=seed
    )
    result = st.run_all_strategies(df, lookback=lookback, cost_bps=0)
    out = {}
    for col in result.columns:
        m = st.summarize(result[col])
        out[col] = m
    return out


def main() -> None:
    print("Study 171 -- Naive-1-Over-N -- synthetic spine\n")
    print("=" * 72)
    print("SCENARIO A: null (signal_strength=0) -- 1/N should be competitive")
    print("  n_periods=1260 (~5yr), lookback=24m => ~3 out-of-sample years")
    print("-" * 72)
    # n_periods=1260 ~5yr trading days; lookback=24 months leaves ~3yr OOS
    stats = run(signal_strength=0.0, lookback=24, n_periods=1260, seed=171)
    print(f"{'Strategy':>12} | {'CAGR%':>7} | {'Sharpe':>7} | {'MaxDD%':>7}")
    print("-" * 46)
    for strat, m in stats.items():
        cagr = m["cagr"] * 100 if m["cagr"] == m["cagr"] else float("nan")
        sharpe = m["sharpe"]
        mdd = m["max_dd"] * 100 if m["max_dd"] == m["max_dd"] else float("nan")
        print(f"{strat:>12} | {cagr:>7.2f} | {sharpe:>7.3f} | {mdd:>7.2f}")

    print()
    print("SCENARIO B: oracle-style (signal_strength=0.003, lookback=60m, 12yr)")
    print("  n_periods=3024 (~12yr); lookback=60m leaves ~7yr OOS")
    print("-" * 72)
    stats2 = run(signal_strength=0.003, lookback=60, n_periods=3024, seed=42)
    print(f"{'Strategy':>12} | {'CAGR%':>7} | {'Sharpe':>7} | {'MaxDD%':>7}")
    print("-" * 46)
    for strat, m in stats2.items():
        cagr = m["cagr"] * 100 if m["cagr"] == m["cagr"] else float("nan")
        sharpe = m["sharpe"]
        mdd = m["max_dd"] * 100 if m["max_dd"] == m["max_dd"] else float("nan")
        print(f"{strat:>12} | {cagr:>7.2f} | {sharpe:>7.3f} | {mdd:>7.2f}")

    print()
    print("Key insight: under null (A) the optimiser cannot beat 1/N -- estimation")
    print("error dominates. With enough data and a true signal (B), max-Sharpe can")
    print("outperform. The real-tape result mirrors scenario A (MIXED: sometimes 1/N")
    print("wins on Sharpe, sometimes max-Sharpe wins on return, never convincingly).")


if __name__ == "__main__":
    main()
