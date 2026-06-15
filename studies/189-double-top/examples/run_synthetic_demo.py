"""Offline, deterministic demo — Study 189 (Double-Top / Double-Bottom).

No network.  Builds a synthetic daily tape and shows the study's spine in one
screen: the double-top / double-bottom detector only edges a random-day placebo
when the tape actually carries mean-reversion — and on a martingale (reversal=0)
it is a fair coin.  Run:

    python studies/189-double-top/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from double_top import data, strategy as st  # noqa: E402


def _edge(reversal: float, seed: int = 189) -> tuple[float, float, int]:
    """Return (signal_mean_bps, placebo_mean_bps, n_signals) on a synthetic tape."""
    bars, _ = data.synthetic_daily(n_days=2000, reversal=reversal, seed=seed)
    study = st.run_pattern_study(
        bars, horizons=(5,), cost_bps=0.0, seed=0,
        min_bars=5, tolerance=0.10, lookback=60, atr_prominence_mult=0.1,
    )
    sig_mean = rand_mean = 0.0
    n_total = 0
    for df in study.values():
        n = len(df)
        n_total += n
        sig_mean += df["ret_signal_5d"].mean() * n
        rand_mean += df["ret_random_5d"].mean() * n
    if n_total > 0:
        sig_mean /= n_total
        rand_mean /= n_total
    return sig_mean * 1e4, rand_mean * 1e4, n_total


def main() -> None:
    print("Study 189 — Double-Top / Double-Bottom — synthetic positive control")
    print("(5-day forward return, gross, pooled across both patterns)\n")
    print(f"{'planted reversal':>17} | {'n signals':>9} | "
          f"{'signal bps':>10} | {'placebo bps':>11} | beats placebo?")
    print("-" * 73)
    for rev in (0.00, 0.10, 0.20, 0.30, -0.10):
        sig, rand, n = _edge(rev)
        beats = "yes" if sig > rand + 0.5 else "no"
        print(f"{rev:>17.2f} | {n:>9d} | {sig:>+10.2f} | {rand:>+11.2f} | {beats}")

    print()
    print("The detector is a faithful mean-reversion harvester: it only edges the")
    print("placebo when reversion is planted.  On a martingale (0.00) it reads noise.")
    print("On the real daily tape there is no such reliable short-term reversion")
    print("after pattern confirmation — see docs/results.md.")


if __name__ == "__main__":
    main()
