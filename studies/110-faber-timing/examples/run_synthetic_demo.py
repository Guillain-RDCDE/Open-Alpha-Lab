"""Offline, deterministic demo — Study 110 (Faber-Timing).

No network. Builds a synthetic daily price tape and shows the study's spine:
- On a two-regime tape (bull/bear separation), the 200-day SMA timing rule reduces
  max drawdown substantially relative to buy-and-hold.
- It also beats a random-timing control in terms of drawdown reduction, confirming
  the SMA has genuine regime-detection value, not merely reduced exposure.
- The Sharpe advantage is smaller and less reliable — the rule earns less in bull
  markets (sits in T-bills too long), so timing skill shows primarily in the
  *risk* dimension, not the *return* dimension.

Run:
    python studies/110-faber-timing/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from faber_timing import data, strategy as st  # noqa: E402


def _row(label: str, s: dict) -> str:
    return (
        f"  {label:20s} | CAGR={s['cagr']:+.1%}  Sharpe={s['sharpe']:+.2f}"
        f"  MaxDD={s['max_drawdown']:+.1%}  vol={s['vol_ann']:.1%}"
    )


def main() -> None:
    print("Study 110 — Faber-Timing — synthetic positive / negative control\n")
    for strength, label in [(1.0, "TWO-REGIME tape (signal_strength=1.0)"),
                             (0.0, "FLAT-VOL tape   (signal_strength=0.0, null)")]:
        prices, truth = data.synthetic_daily(n_years=30, signal_strength=strength, seed=110)
        res = st.compare_strategies(prices["close"], tbill_daily=0.04 / 252,
                                     sma_n=200, cost_bps=5.0)
        print(f"  {label}  |  in-market frac: {res['in_market_frac']:.1%}")
        print(_row("buy-and-hold", res["bh"]))
        print(_row("SMA 200-day timing", res["timing"]))
        print(_row("random-timing (null)", res["random"]))
        dd_improvement = res["bh"]["max_drawdown"] - res["timing"]["max_drawdown"]
        sharpe_diff = res["timing"]["sharpe"] - res["bh"]["sharpe"]
        print(f"  => DD improvement vs BH: {dd_improvement:+.1%}  |"
              f"  Sharpe delta vs BH: {sharpe_diff:+.2f}\n")

    print("Interpretation:")
    print("  - Two-regime tape: SMA timing materially cuts drawdown; Sharpe effect is mixed.")
    print("  - Flat-vol tape: the rule has no reliable advantage on any dimension.")
    print("  - This mirrors the real-tape result: real Sharpe improvement is at best WEAK,")
    print("    but drawdown reduction vs BH is economically large and genuine (Signal=REAL-on-risk).")
    print("\nSee docs/results.md for the real-data numbers.")


if __name__ == "__main__":
    main()
